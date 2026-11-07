import argparse
import json
import re
from pathlib import Path

from scripts.retained_ifs_repair import Manifest, file_digest, require


def job_definition(
    *,
    name: str,
    namespace: str,
    image: str,
    manifest: Manifest,
    manifest_sha256: str,
    input_claim: str,
    output_claim: str,
    memory: str,
    shared_memory: str,
    scratch: str,
    storage_secret: str | None,
    commit: bool,
    capacity: str,
) -> dict:
    require(
        bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,61}[a-z0-9]", name)), "Invalid Job name"
    )
    require(
        bool(re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image)),
        "Image must be pinned by digest",
    )
    require(
        bool(re.fullmatch(r"[a-f0-9]{64}", manifest_sha256)), "Invalid manifest hash"
    )
    require(input_claim != output_claim, "Input and output volumes must be distinct")
    require(
        bool(storage_secret) == commit,
        "Only commit mode requires an explicitly mounted storage secret",
    )
    require(
        bool(
            namespace
            and input_claim
            and output_claim
            and memory
            and shared_memory
            and scratch
        ),
        "Missing Job resources or volume",
    )
    require(capacity in {"spot", "on-demand"}, "Invalid capacity choice")
    require(
        not commit or capacity == "on-demand", "Commit Job requires on-demand capacity"
    )
    command = [
        "python",
        "-m",
        "scripts.retained_ifs_repair",
        "--manifest",
        "/inputs/manifest.json",
        "--manifest-sha256",
        manifest_sha256,
        "--work-dir",
        f"/work/{name}",
    ]
    if storage_secret:
        command.extend(["--auth", "secret", "--k8s-secret", storage_secret])
    if commit:
        command.append("--commit-to-new-branch")
    volumes = [
        {
            "name": "inputs",
            "persistentVolumeClaim": {"claimName": input_claim, "readOnly": True},
        },
        {"name": "output", "persistentVolumeClaim": {"claimName": output_claim}},
        {
            "name": "shared",
            "emptyDir": {"medium": "Memory", "sizeLimit": shared_memory},
        },
    ]
    mounts = [
        {"name": "inputs", "mountPath": "/inputs", "readOnly": True},
        {"name": "output", "mountPath": "/work"},
        {"name": "shared", "mountPath": "/dev/shm"},  # noqa: S108,
    ]
    if storage_secret:
        volumes.append(
            {
                "name": "storage",
                "secret": {
                    "secretName": storage_secret,
                    "items": [{"key": "contents", "path": f"{storage_secret}.json"}],
                },
            }
        )
        mounts.append({"name": "storage", "mountPath": "/secrets", "readOnly": True})
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": {
                "repair-manifest-sha256": manifest_sha256,
                "repair-base-snapshot": manifest.base_snapshot,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "activeDeadlineSeconds": 86400,
            "template": {
                "metadata": {"annotations": {"karpenter.sh/do-not-disrupt": "true"}},
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": {
                        "eks.amazonaws.com/compute-type": "auto",
                        "karpenter.sh/capacity-type": capacity,
                    },
                    "automountServiceAccountToken": False,
                    "securityContext": {
                        "fsGroup": 999,
                        "runAsNonRoot": True,
                        "runAsUser": 999,
                        "runAsGroup": 999,
                    },
                    "containers": [
                        {
                            "name": "repair",
                            "image": image,
                            "command": command,
                            "env": [
                                {"name": "DYNAMICAL_ENV", "value": "prod"},
                                {"name": "PYTHONPATH", "value": "/app/src"},
                                {"name": "JOB_NAME", "value": name},
                            ]
                            + [
                                {"name": key, "value": "1"}
                                for key in (
                                    "OMP_NUM_THREADS",
                                    "OPENBLAS_NUM_THREADS",
                                    "MKL_NUM_THREADS",
                                    "NUMBA_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS",
                                )
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": memory,
                                    "ephemeral-storage": scratch,
                                },
                                "limits": {
                                    "cpu": "4",
                                    "memory": memory,
                                    "ephemeral-storage": scratch,
                                },
                            },
                            "volumeMounts": mounts,
                        }
                    ],
                    "volumes": volumes,
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a repair Job without submitting it"
    )
    for flag in (
        "name",
        "namespace",
        "image",
        "input-claim",
        "output-claim",
        "memory",
        "shared-memory",
        "scratch",
    ):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--storage-secret")
    parser.add_argument("--capacity", choices=("spot", "on-demand"), required=True)
    parser.add_argument("--commit-to-new-branch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.manifest.name == "manifest.json",
        "Mounted input must be named manifest.json",
    )
    manifest = Manifest.model_validate_json(args.manifest.read_text())
    definition = job_definition(
        name=args.name,
        namespace=args.namespace,
        image=args.image,
        manifest=manifest,
        manifest_sha256=file_digest(args.manifest),
        input_claim=args.input_claim,
        output_claim=args.output_claim,
        memory=args.memory,
        shared_memory=args.shared_memory,
        scratch=args.scratch,
        storage_secret=args.storage_secret,
        commit=args.commit_to_new_branch,
        capacity=args.capacity,
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(definition, indent=2) + "\n")


if __name__ == "__main__":
    main()
