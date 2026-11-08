import hashlib
import json
import re
from typing import Any

from scripts.retained_ifs_repair import require

BOOTSTRAP = (
    "import os;from pathlib import Path;"
    "Path('/tmp/repair-manifest.json').write_text(os.environ.pop('REPAIR_MANIFEST'));"
    "from scripts.retained_ifs_patch import main;main()"
)
THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GDAL_NUM_THREADS",
)


def job_definition(
    *,
    name: str,
    namespace: str,
    image: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    memory: str,
    scratch: str,
    deadline_seconds: int,
    storage_secret: str,
    architecture: str,
    capacity: str,
) -> dict[str, Any]:
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
    require(architecture in {"amd64", "arm64"}, "Invalid architecture")
    require(capacity in {"spot", "on-demand"}, "Invalid capacity")
    require(
        bool(namespace and memory and scratch and storage_secret),
        "Missing resource/Secret name",
    )
    require(deadline_seconds > 0, "Missing measured deadline")
    require(bool(manifest.get("approved_shards")), "Missing preapproved shard hashes")
    inline = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    require(
        hashlib.sha256(inline.encode()).hexdigest() == manifest_sha256,
        "Inline manifest hash mismatch",
    )
    require(len(inline.encode()) < 100_000, "Manifest exceeds environment size budget")
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": {
                "repair-manifest-sha256": manifest_sha256,
                "repair-base-snapshot": manifest["base_snapshot"],
            },
        },
        "spec": {
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "activeDeadlineSeconds": deadline_seconds,
            "template": {
                "metadata": {"annotations": {"karpenter.sh/do-not-disrupt": "true"}},
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "nodeSelector": {
                        "eks.amazonaws.com/compute-type": "auto",
                        "kubernetes.io/arch": architecture,
                        "karpenter.sh/capacity-type": capacity,
                    },
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
                            "command": [
                                "python",
                                "-c",
                                BOOTSTRAP,
                                "--manifest",
                                "/tmp/repair-manifest.json",  # noqa: S108
                                "--manifest-sha256",
                                manifest_sha256,
                                "--work-dir",
                                "/tmp/retained-patch",  # noqa: S108
                                "--commit-to-new-branch",
                                "--auth",
                                "secret",
                                "--k8s-secret",
                                storage_secret,
                            ],
                            "env": [
                                {"name": "REPAIR_MANIFEST", "value": inline},
                                {"name": "DYNAMICAL_ENV", "value": "prod"},
                                {"name": "PYTHONPATH", "value": "/app/src"},
                                {"name": "JOB_NAME", "value": name},
                            ]
                            + [{"name": key, "value": "1"} for key in THREADS],
                            "resources": {
                                side: {
                                    "cpu": "4",
                                    "memory": memory,
                                    "ephemeral-storage": scratch,
                                }
                                for side in ("requests", "limits")
                            },
                            "volumeMounts": [
                                {
                                    "name": "storage",
                                    "mountPath": "/secrets",
                                    "readOnly": True,
                                }
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "storage",
                            "secret": {
                                "secretName": storage_secret,
                                "items": [
                                    {
                                        "key": "contents",
                                        "path": f"{storage_secret}.json",
                                    }
                                ],
                            },
                        }
                    ],
                },
            },
        },
    }
