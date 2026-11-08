import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import icechunk

from reformatters.common.logging import get_logger
from scripts.retained_ifs_patch_job import THREADS, job_definition
from scripts.retained_ifs_patch_verify import check, validate_manifest, verify
from scripts.retained_ifs_repair import runtime_digest

log = get_logger(__name__)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def execute(argv: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(  # noqa: S603
        argv, cwd=cwd, check=True, text=True, capture_output=True
    ).stdout


def save(path: Path, value: object) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def load_bundle(path: Path, expected: str) -> dict[str, Any]:
    body = path.read_bytes()
    check(sha(body) == expected, "Handoff bundle hash mismatch")
    bundle = json.loads(body)
    check(bundle["version"] == 1, "Unknown bundle version")
    check(
        bool(re.fullmatch(r"[0-9a-f]{40}", bundle["commit"])), "Invalid source commit"
    )
    check(bundle["architecture"] == "arm64", "This handoff targets arm64")
    check(bundle["capacity"] == "spot", "This handoff targets spot")
    check(
        bool(re.fullmatch(r"[A-Za-z0-9./_-]+:[A-Za-z0-9_.-]+", bundle["image_tag"])),
        "Invalid image tag",
    )
    validate_manifest(bundle["manifest"])
    check(
        sha(canonical(bundle["manifest"])) == bundle["manifest_sha256"],
        "Bundle manifest hash mismatch",
    )
    for name in ("implementation_review", "independent_stage"):
        pin = bundle[name]
        evidence_path = (path.parent / pin["path"]).resolve()
        check(
            evidence_path.is_relative_to(path.parent.resolve()),
            "Evidence path escapes bundle",
        )
        evidence = evidence_path.read_bytes()
        check(sha(evidence) == pin["sha256"], f"Evidence hash mismatch: {name}")
        if name == "independent_stage":
            receipt = json.loads(evidence)
            check(
                receipt["status"] == "pass-local-stage"
                and receipt["base_snapshot"] == bundle["manifest"]["base_snapshot"]
                and receipt["approved_shards"] == bundle["manifest"]["approved_shards"],
                "Independent stage receipt binding mismatch",
            )
    check(
        isinstance(bundle["kube_context"], str), "Invalid Kubernetes context selector"
    )
    check(
        bool(re.fullmatch(r"[a-f0-9]{64}", bundle["kube_server_sha256"])),
        "Missing approved cluster endpoint hash",
    )
    check(bundle["expected_seconds"] > 0, "Missing measured duration")
    # Validate the complete Job before any image push or submission.
    render(bundle, "sha256:" + "0" * 64)
    return bundle


def render(bundle: dict[str, Any], image_digest: str) -> dict[str, Any]:
    repository = bundle["image_tag"].rsplit(":", 1)[0]
    return job_definition(
        name=bundle["job_name"],
        namespace=bundle["namespace"],
        image=f"{repository}@{image_digest}",
        manifest=bundle["manifest"],
        manifest_sha256=bundle["manifest_sha256"],
        memory=bundle["memory"],
        scratch=bundle["scratch"],
        deadline_seconds=bundle["deadline_seconds"],
        storage_secret=bundle["storage_secret"],
        architecture=bundle["architecture"],
        capacity=bundle["capacity"],
    )


def anonymous_repo(manifest: dict[str, Any]) -> icechunk.Repository:
    url = urlparse(manifest["repository"])
    return icechunk.Repository.open(
        icechunk.s3_storage(
            bucket=url.netloc, prefix=url.path.strip("/"), anonymous=True
        )
    )


def fresh_launch_checks(bundle: dict[str, Any]) -> None:
    server = execute(
        [
            "kubectl",
            "config",
            "view",
            "--minify",
            "--context",
            bundle["kube_context"],
            "-o",
            "jsonpath={.clusters[0].cluster.server}",
        ]
    ).strip()
    check(
        server.startswith("https://")
        and sha(server.encode()) == bundle["kube_server_sha256"],
        "Kubernetes context points to a different cluster",
    )
    manifest = bundle["manifest"]
    repo = anonymous_repo(manifest)
    check(
        repo.lookup_branch(manifest["protected_branch"]) == manifest["base_snapshot"],
        "Protected ref moved",
    )
    check(
        manifest["candidate_branch"] not in repo.list_branches(),
        "Candidate already exists; no retry",
    )
    existing = execute(
        [
            "kubectl",
            "get",
            "job",
            bundle["job_name"],
            "--namespace",
            bundle["namespace"],
            "--context",
            bundle["kube_context"],
            "--ignore-not-found",
            "-o",
            "json",
        ]
    )
    check(not existing.strip(), "Job already exists; no retry")


def parity_command(bundle: dict[str, Any], image_id: str, work: Path) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/arm64",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--cpus=4",
        "--memory",
        bundle["memory"],
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={work},dst=/evidence",
        "--tmpfs",
        "/tmp:rw,nosuid,size=256m",  # noqa: S108
        "--env",
        "PYTHONPATH=/app/src",
        "--env",
        "NUMBA_CACHE_DIR=/evidence/numba-cache",
    ]
    for name in THREADS:
        command.extend(["--env", f"{name}=1"])
    command.extend(
        [
            "--entrypoint",
            "python",
            image_id,
            "-m",
            "scripts.retained_ifs_patch",
            "--manifest",
            "/evidence/manifest.json",
            "--manifest-sha256",
            bundle["manifest_sha256"],
            "--work-dir",
            "/evidence/source-check",
            "--check-sources-only",
        ]
    )
    return command


def launch(
    bundle: dict[str, Any], work: Path, checkout: Path, *, dry_run: bool
) -> dict[str, Any]:
    if dry_run:
        log.info(
            "DRY RUN: validate bundle; archive pinned commit; build arm64 locally; check eight source messages/seven field hashes; push; verify registry digest; check refs/Job; submit once. No commands executed."
        )
        log.info(
            f"Expected repair duration: {bundle['expected_seconds']} seconds; candidate {bundle['manifest']['candidate_branch']}"
        )
        return {"status": "dry-run", "external_commands": 0}
    check(
        work.resolve().is_relative_to(Path("/tmp")),  # noqa: S108
        "Source payload evidence must remain under /tmp",
    )
    work.mkdir(parents=True, exist_ok=False)
    try:
        return launch_in_directory(bundle, work, checkout)
    except BaseException as error:
        save(
            work / "STOP.json",
            {
                "status": "stopped",
                "error_type": type(error).__name__,
                "message": str(error),
                "automatic_retry": False,
                "candidate": bundle["manifest"]["candidate_branch"],
            },
        )
        log.error(
            "STOP. No automatic retry or cleanup; inspect the candidate and Job read-only."
        )
        raise


def push_checked_image(
    bundle: dict[str, Any], image_id: str, config_digest: str
) -> str:
    log.info("Source check PASS; pushing the checked image")
    execute(["docker", "push", bundle["image_tag"]])
    repository = bundle["image_tag"].rsplit(":", 1)[0]
    pushed = json.loads(execute(["docker", "image", "inspect", image_id]))[0]
    digests = [
        value.split("@", 1)[1]
        for value in pushed["RepoDigests"]
        if value.startswith(repository + "@")
    ]
    check(len(digests) == 1, "Pushed image has no unique repository digest")
    image_digest = digests[0]
    check(
        bool(re.fullmatch(r"sha256:[a-f0-9]{64}", image_digest)),
        "Invalid registry digest",
    )
    pinned = execute(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            f"{repository}@{image_digest}",
            "--raw",
        ]
    )
    registry_manifest = json.loads(pinned)
    if "manifests" in registry_manifest:
        children = registry_manifest["manifests"]
        check(
            len(children) == 1
            and children[0]["platform"]["architecture"] == "arm64"
            and children[0]["platform"]["os"] == "linux",
            "Unexpected registry image index",
        )
        image_digest = children[0]["digest"]
        check(
            bool(re.fullmatch(r"sha256:[a-f0-9]{64}", image_digest)),
            "Invalid child digest",
        )
        registry_manifest = json.loads(
            execute(
                [
                    "docker",
                    "buildx",
                    "imagetools",
                    "inspect",
                    f"{repository}@{image_digest}",
                    "--raw",
                ]
            )
        )
    check(
        registry_manifest.get("config", {}).get("digest") == config_digest,
        "Registry image differs from checked local image",
    )
    return image_digest


def launch_in_directory(
    bundle: dict[str, Any], work: Path, checkout: Path
) -> dict[str, Any]:
    architecture = execute(["docker", "info", "--format", "{{.Architecture}}"]).strip()
    check(
        architecture in {"aarch64", "arm64"},
        "Launch requires a native arm64 Docker host for the light source check; no builder or emulator is installed automatically",
    )
    if not bundle["kube_context"]:
        bundle = bundle | {
            "kube_context": execute(["kubectl", "config", "current-context"]).strip()
        }
    check(bool(bundle["kube_context"]), "No active Kubernetes context")
    fresh_launch_checks(bundle)
    commit = bundle["commit"]
    check(
        execute(["git", "cat-file", "-t", commit], cwd=checkout).strip() == "commit",
        "Missing pinned source commit",
    )
    archive = work / "source.tar"
    execute(
        ["git", "archive", "--format=tar", "--output", str(archive), commit],
        cwd=checkout,
    )
    source = work / "source"
    source.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter="data")
    (work / "manifest.json").write_bytes(canonical(bundle["manifest"]))
    log.info("Building the pinned unmerged branch for arm64 locally; no push yet")
    execute(
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/arm64",
            "--provenance=false",
            "--sbom=false",
            "--load",
            "--metadata-file",
            str(work / "build-metadata.json"),
            "--label",
            f"org.opencontainers.image.revision={commit}",
            "--tag",
            bundle["image_tag"],
            "--file",
            str(source / "deploy/Dockerfile"),
            str(source),
        ]
    )
    info = json.loads(execute(["docker", "image", "inspect", bundle["image_tag"]]))[0]
    check(
        info["Architecture"] == "arm64" and info["Os"] == "linux",
        "Built image platform mismatch",
    )
    check(
        info["Config"]["Labels"]["org.opencontainers.image.revision"] == commit,
        "Image source label mismatch",
    )
    build_metadata = json.loads((work / "build-metadata.json").read_text())
    config_digest = build_metadata["containerimage.config.digest"]
    check(
        bool(re.fullmatch(r"sha256:[a-f0-9]{64}", config_digest)),
        "Missing build config digest",
    )
    image_id = info["Id"]
    check(
        image_id in {config_digest, build_metadata["containerimage.digest"]},
        "Loaded image differs from build metadata",
    )
    check(
        bool(re.fullmatch(r"sha256:[a-f0-9]{64}", image_id)),
        "Invalid local image identity",
    )
    log.info("Checking source field bit identity in the exact arm64 image (read-only)")
    execute(parity_command(bundle, image_id, work))
    receipt = json.loads((work / "source-check/result.json").read_text())
    check(receipt["status"] == "sources-verified", "Source bit-identity check failed")
    check(
        receipt["runtime_sha256"] == bundle["manifest"]["runtime_sha256"]
        and receipt["manifest_sha256"] == bundle["manifest_sha256"],
        "Source check binding mismatch",
    )
    check(receipt["machine"] in {"aarch64", "arm64"}, "Source check was not arm64")
    check(
        receipt["source_count"] == 8 and receipt["field_count"] == 7,
        "Incomplete source check",
    )
    check(
        json.loads(execute(["docker", "image", "inspect", bundle["image_tag"]]))[0][
            "Id"
        ]
        == image_id,
        "Image tag moved before push",
    )
    image_digest = push_checked_image(bundle, image_id, config_digest)
    repository = bundle["image_tag"].rsplit(":", 1)[0]
    definition = render(bundle, image_digest)
    save(work / "job.json", definition)
    save(
        work / "launch-binding.json",
        {
            "commit": commit,
            "image": f"{repository}@{image_digest}",
            "image_config_digest": config_digest,
            "local_image_id": image_id,
            "job_sha256": sha((work / "job.json").read_bytes()),
            "manifest_sha256": bundle["manifest_sha256"],
            "source_check_sha256": sha(
                (work / "source-check/result.json").read_bytes()
            ),
        },
    )
    fresh_launch_checks(bundle)
    log.info(
        f"Image {repository}@{image_digest}; Job SHA256 {sha((work / 'job.json').read_bytes())}; manifest SHA256 {bundle['manifest_sha256']}"
    )
    log.info("Fresh ref and Job checks PASS; submitting exactly one Job")
    response = json.loads(
        execute(
            [
                "kubectl",
                "create",
                "--context",
                bundle["kube_context"],
                "--filename",
                str(work / "job.json"),
                "-o",
                "json",
            ]
        )
    )
    check(response["metadata"]["name"] == bundle["job_name"], "Unexpected created Job")
    result = {
        "status": "launched",
        "job": bundle["job_name"],
        "namespace": bundle["namespace"],
        "kube_context": bundle["kube_context"],
        "uid": response["metadata"]["uid"],
        "candidate": bundle["manifest"]["candidate_branch"],
        "image": f"{repository}@{image_digest}",
        "expected_seconds": bundle["expected_seconds"],
    }
    save(work / "launch.json", result)
    log.info(
        f"LAUNCHED {result['job']} ({result['uid']}); run verify. Expected {result['expected_seconds']} seconds. No automatic retry."
    )
    return result


def verify_when_available(
    bundle: dict[str, Any], output: Path, wait_seconds: int
) -> dict[str, Any]:
    check(not output.exists(), "Verification output exists")
    manifest = bundle["manifest"]
    deadline = time.monotonic() + wait_seconds
    while True:
        repo = anonymous_repo(manifest)
        check(
            repo.lookup_branch(manifest["protected_branch"])
            == manifest["base_snapshot"],
            "Protected ref moved",
        )
        branches = repo.list_branches()
        snapshot = (
            repo.lookup_branch(manifest["candidate_branch"])
            if manifest["candidate_branch"] in branches
            else None
        )
        if snapshot is not None and snapshot != manifest["base_snapshot"]:
            break
        check(
            time.monotonic() < deadline,
            "INDETERMINATE: candidate absent or still at base; inspect Job, never auto-retry",
        )
        log.info("Waiting read-only for candidate commit")
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    result = verify(repo, manifest, snapshot)
    result["manifest_sha256"] = bundle["manifest_sha256"]
    save(output, result)
    log.info(
        f"PASS: independent full-unit verification at {snapshot}; protected branch unchanged"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Operator-only launch and anonymous verification of the retained patch"
    )
    parser.add_argument("action", choices=("launch", "verify"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approve-retention-and-shared-secret", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle, args.bundle_sha256)
    check(
        runtime_digest(Path(__file__).resolve().parents[2])
        == bundle["manifest"]["runtime_sha256"],
        "Operator code differs from approved runtime",
    )
    if args.action == "launch":
        check(args.work_dir is not None, "Launch requires a fresh --work-dir")
        check(
            args.dry_run or args.approve_retention_and_shared_secret,
            "Operator must approve retention and application-level isolation with the shared Secret",
        )
        launch(bundle, args.work_dir, args.checkout, dry_run=args.dry_run)
    else:
        check(args.output is not None, "Verify requires a fresh --output")
        verify_when_available(bundle, args.output, args.wait_seconds)


if __name__ == "__main__":
    main()
