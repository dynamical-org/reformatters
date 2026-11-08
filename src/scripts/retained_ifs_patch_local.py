import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import icechunk

log = logging.getLogger(__name__)
THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GDAL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "BLOSC_NTHREADS",
    "RAYON_NUM_THREADS",
)
LOCK_ROOT = Path("/tmp/retained-ifs-patch-locks")  # noqa: S108
CHECKOUT = Path(__file__).resolve().parents[2]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def save(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def configure_resources() -> None:
    for name in THREADS:
        os.environ[name] = "1"
    os.setpriority(os.PRIO_PROCESS, 0, max(10, os.getpriority(os.PRIO_PROCESS, 0)))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])


def runtime_digest(root: Path) -> str:
    paths = [root / "uv.lock", root / "pyproject.toml"]
    paths.extend(sorted((root / "src").rglob("*.py")))
    return sha(
        canonical({str(p.relative_to(root)): sha(p.read_bytes()) for p in paths})
    )


def load_bundle(path: Path, expected: str) -> dict[str, Any]:
    from scripts.retained_ifs_patch_verify import validate_manifest  # noqa: PLC0415

    body = path.read_bytes()
    check(sha(body) == expected, "Bundle hash mismatch")
    bundle = json.loads(body)
    check(bundle["version"] == 2, "Expected local bundle version 2")
    check(bool(re.fullmatch(r"[0-9a-f]{40}", bundle["commit"])), "Invalid commit pin")
    manifest = bundle["manifest"]
    validate_manifest(manifest)
    check(manifest["version"] == 2, "Expected manifest version 2")
    check(
        sha(canonical(manifest)) == bundle["manifest_sha256"],
        "Manifest hash mismatch",
    )
    check(
        bool(re.fullmatch(r"[0-9a-f]{64}", manifest["runtime_sha256"])),
        "Invalid runtime pin",
    )
    check(bool(manifest["approved_shards"]), "Missing approved shards")
    check(
        bool(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", manifest["candidate_branch"]
            )
        )
        and manifest["candidate_branch"] not in {"main", manifest["protected_branch"]},
        "Invalid candidate branch",
    )
    url = urlparse(manifest["repository"])
    check(
        url.scheme == "s3" and bool(url.netloc) and bool(url.path.strip("/")),
        "Invalid repository",
    )
    for name in ("expected_seconds", "memory_bytes", "scratch_bytes"):
        check(type(bundle[name]) is int and bundle[name] > 0, f"Invalid {name}")
    for name in ("implementation_review", "independent_stage"):
        pin = bundle[name]
        relative = Path(pin["path"])
        evidence_path = (path.parent / relative).resolve()
        check(
            not relative.is_absolute()
            and evidence_path.is_relative_to(path.parent.resolve()),
            "Evidence must be relative and inside the bundle directory",
        )
        evidence = evidence_path.read_bytes()
        check(sha(evidence) == pin["sha256"], f"Evidence hash mismatch: {name}")
        if name == "independent_stage":
            receipt = json.loads(evidence)
            check(
                receipt["status"] == "pass-local-stage"
                and receipt["base_snapshot"] == manifest["base_snapshot"]
                and receipt["approved_shards"] == manifest["approved_shards"],
                "Independent stage binding mismatch",
            )
    return bundle


def child_environment(source: Path, attempt: Path, *, writable: bool) -> dict[str, str]:
    env = dict(os.environ) if writable else {"PATH": os.defpath, "LANG": "C.UTF-8"}
    env.update(dict.fromkeys(THREADS, "1"))
    env.update(
        PYTHONPATH=str(source / "src"),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        NUMBA_CACHE_DIR=str(attempt / "numba-cache"),
        TMPDIR=str(attempt / "tmp"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_NO_LAZY_FETCH="1",
    )
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONOPTIMIZE", None)
    return env


def archive_runtime(bundle: dict[str, Any], attempt: Path) -> Path:
    git = shutil.which("git")
    check(git is not None, "git is required")
    assert git is not None
    archive = attempt / "runtime.tar"
    env = child_environment(attempt / "runtime", attempt, writable=False)
    with (attempt / "git.log").open("xb") as errors:
        commit_type = subprocess.run(  # noqa: S603
            [git, "cat-file", "-t", bundle["commit"]],
            cwd=CHECKOUT,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=errors,
        ).stdout
        check(commit_type.strip() == b"commit", "Pin is not a git commit")
        with archive.open("xb") as stream:
            subprocess.run(  # noqa: S603
                [
                    git,
                    "archive",
                    "--format=tar",
                    bundle["commit"],
                    "src",
                    "uv.lock",
                    "pyproject.toml",
                ],
                cwd=CHECKOUT,
                env=env,
                check=True,
                stdout=stream,
                stderr=errors,
            )
    source = attempt / "runtime"
    source.mkdir()
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        for member in members:
            check(
                (member.isfile() or member.isdir())
                and not Path(member.name).is_absolute()
                and ".." not in Path(member.name).parts
                and (source / member.name).resolve().is_relative_to(source),
                "Unsafe git archive member",
            )
        stream.extractall(source, members=members, filter="data")
    check(
        runtime_digest(source) == bundle["manifest"]["runtime_sha256"],
        "Archived runtime hash mismatch",
    )
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    save(
        attempt / "runtime-binding.json",
        {
            "commit": bundle["commit"],
            "archive_sha256": sha(archive.read_bytes()),
            "runtime_sha256": runtime_digest(source),
        },
    )
    archive.chmod(0o444)
    return source


@contextmanager
def candidate_lock(manifest: dict[str, Any]) -> Generator[int]:
    LOCK_ROOT.mkdir(mode=0o700, exist_ok=True)
    info = LOCK_ROOT.lstat()
    check(
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == 0o700,
        "Unsafe local lock directory",
    )
    key = sha(canonical([manifest["repository"], manifest["candidate_branch"]]))
    fd = os.open(
        LOCK_ROOT / f"{key}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        # Closing preserves the lock while a surviving child holds the inherited fd.
        os.close(fd)


def bind_work(work: Path, bundle: dict[str, Any], bundle_sha256: str) -> None:
    check(work.is_relative_to(Path("/tmp")), "Work directory must be under /tmp")  # noqa: S108
    binding = {
        "version": 2,
        "bundle_sha256": bundle_sha256,
        "commit": bundle["commit"],
        "manifest_sha256": bundle["manifest_sha256"],
        "runtime_sha256": bundle["manifest"]["runtime_sha256"],
    }
    if work.exists():
        path = work / "binding.json"
        check(
            path.is_file() and not path.is_symlink(),
            "Existing work directory is unbound",
        )
        check(
            json.loads(path.read_bytes()) == binding, "Work directory binding changed"
        )
    else:
        work.mkdir(mode=0o700)
        save(work / "binding.json", binding)
        (work / "binding.json").chmod(0o444)


def anonymous_repo(manifest: dict[str, Any]) -> icechunk.Repository:
    import icechunk  # noqa: PLC0415

    url = urlparse(manifest["repository"])
    return icechunk.Repository.open(
        icechunk.s3_storage(
            bucket=url.netloc, prefix=url.path.strip("/"), anonymous=True
        )
    )


def candidate_snapshot(manifest: dict[str, Any]) -> str | None:
    repo = anonymous_repo(manifest)
    check(
        repo.lookup_branch(manifest["protected_branch"]) == manifest["base_snapshot"],
        "Protected ref moved",
    )
    branches = repo.list_branches()
    snapshot = (
        repo.lookup_branch(manifest["candidate_branch"])
        if manifest["candidate_branch"] in branches
        else None
    )
    check(
        snapshot != manifest["base_snapshot"],
        "STOP: candidate is at BASE; manual investigation required",
    )
    check(snapshot is None or bool(snapshot), "Ambiguous candidate ref")
    return snapshot


def check_expiration(expected_seconds: int) -> None:
    value = os.environ.get("AWS_CREDENTIAL_EXPIRATION")
    if value is None:
        check(
            "AWS_SESSION_TOKEN" not in os.environ,
            "Temporary credentials require AWS_CREDENTIAL_EXPIRATION",
        )
        log.info(
            "Operator SDK credentials must remain valid for the complete run; expiry is not declared"
        )
        return
    try:
        expiry = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("Invalid AWS_CREDENTIAL_EXPIRATION") from None
    check(expiry.tzinfo is not None, "Credential expiration requires a timezone")
    check(
        expiry > datetime.now(UTC) + timedelta(seconds=expected_seconds + 300),
        "Credentials expire before expected completion plus five-minute margin",
    )


def check_capacity(bundle: dict[str, Any], work: Path) -> None:
    check(
        shutil.disk_usage(work).free >= bundle["scratch_bytes"],
        "Insufficient free scratch space for a fresh attempt",
    )
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    check(available >= bundle["memory_bytes"], "Insufficient available memory")


def run_child(
    bundle: dict[str, Any],
    attempt: Path,
    source: Path,
    lock_fd: int,
    *,
    snapshot: str | None,
) -> None:
    check(
        runtime_digest(source) == bundle["manifest"]["runtime_sha256"],
        "Archived runtime changed",
    )
    writable = snapshot is None
    name = "core" if writable else "verify"
    argv = [
        sys.executable,
        "-s",
        "-u",
        "-m",
        f"scripts.retained_ifs_patch{'_verify' if not writable else ''}",
        "--manifest",
        str(attempt / "manifest.json"),
        "--manifest-sha256",
        bundle["manifest_sha256"],
    ]
    argv.extend(
        ["--work-dir", str(attempt / "core"), "--auth", "env", "--commit-to-new-branch"]
        if writable
        else ["--snapshot", snapshot, "--output", str(attempt / "verification.json")]
    )
    started = time.monotonic()
    output = attempt / f"{name}.log"
    log.info(f"Starting {name}; streaming child output to {output}")
    with (
        output.open("xb", buffering=0) as stream,
        subprocess.Popen(  # noqa: S603
            argv,
            cwd=source,
            env=child_environment(source, attempt, writable=writable),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            pass_fds=(lock_fd,),
        ) as child,
    ):
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                log.info(
                    f"{name} running; elapsed={int(time.monotonic() - started)}s log_bytes={output.stat().st_size}"
                )
        check(code == 0, f"{name} failed; inspect {output}; no automatic retry")


def run(
    action: str,
    bundle: dict[str, Any],
    bundle_sha256: str,
    work: Path,
    *,
    dry_run: bool,
    approve: bool,
) -> dict[str, Any]:
    check(action in {"launch", "verify"}, "Unknown action")
    check(action == "launch" or not approve, "Approval flag is launch-only")
    check(
        action != "launch" or dry_run or approve,
        "Launch requires explicit retention and shared-credentials approval",
    )
    if dry_run:
        log.info(
            f"DRY RUN {action}: bundle/evidence validated; commit/archive/runtime/ref checks deferred. No commands or remote reads. Expected seconds={bundle['expected_seconds']} memory_bytes={bundle['memory_bytes']} scratch_bytes={bundle['scratch_bytes']}; nice>=10, CPUs<=4, threads=1"
        )
        return {"status": "dry-run", "external_commands": 0}
    check(sys.flags.optimize == 0, "Optimized Python is forbidden")
    check(
        runtime_digest(CHECKOUT) == bundle["manifest"]["runtime_sha256"],
        "Current wrapper runtime hash mismatch",
    )
    work = work.resolve()
    with candidate_lock(bundle["manifest"]) as lock_fd:
        bind_work(work, bundle, bundle_sha256)
        attempt = Path(tempfile.mkdtemp(prefix=f"{action}-", dir=work))
        (attempt / "tmp").mkdir()
        handler = logging.FileHandler(attempt / "parent.log", mode="x")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
        try:
            log.info(f"Attempt {attempt}; no partial shards are reused")
            save(attempt / "manifest.json", bundle["manifest"])
            (attempt / "manifest.json").chmod(0o444)
            source = archive_runtime(bundle, attempt)
            snapshot = candidate_snapshot(bundle["manifest"])
            if action == "launch" and snapshot is None:
                check_expiration(bundle["expected_seconds"])
                check_capacity(bundle, work)
                run_child(bundle, attempt, source, lock_fd, snapshot=None)
                snapshot = candidate_snapshot(bundle["manifest"])
            check(
                snapshot is not None,
                "Candidate absent; no committed snapshot to verify",
            )
            run_child(bundle, attempt, source, lock_fd, snapshot=snapshot)
            receipt = json.loads((attempt / "verification.json").read_bytes())
            check(
                receipt["status"] == "pass"
                and receipt["snapshot"] == snapshot
                and receipt["base_snapshot"] == bundle["manifest"]["base_snapshot"]
                and receipt["candidate_branch"]
                == bundle["manifest"]["candidate_branch"]
                and receipt["manifest_sha256"] == bundle["manifest_sha256"],
                "Independent verification receipt binding mismatch",
            )
            result = {
                "status": "pass",
                "snapshot": snapshot,
                "attempt": str(attempt),
                "bundle_sha256": bundle_sha256,
            }
            save(attempt / "result.json", result)
            log.info(
                f"PASS independent verification at {snapshot}; receipt {attempt / 'verification.json'}"
            )
            return result
        except BaseException as error:
            save(
                attempt / "STOP.json",
                {
                    "status": "stopped",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "automatic_retry": False,
                },
            )
            log.exception(
                f"STOP; inspect {attempt}; manual investigation, no automatic retry"
            )
            raise
        finally:
            log.removeHandler(handler)
            handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Operator-only foreground local retained repair and anonymous verification"
    )
    parser.add_argument("action", choices=("launch", "verify"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--approve-retention-and-shared-credentials", action="store_true"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_resources()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    bundle = load_bundle(args.bundle, args.bundle_sha256)
    run(
        args.action,
        bundle,
        args.bundle_sha256,
        args.work_dir,
        dry_run=args.dry_run,
        approve=args.approve_retention_and_shared_credentials,
    )


if __name__ == "__main__":
    main()
