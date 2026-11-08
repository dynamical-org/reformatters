import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from scripts import retained_ifs_patch_local as local
from scripts import retained_ifs_patch_verify as verifier


def options() -> dict[str, Any]:
    manifest = {
        "version": 2,
        "dataset_id": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "scientific_manifest_sha256": verifier.SCIENCE_SHA256,
        "base_snapshot": verifier.BASE,
        "protected_branch": verifier.PROTECTED,
        "repository": verifier.REPOSITORY,
        "candidate_branch": "local-repair-candidate",
        "runtime_sha256": "b" * 64,
        "metadata_sha256": "a" * 64,
        "approved_shards": {
            "temperature_2m/c/0/0/0/0/0": {
                "encoded_sha256": "a" * 64,
                "decoded_sha256": "b" * 64,
            }
        },
        "units": [
            {
                "variable": var,
                "init": date + "T00:00:00",
                "fields": [
                    {
                        "member": member,
                        "lead_hours": lead,
                        "expected_sha256": verifier.EXPECTED_HASHES[
                            var, date + "T00:00:00", member, lead
                        ],
                    }
                    for v, d, member, lead in sorted(verifier.SCOPE)
                    if (v, d) == (var, date)
                ],
            }
            for var, date in sorted({(v, d) for v, d, _, _ in verifier.SCOPE})
        ],
    }
    return {
        "version": 2,
        "commit": "c" * 40,
        "manifest": manifest,
        "manifest_sha256": local.sha(local.canonical(manifest)),
        "expected_seconds": 600,
        "memory_bytes": 8_000_000_000,
        "scratch_bytes": 20_000_000_000,
        "implementation_review": {"path": "review.md", "sha256": "d" * 64},
        "independent_stage": {"path": "stage.json", "sha256": "d" * 64},
    }


def write_bundle(root: Path, value: dict[str, Any]) -> tuple[Path, str]:
    local.save(
        root / "stage.json",
        {
            "status": "pass-local-stage",
            "base_snapshot": verifier.BASE,
            "approved_shards": value["manifest"]["approved_shards"],
        },
    )
    (root / "review.md").write_text("Independent implementation review PASS\n")
    for name in ("implementation_review", "independent_stage"):
        value[name]["sha256"] = local.sha((root / value[name]["path"]).read_bytes())
    path = root / "bundle.json"
    local.save(path, value)
    return path, local.sha(path.read_bytes())


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "LOCK_ROOT", tmp_path / "locks")
    monkeypatch.setattr(local, "runtime_digest", lambda _: "b" * 64)


def test_bundle_and_command_free_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = write_bundle(tmp_path, options())
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("Command executed")
    )
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: pytest.fail("Child executed")
    )
    monkeypatch.setattr(local, "anonymous_repo", lambda _: pytest.fail("Remote read"))
    monkeypatch.setattr(local, "candidate_lock", lambda _: pytest.fail("Lock/write"))
    monkeypatch.setattr(
        local, "runtime_digest", lambda _: pytest.fail("Non-dry runtime check")
    )
    value = local.load_bundle(path, digest)
    for action in ("launch", "verify"):
        assert local.run(
            action, value, digest, tmp_path / "work", dry_run=True, approve=False
        ) == {"status": "dry-run", "external_commands": 0}
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    "failure",
    [
        "bundle",
        "commit",
        "manifest",
        "scope",
        "evidence",
        "absolute",
        "escape",
        "stage",
        "duration",
        "resources",
    ],
)
def test_bundle_pin_and_scope_failures(failure: str, tmp_path: Path) -> None:
    value = options()
    path, digest = write_bundle(tmp_path, value)
    if failure == "bundle":
        digest = "0" * 64
    else:
        if failure == "commit":
            value["commit"] = "main"
        elif failure == "manifest":
            value["manifest_sha256"] = "0" * 64
        elif failure == "scope":
            value["manifest"]["units"][0]["fields"][0]["member"] = 99
            value["manifest_sha256"] = local.sha(local.canonical(value["manifest"]))
        elif failure == "evidence":
            (tmp_path / "review.md").write_text("changed")
        elif failure == "absolute":
            value["implementation_review"]["path"] = str(tmp_path / "review.md")
        elif failure == "escape":
            value["implementation_review"]["path"] = "../outside"
        elif failure == "stage":
            receipt = json.loads((tmp_path / "stage.json").read_bytes())
            receipt["status"] = "failed"
            (tmp_path / "stage.json").write_bytes(local.canonical(receipt))
            value["independent_stage"]["sha256"] = local.sha(
                (tmp_path / "stage.json").read_bytes()
            )
        elif failure == "duration":
            value["expected_seconds"] = 0
        elif failure == "resources":
            value["memory_bytes"] = True
        path.write_bytes(local.canonical(value))
        digest = local.sha(path.read_bytes())
    with pytest.raises(ValueError, match=r"hash|pin|scope|relative|binding|Invalid"):
        local.load_bundle(path, digest)


def test_current_runtime_failure_precedes_commands_locks_and_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local, "runtime_digest", lambda _: "changed")
    monkeypatch.setattr(local, "candidate_lock", lambda _: pytest.fail("Lock/write"))
    monkeypatch.setattr(
        local, "archive_runtime", lambda *a: pytest.fail("Archive command")
    )
    monkeypatch.setattr(local, "anonymous_repo", lambda _: pytest.fail("Remote read"))
    with pytest.raises(ValueError, match="Current wrapper runtime"):
        local.run(
            "launch",
            options(),
            "a" * 64,
            tmp_path / "work",
            dry_run=False,
            approve=True,
        )


def test_approval_is_launch_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit"):
        local.run("launch", options(), "a" * 64, tmp_path, dry_run=False, approve=False)
    with pytest.raises(ValueError, match="launch-only"):
        local.run("verify", options(), "a" * 64, tmp_path, dry_run=True, approve=True)


def test_binding_and_unbound_directory_preservation(tmp_path: Path) -> None:
    value = options()
    work = tmp_path / "work"
    work.mkdir()
    (work / "prior").write_text("keep")
    with pytest.raises(ValueError, match="unbound"):
        local.bind_work(work, value, "a" * 64)
    assert sorted(p.name for p in work.iterdir()) == ["prior"]
    bound = tmp_path / "bound"
    local.bind_work(bound, value, "a" * 64)
    original = (bound / "binding.json").read_bytes()
    local.bind_work(bound, value, "a" * 64)
    with pytest.raises(ValueError, match="binding changed"):
        local.bind_work(bound, value, "b" * 64)
    assert (bound / "binding.json").read_bytes() == original


@pytest.mark.parametrize(
    "state", ["absent", "base", "advanced", "protected-moved", "lookup-error", "empty"]
)
def test_ref_states(state: str, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = options()["manifest"]
    repo = Mock()
    repo.list_branches.return_value = (
        [] if state == "absent" else [manifest["candidate_branch"]]
    )

    def lookup(branch: str) -> str:
        if branch == verifier.PROTECTED:
            return "moved" if state == "protected-moved" else verifier.BASE
        if state == "lookup-error":
            raise RuntimeError("Lookup failed")
        return {"base": verifier.BASE, "empty": ""}.get(state, "candidate-snapshot")

    repo.lookup_branch.side_effect = lookup
    monkeypatch.setattr(local, "anonymous_repo", lambda _: repo)
    if state in {"base", "protected-moved", "lookup-error", "empty"}:
        with pytest.raises((ValueError, RuntimeError)):
            local.candidate_snapshot(manifest)
    else:
        assert local.candidate_snapshot(manifest) == (
            None if state == "absent" else "candidate-snapshot"
        )


def receipt(attempt: Path, value: dict[str, Any], snapshot: str) -> None:
    local.save(
        attempt / "verification.json",
        {
            "status": "pass",
            "snapshot": snapshot,
            "base_snapshot": value["manifest"]["base_snapshot"],
            "candidate_branch": value["manifest"]["candidate_branch"],
            "manifest_sha256": value["manifest_sha256"],
        },
    )


@pytest.mark.usefixtures("isolated")
@pytest.mark.parametrize(
    "failure", ["none", "stage", "verify", "receipt", "absent-after-core", "base"]
)
def test_launch_and_duplicate_rerun_are_fresh(  # noqa: PLR0915
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = options()
    state: str | None = None
    calls: list[tuple[str | None, Path]] = []
    archives = []

    def archive(bundle: dict[str, Any], attempt: Path) -> Path:
        archives.append(attempt)
        return attempt / "runtime"

    def current(_: dict[str, Any]) -> str | None:
        if failure == "base":
            raise ValueError("STOP: candidate is at BASE")
        return state

    def child(
        bundle: dict[str, Any],
        attempt: Path,
        source: Path,
        lock_fd: int,
        *,
        snapshot: str | None,
    ) -> None:
        nonlocal state
        assert source == attempt / "runtime"
        assert os.fstat(lock_fd).st_ino > 0
        assert (
            local.sha((attempt / "manifest.json").read_bytes())
            == value["manifest_sha256"]
        )
        calls.append((snapshot, attempt))
        if snapshot is None:
            assert not (attempt / "core").exists()
            (attempt / "core").mkdir()
            (attempt / "core" / "partial").write_text("new stage")
            if failure == "stage":
                raise RuntimeError("Stage failed")
            if failure != "absent-after-core":
                state = "snapshot"
        else:
            if failure == "verify":
                raise RuntimeError("Independent verifier failed")
            receipt(attempt, value, "wrong" if failure == "receipt" else snapshot)

    monkeypatch.setattr(local, "archive_runtime", archive)
    monkeypatch.setattr(local, "candidate_snapshot", current)
    monkeypatch.setattr(local, "run_child", child)
    monkeypatch.setattr(local, "check_expiration", lambda _: None)
    monkeypatch.setattr(local, "check_capacity", lambda *_: None)
    work = tmp_path / "work"
    if failure == "none":
        first = local.run("launch", value, "a" * 64, work, dry_run=False, approve=True)
        first_bytes = (Path(first["attempt"]) / "result.json").read_bytes()
        second = local.run("launch", value, "a" * 64, work, dry_run=False, approve=True)
        third = local.run("verify", value, "a" * 64, work, dry_run=False, approve=False)
        assert [s for s, _ in calls] == [None, "snapshot", "snapshot", "snapshot"]
        assert len(set(archives)) == 3
        assert len({first["attempt"], second["attempt"], third["attempt"]}) == 3
        assert (Path(first["attempt"]) / "result.json").read_bytes() == first_bytes
    else:
        with pytest.raises((ValueError, RuntimeError)):
            local.run("launch", value, "a" * 64, work, dry_run=False, approve=True)
        if failure == "stage":
            with pytest.raises(RuntimeError, match="Stage failed"):
                local.run("launch", value, "a" * 64, work, dry_run=False, approve=True)
            assert len({attempt for _, attempt in calls}) == 2
        for attempt in archives:
            assert (attempt / "STOP.json").exists()
            assert not (attempt / "result.json").exists()
        if failure == "base":
            assert calls == []


@pytest.mark.usefixtures("isolated")
def test_saved_success_does_not_hide_fresh_verification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = options()
    work = tmp_path / "work"
    local.bind_work(work, value, "a" * 64)
    local.save(work / "result.json", {"status": "pass"})
    monkeypatch.setattr(local, "archive_runtime", lambda _, attempt: attempt)
    monkeypatch.setattr(local, "candidate_snapshot", lambda _: "existing-snapshot")
    monkeypatch.setattr(
        local,
        "check_expiration",
        lambda _: pytest.fail("Credential access during verify"),
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Independent verification failed")

    monkeypatch.setattr(local, "run_child", fail)
    with pytest.raises(RuntimeError, match="Independent verification failed"):
        local.run("launch", value, "a" * 64, work, dry_run=False, approve=True)
    assert json.loads((work / "result.json").read_bytes()) == {"status": "pass"}


@pytest.mark.parametrize(
    "expiry", ["absent", "future", "expired", "short", "malformed", "naive"]
)
def test_credential_expiration_is_checked_without_disclosure(
    expiry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AWS_CREDENTIAL_EXPIRATION", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    value = ""
    if expiry != "absent":
        value = {
            "future": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "expired": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            "short": (datetime.now(UTC) + timedelta(seconds=800)).isoformat(),
            "malformed": "not-a-date-sensitive-value",
            "naive": "2099-01-01T00:00:00",
        }[expiry]
        monkeypatch.setenv("AWS_CREDENTIAL_EXPIRATION", value)
    if expiry in {"absent", "future"}:
        local.check_expiration(600)
    else:
        with pytest.raises(ValueError, match=r"expiration|expire|Invalid") as caught:
            local.check_expiration(600)
        assert value not in str(caught.value)


def test_temporary_credentials_require_declared_expiration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_CREDENTIAL_EXPIRATION", raising=False)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "synthetic-session-token")
    with pytest.raises(ValueError, match="require AWS_CREDENTIAL_EXPIRATION"):
        local.check_expiration(600)
    monkeypatch.setenv(
        "AWS_CREDENTIAL_EXPIRATION",
        (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )
    local.check_expiration(600)


@pytest.mark.parametrize("failure", ["none", "disk", "memory"])
def test_capacity_preflight(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local.shutil,
        "disk_usage",
        lambda _: Mock(free=0 if failure == "disk" else 4096),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _: (
            "MemAvailable: 0 kB\n" if failure == "memory" else "MemAvailable: 4 kB\n"
        ),
    )
    bundle = {"scratch_bytes": 2048, "memory_bytes": 2048}
    if failure == "none":
        local.check_capacity(bundle, tmp_path)
    else:
        with pytest.raises(ValueError, match="Insufficient"):
            local.check_capacity(bundle, tmp_path)


@pytest.mark.usefixtures("isolated")
def test_capacity_failure_stops_before_core_and_records_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local, "archive_runtime", lambda _, attempt: attempt)
    monkeypatch.setattr(local, "candidate_snapshot", lambda _: None)
    monkeypatch.setattr(local, "check_expiration", lambda _: None)
    monkeypatch.setattr(
        local, "check_capacity", Mock(side_effect=ValueError("Insufficient scratch"))
    )
    monkeypatch.setattr(
        local, "run_child", lambda *a, **kw: pytest.fail("Core started")
    )
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="Insufficient scratch"):
        local.run("launch", options(), "a" * 64, work, dry_run=False, approve=True)
    (stop,) = work.glob("launch-*/STOP.json")
    assert json.loads(stop.read_bytes())["message"] == "Insufficient scratch"
    assert "Insufficient scratch" in (stop.parent / "parent.log").read_text()


def test_resources_and_credential_free_child_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []
    monkeypatch.setattr(os, "getpriority", lambda *a: 15)
    monkeypatch.setattr(os, "setpriority", lambda *a: events.append(("nice", a[-1])))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 3, 4, 5, 6, 7})
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda _, cpus: events.append(("affinity", cpus))
    )
    for name in local.THREADS:
        monkeypatch.setenv(name, "8")
    local.configure_resources()
    assert events == [("nice", 15), ("affinity", [2, 3, 4, 5])]
    assert all(os.environ[name] == "1" for name in local.THREADS)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic-test-value")
    monkeypatch.setenv("PYTHONPATH", "/editable-worktree")
    monkeypatch.setenv("PYTHONHOME", "/unapproved")
    anonymous = local.child_environment(tmp_path, tmp_path, writable=False)
    assert not any(name.startswith("AWS_") for name in anonymous)
    assert anonymous["PYTHONPATH"] == str(tmp_path / "src")
    writable = local.child_environment(tmp_path, tmp_path, writable=True)
    assert writable["AWS_ACCESS_KEY_ID"] == "synthetic-test-value"
    assert "PYTHONHOME" not in writable
    assert all(writable[name] == "1" for name in local.THREADS)


@pytest.mark.parametrize(
    "failure", ["none", "hash", "traversal", "absolute", "symlink", "hardlink"]
)
def test_git_archive_pinning_safe_extraction_and_readonly_runtime(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = options()
    contents = {
        "uv.lock": b"locked",
        "pyproject.toml": b"project",
        "src/scripts/core.py": b"approved source",
    }
    value["manifest"]["runtime_sha256"] = local.sha(
        local.canonical({k: local.sha(v) for k, v in sorted(contents.items())})
    )
    commands = []

    def execute(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:  # noqa: ANN401
        commands.append(argv)
        assert kwargs["cwd"] == local.CHECKOUT
        assert not any(k.startswith("AWS_") for k in kwargs["env"])
        if argv[1] == "cat-file":
            return subprocess.CompletedProcess(argv, 0, b"commit\n")
        assert Path(argv[0]).name == "git"
        assert argv[1:] == [
            "archive",
            "--format=tar",
            value["commit"],
            "src",
            "uv.lock",
            "pyproject.toml",
        ]
        with tarfile.open(fileobj=kwargs["stdout"], mode="w") as archive:
            for name, body in contents.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(body)
                archive.addfile(entry, io.BytesIO(body))
            if failure in {"traversal", "absolute", "symlink", "hardlink"}:
                entry = tarfile.TarInfo(
                    {"traversal": "../escape", "absolute": "/escape"}.get(
                        failure, "src/link"
                    )
                )
                if failure in {"symlink", "hardlink"}:
                    entry.type = (
                        tarfile.SYMTYPE if failure == "symlink" else tarfile.LNKTYPE
                    )
                    entry.linkname = "../outside"
                archive.addfile(entry)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    if failure == "hash":
        value["manifest"]["runtime_sha256"] = "0" * 64
    if failure == "none":
        source = local.archive_runtime(value, tmp_path)
        try:
            assert (source / "src/scripts/core.py").read_bytes() == contents[
                "src/scripts/core.py"
            ]
            assert (source.stat().st_mode & 0o222) == 0
            assert ((source / "src/scripts/core.py").stat().st_mode & 0o222) == 0
            assert (
                json.loads((tmp_path / "runtime-binding.json").read_bytes())["commit"]
                == value["commit"]
            )
        finally:
            source.chmod(0o700)
            for path in source.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
    else:
        with pytest.raises(ValueError, match=r"runtime hash|Unsafe git archive"):
            local.archive_runtime(value, tmp_path)
    assert len(commands) == 2


@pytest.mark.usefixtures("isolated")
def test_local_lock_excludes_other_workdirs_and_bundle_revisions() -> None:
    manifest = options()["manifest"]
    with (
        local.candidate_lock(manifest),
        pytest.raises(BlockingIOError),
        local.candidate_lock(manifest | {"runtime_sha256": "changed"}),
    ):
        pytest.fail("Concurrent candidate lock")
    with local.candidate_lock(manifest):
        pass


def test_inherited_lock_survives_parent_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local, "LOCK_ROOT", tmp_path / "locks")
    ready, release = tmp_path / "ready", tmp_path / "release"
    source = tmp_path / "runtime"
    package = source / "src/scripts"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "retained_ifs_patch.py").write_text(
        f"import time\nfrom pathlib import Path\nPath({str(ready)!r}).touch()\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(0.02)\n"
    )
    (source / "uv.lock").touch()
    (source / "pyproject.toml").touch()
    value = options()
    value["manifest"]["runtime_sha256"] = local.runtime_digest(source)
    local.save(tmp_path / "bundle.json", value)
    code = """
import json, sys
from pathlib import Path
from scripts import retained_ifs_patch_local as local
attempt = Path(sys.argv[1])
local.LOCK_ROOT = attempt / 'locks'
bundle = json.loads((attempt / 'bundle.json').read_bytes())
with local.candidate_lock(bundle['manifest']) as fd:
    local.run_child(bundle, attempt, attempt / 'runtime', fd, snapshot=None)
"""
    env = local.child_environment(local.CHECKOUT, tmp_path, writable=False)
    parent = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", code, str(tmp_path)],
        env=env,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert time.monotonic() < deadline
            assert parent.poll() is None
            time.sleep(0.02)
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=5)
        with (
            pytest.raises(BlockingIOError),
            local.candidate_lock(value["manifest"]),
        ):
            pytest.fail("Child lost inherited lock")
    finally:
        release.touch()
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
    deadline = time.monotonic() + 10
    while True:
        try:
            with local.candidate_lock(value["manifest"]):
                break
        except BlockingIOError:
            assert time.monotonic() < deadline
            time.sleep(0.02)


@pytest.mark.usefixtures("isolated")
@pytest.mark.parametrize("snapshot", [None, "committed-snapshot"])
def test_real_child_uses_archive_venv_log_and_inherited_resources(
    snapshot: str | None, tmp_path: Path
) -> None:
    source = tmp_path / "runtime"
    package = source / "src/scripts"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    name = "retained_ifs_patch" + ("_verify" if snapshot else "")
    (package / f"{name}.py").write_text("""
import json, os, sys
from pathlib import Path
assert os.environ['NUMBA_NUM_THREADS'] == '1'
assert Path(__file__).resolve().is_relative_to(Path.cwd())
print(json.dumps({'argv':sys.argv, 'executable':sys.executable, 'cwd':str(Path.cwd()),
                  'nice':os.getpriority(os.PRIO_PROCESS, 0), 'affinity':sorted(os.sched_getaffinity(0))}), flush=True)
""")
    with local.candidate_lock(options()["manifest"]) as fd:
        local.run_child(options(), tmp_path, source, fd, snapshot=snapshot)
    record = json.loads(
        (tmp_path / ("verify.log" if snapshot else "core.log")).read_text()
    )
    assert record["executable"] == sys.executable
    assert record["cwd"] == str(source)
    assert record["nice"] == os.getpriority(os.PRIO_PROCESS, 0)
    assert record["affinity"] == sorted(os.sched_getaffinity(0))
    if snapshot:
        assert "--auth" not in record["argv"]
        assert "--commit-to-new-branch" not in record["argv"]
    else:
        assert record["argv"][record["argv"].index("--auth") + 1] == "env"
        assert "--commit-to-new-branch" in record["argv"]
