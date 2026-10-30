import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from reformatters.ecmwf.ifs_ens.profiling.verify import verify_evidence

METADATA = {"job_name": "test", "workers": 2, "worker_index": 0}


def _write_attempt(directory: Path, **final_overrides: Any) -> Path:  # noqa: ANN401
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest-initial.json").write_text(json.dumps({"metadata": METADATA}))
    payload = directory / "resource-000000.json"
    payload.write_text("[]")
    for name in ("py-spy.speedscope.json", "workload-exit.json"):
        (directory / name).write_text("{}")
    final = {
        "metadata": METADATA,
        "returncode": 0,
        "evidence_error": None,
        "signal": None,
        "profile_samples": 1,
        "py_spy_returncode": 0,
        "killed_after_drain": False,
        "sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (payload.name, "py-spy.speedscope.json", "workload-exit.json")
        },
        **final_overrides,
    }
    (directory / "manifest-final.json").write_text(json.dumps(final))
    return payload


def test_evidence_verification_rejects_corruption(tmp_path: Path) -> None:
    payload = _write_attempt(tmp_path)
    assert verify_evidence(tmp_path) == {
        "attempts": 1,
        "complete": 1,
        "incomplete": 0,
        "files_verified": 3,
        "overhead_warning_attempts": [],
        "jobs": {
            "test": {"workers": 2, "complete_workers": [0], "missing_workers": [1]}
        },
    }
    payload.write_text("corrupted")
    with pytest.raises(AssertionError, match="checksum mismatch"):
        verify_evidence(tmp_path)


def test_failed_attempt_is_incomplete(tmp_path: Path) -> None:
    _write_attempt(tmp_path / "attempt", signal="SIGTERM")
    summary = verify_evidence(tmp_path)
    assert (summary["complete"], summary["incomplete"]) == (0, 1)
    assert summary["jobs"]["test"]["missing_workers"] == [0, 1]


@pytest.mark.parametrize("has_final", [False, True])
def test_overhead_warning_is_reported_without_failing_evidence(
    tmp_path: Path, has_final: bool
) -> None:
    _write_attempt(tmp_path)
    (tmp_path / "manifest-initial.json").write_text(
        json.dumps({"metadata": METADATA, "sampler_overhead_warning": True})
    )
    if not has_final:
        (tmp_path / "manifest-final.json").unlink()
    summary = verify_evidence(tmp_path)
    assert summary["overhead_warning_attempts"] == [METADATA]
    assert summary["complete"] == int(has_final)


def test_attempt_without_final_manifest_is_incomplete(tmp_path: Path) -> None:
    _write_attempt(tmp_path)
    (tmp_path / "manifest-final.json").unlink()
    summary = verify_evidence(tmp_path)
    assert (summary["complete"], summary["files_verified"]) == (0, 0)


def test_rejects_mismatched_metadata(tmp_path: Path) -> None:
    _write_attempt(tmp_path, metadata={**METADATA, "worker_index": 1})
    with pytest.raises(AssertionError, match="metadata differs"):
        verify_evidence(tmp_path)


def test_rejects_non_basename_hash_entry(tmp_path: Path) -> None:
    _write_attempt(tmp_path, sha256={"../escape.json": "0" * 64})
    with pytest.raises(AssertionError, match="not a basename"):
        verify_evidence(tmp_path)


def test_rejects_out_of_range_worker_index(tmp_path: Path) -> None:
    _write_attempt(tmp_path)
    (tmp_path / "manifest-initial.json").write_text(
        json.dumps({"metadata": {**METADATA, "worker_index": 2}})
    )
    with pytest.raises(AssertionError):
        verify_evidence(tmp_path)


def test_empty_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="No initial manifests"):
        verify_evidence(tmp_path)


def test_cli_prints_summary(tmp_path: Path) -> None:
    _write_attempt(tmp_path)
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "reformatters.ecmwf.ifs_ens.profiling.verify",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["complete"] == 1
