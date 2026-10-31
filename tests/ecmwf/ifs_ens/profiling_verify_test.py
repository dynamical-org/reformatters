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
        "invalid_utf8_attempts": [],
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


def _invalid_profile() -> bytes:
    return (
        b'{"profiles": [{"name": "Process 1 Thread 1", "samples": [[0], [0, 1]]}],'
        b' "shared": {"frames": [{"name": "busy", "file": "x.py"},'
        b' {"name": "bad\xff", "file": "x.py"}]}}'
    )


def _write_invalid_attempt(directory: Path, **final_overrides: Any) -> None:  # noqa: ANN401
    _write_attempt(directory)
    profile = directory / "py-spy.speedscope.json"
    profile.write_bytes(_invalid_profile())
    final = json.loads((directory / "manifest-final.json").read_text())
    final["sha256"][profile.name] = hashlib.sha256(_invalid_profile()).hexdigest()
    (directory / "manifest-final.json").write_text(
        json.dumps({**final, **final_overrides})
    )


def test_invalid_utf8_is_listed_from_raw_bytes_for_manifest_without_quality_fields(
    tmp_path: Path,
) -> None:
    _write_invalid_attempt(tmp_path, profile_samples=0)
    summary = verify_evidence(tmp_path)
    assert summary["complete"] == 0
    assert summary["invalid_utf8_attempts"] == [
        {
            "metadata": METADATA,
            "profile_samples": 2,
            "profile_decode": "replacement",
            "profile_error": None,
            "profile_replacement_chars": 1,
            "profile_affected_frames": 1,
            "profile_affected_samples": 1,
        }
    ]


def test_invalid_utf8_attempt_stays_complete_but_visible(tmp_path: Path) -> None:
    _write_invalid_attempt(tmp_path)
    summary = verify_evidence(tmp_path)
    assert summary["complete"] == 1
    assert len(summary["invalid_utf8_attempts"]) == 1


def test_changed_invalid_profile_fails_its_raw_hash(tmp_path: Path) -> None:
    _write_invalid_attempt(tmp_path)
    (tmp_path / "py-spy.speedscope.json").write_bytes(
        _invalid_profile().replace(b"\xff", b"\xfe")
    )
    with pytest.raises(AssertionError, match="checksum mismatch"):
        verify_evidence(tmp_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"profile_decode": "strict"}, "quality differs"),
        ({"profile_samples": 1}, "sample count differs"),
    ],
)
def test_rejects_manifest_that_disagrees_with_raw_profile(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    quality = {
        "profile_samples": 2,
        "profile_decode": "replacement",
        "profile_error": None,
        "profile_replacement_chars": 1,
        "profile_affected_frames": 1,
        "profile_affected_samples": 1,
    }
    _write_invalid_attempt(tmp_path, **{**quality, **overrides})
    with pytest.raises(AssertionError, match=message):
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
