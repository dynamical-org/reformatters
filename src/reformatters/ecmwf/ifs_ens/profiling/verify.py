import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from reformatters.common.profiling import PY_SPY_OUTPUT, WORKLOAD_RECEIPT


def verify_evidence(directory: Path) -> dict[str, Any]:
    """Check every downloaded evidence attempt under ``directory`` and summarize
    worker coverage per job. Raises ``AssertionError`` on corrupt or inconsistent
    evidence."""
    initial = list(directory.rglob("manifest-initial.json"))
    assert initial, "No initial manifests found"
    complete = checked = 0
    overhead_warning_attempts = []
    jobs: dict[str, dict[str, Any]] = {}
    for path in initial:
        initial_manifest = json.loads(path.read_text())
        metadata = initial_manifest["metadata"]
        if initial_manifest.get("sampler_overhead_warning"):
            overhead_warning_attempts.append(metadata)
        name = metadata["job_name"]
        workers = metadata["workers"]
        index = metadata["worker_index"]
        assert 0 <= index < workers
        coverage = jobs.setdefault(
            name, {"workers": workers, "complete_workers": set()}
        )
        assert coverage["workers"] == workers
        final_path = path.with_name("manifest-final.json")
        if not final_path.exists():
            continue
        final = json.loads(final_path.read_text())
        assert final["metadata"] == metadata, "Evidence attempt metadata differs"
        for file_name, expected in final["sha256"].items():
            assert Path(file_name).name == file_name, (
                "Evidence filename is not a basename"
            )
            actual = hashlib.sha256((path.parent / file_name).read_bytes()).hexdigest()
            assert actual == expected, f"Evidence checksum mismatch: {file_name}"
            checked += 1
        if (
            final["returncode"] == 0
            and final["evidence_error"] is None
            and final["signal"] is None
            and final["profile_samples"] > 0
            and final["py_spy_returncode"] == 0
            and not final["killed_after_drain"]
        ):
            names = set(final["sha256"])
            assert {PY_SPY_OUTPUT, WORKLOAD_RECEIPT} <= names
            assert any(n.startswith("resource-") for n in names)
            complete += 1
            coverage["complete_workers"].add(index)
    for coverage in jobs.values():
        coverage["missing_workers"] = sorted(
            set(range(coverage["workers"])) - coverage["complete_workers"]
        )
        coverage["complete_workers"] = sorted(coverage["complete_workers"])
    return {
        "attempts": len(initial),
        "complete": complete,
        "incomplete": len(initial) - complete,
        "files_verified": checked,
        "overhead_warning_attempts": overhead_warning_attempts,
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.write(json.dumps(verify_evidence(args.output_dir), indent=2) + "\n")


if __name__ == "__main__":
    main()
