import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import xarray as xr

from reformatters.common import operational_run_guard as guard


class FakeStoreFactory:
    def __init__(self) -> None:
        self.files: dict[str, dict[str, bytes]] = {}

    def write_coordination_file(self, job_name: str, key: str, data: bytes) -> None:
        self.files.setdefault(job_name, {})[key] = data

    def read_all_coordination_files(self, job_name: str, prefix: str) -> list[bytes]:
        files = self.files.get(job_name, {})
        matching = {k: v for k, v in files.items() if k.startswith(f"{prefix}/")}
        return [matching[k] for k in sorted(matching)]

    def count_coordination_files(self, job_name: str, prefix: str) -> int:
        return len(self.read_all_coordination_files(job_name, prefix))

    def delete_coordination_file(self, job_name: str, key: str) -> None:
        self.files.get(job_name, {}).pop(key, None)


def _factory() -> guard.StoreFactory:
    return FakeStoreFactory()  # ty: ignore[invalid-return-type]


RUN = "2026-06-10T06:00:00+00:00"


def test_run_key_uses_newest_written_append_dim_value() -> None:
    template_ds = xr.Dataset(
        coords={
            "init_time": pd.to_datetime(
                ["2026-06-10T00:00", "2026-06-10T06:00", "2026-06-10T12:00"]
            )
        }
    )
    jobs = [SimpleNamespace(region=slice(0, 2)), SimpleNamespace(region=slice(1, 3))]
    assert (
        guard.run_key(jobs, template_ds, "init_time")  # ty: ignore[invalid-argument-type]
        == "2026-06-10T12:00:00"
    )


def test_claim_then_skip_for_other_job_but_not_self() -> None:
    sf = _factory()
    guard.claim(sf, RUN, "job-a")
    # A different invocation sees the fresh lease and skips.
    assert guard.should_skip(sf, RUN, "job-b") is True
    # The owning invocation's other workers proceed (same job name).
    assert guard.should_skip(sf, RUN, "job-a") is False


def test_stale_lease_is_ignored() -> None:
    sf = _factory()
    stale = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    sf.write_coordination_file(
        guard._NAMESPACE,
        guard._lease_key(RUN),
        json.dumps({"job_name": "dead-job", "started_at": stale}).encode(),
    )
    assert guard.should_skip(sf, RUN, "job-b") is False


def test_complete_marker_always_skips() -> None:
    sf = _factory()
    guard.mark_complete(sf, RUN)
    assert guard.should_skip(sf, RUN, "job-b") is True


def test_release_clears_lease() -> None:
    sf = _factory()
    guard.claim(sf, RUN, "job-a")
    guard.release(sf, RUN)
    assert guard.should_skip(sf, RUN, "job-b") is False
