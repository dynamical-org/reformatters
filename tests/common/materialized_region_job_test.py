import json

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.materialized_region_job import worker_commit_metadata
from reformatters.common.region_job import SourceFileResult, SourceFileStatus

NOW = pd.Timestamp("2026-10-23T05:39:52.123", tz="UTC")


def _result(status: SourceFileStatus, **out_loc: object) -> SourceFileResult:
    return SourceFileResult(status=status, out_loc=out_loc, url="https://example.com")


def _forecast_template(n_leads: int) -> xr.DataTree:
    return xr.DataTree(
        xr.Dataset(coords={"lead_time": pd.timedelta_range("0h", periods=n_leads)})
    )


def test_worker_commit_metadata_single_init_lead_range() -> None:
    init = pd.Timestamp("2026-10-22T18:00")
    worker_results = {
        "temperature_2m": [
            _result(
                SourceFileStatus.Succeeded, init_time=init, lead_time=pd.Timedelta(h)
            )
            for h in ("0h", "3h", "6h")
        ],
        "precipitation_surface": [
            _result(
                SourceFileStatus.DownloadFailed,
                init_time=init,
                lead_time=pd.Timedelta("9h"),
            )
        ],
    }

    metadata = worker_commit_metadata(worker_results, _forecast_template(4), 2, NOW)

    assert metadata == {
        "timestamp": "2026-10-23T05:39:52Z",
        "worker": 2,
        "data_vars": ["temperature_2m"],
        "source_files": {"Succeeded": 3, "DownloadFailed": 1},
        "init_time": "2026-10-22T18:00:00",
        "lead_time": ["P0DT0H0M0S", "P0DT6H0M0S"],
        "leads_written": 3,
        "leads_expected": 4,
    }
    json.dumps(metadata)


def test_worker_commit_metadata_single_lead_is_scalar() -> None:
    worker_results = {
        "temperature_2m": [
            _result(
                SourceFileStatus.Succeeded,
                init_time=np.datetime64("2026-10-22T18:00"),
                lead_time=np.timedelta64(6, "h"),
            )
        ]
    }

    metadata = worker_commit_metadata(worker_results, _forecast_template(2), 0, NOW)

    assert metadata["lead_time"] == "P0DT6H0M0S"
    assert metadata["leads_written"] == 1


def test_worker_commit_metadata_analysis_time_range() -> None:
    worker_results = {
        "temperature_2m": [
            _result(SourceFileStatus.Succeeded, time=pd.Timestamp(t))
            for t in ("2026-10-22T01:00", "2026-10-22T00:00")
        ]
    }

    metadata = worker_commit_metadata(worker_results, xr.DataTree(), 0, NOW)

    assert metadata["time"] == ["2026-10-22T00:00:00", "2026-10-22T01:00:00"]
    assert "init_time" not in metadata
    assert "lead_time" not in metadata
    assert "leads_expected" not in metadata
