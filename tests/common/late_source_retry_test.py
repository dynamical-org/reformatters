from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from reformatters.common import kubernetes, validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.types import DatetimeLike
from reformatters.noaa.refs.forecast_hourly_virtual.dynamical_dataset import (
    NoaaRefsForecastHourlyVirtualDataset,
)


@pytest.mark.parametrize("now", ["19:43", "19:45", "19:47"])
def test_final_validation_does_not_spawn_more_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, now: str
) -> None:
    dataset = NoaaRefsForecastHourlyVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    dataset._retry_timing = kubernetes.RetryTiming(
        pd.Timestamp("2026-10-19 13:45"),
        pd.Timestamp("2026-10-19 19:45"),
        pd.Timestamp("2026-10-19 19:43"),
    )
    current = pd.Timestamp(f"2026-10-19 {now}")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: current))
    create = Mock()
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", create)
    dataset._retry_late_source_update("update-r2")
    create.assert_not_called()


def test_validation_after_next_fire_keeps_original_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = NoaaRefsForecastHourlyVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    dataset._retry_timing = kubernetes.RetryTiming(
        pd.Timestamp("2026-10-19 13:45"),
        pd.Timestamp("2026-10-19 19:45"),
        pd.Timestamp("2026-10-19 19:43"),
    )
    now = pd.Timestamp("2026-10-19 19:47")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: now))

    def template(self: object, end: DatetimeLike) -> xr.DataTree:
        return xr.DataTree(
            xr.Dataset(
                coords={
                    "init_time": pd.date_range(
                        "2026-10-18", end, freq="6h", inclusive="left"
                    )
                }
            )
        )

    monkeypatch.setattr(type(dataset), "_get_template", template)
    monkeypatch.setattr(
        type(dataset.store_factory), "primary_store", lambda self: MemoryStore()
    )
    job = dataset._virtual_validation_region_job(dataset.validators(), "update-r2")
    assert job is not None
    inits = job.template_ds.to_dataset().isel(init_time=job.region).init_time.values
    assert list(pd.DatetimeIndex(inits)) == list(
        pd.date_range("2026-10-19", periods=3, freq="6h")
    )
    assert dataset._virtual_poll_deadline(now) == pd.Timestamp("2026-10-19 19:43")


@pytest.mark.parametrize(
    ("checks", "missing"),
    [
        ((), False),
        (("CheckVirtualManifestCompleteness",), True),
        (("CheckCurrentData", "CheckVirtualManifestCompleteness"), True),
        (("CheckVirtualManifestCompleteness", "CheckVirtualDecodeHealth"), False),
    ],
)
def test_only_missing_data_classification(
    checks: tuple[str, ...], missing: bool
) -> None:
    assert (
        validation.OperationalValidationError("failure", checks).only_missing_data
        is missing
    )


def test_late_source_retries_require_single_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        NoaaRefsForecastHourlyVirtualDataset, "retry_late_sources", True
    )
    storage = StorageConfig(base_path=str(tmp_path), format=DatasetFormat.ICECHUNK)
    with pytest.raises(ValueError, match="single store without replicas"):
        NoaaRefsForecastHourlyVirtualDataset(
            primary_storage_config=storage, replica_storage_configs=[storage]
        )


def test_validation_keeps_snapshot_when_successor_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    first = repo.writable_session("main")
    xr.Dataset({"value": ("time", [1])}).to_zarr(first.store, consolidated=False)
    first.commit("initial")
    pinned = repo.readonly_session("main").store
    observed: list[int] = []

    def check(
        self: validation.CheckCurrentData, context: validation.ValidationContext
    ) -> validation.ValidationResult:
        observed.append(int(context.ds["value"].item()))
        if len(observed) == 1:
            successor = repo.writable_session("main")
            xr.Dataset({"value": ("time", [2])}).to_zarr(
                successor.store, mode="w", consolidated=False
            )
            successor.commit("successor")
        return validation.ValidationResult(passed=True, message="ok")

    monkeypatch.setattr(validation.CheckCurrentData, "check", check)
    validation.validate_dataset(
        [validation.CheckCurrentData(max_delay=timedelta(0))] * 2,
        store=pinned,
        append_dim="time",
        dataset_id="test",
    )
    assert observed == [1, 1]
    latest = xr.open_zarr(
        repo.readonly_session("main").store, consolidated=False, chunks=None
    )
    assert latest["value"].item() == 2
