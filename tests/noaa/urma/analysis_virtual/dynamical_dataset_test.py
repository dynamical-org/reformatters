import json
from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from icechunk import IcechunkStore

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.urma.analysis_virtual import NoaaUrmaAnalysisVirtualDataset
from reformatters.noaa.urma.analysis_virtual.region_job import (
    NoaaUrmaAnalysisVirtualRegionJob,
)
from reformatters.noaa.urma.analysis_virtual.template_config import (
    NoaaUrmaAnalysisVirtualTemplateConfig,
)


def make_dataset(
    tmp_path: Path, start: str = "2017-05-02T00:00"
) -> NoaaUrmaAnalysisVirtualDataset:
    return NoaaUrmaAnalysisVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        template_config=NoaaUrmaAnalysisVirtualTemplateConfig(
            append_dim_start=pd.Timestamp(start)
        ),
    )


def test_urma_operations_storage_and_validators(tmp_path: Path) -> None:
    dataset = NoaaUrmaAnalysisVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        template_config=NoaaUrmaAnalysisVirtualTemplateConfig(),
    )
    assert dataset.dataset_id == "noaa-urma-analysis-virtual"
    assert not dataset.replica_storage_configs
    (cron,) = dataset.operational_kubernetes_resources("test-image")
    assert cron.as_kubernetes_object()["spec"]["suspend"] is True
    assert cron.schedule == "45 * * * *"
    assert cron.pod_active_deadline == timedelta(minutes=59)
    assert dataset.virtual_poll_deadline_grace == timedelta(minutes=2)
    assert (cron.workers_total, cron.parallelism) == (1, 1)
    assert (cron.cpu, cron.memory) == ("1.7", "7G")
    current, completeness, decode = dataset.validators()
    assert isinstance(current, validation.CheckCurrentData)
    assert current.max_delay == timedelta(hours=7, minutes=40)
    assert isinstance(completeness, validation.CheckVirtualManifestCompleteness)
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    assert not decode.allow_all_nan_vars
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-urma-pds/"
    ((_condition, splits),) = dataset.icechunk_virtual_config.manifest_split.split_sizes
    ((_dim, split_size),) = splits
    assert split_size == 30000


@pytest.mark.slow
def test_local_full_dataset_reads_all_fields_and_times_validators(
    tmp_path: Path,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    dataset = make_dataset(tmp_path, source_fixture["time"])
    time = pd.Timestamp(source_fixture["time"])
    end = time + pd.Timedelta("1h")
    dataset.backfill_local(append_dim_end=end)
    store = dataset.store_factory.primary_store()
    assert isinstance(store, IcechunkStore)
    ds = xr.open_zarr(store, consolidated=False, chunks=None)
    assert dict(ds.sizes) == {"time": 1, "y": 1597, "x": 2345}
    assert len(ds.data_vars) == 14
    job = dataset.region_job_class(
        tmp_store=tmp_path,
        template_ds=dataset.template_config.get_template(end),
        data_vars=dataset.template_config.data_vars,
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test-full-validation",
    )
    context = validation.ValidationContext(
        store=store,
        ds=ds,
        append_dim="time",
        append_dim_frequency=pd.Timedelta("1h"),
        data_vars=dataset.template_config.data_vars,
        region_job=job,
    )
    timings = {}
    now = time + pd.Timedelta("7h42min")
    total_start = perf_counter()
    with monkeypatch.context() as clock:
        clock.setattr(
            pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: now)
        )
        for check in dataset.validators():
            check_start = perf_counter()
            result = check.check(context)
            timings[type(check).__name__] = perf_counter() - check_start
            assert result.passed, result.message
    timings["total"] = perf_counter() - total_start
    record_property("source_time", source_fixture["time"])
    record_property("validator_wall_seconds", json.dumps(timings))
    for var in dataset.template_config.data_vars:
        array = ds[var.name].isel(time=0).values
        assert array.dtype == np.float64
        if (
            var.internal_attrs.available_from is not None
            and time < var.internal_attrs.available_from
        ):
            assert np.isnan(array).all()
            continue
        message = next(
            m
            for m in source_fixture["messages"]
            if m["element"] == var.internal_attrs.grib_element
        )
        expected = np.array([np.nan if v is None else v for v in message["values"]])
        if var.attrs.units == "degree_Celsius":
            expected -= 273.15
        actual = [array[p["y"], p["x"]] for p in source_fixture["points"]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        if var.internal_attrs.requires_bitmap:
            assert int(np.isnan(array).sum()) == message["nan_count"]
        else:
            assert np.isfinite(array).all()
    for point in source_fixture["points"]:
        np.testing.assert_allclose(
            ds.latitude.values[point["y"], point["x"]],
            point["latitude"],
            rtol=0,
            atol=4e-6,
        )
        np.testing.assert_allclose(
            ds.longitude.values[point["y"], point["x"]],
            point["longitude"],
            rtol=0,
            atol=8e-6,
        )
    with xr.open_zarr(store, consolidated=False, chunks=None, decode_cf=False) as raw:
        for var in dataset.template_config.data_vars:
            assert np.isnan(raw[var.name].attrs["_FillValue"])
    dataset.backfill_local(append_dim_end=end)
    reopened = xr.open_zarr(
        dataset.store_factory.primary_store(), consolidated=False, chunks=None
    )
    np.testing.assert_array_equal(reopened.time.values, ds.time.values)
    for var in dataset.template_config.data_vars:
        np.testing.assert_array_equal(
            reopened[var.name].isel(y=798, x=1172).values,
            ds[var.name].isel(y=798, x=1172).values,
        )


@pytest.mark.slow
def test_full_dataset_append_idempotence_and_freshness_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path, "2026-10-12T11:00")
    dataset.backfill_local(append_dim_end=pd.Timestamp("2026-10-12T12:00"))
    now = pd.Timestamp("2026-10-12T19:42")
    discover = NoaaUrmaAnalysisVirtualRegionJob.discover_available
    monkeypatch.setattr(
        NoaaUrmaAnalysisVirtualRegionJob,
        "discover_available",
        lambda self, pending: discover(
            self,
            [
                coord
                for coord in pending
                if coord.time <= pd.Timestamp("2026-10-12T12:00")
            ],
        ),
    )
    before_store = dataset.store_factory.primary_store()
    before = xr.open_zarr(before_store, consolidated=False, chunks=None)
    with monkeypatch.context() as clock:
        clock.setattr(
            pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: now)
        )
        current = dataset.validators()[0]
        stale = current.check(
            validation.ValidationContext(
                store=before_store,
                ds=before,
                append_dim="time",
                append_dim_frequency=pd.Timedelta("1h"),
            )
        )
        assert not stale.passed
        assert "2026-10-12T12:00:00" in stale.message
        dataset.update("test-full-update")
        repo = dataset.store_factory.icechunk_repos(sort="primary-first")[0][1]
        snapshots = sum(1 for _ in repo.ancestry(branch="main"))
        dataset.update("test-idempotent-update")
        assert sum(1 for _ in repo.ancestry(branch="main")) == snapshots
        validation_job = dataset._virtual_validation_region_job(
            dataset.validators(), "test-validated"
        )
    store = dataset.store_factory.primary_store()
    ds = xr.open_zarr(store, consolidated=False, chunks=None)
    np.testing.assert_array_equal(
        ds.time.values,
        pd.date_range("2026-10-12T11:00", periods=2, freq="h").values,
    )
    assert len(ds.data_vars) == 14
    context = validation.ValidationContext(
        store=store,
        ds=ds,
        append_dim="time",
        append_dim_frequency=pd.Timedelta("1h"),
        data_vars=dataset.template_config.data_vars,
        region_job=validation_job,
    )
    with monkeypatch.context() as clock:
        clock.setattr(
            pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: now)
        )
        for check in dataset.validators():
            result = check.check(context)
            assert result.passed, result.message
    missing = ds.isel(time=slice(0, 1))
    with monkeypatch.context() as clock:
        clock.setattr(
            pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: now)
        )
        assert not dataset.validators()[0].check(replace(context, ds=missing)).passed
