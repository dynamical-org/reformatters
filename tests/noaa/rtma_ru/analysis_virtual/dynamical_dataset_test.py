import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr.core.sync
from icechunk import IcechunkStore

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.rtma_ru.analysis_virtual import NoaaRtmaRuAnalysisVirtualDataset
from reformatters.noaa.rtma_ru.analysis_virtual.template_config import (
    NoaaRtmaRuAnalysisVirtualTemplateConfig,
)


def make_dataset(
    tmp_path: Path, start: str = "2017-12-13T00:00"
) -> NoaaRtmaRuAnalysisVirtualDataset:
    return NoaaRtmaRuAnalysisVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        template_config=NoaaRtmaRuAnalysisVirtualTemplateConfig(
            append_dim_start=pd.Timestamp(start)
        ),
    )


def test_active_quarter_hour_overlapping_polling_configuration(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    (cron,) = dataset.operational_kubernetes_resources("test-image")
    assert not cron.suspend
    assert cron.as_kubernetes_object()["spec"]["suspend"] is False
    assert (cron.workers_total, cron.parallelism) == (1, 1)
    assert (cron.cpu, cron.memory) == ("1.7", "7G")
    assert cron.schedule == "6,21,36,51 * * * *"
    assert cron.pod_active_deadline == timedelta(minutes=17, seconds=9)
    assert cron.active_deadline == cron.pod_active_deadline
    assert dataset.virtual_poll_deadline_grace == timedelta(seconds=43)
    assert dataset.region_job_class.operational_update_window == pd.Timedelta("24h")
    for minute in (6, 21, 36, 51):
        fire = pd.Timestamp("2026-10-16T00:00") + timedelta(minutes=minute)
        deadline = dataset._virtual_poll_deadline(fire)
        assert cron.previous_fire_time(fire) == fire
        assert deadline == fire + timedelta(minutes=16, seconds=26)
        assert (
            fire
            < cron.next_fire_time(fire)
            < deadline
            < fire + cron.pod_active_deadline
            < cron.next_fire_time(cron.next_fire_time(fire))
        )
    current, complete, decode = dataset.validators()
    assert isinstance(current, validation.CheckCurrentData)
    assert current.max_delay == timedelta(minutes=43)
    assert isinstance(complete, validation.CheckVirtualManifestCompleteness)
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    assert not decode.allow_all_nan_vars
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-rtma-pds/"
    ((_condition, splits),) = dataset.icechunk_virtual_config.manifest_split.split_sizes
    ((_dimension, split_size),) = splits
    assert split_size == 30000
    assert split_size * len(dataset.template_config.data_vars) == 390000


@pytest.mark.slow
def test_local_icechunk_reads_all_available_fields_and_validates(
    tmp_path: Path, source_fixture: dict[str, Any]
) -> None:
    dataset = make_dataset(tmp_path, source_fixture["time"])
    time = pd.Timestamp(source_fixture["time"])
    end = time + pd.Timedelta("15min")
    dataset.backfill_local(append_dim_end=end)
    store = dataset.store_factory.primary_store()
    ds = xr.open_zarr(store, consolidated=False, chunks=None)
    assert dict(ds.sizes) == {"time": 1, "y": 1597, "x": 2345}
    np.testing.assert_array_equal(ds.time.values, [time.to_datetime64()])
    assert np.all(np.diff(ds.y.values) < 0)
    assert np.all(np.diff(ds.x.values) > 0)
    for var in dataset.template_config.data_vars:
        array = ds[var.name].isel(time=0).values
        assert array.dtype == np.float64
        assert array.shape == (1597, 2345)
        available_from = var.internal_attrs.available_from
        if available_from is not None and time < available_from:
            assert np.isnan(array).all()
            continue
        assert np.isfinite(array).all()
        message = next(
            m
            for m in source_fixture["messages"]
            if m["element"] == var.internal_attrs.grib_element
        )
        expected = np.array(message["values"])
        if var.attrs.units == "degree_Celsius":
            expected -= 273.15
        actual = [array[p["y"], p["x"]] for p in source_fixture["points"]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
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
    job = dataset.region_job_class(
        tmp_store=tmp_path,
        template_ds=dataset.template_config.get_template(end),
        data_vars=dataset.template_config.data_vars,
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test-validation",
    )
    context = validation.ValidationContext(
        store=store,
        ds=ds,
        append_dim="time",
        append_dim_frequency=pd.Timedelta("15min"),
        data_vars=dataset.template_config.data_vars,
        region_job=job,
    )
    for check in dataset.validators()[1:]:
        result = check.check(context)
        assert result.passed, result.message
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
    if time == pd.Timestamp("2020-07-28T00:00"):
        broken_store = dataset.store_factory.primary_store(writable=True)
        assert isinstance(broken_store, IcechunkStore)
        zarr.core.sync.sync(broken_store.delete("total_cloud_cover_atmosphere/c/0/0/0"))
        broken_ds = xr.open_zarr(broken_store, consolidated=False, chunks=None)
        result = validation.CheckVirtualDecodeHealth().check(
            replace(context, store=broken_store, ds=broken_ds)
        )
        assert not result.passed
        assert "total_cloud_cover_atmosphere" in result.message
        assert "entirely NaN" in result.message


@pytest.mark.slow
def test_real_source_missing_first_day_slots(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    job = dataset.region_job_class(
        tmp_store=tmp_path,
        template_ds=dataset.template_config.get_template(
            pd.Timestamp("2017-12-13T01:30")
        ),
        data_vars=dataset.template_config.data_vars,
        append_dim="time",
        region=slice(0, 6),
        reformat_job_name="test-source-gaps",
    )
    available = job.discover_available(list(job.source_file_coords()))
    assert [c.time for c, _ in available] == list(
        pd.to_datetime(
            [
                "2017-12-13T00:00",
                "2017-12-13T00:15",
                "2017-12-13T00:45",
                "2017-12-13T01:00",
            ]
        )
    )


@pytest.mark.slow
def test_operational_update_appends_quarter_hour_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path, "2020-07-28T12:30")
    dataset.backfill_local(append_dim_end=pd.Timestamp("2020-07-28T12:45"))
    now = pd.Timestamp("2020-07-28T13:35")
    (cron,) = dataset.operational_kubernetes_resources("test-image")
    assert cron.previous_fire_time(now) == pd.Timestamp("2020-07-28T13:21")
    assert dataset._virtual_poll_deadline(now) == pd.Timestamp("2020-07-28T13:37:26")
    assert now < dataset._virtual_poll_deadline(now)
    with monkeypatch.context() as clock:
        clock.setattr(pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now))
        before = xr.open_zarr(
            dataset.store_factory.primary_store(), consolidated=False, chunks=None
        )
        stale = dataset.validators()[0].check(
            validation.ValidationContext(
                store=dataset.store_factory.primary_store(),
                ds=before,
                append_dim="time",
                append_dim_frequency=pd.Timedelta("15min"),
            )
        )
        assert not stale.passed
        assert "2020-07-28T12:45:00" in stale.message
        dataset.update("test-update")
        dataset.update("test-idempotent-update")
    ds = xr.open_zarr(
        dataset.store_factory.primary_store(), consolidated=False, chunks=None
    )
    np.testing.assert_array_equal(
        ds.time.values,
        pd.date_range("2020-07-28T12:30", periods=4, freq="15min").values,
    )
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/20200728t1245.json").read_text()
    )
    point = ds.sel(time="2020-07-28T12:45").isel(y=798, x=1172)
    for var in dataset.template_config.data_vars:
        message = next(
            (
                m
                for m in fixture["messages"]
                if m["element"] == var.internal_attrs.grib_element
            ),
            None,
        )
        if message is None:
            assert np.isnan(point[var.name].item())
            continue
        expected = message["values"][1] - (
            273.15 if var.attrs.units == "degree_Celsius" else 0
        )
        np.testing.assert_allclose(point[var.name].item(), expected, rtol=0, atol=1e-12)
