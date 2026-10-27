from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.rtma.analysis_virtual import NoaaRtmaAnalysisVirtualDataset
from reformatters.noaa.rtma.analysis_virtual.template_config import (
    NoaaRtmaAnalysisVirtualTemplateConfig,
)


def make_dataset(
    tmp_path: Path, start: str = "2017-05-02T00:00"
) -> NoaaRtmaAnalysisVirtualDataset:
    return NoaaRtmaAnalysisVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        template_config=NoaaRtmaAnalysisVirtualTemplateConfig(
            append_dim_start=pd.Timestamp(start)
        ),
    )


def test_active_writer_polls_across_the_next_fire(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    (cron,) = dataset.operational_kubernetes_resources("test-image")
    assert not cron.suspend
    assert cron.as_kubernetes_object()["spec"]["suspend"] is False
    assert (cron.workers_total, cron.parallelism) == (1, 1)
    assert cron.schedule == "41 * * * *"
    assert cron.pod_active_deadline == timedelta(minutes=62, seconds=9)
    fire = pd.Timestamp("2026-10-16T00:41")
    deadline = dataset._virtual_poll_deadline(fire)
    assert fire + timedelta(hours=1) < deadline < fire + cron.pod_active_deadline
    assert deadline < fire + cron.pod_active_deadline
    assert dataset.region_job_class.operational_update_window == pd.Timedelta("24h")
    current, complete, decode = dataset.validators()
    assert isinstance(current, validation.CheckCurrentData)
    assert current.max_delay == timedelta(minutes=73)
    assert isinstance(complete, validation.CheckVirtualManifestCompleteness)
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-rtma-pds/"


@pytest.mark.slow
def test_local_icechunk_virtual_reads_all_real_source_fields(
    tmp_path: Path, source_fixture: dict[str, Any]
) -> None:
    dataset = make_dataset(tmp_path, source_fixture["time"])
    time = pd.Timestamp(source_fixture["time"])
    dataset.backfill_local(append_dim_end=time + pd.Timedelta("1h"))
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
    with xr.open_zarr(store, consolidated=False, chunks=None, decode_cf=False) as raw:
        for var in dataset.template_config.data_vars:
            assert np.isnan(raw[var.name].attrs["_FillValue"])
    dataset.backfill_local(append_dim_end=time + pd.Timedelta("1h"))
    reopened = xr.open_zarr(
        dataset.store_factory.primary_store(), consolidated=False, chunks=None
    )
    np.testing.assert_array_equal(reopened.time.values, ds.time.values)


@pytest.mark.slow
def test_operational_update_appends_new_hour_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)
    dataset.backfill_local(append_dim_end=pd.Timestamp("2017-05-02T01:00"))
    now = pd.Timestamp("2017-05-02T02:40")
    (cron,) = dataset.operational_kubernetes_resources("test-image")
    assert cron.previous_fire_time(now) == pd.Timestamp("2017-05-02T01:41")
    deadline = dataset._virtual_poll_deadline(now)
    assert deadline == pd.Timestamp("2017-05-02T02:42:26")
    assert now < deadline
    with monkeypatch.context() as clock:
        clock.setattr(
            pd.Timestamp,
            "now",
            classmethod(lambda *args, **kwargs: now),
        )
        dataset.update("test-update")
    ds = xr.open_zarr(
        dataset.store_factory.primary_store(), consolidated=False, chunks=None
    )
    np.testing.assert_array_equal(
        ds.time.values, pd.date_range("2017-05-02", periods=2, freq="h").values
    )
    cell = ds.isel(y=798, x=1172).sel(time="2017-05-02T01:00")
    np.testing.assert_allclose(cell.temperature_2m.item(), 11.35, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        cell.cloud_ceiling_height.item(), 4400.6, rtol=0, atol=1e-12
    )
