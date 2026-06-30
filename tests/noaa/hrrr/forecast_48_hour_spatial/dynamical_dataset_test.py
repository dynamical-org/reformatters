from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
)
from reformatters.noaa.hrrr.forecast_48_hour_spatial.dynamical_dataset import (
    NoaaHrrrForecast48HourSpatialDataset,
)
from reformatters.noaa.hrrr.forecast_48_hour_spatial.region_job import (
    NoaaHrrrForecast48HourSpatialRegionJob,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from tests.common.dynamical_dataset_test import (
    NOOP_STORAGE_CONFIG,
    assert_configured_validators,
)

# A heavy-rain cell in the 2024-06-01T00 init (gribberish row/col, south-first y).
_Y, _X = 423, 1062
_INIT = "2024-06-01T00:00"

# Variables spanning root + pressure_level + model_level. "temperature" matches both
# the pressure_level and model_level group vars (un-suffixed group var names).
_FILTER_VARS = [
    "temperature_2m",
    "wind_u_10m",
    "total_precipitation_surface",
    "temperature",
]


def make_dataset(tmp_path: Path) -> NoaaHrrrForecast48HourSpatialDataset:
    return NoaaHrrrForecast48HourSpatialDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaHrrrForecast48HourSpatialDataset:
    return make_dataset(tmp_path)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)

    # Trim to leads 0h and 6h to limit work (virtual backfill downloads only .idx
    # sidecars; decode happens when the snapshot cells are read).
    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).isel(lead_time=[0, 6]),
    )

    # 1. Backfill the single 2024-06-01T00 init.
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-06-01T01:00"),
        filter_start=pd.Timestamp(_INIT),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert ds.init_time.values[-1] == np.datetime64("2024-06-01T00:00")

    cell = ds.isel(y=_Y, x=_X).sel(init_time=_INIT)
    f6 = cell.sel(lead_time=pd.Timedelta("6h"))
    # Snapshot values (decoded raw GRIB; temperature converted K->C by the codec).
    np.testing.assert_allclose(f6["temperature_2m"].values, 20.752984619140648)
    np.testing.assert_allclose(f6["wind_u_10m"].values, -7.187067031860352)
    # Raw window accumulation (kg m-2), not the materialized deaccumulated rate.
    np.testing.assert_allclose(f6["total_precipitation_surface"].values, 98.101)
    np.testing.assert_allclose(
        f6["pressure_level/temperature"].sel(pressure_level=500).values,
        -10.818884277343727,
    )
    np.testing.assert_allclose(
        f6["model_level/temperature"].sel(model_level=1).values, 20.325891113281273
    )

    # Hour-0 handling: accumulated precip is absent (excluded in coord generation),
    # instant fields are present.
    f0 = cell.sel(lead_time=pd.Timedelta("0h"))
    assert np.isnan(f0["total_precipitation_surface"].values)
    assert not np.isnan(f0["temperature_2m"].values)
    assert not np.isnan(f0["pressure_level/temperature"].sel(pressure_level=500).values)

    # 2. Operational update: "now" during the 06z publication window.
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2024-06-01T08:00")),
    )
    orig_update_jobs = (
        NoaaHrrrForecast48HourSpatialRegionJob.operational_update_jobs.__func__  # type: ignore[attr-defined]
    )

    def filtered_update_jobs(
        cls: type[NoaaHrrrForecast48HourSpatialRegionJob],
        *,
        all_data_vars: Sequence[NoaaHrrrDataVar],
        **kwargs: Any,  # noqa: ANN401 - passthrough to the wrapped classmethod
    ) -> object:
        return orig_update_jobs(
            cls,
            all_data_vars=[v for v in all_data_vars if v.name in _FILTER_VARS],
            **kwargs,
        )

    monkeypatch.setattr(
        NoaaHrrrForecast48HourSpatialRegionJob,
        "operational_update_jobs",
        classmethod(filtered_update_jobs),
    )

    dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    # The update window (24h before 08:00) ingests the 06z init too.
    assert updated.init_time.values[-1] == np.datetime64("2024-06-01T06:00")
    new_cell = updated.isel(y=_Y, x=_X).sel(
        init_time="2024-06-01T06:00", lead_time=pd.Timedelta("6h")
    )
    t6 = float(new_cell["temperature_2m"].values)
    assert -60.0 < t6 < 60.0  # plausible Celsius
    assert not np.isnan(new_cell["model_level/temperature"].sel(model_level=1).values)

    assert_configured_validators(dataset)


@pytest.mark.slow
def test_dropin_matches_materialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointwise drop-in: temperature/wind match the materialized noaa-hrrr-forecast-48-hour
    at the same init/lead/location; the precipitation accumulation differs from the
    materialized deaccumulated rate (and is a distinct variable)."""
    lead = pd.Timedelta("6h")

    # Virtual dataset: temperature_2m (codec K->C), wind_u_10m (raw), accumulation.
    vds = make_dataset(tmp_path / "virtual")
    v_orig = vds.template_config.get_template
    monkeypatch.setattr(
        type(vds.template_config),
        "get_template",
        lambda self, end_time: v_orig(end_time).isel(lead_time=[0, 6]),
    )
    vds.backfill_local(
        append_dim_end=pd.Timestamp("2024-06-01T01:00"),
        filter_start=pd.Timestamp(_INIT),
        filter_variable_names=[
            "temperature_2m",
            "wind_u_10m",
            "total_precipitation_surface",
            "snow_water_equivalent_surface",
            "snow_area_fraction_surface",
        ],
    )
    v = validation.open_flattened_dataset(
        vds.store_factory.primary_store(), consolidated=False
    )
    y_value = float(v.y.values[_Y])
    x_value = float(v.x.values[_X])
    v_cell = v.sel(init_time=_INIT, lead_time=lead).sel(
        y=y_value, x=x_value, method="nearest"
    )

    # Materialized dataset, same init/leads, pointwise vars.
    mds = NoaaHrrrForecast48HourDataset(primary_storage_config=NOOP_STORAGE_CONFIG)
    m_orig = mds.template_config.get_template
    monkeypatch.setattr(
        type(mds.template_config),
        "get_template",
        lambda self, end_time: m_orig(end_time).isel(lead_time=[0, 6]),
    )
    mds.backfill_local(
        append_dim_end=pd.Timestamp("2024-06-01T01:00"),
        filter_start=pd.Timestamp(_INIT),
        filter_variable_names=[
            "temperature_2m",
            "wind_u_10m",
            "precipitation_surface",
            "snow_water_equivalent_surface",
            "snow_area_fraction_surface",
        ],
    )
    m = xr.open_zarr(
        mds.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    m_cell = m.sel(init_time=_INIT, lead_time=lead).sel(
        y=y_value, x=x_value, method="nearest"
    )

    # Pointwise drop-in (materialized rounds mantissa bits, hence the tolerance).
    np.testing.assert_allclose(
        v_cell["temperature_2m"].values, m_cell["temperature_2m"].values, atol=0.3
    )
    np.testing.assert_allclose(
        v_cell["wind_u_10m"].values, m_cell["wind_u_10m"].values, atol=0.3
    )
    # Accumulation (kg m-2) vs the materialized deaccumulated rate (kg m-2 s-1) differ
    # by orders of magnitude - the virtual dataset cannot deaccumulate on read, so it
    # serves the raw window total under a distinct name, by design.
    accumulation = float(v_cell["total_precipitation_surface"].values)
    rate = float(m_cell["precipitation_surface"].values)
    assert accumulation > 1.0
    assert accumulation > rate * 100

    # Snow vars are a true drop-in: the virtual ScaleOffset filter applies the same
    # conversion the materialized dataset bakes in (WEASD kg m-2 -> m, SNOWC % ->
    # fraction). Compare full-field means, which are orientation-independent and
    # nonzero thanks to early-June high-elevation snow, so a wrong scale direction
    # (off by 100x / 1000x) would be caught. Materialized rounds mantissa bits.
    v_full = v.sel(init_time=_INIT, lead_time=lead)
    m_full = m.sel(init_time=_INIT, lead_time=lead)
    for name in ("snow_water_equivalent_surface", "snow_area_fraction_surface"):
        v_mean = float(np.nanmean(v_full[name].values))
        m_mean = float(np.nanmean(m_full[name].values))
        assert m_mean > 0.0
        np.testing.assert_allclose(v_mean, m_mean, rtol=0.02)


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrForecast48HourSpatialDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))
    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    # Single-writer virtual update: no fan-out.
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert update_cron_job.pod_active_deadline < timedelta(hours=6)
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert len(update_cron_job.secret_names) > 0
    # Suspended until the initial backfill completes; flip when un-suspending.
    assert update_cron_job.suspend is True
    assert validation_cron_job.suspend is True


def test_validators(dataset: NoaaHrrrForecast48HourSpatialDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    assert any(
        isinstance(v, validation.CheckVirtualManifestCompleteness) for v in validators
    )
    assert any(isinstance(v, validation.CheckVirtualDecodeHealth) for v in validators)


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaHrrrForecast48HourSpatialDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-hrrr-bdp-pds/"
