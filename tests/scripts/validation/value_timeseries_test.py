from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.validation.utils import RunContext
from scripts.validation.value_timeseries import (
    _compute_value_series,
    run_value_timeseries,
)


def _analysis_dataset() -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=8, freq="D")
    lat = np.array([10.0, 20.0])
    lon = np.array([30.0, 40.0])
    rng = np.random.default_rng(0)
    data = rng.standard_normal((time.size, lat.size, lon.size))
    return xr.Dataset(
        {"temperature_2m": (("time", "latitude", "longitude"), data)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def _forecast_dataset() -> xr.Dataset:
    init_time = pd.date_range("2020-01-01", periods=4, freq="D")
    lead_time = pd.to_timedelta([0, 6, 12], unit="h")
    lat = np.array([10.0, 20.0])
    lon = np.array([30.0, 40.0])
    rng = np.random.default_rng(1)
    data = rng.standard_normal((init_time.size, lead_time.size, lat.size, lon.size))
    return xr.Dataset(
        {"temperature_2m": (("init_time", "lead_time", "latitude", "longitude"), data)},
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_compute_value_series_analysis_has_zero_std() -> None:
    ds = _analysis_dataset()
    da_point = ds["temperature_2m"].isel(latitude=0, longitude=0)
    mean_series, std_series = _compute_value_series(da_point)
    assert mean_series.dims == ("time",)
    np.testing.assert_allclose(mean_series.values, da_point.values)
    np.testing.assert_allclose(std_series.values, 0.0)


def test_compute_value_series_forecast_has_nonzero_band() -> None:
    ds = _forecast_dataset()
    da_point = ds["temperature_2m"].isel(latitude=0, longitude=0)
    mean_series, std_series = _compute_value_series(da_point)
    assert mean_series.dims == ("init_time",)
    assert std_series.dims == ("init_time",)
    assert (std_series.values > 0).all()


def _ctx(ds: xr.Dataset, output_dir: Path) -> RunContext:
    return RunContext(
        output_dir=output_dir,
        validation_url="s3://bucket/noaa-test/v1.zarr",
        reference_url=None,
        validation_ds=ds,
        reference_ds=None,
        started_at=pd.Timestamp.now(tz="UTC"),
        point1_sel={"latitude": 0, "longitude": 0},
        point2_sel={"latitude": 1, "longitude": 1},
        point1_lat=10.0,
        point1_lon=30.0,
        point2_lat=20.0,
        point2_lon=40.0,
        ensemble_member=None,
        variables=["temperature_2m"],
    )


def test_run_value_timeseries_writes_plots_and_stats(tmp_path: Path) -> None:
    ds = _analysis_dataset()
    ctx = _ctx(ds, tmp_path)
    run_value_timeseries(ctx)

    assert (tmp_path / "value_timeseries_temperature_2m.png").exists()
    assert (tmp_path / "combined_value_timeseries.png").exists()
    assert ctx.combined_value_timeseries_plot == "combined_value_timeseries.png"

    stats = ctx.stats["temperature_2m"]
    assert stats.value_ts_plot == "value_timeseries_temperature_2m.png"
    assert stats.value_mean_p1 is not None
    assert stats.value_std_p1 == 0.0
