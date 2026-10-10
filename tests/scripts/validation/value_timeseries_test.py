from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.validation.utils import RunContext
from scripts.validation.value_timeseries import (
    _compute_value_series,
    _point_arrays,
    _sample_virtual_points,
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


def _grouped_forecast_dataset() -> xr.Dataset:
    init_time = pd.date_range("2020-01-01", periods=4, freq="D")
    lead_time = pd.to_timedelta([0, 6, 12], unit="h")
    lat = np.array([10.0, 20.0])
    lon = np.array([30.0, 40.0])
    pressure_level = np.array([1000, 500, 100])
    rng = np.random.default_rng(2)
    data = rng.standard_normal((4, 3, 2, 2, 3))
    return xr.Dataset(
        {
            "pressure_level/temperature": (
                ("init_time", "lead_time", "latitude", "longitude", "pressure_level"),
                data,
            )
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "latitude": lat,
            "longitude": lon,
            "pressure_level": pressure_level,
        },
    )


def _ctx(
    ds: xr.Dataset, output_dir: Path, variables: list[str] | None = None
) -> RunContext:
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
        variables=variables or ["temperature_2m"],
    )


def test_run_value_timeseries_writes_plots_and_stats(tmp_path: Path) -> None:
    ds = _analysis_dataset()
    ctx = _ctx(ds, tmp_path)
    run_value_timeseries(ctx)

    assert (tmp_path / "value_timeseries_temperature_2m.png").exists()

    stats = ctx.stats["temperature_2m"]
    assert stats.value_ts_plot == "value_timeseries_temperature_2m.png"
    assert stats.value_mean_p1 is not None
    # A single value per timestep has no meaningful spread: n/a, not 0.
    assert stats.value_std_p1 is None


def test_run_value_timeseries_forecast_draws_std(tmp_path: Path) -> None:
    ds = _forecast_dataset()
    ctx = _ctx(ds, tmp_path)
    run_value_timeseries(ctx)

    assert (tmp_path / "value_timeseries_temperature_2m.png").exists()
    stats = ctx.stats["temperature_2m"]
    assert stats.value_std_p1 is not None
    assert stats.value_std_p1 > 0.0


def test_run_value_timeseries_selects_and_records_level(tmp_path: Path) -> None:
    ds = _grouped_forecast_dataset()
    ctx = _ctx(ds, tmp_path, variables=["pressure_level/temperature"])
    run_value_timeseries(ctx)

    # Filename slugs the '/'; the middle of [1000, 500, 100] is recorded as the level.
    assert (tmp_path / "value_timeseries_pressure_level__temperature.png").exists()
    stats = ctx.stats["pressure_level/temperature"]
    assert stats.level_dim == "pressure_level"
    assert stats.level_value == 500
    # The level is pinned, so the band still spans lead time -> non-zero std.
    assert stats.value_std_p1 is not None
    assert stats.value_std_p1 > 0.0


def test_run_value_timeseries_virtual_samples_one_message_per_position(
    tmp_path: Path,
) -> None:
    ds = _grouped_forecast_dataset()
    ctx = _ctx(ds, tmp_path, variables=["pressure_level/temperature"])
    ctx.is_virtual = True
    run_value_timeseries(ctx)

    stats = ctx.stats["pressure_level/temperature"]
    # The level is pinned to the middle of [1000, 500, 100] and recorded.
    assert stats.level_dim == "pressure_level"
    assert stats.level_value == 500
    # One message per position: a single value per timestep, so std is n/a.
    assert stats.value_std_p1 is None
    assert (tmp_path / "value_timeseries_pressure_level__temperature.png").exists()


def test_sample_virtual_points_pins_representative_slice() -> None:
    ds = _grouped_forecast_dataset()
    ctx = _ctx(ds, Path("/unused"), variables=["pressure_level/temperature"])
    ctx.is_virtual = True
    stats = ctx.stats_for("pressure_level/temperature")

    da = _sample_virtual_points(ctx, "pressure_level/temperature", stats)

    # One value per (position, point): lead/level pinned to a single representative slice.
    assert dict(da.sizes) == {"init_time": 4, "point": 2}
    # Smallest nonzero lead (index 1) and middle level (index 1 of [1000, 500, 100]).
    assert da.lead_time.item() == ds.lead_time.values[1]
    assert da.pressure_level.item() == ds.pressure_level.values[1]
    assert ctx.value_ts_lead_label == "6h"
    assert stats.level_dim == "pressure_level"
    assert stats.level_value == 500
    # Both points come from the same message; values match direct selection.
    expected_p1 = ds["pressure_level/temperature"].values[np.arange(4), 1, 0, 0, 1]
    np.testing.assert_array_equal(da.isel(point=0).values, expected_p1)


def test_sample_virtual_points_pins_nonzero_lead_for_accum() -> None:
    ds = _forecast_dataset()
    ds["temperature_2m"].attrs["step_type"] = "accum"
    ctx = _ctx(ds, Path("/unused"))
    ctx.is_virtual = True
    stats = ctx.stats_for("temperature_2m")

    da = _sample_virtual_points(ctx, "temperature_2m", stats)

    assert da.lead_time.item() > pd.Timedelta(0)


@pytest.mark.parametrize("cached", [False, True])
def test_point_arrays_select_statistic_without_averaging_slices(
    statistic_context: RunContext,
    cached: bool,
) -> None:
    ctx = statistic_context
    var = "pressure_level/wind_u"
    if cached:
        da = ctx.validation_ds[var].sel(pressure_level=500)
        ctx.loaded_point_data[var] = (da.isel(ctx.point1_sel), da.isel(ctx.point2_sel))
    stats = ctx.stats_for(var)
    p1, p2 = _point_arrays(ctx, var, stats)
    assert p1.dims == p2.dims == ("init_time", "lead_time")
    assert float(p1.mean()) == float(p2.mean()) == 2.0
    assert stats.level_value == 500
    assert stats.label_value == "mean"


@pytest.mark.parametrize("var", ["pressure_level/wind_u", "composite_reflectivity"])
def test_virtual_sample_and_value_plot_record_statistic(
    statistic_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    var: str,
) -> None:
    ctx = statistic_context
    ctx.is_virtual = True
    ctx.variables = [var]
    stats = ctx.stats_for(var)
    sampled = _sample_virtual_points(ctx, var, stats)
    assert sampled.dims == ("init_time", "point")
    assert sampled.statistic.item() == stats.label_value
    expected = "mean" if var.startswith("pressure_level/") else "standard_deviation"
    assert stats.label_value == expected
    figure = Mock()
    monkeypatch.setattr(
        "scripts.validation.value_timeseries.plt.subplots",
        Mock(return_value=(figure, np.full((1, 2), Mock()))),
    )
    monkeypatch.setattr("scripts.validation.value_timeseries.plt.close", Mock())
    monkeypatch.setattr("scripts.validation.value_timeseries._draw_value_trace", Mock())
    run_value_timeseries(ctx)
    assert f"[statistic={expected}]" in figure.suptitle.call_args.args[0]
    assert stats.value_std_p1 is None
