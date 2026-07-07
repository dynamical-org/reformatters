from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.validation.availability import (
    HEATMAP_FILENAME,
    run_value_availability,
    write_availability_artifacts,
)
from scripts.validation.utils import AvailabilitySeries, RunContext


def _forecast_dataset() -> xr.Dataset:
    init_time = pd.date_range("2020-01-01", periods=6, freq="D")
    lead_time = pd.to_timedelta([0, 6, 12], unit="h")
    lat = np.array([10.0, 20.0])
    lon = np.array([30.0, 40.0])
    rng = np.random.default_rng(0)
    temperature = rng.standard_normal(
        (init_time.size, lead_time.size, lat.size, lon.size)
    )
    temperature[2, :, :, :] = np.nan  # one fully-missing position
    precipitation = rng.standard_normal(
        (init_time.size, lead_time.size, lat.size, lon.size)
    )
    precipitation[:, 0, :, :] = np.nan  # structural hour-0 NaN for an accum var
    ds = xr.Dataset(
        {
            "temperature_2m": (
                ("init_time", "lead_time", "latitude", "longitude"),
                temperature,
            ),
            "precipitation_surface": (
                ("init_time", "lead_time", "latitude", "longitude"),
                precipitation,
            ),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds["temperature_2m"].attrs["step_type"] = "instant"
    ds["precipitation_surface"].attrs["step_type"] = "accum"
    return ds


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
        variables=["precipitation_surface", "temperature_2m"],
    )


def test_run_value_availability_flags_missing_position(tmp_path: Path) -> None:
    ctx = _ctx(_forecast_dataset(), tmp_path)
    run_value_availability(ctx)

    assert (tmp_path / HEATMAP_FILENAME).exists()
    assert ctx.combined_availability_plot == HEATMAP_FILENAME

    temperature = ctx.stats["temperature_2m"]
    assert temperature.positions_total == 6
    assert temperature.positions_complete == 5
    assert (
        temperature.first_incomplete
        == temperature.last_incomplete
        == "2020-01-03T00:00"
    )
    assert temperature.availability_plot == "availability_temperature_2m.png"
    assert (tmp_path / "availability_temperature_2m.png").exists()

    series = ctx.availability["temperature_2m"]
    np.testing.assert_allclose(series.fraction, [1, 1, 0, 1, 1, 1])

    # Nulls listed for retry + both points recorded.
    assert ctx.unavailable_timestamps_file == "unavailable_timestamps.txt"
    assert "2020-01-03" in (tmp_path / "unavailable_timestamps.txt").read_text()
    assert temperature.null_count_p1 == 3
    assert temperature.total_count_p1 == 18

    # Point data is cached for run_value_timeseries to reuse.
    assert "temperature_2m" in ctx.loaded_point_data


def test_run_value_availability_exempts_accum_hour_zero(tmp_path: Path) -> None:
    ctx = _ctx(_forecast_dataset(), tmp_path)
    run_value_availability(ctx)

    precipitation = ctx.stats["precipitation_surface"]
    # Structural hour-0 NaNs are excluded: the variable is fully available.
    assert precipitation.positions_complete == precipitation.positions_total == 6
    assert precipitation.availability_plot is None
    assert not (tmp_path / "availability_precipitation_surface.png").exists()
    np.testing.assert_allclose(
        ctx.availability["precipitation_surface"].fraction, np.ones(6)
    )


def test_write_availability_artifacts_ignores_unprobed_positions(
    tmp_path: Path,
) -> None:
    positions = np.array(
        pd.date_range("2020-01-01", periods=4, freq="D"), dtype="datetime64[ns]"
    )
    series = {
        "temperature_2m": AvailabilitySeries(
            positions=positions, fraction=np.array([1.0, 0.0, np.nan, 1.0])
        ),
    }

    heatmap, summaries = write_availability_artifacts(tmp_path, series)

    assert heatmap == HEATMAP_FILENAME
    assert (tmp_path / HEATMAP_FILENAME).exists()
    summary = summaries["temperature_2m"]
    assert summary.positions_total == 3  # NaN (not probed) excluded
    assert summary.positions_complete == 2
    assert summary.first_incomplete == summary.last_incomplete == "2020-01-02T00:00"
    assert summary.plot == "availability_temperature_2m.png"
    assert (tmp_path / summary.plot).exists()
