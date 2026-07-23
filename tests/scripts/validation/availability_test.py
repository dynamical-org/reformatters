from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
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


def test_run_value_availability_analysis_dataset_fraction_is_float(
    tmp_path: Path,
) -> None:
    """An analysis point series has no non-time dims, so the null-fraction mean is an
    identity op that keeps bool dtype; bool arithmetic then reports fully-missing
    positions as fraction 0.5 instead of 0.0."""
    time = pd.date_range("2020-01-01", periods=6, freq="h")
    lat = np.array([10.0, 20.0])
    lon = np.array([30.0, 40.0])
    values = np.ones((time.size, lat.size, lon.size))
    values[2, :, :] = np.nan  # missing at both points
    values[4, 0, 0] = np.nan  # missing at point 1 only
    ds = xr.Dataset(
        {"temperature_2m": (("time", "latitude", "longitude"), values)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds["temperature_2m"].attrs["step_type"] = "instant"
    ctx = _ctx(ds, tmp_path)
    ctx.variables = ["temperature_2m"]

    run_value_availability(ctx)

    series = ctx.availability["temperature_2m"]
    np.testing.assert_allclose(series.fraction, [1, 1, 0, 1, 0.5, 1])


def test_run_value_availability_exempts_accum_hour_zero(tmp_path: Path) -> None:
    ctx = _ctx(_forecast_dataset(), tmp_path)
    run_value_availability(ctx)

    precipitation = ctx.stats["precipitation_surface"]
    # Structural hour-0 NaNs are excluded: the variable is fully available.
    assert precipitation.positions_complete == precipitation.positions_total == 6
    assert precipitation.first_incomplete is None
    # The plot is still written (every variable gets one for rendering consistency).
    assert precipitation.availability_plot == "availability_precipitation_surface.png"
    assert (tmp_path / "availability_precipitation_surface.png").exists()
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


def _stub_registry(
    monkeypatch: pytest.MonkeyPatch, data_vars: list[SimpleNamespace]
) -> None:
    """Register a stub dataset: all data_vars share one source file group."""
    dataset = SimpleNamespace(
        template_config=SimpleNamespace(data_vars=data_vars),
        region_job_class=SimpleNamespace(source_file_var_groups=lambda dvs: [dvs]),
    )
    monkeypatch.setattr(
        "scripts.validation.availability.find_registered_dataset", lambda _id: dataset
    )


def _stub_var(name: str, has_hour_0: bool) -> SimpleNamespace:
    return SimpleNamespace(name=name, path=name, has_hour_0_values=lambda: has_hour_0)


def test_run_value_availability_exempts_hour_0_override_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An instant var the provider marks hour-0-NaN (hour_0_values_override=False)
    has its structural lead-0 NaN excluded, like an accumulated var's."""
    ds = _forecast_dataset()
    flag = np.ones_like(ds["temperature_2m"].values)
    flag[:, 0, :, :] = np.nan  # structural hour-0 NaN on an *instant* var
    ds["categorical_rain_surface"] = (ds["temperature_2m"].dims, flag)
    ds["categorical_rain_surface"].attrs["step_type"] = "instant"
    _stub_registry(
        monkeypatch,
        [
            _stub_var("temperature_2m", has_hour_0=True),
            _stub_var("precipitation_surface", has_hour_0=False),
            _stub_var("categorical_rain_surface", has_hour_0=False),
        ],
    )
    ctx = _ctx(ds, tmp_path)
    ctx.variables = [*ctx.variables, "categorical_rain_surface"]

    run_value_availability(ctx)

    categorical = ctx.stats["categorical_rain_surface"]
    assert categorical.positions_complete == categorical.positions_total == 6


def test_run_value_availability_sentinel_masked_uses_co_ingested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A masked-sentinel var (missing_value) can't be value-scanned — NaN is ambiguous
    between not-ingested and legitimately-no-data — so its availability is the mean of
    its source-file group co-members'."""
    ds = _forecast_dataset()
    sentinel = np.full(ds["temperature_2m"].shape, np.nan)  # all "no data" at points
    ds["percent_frozen_precipitation_surface"] = (
        ds["temperature_2m"].dims,
        sentinel,
    )
    ds["percent_frozen_precipitation_surface"].attrs["step_type"] = "instant"
    ds["percent_frozen_precipitation_surface"].attrs["missing_value"] = -50.0
    _stub_registry(
        monkeypatch,
        [
            _stub_var("temperature_2m", has_hour_0=True),
            _stub_var("precipitation_surface", has_hour_0=False),
            _stub_var("percent_frozen_precipitation_surface", has_hour_0=False),
        ],
    )
    ctx = _ctx(ds, tmp_path)
    ctx.variables = [*ctx.variables, "percent_frozen_precipitation_surface"]

    run_value_availability(ctx)

    stats = ctx.stats["percent_frozen_precipitation_surface"]
    # Mean of temperature ([1,1,0,1,1,1]) and precipitation ([1]*6) series.
    np.testing.assert_allclose(
        ctx.availability["percent_frozen_precipitation_surface"].fraction,
        [1, 1, 0.5, 1, 1, 1],
    )
    # No direct value scan: no per-point null counts, nothing in the retry list.
    assert stats.null_count_p1 is None
    assert stats.unavailable_timestamps_p1 == []
    assert "percent_frozen" not in (tmp_path / "unavailable_timestamps.txt").read_text()


def test_run_value_availability_sentinel_masked_unregistered_store(
    tmp_path: Path,
) -> None:
    """Without a registered dataset there is nothing co-ingested to measure through;
    the variable reports n/a instead of a fabricated number."""
    ds = _forecast_dataset()
    ds["percent_frozen_precipitation_surface"] = (
        ds["temperature_2m"].dims,
        np.full(ds["temperature_2m"].shape, np.nan),
    )
    ds["percent_frozen_precipitation_surface"].attrs["missing_value"] = -50.0
    ctx = _ctx(ds, tmp_path)
    ctx.variables = [*ctx.variables, "percent_frozen_precipitation_surface"]

    run_value_availability(ctx)

    stats = ctx.stats["percent_frozen_precipitation_surface"]
    assert stats.positions_total is None
    assert "percent_frozen_precipitation_surface" not in ctx.availability
