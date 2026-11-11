from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.validation import compare_spatial
from scripts.validation.compare_spatial import (
    _compute_spatial_stats,
    _select_analysis_snapshot,
    _select_forecast_snapshot,
    _spatial_color_range,
    _variable_spatial_snapshot,
    align_reference_spatially,
    run_compare_spatial,
    select_reference_time,
)
from scripts.validation.summary import _spatial_table
from scripts.validation.utils import RunContext, VariableStats, load_zarr_dataset


def _reference_ds() -> xr.Dataset:
    latitude = np.arange(90, -91, -45)
    longitude = np.arange(-180, 180, 45)
    return xr.Dataset(
        {
            "temperature_2m": (
                ("latitude", "longitude"),
                np.zeros((len(latitude), len(longitude))),
            )
        },
        coords={"latitude": latitude, "longitude": longitude},
    )


def test_align_reference_spatially_keeps_full_longitude_for_geographic_xy(
    geographic_xy_store: str,
) -> None:
    ds = load_zarr_dataset(geographic_xy_store)
    reference_ds = _reference_ds()

    aligned = align_reference_spatially(ds, reference_ds)

    # The dataset's own 0 to 360 longitude bounds would have cropped this to one
    # hemisphere; latitude bounds still apply.
    np.testing.assert_array_equal(
        aligned.longitude.values, reference_ds.longitude.values
    )
    np.testing.assert_array_equal(aligned.latitude.values, reference_ds.latitude.values)


@pytest.mark.parametrize(
    ("var", "label", "comparable"),
    [
        ("pressure_level/wind_u", "mean", True),
        ("derived_radar_reflectivity_1000m", "standard_deviation", False),
        ("temperature_2m_standard_deviation", None, False),
    ],
)
def test_spatial_statistic_selection_labels_and_reference(
    statistic_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    var: str,
    label: str | None,
    comparable: bool,
) -> None:
    ctx = statistic_context
    ctx.variables = [var]
    figure = Mock()
    draw = Mock()
    monkeypatch.setattr(
        "scripts.validation.compare_spatial.plt.subplots",
        Mock(return_value=(figure, np.full((1, 3), Mock()))),
    )
    monkeypatch.setattr("scripts.validation.compare_spatial.plt.close", Mock())
    monkeypatch.setattr(
        "scripts.validation.compare_spatial._draw_spatial_triplet", draw
    )
    run_compare_spatial(ctx)
    args = draw.call_args.args
    data, ref_data = args[4:6]
    assert data.dims == ("latitude", "longitude")
    assert (ref_data is not None) == comparable
    assert ctx.stats[var].ref_available_spatial == comparable
    assert ctx.stats[var].label_value == label
    if label is not None:
        assert f"[statistic={label}]" in args[9]
    if not comparable:
        assert args[10] == "Validation only\nNo comparable reference"
    if var.startswith("pressure_level/"):
        assert "[pressure_level=500]" in args[9]
        assert float(ref_data.mean()) == 2.0


def test_all_nan_declared_mean_stays_visible(
    statistic_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = statistic_context
    ctx.variables = ["wind_u_10m"]
    ctx.validation_ds["wind_u_10m"].loc[{"statistic": "mean"}] = np.nan
    draw = Mock()
    monkeypatch.setattr(
        "scripts.validation.compare_spatial.plt.subplots",
        Mock(return_value=(Mock(), np.full((1, 3), Mock()))),
    )
    monkeypatch.setattr("scripts.validation.compare_spatial.plt.close", Mock())
    monkeypatch.setattr(
        "scripts.validation.compare_spatial._draw_spatial_triplet", draw
    )
    run_compare_spatial(ctx)
    assert draw.call_args.args[4].isnull().all()
    assert ctx.stats["wind_u_10m"].label_value == "mean"
    mean = ctx.stats["wind_u_10m"].val_spatial_mean
    assert mean is not None
    assert np.isnan(mean)


@pytest.mark.parametrize("reference", [False, True])
def test_all_nan_selected_mean_renders_without_switching_statistic(
    statistic_context: RunContext,
    reference: bool,
) -> None:
    ctx = statistic_context
    ctx.variables = ["wind_u_10m"]
    ctx.validation_ds["wind_u_10m"].loc[{"statistic": "mean"}] = np.nan
    if not reference:
        assert ctx.reference_ds is not None
        ctx.reference_ds = ctx.reference_ds.drop_vars("wind_u_10m")
    run_compare_spatial(ctx)
    assert ctx.stats["wind_u_10m"].label_value == "mean"
    assert (ctx.output_dir / "spatial_wind_u_10m.png").exists()


def test_spatial_color_range_uses_available_values_without_hiding_nan_slice() -> None:
    assert _spatial_color_range("wind_u_10m", np.array([]), np.array([2.0, 7.0])) == (
        2.0,
        7.0,
    )
    assert _spatial_color_range("wind_u_10m", np.array([]), np.array([])) == (0.0, 1.0)


@pytest.mark.parametrize("wind", [False, True])
def test_spatial_statistics_preserve_values_and_extrema(wind: bool) -> None:
    data = xr.DataArray([[350.0, np.nan], [10.0, np.nan]], dims=("y", "x"))
    if wind:
        data.attrs.update(standard_name="wind_from_direction", units="degree")
    stats = VariableStats(name="test")
    val_clean, ref_clean = _compute_spatial_stats(data, data.copy(), stats)
    np.testing.assert_array_equal(val_clean, [350.0, 10.0])
    np.testing.assert_array_equal(ref_clean, val_clean)
    for prefix in ("val", "ref"):
        assert getattr(stats, f"{prefix}_spatial_min") == 10.0
        assert getattr(stats, f"{prefix}_spatial_max") == 350.0
        assert getattr(stats, f"{prefix}_spatial_mean") == pytest.approx(
            0.0 if wind else 180.0, abs=1e-10
        )


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize(
    ("delta", "outcome"),
    [
        ("0h", "pass"),
        ("6h", "pass"),
        ("6h1s", "warn"),
        ("24h", "warn"),
        ("24h1s", "fail"),
    ],
)
def test_reference_time_drift_boundaries(
    direction: int,
    delta: str,
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    requested = pd.Timestamp("2026-11-03T12")
    actual = requested + direction * pd.Timedelta(delta)
    reference = xr.Dataset(coords={"time": [actual]})
    warning = Mock()
    monkeypatch.setattr(compare_spatial.log, "warning", warning)
    if outcome == "fail":
        with pytest.raises(ValueError, match="exceeds 24 hours") as error:
            select_reference_time(reference, requested)
        assert str(requested) in str(error.value)
        assert str(actual) in str(error.value)
    else:
        selected, message = select_reference_time(reference, requested)
        assert (message is not None) == (outcome == "warn")
        assert pd.Timestamp(selected.time.item()) == actual
        assert warning.call_count == (1 if outcome == "warn" else 0)


def test_empty_reference_fails_clearly() -> None:

    with pytest.raises(ValueError, match="Reference archive is empty"):
        select_reference_time(
            xr.Dataset(coords={"time": pd.DatetimeIndex([])}),
            pd.Timestamp("2026-11-03"),
        )


def test_sparse_spatial_selection_is_per_variable(sparse_context: RunContext) -> None:
    ctx = sparse_context
    run_compare_spatial(ctx)
    static = ctx.stats["geopotential_height_surface"]
    ordinary = ctx.stats["temperature_2m"]
    control = ctx.stats["pressure_level/geopotential"]
    assert static.spatial_member == control.spatial_member == 0
    assert ordinary.spatial_member == 25
    assert static.val_spatial_mean == 100.0
    assert control.val_spatial_mean == 5000.0
    assert ordinary.val_spatial_mean == 2.0
    assert "lead=0h" in (static.spatial_time_label or "")
    assert "lead=150h" in (ordinary.spatial_time_label or "")
    assert static.ref_spatial_time_label == "2026-11-03T00:00:00"
    assert ordinary.ref_spatial_time_label == "2026-11-09T06:00:00"
    assert ctx.ensemble_member == 25
    assert "lead=150h" in (ctx.spatial_time_label or "")


def test_drift_warning_persists_in_report(sparse_context: RunContext) -> None:

    ctx = sparse_context
    ctx.variables = ["temperature_2m"]
    assert ctx.reference_ds is not None
    ctx.reference_ds = ctx.reference_ds.sel(time=slice(None, "2026-11-08T18"))
    run_compare_spatial(ctx)
    stats = ctx.stats_for("temperature_2m")
    assert stats.reference_time_warning is not None
    report = "\n".join(_spatial_table(stats, ctx))
    assert "Warning" in report
    assert "2026-11-09 06:00:00" in report
    assert "2026-11-08 18:00:00" in report
    assert str(pd.Timedelta("12h")) in report


def test_variable_without_reference_skips_drift_check(
    sparse_context: RunContext,
) -> None:
    ctx = sparse_context
    ctx.variables = ["temperature_2m"]
    assert ctx.reference_ds is not None
    ctx.reference_ds = ctx.reference_ds.drop_vars("temperature_2m").isel(
        time=slice(0, 0)
    )
    run_compare_spatial(ctx)
    stats = ctx.stats_for("temperature_2m")
    assert stats.val_spatial_mean == 2.0
    assert not stats.ref_available_spatial
    assert stats.reference_time_warning is None


@pytest.mark.parametrize("unit", ["ns", "us"])
@pytest.mark.parametrize("virtual", [False, True])
def test_spatial_snapshot_preserves_requested_lead_and_checks_reference(
    sparse_context: RunContext,
    unit: str,
    virtual: bool,
) -> None:
    ctx = sparse_context
    ctx.is_virtual = virtual
    ctx.validation_ds = ctx.validation_ds.assign_coords(
        lead_time=ctx.validation_ds.lead_time.values.astype(f"timedelta64[{unit}]")
    )
    assert ctx.validation_ds.lead_time.dtype == np.dtype(f"timedelta64[{unit}]")
    assert ctx.reference_ds is not None
    snapshot = _select_forecast_snapshot(
        ctx.validation_ds, ctx.reference_ds, ctx.init_time, "150"
    )
    var = "temperature_2m"
    stats = ctx.stats_for(var)
    selected, reference, comparable = _variable_spatial_snapshot(
        ctx, var, snapshot, ctx.reference_ds, stats
    )
    assert pd.Timedelta(selected.lead_time.item()) == pd.Timedelta("150h")
    assert pd.Timestamp(selected.valid_time.item()) == pd.Timestamp("2026-11-09T06")
    assert pd.Timestamp(reference.time.item()) == pd.Timestamp("2026-11-09T06")
    assert comparable
    assert stats.spatial_valid_time == "2026-11-09T06:00:00"
    stale_reference = ctx.reference_ds.isel(time=slice(0, 1))
    with pytest.raises(ValueError, match="exceeds 24 hours") as error:
        _variable_spatial_snapshot(ctx, var, snapshot, stale_reference, stats)
    assert "2026-11-09 06:00:00" in str(error.value)
    assert "2026-11-03 00:00:00" in str(error.value)


@pytest.mark.parametrize("time", [None, "2026-11-03T06:00:00"])
def test_analysis_snapshot_selects_requested_time_and_checks_reference(
    sparse_context: RunContext,
    time: str | None,
) -> None:
    ctx = sparse_context
    ctx.is_virtual = False
    ctx.validation_ds = (
        ctx.validation_ds[["temperature_2m"]]
        .isel(lead_time=0, ensemble_member=0, drop=True)
        .drop_vars("valid_time")
        .rename({"init_time": "time"})
    )
    assert ctx.reference_ds is not None
    requested = pd.Timestamp(time or "2026-11-03T00:00:00")
    snapshot = _select_analysis_snapshot(ctx.validation_ds, time)
    stats = ctx.stats_for("temperature_2m")
    selected, reference, comparable = _variable_spatial_snapshot(
        ctx, "temperature_2m", snapshot, ctx.reference_ds, stats
    )
    assert pd.Timestamp(selected.time.item()) == requested
    assert pd.Timestamp(reference.time.item()) == requested
    assert stats.spatial_valid_time == requested.isoformat()
    assert comparable
    stale_reference = ctx.reference_ds.isel(time=slice(0, 1)).assign_coords(
        time=[requested - pd.Timedelta("25h")]
    )
    with pytest.raises(ValueError, match="exceeds 24 hours"):
        _variable_spatial_snapshot(
            ctx, "temperature_2m", snapshot, stale_reference, stats
        )
