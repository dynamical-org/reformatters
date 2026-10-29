from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scripts.validation.compare_spatial import (
    _compute_spatial_stats,
    _roll_longitude_seam_to_edge,
    _spatial_color_range,
    align_reference_spatially,
    run_compare_spatial,
)
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

    np.testing.assert_array_equal(
        aligned.longitude.values, reference_ds.longitude.values
    )
    np.testing.assert_array_equal(aligned.latitude.values, reference_ds.latitude.values)


@pytest.mark.parametrize("two_dimensional", [False, True])
def test_roll_longitude_seam_to_edge_makes_longitude_increase(
    geographic_xy_store: str,
    two_dimensional: bool,
) -> None:
    data = load_zarr_dataset(geographic_xy_store)["temperature_2m"].load()
    if two_dimensional:
        latitude, longitude = xr.broadcast(data.latitude, data.longitude)
        data = data.assign_coords(latitude=latitude, longitude=longitude)

    rolled = _roll_longitude_seam_to_edge(data)

    np.testing.assert_array_equal(
        rolled.longitude.isel(y=0, missing_dims="ignore").values,
        [-180, -135, -90, -45, 0, 45, 90, 135],
    )
    np.testing.assert_array_equal(rolled.sel(x=225.0).values, data.sel(x=225.0).values)
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(rolled.longitude, rolled.latitude, rolled.values)
    np.testing.assert_array_equal(mesh.get_array(), rolled.values)
    mesh_longitude = np.asarray(mesh.get_coordinates())[..., 0]
    np.testing.assert_array_equal(np.diff(mesh_longitude, axis=1), 45)
    plt.close(fig)


@pytest.mark.parametrize(
    "longitude_row", [[-100.0, -99.0, -98.0], [-98.0, -99.0, -100.0]]
)
def test_roll_longitude_seam_to_edge_leaves_unwrapped_longitude_unchanged(
    longitude_row: list[float],
) -> None:
    longitude = np.array([longitude_row, longitude_row])
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("y", "x"),
        coords={"longitude": (("y", "x"), longitude)},
    )

    assert _roll_longitude_seam_to_edge(data) is data


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
