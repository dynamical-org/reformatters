from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from scripts.validation.compare_spatial import (
    _spatial_color_range,
    align_reference_spatially,
    run_compare_spatial,
)
from scripts.validation.utils import RunContext, load_zarr_dataset


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
        ("composite_reflectivity", "standard_deviation", False),
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
