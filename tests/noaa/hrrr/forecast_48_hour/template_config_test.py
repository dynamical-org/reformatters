import json
from pathlib import Path

import pandas as pd
import pytest

from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)


def test_update_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Ensure that `uv run main <dataset-id> update-template` has been run and
    all changes to NoaaHrrrForecast48HourTemplateConfig are reflected in the on-disk Zarr template.
    """
    template_config = NoaaHrrrForecast48HourTemplateConfig()
    with open(template_config.template_path() / "zarr.json") as f:
        existing_template = json.load(f)

    test_template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(
        NoaaHrrrForecast48HourTemplateConfig,
        "template_path",
        lambda _self: test_template_path,
    )

    template_config.update_template()

    with open(template_config.template_path() / "zarr.json") as f:
        updated_template = json.load(f)

    assert existing_template == updated_template


def test_get_template_coordinates() -> None:
    """Ensure the template has the expected coordinate system."""
    template_config = NoaaHrrrForecast48HourTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )

    # HRRR template should have latitude/longitude coordinates
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "x" in ds.coords
    assert "y" in ds.coords

    # Check that coordinates have the expected dimensions
    assert ds.latitude.dims == ("y", "x")
    assert ds.longitude.dims == ("y", "x")
    assert ds.x.dims == ("x",)
    assert ds.y.dims == ("y",)


def test_template_config_attrs() -> None:
    """Test basic template configuration attributes."""
    config = NoaaHrrrForecast48HourTemplateConfig()

    # Check dimensions
    assert config.dims == ("init_time", "lead_time", "y", "x")
    assert config.append_dim == "init_time"

    # Check date range
    assert config.append_dim_start == pd.Timestamp("2018-07-13T12:00")
    assert config.append_dim_frequency == pd.Timedelta("6h")

    # Check that we have data variables
    data_vars = config.data_vars
    assert len(data_vars) > 0

    # Check that REFC (composite reflectivity) is present as it's a key HRRR variable
    refc_vars = [v for v in data_vars if v.internal_attrs.grib_element == "REFC"]
    assert len(refc_vars) == 1
    assert refc_vars[0].name == "composite_reflectivity"
    assert refc_vars[0].internal_attrs.hrrr_file_type == "sfc"


def test_dimension_coordinates() -> None:
    """Test dimension coordinates are properly configured."""
    config = NoaaHrrrForecast48HourTemplateConfig()
    dim_coords = config.dimension_coordinates()

    # Check required dimensions
    assert "init_time" in dim_coords
    assert "lead_time" in dim_coords
    assert "x" in dim_coords
    assert "y" in dim_coords

    # Check lead_time goes from 0 to 48 hours
    lead_times = dim_coords["lead_time"]
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("48h")
    assert len(lead_times) == 49  # 0 to 48 hours inclusive

    # Check x and y dimensions (HRRR CONUS grid)
    assert len(dim_coords["x"]) == 1799
    assert len(dim_coords["y"]) == 1059


def test_template_variables_have_required_attrs() -> None:
    """Test that all data variables have required attributes."""
    config = NoaaHrrrForecast48HourTemplateConfig()

    for var in config.data_vars:
        # Check variable has name and encoding
        assert var.name
        assert var.encoding

        # Check internal attributes
        assert var.internal_attrs.grib_element
        assert var.internal_attrs.grib_index_level
        assert var.internal_attrs.hrrr_file_type in ["sfc", "prs", "nat", "subh"]

        # Check that variable attributes are set
        assert var.attrs.short_name
        assert var.attrs.long_name
        assert var.attrs.units
        assert var.attrs.step_type in ["instant", "avg", "accum", "max", "min"]


def test_coordinate_configs() -> None:
    """Test coordinate configurations."""
    config = NoaaHrrrForecast48HourTemplateConfig()
    coords = config.coords

    coord_names = [coord.name for coord in coords]

    # Check required coordinates
    required_coords = [
        "init_time",
        "lead_time",
        "x",
        "y",
        "valid_time",
        "ingested_forecast_length",
        "expected_forecast_length",
        "latitude",
        "longitude",
    ]

    for coord_name in required_coords:
        assert coord_name in coord_names, f"Missing coordinate: {coord_name}"


def test_derive_coordinates_integration() -> None:
    """Integration test for derive_coordinates method (requires network access)."""
    config = NoaaHrrrForecast48HourTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2025-01-01"))

    # This would test the actual coordinate derivation, but requires downloading HRRR files
    # Skip by default since it requires network access
    derived_coords = config.derive_coordinates(template_ds)

    # Check that spatial coordinates are derived
    assert "latitude" in derived_coords
    assert "longitude" in derived_coords
