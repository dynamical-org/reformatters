import numpy as np
import pandas as pd

from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)


def test_spatial_coordinates_from_template() -> None:
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

    # These checks below read from ds AI
    # Check values in x and y coordinates
    assert len(ds.x) == 1799
    assert (ds.x.diff(dim="x") == 3000.0).all()
    assert np.isclose(ds.x.min() - (3000 / 2), -2699020.143)
    assert np.isclose(ds.x.max() + (3000 / 2), 2697979.857)
    assert len(ds.y) == 1059
    assert (ds.y.diff(dim="y") == -3000.0).all()
    assert np.isclose(ds.y.min() - (3000 / 2), -1588806.153)
    assert np.isclose(ds.y.max() + (3000 / 2), 1588193.847)

    # Check values in our computed latitude and longitude coordinates
    assert ds.latitude.min() == 21.138123
    assert ds.latitude.mean() == 37.152527
    # Note the maximum latitude is in the center north of CONUS, so this
    # max is larger than either of the upper corners latitudes.
    assert ds.latitude.max() == 52.615654
    # latitude decreases as we go north to south
    # and the min and max diff in the y direction should be similar
    assert np.isclose(ds.latitude.diff(dim="y").min(), -0.02698135)
    assert np.isclose(ds.latitude.diff(dim="y").max(), -0.0245285)

    assert ds.longitude.min() == -134.09547
    assert ds.longitude.mean() == -97.50583
    assert ds.longitude.max() == -60.917194
    # longitude increases as we go west to east
    # and the min and max diff in the x direction should be similar
    assert np.isclose(ds.longitude.diff(dim="x").min(), 0.02666473)
    assert np.isclose(ds.longitude.diff(dim="x").max(), 0.04299164)


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

    # Check init time
    assert (
        dim_coords["init_time"]
        == pd.date_range("2018-07-13T12:00", "2018-07-13T12:00", freq="6h")
    ).all()

    # Check lead_time goes from 0 to 48 hours
    lead_times = dim_coords["lead_time"]
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("48h")
    assert len(lead_times) == 49  # 0 to 48 hours inclusive

    # These are checked more rigorously in the tests for the base NoaaHrrrTemplateConfig
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
        "spatial_ref",
    ]

    for coord_name in required_coords:
        assert coord_name in coord_names, f"Missing coordinate: {coord_name}"


def test_derive_coordinates_integration() -> None:
    """Integration test for derive_coordinates method (requires network access)."""
    config = NoaaHrrrForecast48HourTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2025-01-01"))

    assert (
        template_ds.coords["init_time"]
        == pd.date_range("2018-07-13T12:00", "2024-12-31T18:00", freq="6h")
    ).all()

    assert (
        template_ds.coords["valid_time"]
        == (template_ds.coords["init_time"] + template_ds.coords["lead_time"])
    ).all()
    assert template_ds.coords["valid_time"].shape == (9454, 49)
