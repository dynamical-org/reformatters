import numpy as np
import pandas as pd

from reformatters.common.config_models import Encoding
from reformatters.common.zarr import BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig
from reformatters.noaa.gfs.template_config import NoaaGfsCommonTemplateConfig


def test_common_template_config_coords_include_lat_lon_spatial_ref() -> None:
    """Test that common template coords include latitude, longitude, and spatial_ref."""
    config = NoaaGfsForecastTemplateConfig()

    common_coords = NoaaGfsCommonTemplateConfig.coords.fget(config)
    coord_names = [c.name for c in common_coords]

    assert "latitude" in coord_names
    assert "longitude" in coord_names
    assert "spatial_ref" in coord_names
    assert len(coord_names) == 3


def test_common_template_config_latitude_coord_properties() -> None:
    """Test latitude coordinate has correct properties."""
    config = NoaaGfsForecastTemplateConfig()
    lat_values = np.flip(np.arange(-90, 90.25, 0.25))

    common_coords = NoaaGfsCommonTemplateConfig.coords.fget(config)
    lat_coord = next(c for c in common_coords if c.name == "latitude")

    assert lat_coord.encoding.dtype == "float64"
    assert lat_coord.encoding.chunks == len(lat_values)
    assert lat_coord.attrs.units == "degree_north"
    assert lat_coord.attrs.statistics_approximate is not None
    assert lat_coord.attrs.statistics_approximate.min == float(lat_values.min())
    assert lat_coord.attrs.statistics_approximate.max == float(lat_values.max())


def test_common_template_config_longitude_coord_properties() -> None:
    """Test longitude coordinate has correct properties."""
    config = NoaaGfsForecastTemplateConfig()
    lon_values = np.arange(-180, 180, 0.25).astype(np.float64)

    common_coords = NoaaGfsCommonTemplateConfig.coords.fget(config)
    lon_coord = next(c for c in common_coords if c.name == "longitude")

    assert lon_coord.encoding.dtype == "float64"
    assert lon_coord.encoding.chunks == len(lon_values)
    assert lon_coord.attrs.units == "degree_east"
    assert lon_coord.attrs.statistics_approximate is not None
    assert lon_coord.attrs.statistics_approximate.min == float(lon_values.min())
    assert lon_coord.attrs.statistics_approximate.max == float(lon_values.max())


def test_common_template_config_spatial_ref_coord_properties() -> None:
    """Test spatial_ref coordinate has correct properties."""
    config = NoaaGfsForecastTemplateConfig()

    common_coords = NoaaGfsCommonTemplateConfig.coords.fget(config)
    spatial_ref_coord = next(c for c in common_coords if c.name == "spatial_ref")

    assert spatial_ref_coord.encoding.dtype == "int64"
    assert spatial_ref_coord.encoding.chunks == ()
    assert spatial_ref_coord.attrs.units is None
    assert spatial_ref_coord.attrs.statistics_approximate is None
    assert spatial_ref_coord.attrs.crs_wkt is not None
    assert "GEOGCS" in spatial_ref_coord.attrs.crs_wkt
    assert spatial_ref_coord.attrs.grid_mapping_name == "latitude_longitude"
    assert spatial_ref_coord.attrs.comment is not None


def test_common_template_config_get_data_vars() -> None:
    """Test that get_data_vars returns all expected data variables."""
    config = NoaaGfsForecastTemplateConfig()

    encoding = Encoding(
        dtype="float32",
        fill_value=0,
        chunks=(1, 105, 121, 121),
        shards=(1, 210, 726, 726),
        compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
    )

    data_vars = config.get_data_vars(encoding)

    # Check that we have all expected variables
    expected_var_names = [
        "pressure_surface",
        "temperature_2m",
        "relative_humidity_2m",
        "maximum_temperature_2m",
        "minimum_temperature_2m",
        "wind_u_10m",
        "wind_v_10m",
        "wind_u_100m",
        "wind_v_100m",
        "percent_frozen_precipitation_surface",
        "precipitation_surface",
        "categorical_snow_surface",
        "categorical_ice_pellets_surface",
        "categorical_freezing_rain_surface",
        "categorical_rain_surface",
        "precipitable_water_atmosphere",
        "total_cloud_cover_atmosphere",
        "geopotential_height_cloud_ceiling",
        "downward_short_wave_radiation_flux_surface",
        "downward_long_wave_radiation_flux_surface",
        "pressure_reduced_to_mean_sea_level",
    ]

    actual_var_names = [v.name for v in data_vars]
    assert actual_var_names == expected_var_names


def test_common_template_config_get_data_vars_encoding() -> None:
    """Test that get_data_vars applies the provided encoding to all variables."""
    config = NoaaGfsForecastTemplateConfig()

    custom_encoding = Encoding(
        dtype="float32",
        fill_value=-999,
        chunks=(2, 50, 60, 60),
        shards=(2, 100, 120, 120),
        compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
    )

    data_vars = config.get_data_vars(custom_encoding)

    for var in data_vars:
        assert var.encoding.dtype == "float32"
        assert var.encoding.fill_value == -999
        assert var.encoding.chunks == (2, 50, 60, 60)
        assert var.encoding.shards == (2, 100, 120, 120)


def test_common_template_config_get_data_vars_internal_attrs() -> None:
    """Test that get_data_vars sets correct internal attributes."""
    config = NoaaGfsForecastTemplateConfig()

    encoding = Encoding(
        dtype="float32",
        fill_value=0,
        chunks=(1, 105, 121, 121),
        shards=(1, 210, 726, 726),
        compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
    )

    data_vars = config.get_data_vars(encoding)
    var_by_name = {v.name: v for v in data_vars}

    # Test a few key variables for correct internal attrs
    temp_2m = var_by_name["temperature_2m"]
    assert temp_2m.internal_attrs.grib_element == "TMP"
    assert temp_2m.internal_attrs.grib_index_level == "2 m above ground"
    assert temp_2m.attrs.step_type == "instant"

    precip = var_by_name["precipitation_surface"]
    assert precip.internal_attrs.grib_element == "APCP"
    assert precip.internal_attrs.deaccumulate_to_rate is True
    assert precip.internal_attrs.window_reset_frequency == pd.Timedelta("6h")
    assert precip.attrs.step_type == "avg"

    # Check categorical variables have no-rounding
    csnow = var_by_name["categorical_snow_surface"]
    assert csnow.internal_attrs.keep_mantissa_bits == "no-rounding"


def test_common_template_config_inheritance() -> None:
    """Test that NoaaGfsForecastTemplateConfig inherits from NoaaGfsCommonTemplateConfig."""
    assert issubclass(NoaaGfsForecastTemplateConfig, NoaaGfsCommonTemplateConfig)


def test_forecast_template_config_coords_include_common_coords() -> None:
    """Test that forecast coords property includes latitude, longitude, spatial_ref from common config."""
    config = NoaaGfsForecastTemplateConfig()
    coords = config.coords
    coord_names = [c.name for c in coords]

    # Should include common coords
    assert "latitude" in coord_names
    assert "longitude" in coord_names
    assert "spatial_ref" in coord_names

    # Should also include forecast-specific coords
    assert "init_time" in coord_names
    assert "lead_time" in coord_names
    assert "valid_time" in coord_names


def test_forecast_template_config_data_vars_match_common_get_data_vars() -> None:
    """Test that get_data_vars returns same variables as forecast data_vars property."""
    config = NoaaGfsForecastTemplateConfig()

    # Get data vars from the property
    forecast_data_vars = config.data_vars

    # Get data vars from the common method with same encoding
    common_data_vars = config.get_data_vars(forecast_data_vars[0].encoding)

    # Names should match
    forecast_names = [v.name for v in forecast_data_vars]
    common_names = [v.name for v in common_data_vars]
    assert forecast_names == common_names
