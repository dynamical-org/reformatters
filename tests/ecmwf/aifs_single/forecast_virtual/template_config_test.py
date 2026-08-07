import numpy as np
import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.ecmwf.aifs_single.forecast_virtual.template_config import (
    AIFS_2026_UPGRADE_DATE,
    AIFS_SINGLE_FORMAT_CHANGE_DATE,
    PRESSURE_LEVELS,
    EcmwfAifsSingleForecastVirtualTemplateConfig,
    EcmwfAifsSingleVirtualDataVar,
)

CONFIG = EcmwfAifsSingleForecastVirtualTemplateConfig()


def get_var(path: str) -> EcmwfAifsSingleVirtualDataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_group_structure_and_counts() -> None:
    by_group: dict[object, int] = {}
    for var in CONFIG.data_vars:
        by_group[var.group] = by_group.get(var.group, 0) + 1
    assert by_group[ROOT] == 29
    assert by_group["pressure_level"] == 6


def test_pressure_levels() -> None:
    assert PRESSURE_LEVELS[0] == 1000
    assert PRESSURE_LEVELS[-1] == 10
    assert len(PRESSURE_LEVELS) == 14


def test_one_chunk_per_message_root() -> None:
    var = get_var("temperature_2m")
    assert var.encoding.chunks == (1, 1, 721, 1440)
    assert var.encoding.shards is None
    assert var.encoding.compressors == ()
    assert var.encoding.serializer is not None


def test_one_chunk_per_message_group_includes_level() -> None:
    var = get_var("pressure_level/temperature")
    assert var.encoding.chunks == (1, 1, 721, 1440, 1)
    assert var.encoding.shards is None


def test_temperatures_have_kelvin_to_celsius_filter() -> None:
    for path in (
        "temperature_2m",
        "dew_point_temperature_2m",
        "skin_temperature_surface",
        "soil_temperature_layer_1",
        "soil_temperature_layer_2",
        "pressure_level/temperature",
    ):
        var = get_var(path)
        assert var.attrs.units == "degree_Celsius"
        filters = var.encoding.filters
        assert filters is not None
        assert filters[0]["name"] == "scale_offset"
        assert filters[0]["configuration"]["offset"] == -273.15


def test_geopotential_serves_height_in_metres() -> None:
    # Source z is geopotential (m2 s-2); ScaleOffset divides by standard gravity on
    # read, matching the materialized geopotential_height_* variables.
    for path in ("geopotential_height_surface", "pressure_level/geopotential_height"):
        var = get_var(path)
        assert var.attrs.units == "m"
        assert var.attrs.standard_name == "geopotential_height"
        filters = var.encoding.filters
        assert filters is not None
        assert filters[0]["name"] == "scale_offset"
        assert filters[0]["configuration"]["scale"] == 9.80665


def test_non_temperature_var_has_no_filter() -> None:
    var = get_var("wind_u_10m")
    assert var.encoding.filters in (None, (), [])
    assert var.attrs.units == "m s-1"


def test_coords_match_codec_decoded_grid() -> None:
    dim_coords = CONFIG.dimension_coordinates()
    latitude, longitude = dim_coords["latitude"], dim_coords["longitude"]
    assert len(latitude) == 721
    assert len(longitude) == 1440
    assert np.all(np.diff(latitude) < 0), "latitude must descend (row 0 = north)"
    assert latitude[0] == 90
    assert longitude.min() == -180
    assert longitude.max() < 180
    assert (dim_coords["pressure_level"] == PRESSURE_LEVELS).all()


def test_era_availability() -> None:
    assert get_var("pressure_surface").internal_attrs.date_available is None
    assert get_var("land_sea_mask_surface").internal_attrs.date_available is None
    for path in (
        "wind_u_100m",
        "total_cloud_cover_atmosphere",
        "total_precipitation_run_total_surface",
        "convective_precipitation_run_total_surface",
        "downward_short_wave_radiation_run_total_surface",
        "soil_temperature_layer_1",
        "volumetric_soil_moisture_layer_2",
        "standard_deviation_of_sub_gridscale_orography_surface",
    ):
        assert (
            get_var(path).internal_attrs.date_available
            == AIFS_SINGLE_FORMAT_CHANGE_DATE
        )
    assert (
        get_var("snow_area_fraction_surface").internal_attrs.date_available
        == AIFS_2026_UPGRADE_DATE
    )


def test_statics_are_lead_0_only() -> None:
    static_paths = {
        "land_sea_mask_surface",
        "geopotential_height_surface",
        "standard_deviation_of_sub_gridscale_orography_surface",
        "slope_of_sub_gridscale_orography_surface",
    }
    for var in CONFIG.data_vars:
        assert var.internal_attrs.lead_0_only == (var.path in static_paths)


def test_soil_vars_index_under_sol_levels() -> None:
    sot1 = get_var("soil_temperature_layer_1")
    assert sot1.internal_attrs.grib_index_level_type == "sol"
    assert sot1.internal_attrs.grib_index_level_value == 1
    vsw2 = get_var("volumetric_soil_moisture_layer_2")
    assert vsw2.internal_attrs.grib_index_level_type == "sol"
    assert vsw2.internal_attrs.grib_index_level_value == 2


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "ecmwf-aifs-single-forecast-virtual"
    assert attrs.name == "ECMWF AIFS Single forecast, virtual"


def test_lead_times_and_append_dim() -> None:
    dim_coords = CONFIG.dimension_coordinates()
    assert len(dim_coords["lead_time"]) == 61
    assert dim_coords["lead_time"].max() == pd.Timedelta("360h")
    assert CONFIG.append_dim_start == pd.Timestamp("2024-04-01T00:00")
    assert CONFIG.append_dim_frequency == pd.Timedelta("6h")
