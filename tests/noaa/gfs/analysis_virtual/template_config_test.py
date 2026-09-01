import numpy as np
import pandas as pd
from zarr.codecs import ScaleOffset

from reformatters.common.config_models import ROOT
from reformatters.noaa.gfs.analysis_virtual.template_config import (
    NoaaGfsAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.gfs.virtual_template_config import (
    PRESSURE_LEVELS,
    NoaaGfsVirtualTemplateConfig,
)
from reformatters.noaa.models import NoaaDataVar

CONFIG = NoaaGfsAnalysisVirtualTemplateConfig()
_KELVIN_TO_CELSIUS = ScaleOffset(offset=-273.15, scale=1.0).to_dict()


def get_var(path: str) -> NoaaDataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_hourly_time_structure() -> None:
    assert CONFIG.append_dim == "time"
    assert CONFIG.append_dim_frequency == pd.Timedelta("1h")
    # The first cycle of the 0.25 degree archive. Prepending to an append dim later is a
    # breaking change, so the inclusive start is the only one that stays available.
    assert CONFIG.append_dim_start == pd.Timestamp("2021-03-22T12:00")
    assert CONFIG.dims[ROOT] == ("time", "latitude", "longitude")
    assert CONFIG.dims["pressure_level"] == (
        "time",
        "latitude",
        "longitude",
        "pressure_level",
    )


def test_serves_the_shared_catalog_without_its_running_totals() -> None:
    """The running totals duplicate the 6 hour buckets at every lead an analysis reads."""
    shared = {v.path for v in NoaaGfsVirtualTemplateConfig._catalog_data_vars(CONFIG)}
    served = {v.path for v in CONFIG.data_vars}
    assert len(served) == 293
    assert shared - served == {
        "total_precipitation_run_total_surface",
        "convective_precipitation_run_total_surface",
    }


def test_one_chunk_holds_one_grib_message() -> None:
    assert len(CONFIG.data_vars) == 293
    assert get_var("temperature_2m").encoding.chunks == (1, 721, 1440)
    assert get_var("pressure_level/temperature").encoding.chunks == (1, 721, 1440, 1)
    for var in CONFIG.data_vars:
        assert var.encoding.shards is None, var.path
        assert list(var.encoding.compressors or []) == [], var.path
        assert var.internal_attrs.keep_mantissa_bits == "no-rounding", var.path


def test_pressure_level_coordinate() -> None:
    levels = CONFIG.dimension_coordinates()["pressure_level"]
    assert levels.dtype == np.float64
    assert list(levels) == PRESSURE_LEVELS
    assert len(levels) == 57
    assert (levels[0], levels[-1]) == (1000.0, 0.01)
    (coord,) = [c for c in CONFIG.coords if c.name == "pressure_level"]
    assert (coord.attrs.units, coord.attrs.positive) == ("hPa", "down")


def test_every_windowed_variable_describes_its_window_in_utc_times() -> None:
    windowed = [v for v in CONFIG.data_vars if v.attrs.step_type != "instant"]
    assert len(windowed) == 42
    for var in windowed:
        comment = var.attrs.comment
        assert comment is not None, var.name
        assert comment.startswith(
            (
                "Accumulated since",
                "Averaged over",
                "Maximum value over",
                "Minimum value over",
            )
        ), var.name
        assert "00, 06, 12 or 18 UTC strictly before this time" in comment, var.name
    assert get_var("total_precipitation_surface").attrs.comment == (
        "Accumulated since the most recent 00, 06, 12 or 18 UTC strictly before this "
        "time, so the window is 1 hour at 01, 07, 13 and 19 UTC and lengthens to 6 "
        "hours at 00, 06, 12 and 18 UTC. Subtracting the value at an earlier time with "
        "the same window start gives the exact total between those two times."
    )
    # Only accumulations get the differencing sentence; an average of a lengthening
    # window cannot be differenced.
    assert "Subtracting" not in str(get_var("albedo_surface").attrs.comment)


def test_absolute_temperatures_are_celsius_and_differences_are_kelvin() -> None:
    for path in (
        "temperature_2m",
        "pressure_level/temperature",
        "soil_temperature_0_10cm",
        "ice_temperature_surface",
        "apparent_temperature_2m",
        "maximum_temperature_2m",
        "dew_point_temperature_2m",
    ):
        var = get_var(path)
        assert var.attrs.units == "degree_Celsius", path
        assert list(var.encoding.filters or []) == [_KELVIN_TO_CELSIUS], path
    # Potential temperature is conventionally kelvin; lifted indices are differences.
    for path in (
        "potential_temperature_0p995_sigma",
        "surface_lifted_index_surface",
        "best_4_layer_lifted_index_surface",
    ):
        var = get_var(path)
        assert var.attrs.units == "K", path
        assert list(var.encoding.filters or []) == [], path


def test_snow_water_equivalent_is_scaled_to_metres() -> None:
    var = get_var("snow_water_equivalent_surface")
    assert var.attrs.units == "m"
    assert list(var.encoding.filters or []) == [
        ScaleOffset(offset=0.0, scale=1000.0).to_dict()
    ]


def test_cloud_mixing_ratio_matches_the_pre_2023_element_spelling() -> None:
    for path in (
        "pressure_level/cloud_mixing_ratio",
        "cloud_mixing_ratio_model_level_1",
    ):
        var = get_var(path)
        assert var.internal_attrs.grib_element == "CLMR"
        assert var.internal_attrs.grib_element_alternatives == ("CLWMR",)


def test_flag_variables_only_where_the_source_is_categorical() -> None:
    for path in ("categorical_snow_surface", "instantaneous_categorical_snow_surface"):
        var = get_var(path)
        assert var.attrs.flag_values == (0, 1)
        assert var.attrs.flag_meanings == "no yes"
    assert get_var("land_sea_mask_surface").attrs.flag_values == (0, 1)
    # An area fraction and an interpolated class index are not flag variables.
    for path in ("ice_cover_surface", "soil_type_surface"):
        assert get_var(path).attrs.flag_values is None, path


def test_every_variable_declares_nan_missing_but_the_freezing_level_heights() -> None:
    """The source marks a freezing level at or below ground with an exact 0, so those
    two declare it as the fill value and a CF-aware reader sees NaN there."""
    assert len(CONFIG.data_vars) == 293
    declared_zero = {
        var.path for var in CONFIG.data_vars if var.encoding.fill_value == 0.0
    }
    assert declared_zero == {
        "geopotential_height_0c_isotherm",
        "geopotential_height_highest_tropospheric_freezing_level",
    }
    for path in declared_zero:
        assert get_var(path).attrs.comment == (
            "NaN where the freezing level is at or below ground."
        )
    for var in CONFIG.data_vars:
        assert np.isnan(var.encoding.fill_value) or var.path in declared_zero, var.path
        assert var.internal_attrs.source_fill_value is None, var.path


def test_reflectivity_arrays_name_the_no_echo_floor() -> None:
    """GFS floors reflectivity at -20 dBZ, where HRRR floors it at -10, so the value in
    the comment is GFS's own rather than the one the identically-named HRRR arrays use."""
    floored = [
        var
        for var in CONFIG.data_vars
        if var.internal_attrs.grib_element in ("REFC", "REFD")
    ]
    assert len(floored) == 5
    for var in floored:
        assert "-20 dBZ is the source's no-echo floor" in str(var.attrs.comment), (
            var.path
        )
