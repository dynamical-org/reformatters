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
    assert CONFIG.append_dim_start == pd.Timestamp("2021-05-01T00:00")
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
    assert len(served) == 272
    assert shared - served == {
        "total_precipitation_run_total_surface",
        "convective_precipitation_run_total_surface",
    }


def test_one_chunk_holds_one_grib_message() -> None:
    assert len(CONFIG.data_vars) == 272
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
                "Accumulated over",
                "Averaged over",
                "Maximum value over",
                "Minimum value over",
            )
        ), var.name
        assert "00, 06, 12 or 18 UTC hour before this time" in comment, var.name
    assert get_var("total_precipitation_surface").attrs.comment == (
        "Accumulated over the preceding 1-6 hours, since the 00, 06, 12 or 18 UTC hour "
        "before this time. Subtracting the value at an earlier time with the same "
        "window start gives the exact total between those two times."
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


def test_every_variable_declares_nan_missing() -> None:
    """No variable declares a non-NaN fill value.

    The two freezing-level heights are the case that makes this worth pinning: the
    source writes an exact 0 where the level is at or below the surface, but over water
    the surface is at zero height, so 0 is also a real answer there. Declaring it as the
    fill value turned roughly 130,000 legitimate ocean cells per field into NaN for a
    CF-aware reader.
    """
    assert len(CONFIG.data_vars) == 272
    for var in CONFIG.data_vars:
        assert np.isnan(var.encoding.fill_value), var.path
        assert var.internal_attrs.source_fill_value is None, var.path
    for path in (
        "geopotential_height_0c_isotherm",
        "geopotential_height_highest_tropospheric_freezing_level",
    ):
        assert get_var(path).attrs.comment == (
            "Zero marks a freezing level at or below the surface rather than missing "
            "data; over water, where the surface is at zero height, it is a genuine "
            "value."
        ), path


def test_in_band_markers_are_described_where_the_source_uses_them() -> None:
    """A variable whose no-data condition is physical rather than instrumental has no
    bitmap, so the source encodes it in band and only the comment can carry it."""
    assert "about 24 km" in str(get_var("visibility_surface").attrs.comment)
    assert "20,000m mark no cloud ceiling" in str(
        get_var("geopotential_height_cloud_ceiling").attrs.comment
    )
    assert "Negative values mark no precipitation" in str(
        get_var("percent_frozen_precipitation_surface").attrs.comment
    )


def test_right_censored_variables_say_the_ceiling_is_data() -> None:
    """Three fields pile a large share of cells on their encodable maximum.

    A saturated value is indistinguishable from a sentinel in the value distribution;
    each of these was settled by physical co-occurrence instead (surface pressure above
    the PLPL ceiling, CAPE nonzero there). The comment has to say the ceiling is
    clipped data, because a reader who masks it loses exactly the cells that matter.
    """
    for path in (
        "pressure_of_lifted_parcel_level_255_0mb",
        "visibility_surface",
    ):
        comment = str(get_var(path).attrs.comment)
        assert comment.startswith("Clipped at the"), path
        assert "not absent" in comment, path
        # No mask range and no fill value: masking would discard real data.
        assert "Mask" not in comment, path
        assert np.isnan(get_var(path).encoding.fill_value), path


def test_sunshine_duration_is_a_six_hour_bucket_despite_its_instant_label() -> None:
    """The index labels SUNSD instantaneous, which is what makes the window string
    match, but the values accumulate and reset every 6 hours of lead time."""
    var = get_var("sunshine_duration_surface")
    assert var.attrs.step_type == "instant"
    assert var.internal_attrs.window_reset_frequency is None
    comment = str(var.attrs.comment)
    assert "accumulated over the preceding 1-6 hours" in comment
    assert "at most 21600 s" in comment


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
        assert "-20 dBZ means no echo was detected" in str(var.attrs.comment), var.path
