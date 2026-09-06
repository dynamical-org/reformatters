import numpy as np
import pandas as pd
from zarr.codecs import ScaleOffset

from reformatters.common.config_models import ROOT
from reformatters.noaa.gefs.analysis_0_25_degree_virtual.template_config import (
    NoaaGefsAnalysis025DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_B22_TRANSITION_DATE,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.gefs.virtual_template_config import MSLET_AVAILABLE_FROM

CONFIG = NoaaGefsAnalysis025DegreeVirtualTemplateConfig()

_KELVIN_TO_CELSIUS = ScaleOffset(offset=-273.15, scale=1.0).to_dict()


def get_var(name: str) -> NoaaGefsVirtualDataVar:
    return next(v for v in CONFIG.data_vars if v.name == name)


def test_three_hourly_time_structure() -> None:
    assert CONFIG.append_dim == "time"
    assert CONFIG.append_dim_frequency == pd.Timedelta("3h")
    assert CONFIG.dims == {ROOT: ("time", "latitude", "longitude")}
    assert CONFIG.append_dim_start == pd.Timestamp("2020-10-01T00:00")


def test_grid_follows_from_the_source_file_type() -> None:
    """The 0.25 degree grid falls out of source_file_types, not a hardcoded shape."""
    assert CONFIG.source_file_types == frozenset({"s"})
    assert CONFIG.resolution_degrees == 0.25
    dim_coords = CONFIG.dimension_coordinates()
    assert len(dim_coords["latitude"]) == 721
    assert len(dim_coords["longitude"]) == 1440
    # Latitude descends so the decoded north-up message lands in row order.
    assert dim_coords["latitude"][0] == 90.0
    assert dim_coords["latitude"][-1] == -90.0
    assert dim_coords["longitude"][0] == -180.0


def test_serves_the_s_file_inventory_except_surface_geopotential_height() -> None:
    """The s file publishes 38 messages at lead 3 and beyond. HGT@surface, which it
    publishes only at lead 0, is served by the 0.5 degree datasets instead."""
    assert len(CONFIG.data_vars) == 38
    assert "geopotential_height_surface" not in {v.name for v in CONFIG.data_vars}


def test_one_chunk_per_message_encoding() -> None:
    var = get_var("temperature_2m")
    assert var.encoding.chunks == (1, 721, 1440)
    assert var.encoding.shards is None
    assert var.encoding.compressors == ()
    assert var.encoding.dtype == "float64"
    assert var.encoding.serializer is not None
    assert var.encoding.serializer["name"] == "gribberish"


def test_temperatures_are_converted_to_celsius() -> None:
    for name in (
        "temperature_2m",
        "dew_point_temperature_2m",
        "maximum_temperature_2m",
        "minimum_temperature_2m",
    ):
        var = get_var(name)
        assert var.attrs.units == "degree_Celsius", name
        assert var.encoding.filters == [_KELVIN_TO_CELSIUS], name


def test_soil_temperature_is_converted_despite_the_gdal_unit_label() -> None:
    """GDAL labels TSOIL [C] but hands back Kelvin, so the conversion is explicit here
    and every temperature in the dataset shares one unit."""
    var = get_var("soil_temperature_0_10cm")
    assert var.attrs.units == "degree_Celsius"
    assert var.encoding.filters == [_KELVIN_TO_CELSIUS]


def test_snow_water_equivalent_is_scaled_to_metres() -> None:
    var = get_var("snow_water_equivalent_surface")
    assert var.attrs.units == "m"
    assert var.encoding.filters == [ScaleOffset(offset=0.0, scale=1000.0).to_dict()]


def test_accumulation_does_not_reuse_the_materialized_rate_name() -> None:
    """No transform runs in a virtual store, so APCP is served as the run's bucket
    accumulation and must not take precipitation_surface, which is a rate."""
    names = {v.name for v in CONFIG.data_vars}
    assert "precipitation_surface" not in names
    var = get_var("total_precipitation_surface")
    assert var.attrs.units == "kg m-2"
    assert var.attrs.step_type == "accum"
    assert var.attrs.standard_name == "precipitation_amount"


def test_windowed_variables_declare_the_six_hour_reset_and_say_so() -> None:
    windowed = [v for v in CONFIG.data_vars if v.attrs.step_type != "instant"]
    assert len(windowed) == 15
    for var in windowed:
        assert var.internal_attrs.window_reset_frequency == pd.Timedelta("6h"), var.name
        if var.attrs.flag_values is None:
            assert var.attrs.comment is not None, var.name
            assert "(00, 06, 12, 18 UTC) or 3 hour period (03, 09, 15, 21 UTC)" in (
                var.attrs.comment
            ), var.name
    assert get_var("total_cloud_cover_atmosphere").attrs.comment == (
        "Average value in the last 6 hour period (00, 06, 12, 18 UTC) or 3 hour "
        "period (03, 09, 15, 21 UTC)."
    )
    assert get_var("total_precipitation_surface").attrs.comment == (
        "Total accumulated in the last 6 hour period (00, 06, 12, 18 UTC) or 3 hour "
        "period (03, 09, 15, 21 UTC)."
    )


def test_flag_variables_carry_only_their_codes() -> None:
    """flag_values and flag_meanings are the whole meaning of a categorical variable, so
    it carries no comment: a window sentence would contradict them by describing a
    fraction, and restating the codes in prose would let the two representations drift.
    """
    for name in (
        "categorical_snow_surface",
        "categorical_ice_pellets_surface",
        "categorical_freezing_rain_surface",
        "categorical_rain_surface",
    ):
        var = get_var(name)
        assert var.attrs.flag_values == (0, 1), name
        assert var.attrs.flag_meanings == "no yes", name
        assert var.attrs.comment is None, name


def test_extreme_temperatures_are_not_read_from_the_degenerate_lead_0_window() -> None:
    """The source publishes TMAX/TMIN at lead 0 with a zero-length window ("0-0 day max
    fcst"), which is the instantaneous value rather than an extreme. Treating them as
    having no hour-0 values sends the analysis to lead 3 or 6, so the window is always a
    real one and matches what the comment promises."""
    for name in ("maximum_temperature_2m", "minimum_temperature_2m"):
        assert not get_var(name).has_hour_0_values(), name


def test_instant_variables_carry_no_window_reset() -> None:
    for var in CONFIG.data_vars:
        if var.attrs.step_type == "instant":
            assert var.internal_attrs.window_reset_frequency is None, var.name


def test_variables_the_archive_added_later_declare_their_start() -> None:
    """MSLET arrived on its own date; the other three at the pgrb2b transition."""
    assert {
        v.name: v.internal_attrs.available_from
        for v in CONFIG.data_vars
        if v.internal_attrs.available_from is not None
    } == {
        "pressure_reduced_to_mean_sea_level_eta_model": MSLET_AVAILABLE_FROM,
        "visibility_surface": GEFS_B22_TRANSITION_DATE,
        "percent_frozen_precipitation_surface": GEFS_B22_TRANSITION_DATE,
        "geopotential_height_cloud_ceiling": GEFS_B22_TRANSITION_DATE,
    }
    assert MSLET_AVAILABLE_FROM == pd.Timestamp("2021-07-20T12:00")


def test_missing_value_ranges_are_documented_rather_than_masked() -> None:
    """Both sentinels span a range of values, so no single fill_value can express them
    and neither variable is normalized."""
    for name in (
        "percent_frozen_precipitation_surface",
        "geopotential_height_cloud_ceiling",
    ):
        var = get_var(name)
        assert np.isnan(var.encoding.fill_value), name
        assert var.attrs.comment is not None, name
        assert "Mask values" in var.attrs.comment, name


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-gefs-analysis-0-25-degree-virtual"
    assert attrs.name == "NOAA GEFS analysis, 0.25 degree, virtual"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.license == "CC-BY-4.0"
    assert attrs.spatial_resolution == "0.25 degrees (~20km)"
    assert attrs.time_resolution == "3 hours"
    assert attrs.description == (
        "Weather analysis from the Global Ensemble Forecast System (GEFS) "
        "operated by NOAA NWS NCEP."
    )


def test_template_starts_where_the_materialized_forecast_archive_does() -> None:
    """Aligned with noaa-gefs-forecast-35-day rather than the v12 archive start, which
    falls inside the ragged inits that precede it."""
    template = CONFIG.get_template(pd.Timestamp("2020-10-01T09:00"))
    times = template.to_dataset().get_index("time")
    assert list(times) == [
        pd.Timestamp("2020-10-01T00:00"),
        pd.Timestamp("2020-10-01T03:00"),
        pd.Timestamp("2020-10-01T06:00"),
    ]
