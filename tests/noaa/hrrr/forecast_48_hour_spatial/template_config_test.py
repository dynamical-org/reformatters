from reformatters.common.config_models import ROOT
from reformatters.noaa.hrrr.forecast_48_hour_spatial.template_config import (
    MODEL_LEVELS,
    PRESSURE_LEVELS,
    NoaaHrrrForecast48HourSpatialTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar

CONFIG = NoaaHrrrForecast48HourSpatialTemplateConfig()


def get_var(path: str) -> NoaaHrrrDataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_group_structure_and_counts() -> None:
    by_group: dict[object, int] = {}
    for var in CONFIG.data_vars:
        by_group[var.group] = by_group.get(var.group, 0) + 1
    assert by_group[ROOT] == 143
    assert by_group["pressure_level"] == 14
    assert by_group["model_level"] == 20


def test_levels_exclude_pseudo_level() -> None:
    assert PRESSURE_LEVELS[0] == 1000
    assert PRESSURE_LEVELS[-1] == 50
    assert len(PRESSURE_LEVELS) == 39
    assert 1013 not in PRESSURE_LEVELS  # the 1013.2 mb pseudo-level is excluded
    assert MODEL_LEVELS == list(range(1, 51))


def test_one_chunk_per_message_root() -> None:
    var = get_var("temperature_2m")
    # chunk 1 along init_time/lead_time, full spatial grid, no shards/compressors.
    assert var.encoding.chunks == (1, 1, 1059, 1799)
    assert var.encoding.shards is None
    assert var.encoding.compressors == ()
    assert var.encoding.serializer is not None


def test_one_chunk_per_message_group_includes_level() -> None:
    var = get_var("pressure_level/temperature")
    assert var.encoding.chunks == (1, 1, 1059, 1799, 1)
    assert var.encoding.shards is None


def test_temperature_has_kelvin_to_celsius_filter() -> None:
    # Drop-in with the materialized dataset: GribberishCodec decodes raw Kelvin, then a
    # scale_offset filter subtracts 273.15 so temperature/dew point read as Celsius.
    for path in (
        "temperature_2m",
        "dew_point_temperature_2m",
        "pressure_level/temperature",
        "model_level/temperature",
    ):
        var = get_var(path)
        assert var.attrs.units == "degree_Celsius"
        filters = var.encoding.filters
        assert filters is not None
        assert filters[0]["name"] == "scale_offset"
        assert filters[0]["configuration"]["offset"] == -273.15


def test_non_temperature_var_has_no_filter() -> None:
    var = get_var("wind_u_10m")
    assert var.encoding.filters in (None, (), [])
    assert var.attrs.units == "m s-1"


def test_serializer_adjusts_longitude_range() -> None:
    var = get_var("temperature_2m")
    assert var.encoding.serializer is not None
    assert var.encoding.serializer["configuration"]["adjust_longitude_range"] is True
