import numpy as np

from reformatters.common.config_models import ROOT
from reformatters.noaa.hrrr.forecast_48_hour_virtual.template_config import (
    MODEL_LEVELS,
    PRESSURE_LEVELS,
    NoaaHrrrForecast48HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar

CONFIG = NoaaHrrrForecast48HourVirtualTemplateConfig()


def get_var(path: str) -> NoaaHrrrDataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_group_structure_and_counts() -> None:
    by_group: dict[object, int] = {}
    for var in CONFIG.data_vars:
        by_group[var.group] = by_group.get(var.group, 0) + 1
    assert by_group[ROOT] == 142
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


def test_grid_is_north_first() -> None:
    # north_up decodes every message north-first, so the y/latitude coords descend to
    # match: y from largest (north) to smallest, and the 2D latitude grid's first row
    # is the northernmost. x ascends west to east, so the 2D longitude grid's first
    # column is the westernmost.
    dim_coords = CONFIG.dimension_coordinates()
    y, x = dim_coords["y"], dim_coords["x"]
    assert np.all(np.diff(y) < 0), "y must descend (row 0 = north)"
    assert np.all(np.diff(x) > 0), "x must ascend (col 0 = west)"

    latitudes, longitudes = CONFIG._latitude_longitude_coordinates(x, y)
    assert latitudes[0].mean() > latitudes[-1].mean()
    assert longitudes[:, 0].mean() < longitudes[:, -1].mean()

    spatial_ref = next(c for c in CONFIG.coords if c.name == "spatial_ref")
    geo_transform = spatial_ref.attrs.GeoTransform
    assert geo_transform is not None
    y_pixel_size = float(geo_transform.split()[5])
    assert y_pixel_size < 0, "north-first GeoTransform has a negative y pixel size"
