from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import pytest
import zarr
import zarr.storage
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common.config_models import ROOT
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    PER_INIT_STORE_DATE,
    PRESSURE_LEVELS,
    GoogleWeathernext2DataVar,
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)

CONFIG = GoogleWeathernext2ForecastVirtualTemplateConfig()


def get_var(path: str) -> GoogleWeathernext2DataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_group_structure_and_counts() -> None:
    by_group: dict[object, int] = {}
    for var in CONFIG.data_vars:
        by_group[var.group] = by_group.get(var.group, 0) + 1
    assert by_group[ROOT] == 10
    assert by_group["pressure_level"] == 6


def test_pressure_levels_descend() -> None:
    assert PRESSURE_LEVELS[0] == 1000
    assert PRESSURE_LEVELS[-1] == 50
    assert len(PRESSURE_LEVELS) == 13
    assert PRESSURE_LEVELS == sorted(PRESSURE_LEVELS, reverse=True)


def test_one_chunk_per_source_chunk_root() -> None:
    var = get_var("temperature_2m")
    assert var.encoding.chunks == (1, 1, 721, 1440)
    assert var.encoding.shards is None
    assert var.encoding.dtype == "float32"
    # No custom serializer: the referenced bytes are a zarr chunk, decoded by the
    # standard bytes codec plus the source's own blosc compressor.
    assert var.encoding.serializer is None


def test_one_chunk_per_source_chunk_group_includes_level() -> None:
    var = get_var("pressure_level/temperature")
    assert var.encoding.chunks == (1, 1, 721, 1440, 1)
    assert var.encoding.shards is None


def test_every_var_declares_the_source_blosc_compressor() -> None:
    # A reference points at the source's blosc buffer, so the array must decode with the
    # exact codec the source encoded it under.
    for var in CONFIG.data_vars:
        compressors = var.encoding.compressors
        assert compressors is not None
        (blosc,) = compressors
        assert blosc["name"] == "blosc"
        assert blosc["configuration"] == {
            "typesize": 4,
            "cname": "lz4",
            "clevel": 5,
            "shuffle": "shuffle",
            "blocksize": 0,
        }


def test_temperatures_have_kelvin_to_celsius_filter() -> None:
    for path in (
        "temperature_2m",
        "sea_surface_temperature",
        "pressure_level/temperature",
    ):
        var = get_var(path)
        assert var.attrs.units == "degree_Celsius"
        filters = var.encoding.filters
        assert filters is not None
        assert filters[0]["name"] == "scale_offset"
        assert filters[0]["configuration"]["offset"] == -273.15


def test_geopotential_serves_height_in_metres() -> None:
    var = get_var("pressure_level/geopotential_height")
    assert var.attrs.units == "m"
    assert var.attrs.standard_name == "geopotential_height"
    filters = var.encoding.filters
    assert filters is not None
    assert filters[0]["configuration"]["scale"] == 9.80665


def test_precipitation_serves_kg_m2_from_metres() -> None:
    var = get_var("total_precipitation_surface")
    assert var.attrs.units == "kg m-2"
    assert var.attrs.step_type == "accum"
    filters = var.encoding.filters
    assert filters is not None
    assert filters[0]["configuration"]["scale"] == 0.001


def test_non_converted_var_has_no_filter() -> None:
    var = get_var("wind_u_10m")
    assert var.encoding.filters in (None, (), [])
    assert var.attrs.units == "m s-1"


def test_wind_speed_vars_document_the_jensen_gap() -> None:
    for path in ("wind_speed_10m", "wind_speed_100m"):
        comment = get_var(path).attrs.comment
        assert comment is not None
        assert "not the speed of the mean wind vector" in comment


def test_coords_match_the_source_grid_orientation() -> None:
    # Latitude and longitude live inside a source chunk, so a virtual dataset serves
    # them exactly as the source stored them: ascending latitude and 0-360 longitude,
    # unlike our other global 0.25 degree datasets.
    dim_coords = CONFIG.dimension_coordinates()
    latitude, longitude = dim_coords["latitude"], dim_coords["longitude"]
    assert len(latitude) == 721
    assert len(longitude) == 1440
    assert np.all(np.diff(latitude) > 0)
    assert (latitude[0], latitude[-1]) == (-90, 90)
    assert (longitude.min(), longitude.max()) == (0, 359.75)
    assert (dim_coords["pressure_level"] == PRESSURE_LEVELS).all()


def test_pressure_level_vars_start_at_the_per_init_store_era() -> None:
    for var in CONFIG.data_vars:
        expected = None if var.group is ROOT else PER_INIT_STORE_DATE
        assert var.internal_attrs.date_available == expected, var.path


def test_no_rounding_since_virtual_chunks_are_never_rewritten() -> None:
    for var in CONFIG.data_vars:
        assert var.internal_attrs.keep_mantissa_bits == "no-rounding", var.path


def test_lead_times_start_at_6h_and_append_dim() -> None:
    dim_coords = CONFIG.dimension_coordinates()
    lead_time = dim_coords["lead_time"]
    assert len(lead_time) == 60
    assert lead_time.min() == pd.Timedelta("6h")
    assert lead_time.max() == pd.Timedelta("360h")
    assert CONFIG.append_dim_start == pd.Timestamp("2022-01-01T00:00")
    assert CONFIG.append_dim_frequency == pd.Timedelta("6h")


@pytest.mark.parametrize("path", ["temperature_2m", "pressure_level/temperature"])
def test_encoding_decodes_bytes_the_source_wrote(tmp_path: Path, path: str) -> None:
    """A referenced chunk is the source's own numcodecs-blosc buffer, so the variable's
    zarr v3 codec pipeline must decode those exact bytes (and apply the read-time unit
    conversion). Proving it here means the real-source integration test is confirming
    values, not the codec chain.

    The pressure-level case additionally pins that the trailing size-1 pressure_level
    dim leaves the chunk's memory layout identical to the source's (lat, lon) buffer."""
    var = get_var(path)
    chunks = var.encoding.chunks
    assert isinstance(chunks, tuple)
    kelvin = (200 + np.arange(721 * 1440, dtype=np.float32) % 130).reshape(721, 1440)
    source_bytes = numcodecs.Blosc(
        cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE, blocksize=0
    ).encode(kelvin)

    array = zarr.create_array(
        zarr.storage.LocalStore(str(tmp_path)),
        name=var.name,
        shape=chunks,
        chunks=chunks,
        dtype=var.encoding.dtype,
        fill_value=var.encoding.fill_value,
        filters=var.encoding.filters,
        compressors=var.encoding.compressors,
    )
    metadata = array.metadata
    assert isinstance(metadata, ArrayV3Metadata)
    chunk_key = metadata.chunk_key_encoding.encode_chunk_key((0,) * len(chunks))
    chunk_path = tmp_path / var.name / chunk_key
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(source_bytes)

    np.testing.assert_array_equal(np.squeeze(array[:]), kelvin - np.float32(273.15))


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "google-weathernext2-forecast-virtual"
    assert attrs.name == "Google WeatherNext 2 forecast, virtual"
