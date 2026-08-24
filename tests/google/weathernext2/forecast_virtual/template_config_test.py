from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
import zarr.storage
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common.config_models import ROOT
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    PRESSURE_LEVELS,
    GoogleWeathernext2DataVar,
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)

CONFIG = GoogleWeathernext2ForecastVirtualTemplateConfig()


def get_var(path: str) -> GoogleWeathernext2DataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_group_structure_and_counts() -> None:
    assert CONFIG.dims["pressure_level"] == (
        "init_time",
        "ensemble_member",
        "lead_time",
        "latitude",
        "longitude",
        "pressure_level",
    )
    assert {coord.name for coord in CONFIG.coords} == {
        "expected_forecast_length",
        "init_time",
        "ensemble_member",
        "lead_time",
        "latitude",
        "longitude",
        "spatial_ref",
        "valid_time",
        "pressure_level",
    }
    by_group: dict[object, int] = {}
    for var in CONFIG.data_vars:
        by_group[var.group] = by_group.get(var.group, 0) + 1
    assert by_group[ROOT] == 8
    assert by_group["pressure_level"] == 6


def test_pressure_levels_descend() -> None:
    assert PRESSURE_LEVELS[0] == 1000
    assert PRESSURE_LEVELS[-1] == 50
    assert len(PRESSURE_LEVELS) == 13
    assert PRESSURE_LEVELS == sorted(PRESSURE_LEVELS, reverse=True)


def test_one_output_chunk_per_member_root() -> None:
    var = get_var("temperature_2m")
    assert var.encoding.chunks == (1, 1, 1, 721, 1440)
    assert var.encoding.shards is None
    assert var.encoding.dtype == "float32"
    assert var.encoding.serializer is not None
    assert var.encoding.serializer["name"] == "bytes"
    assert var.encoding.serializer["configuration"] == {"endian": "little"}


def test_one_output_chunk_per_member_and_level() -> None:
    var = get_var("pressure_level/temperature")
    assert var.encoding.chunks == (1, 1, 1, 721, 1440, 1)
    assert var.encoding.shards is None


def test_every_var_accepts_raw_proxy_bytes() -> None:
    for var in CONFIG.data_vars:
        assert var.encoding.compressors == ()


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
    assert var.attrs.comment == (
        "Accumulated over a six-hour forecast interval. Small negative "
        "values are raw model artifacts; set values < 0 to zero."
    )
    filters = var.encoding.filters
    assert filters is not None
    assert filters[0]["configuration"]["scale"] == 0.001


def test_sea_surface_temperature_documents_land_mask() -> None:
    assert get_var("sea_surface_temperature").attrs.comment == (
        "NaN over land where sea surface temperature does not apply."
    )


def test_non_converted_var_has_no_filter() -> None:
    var = get_var("wind_u_10m")
    assert var.encoding.filters in (None, (), [])
    assert var.attrs.units == "m s-1"


def test_mean_only_wind_speed_vars_are_absent() -> None:
    assert "wind_speed_10m" not in {var.name for var in CONFIG.data_vars}
    assert "wind_speed_100m" not in {var.name for var in CONFIG.data_vars}


def test_coords_follow_canonical_global_grid_orientation() -> None:
    dim_coords = CONFIG.dimension_coordinates()
    latitude, longitude = dim_coords["latitude"], dim_coords["longitude"]
    assert len(latitude) == 721
    assert len(longitude) == 1440
    assert np.all(np.diff(latitude) < 0)
    assert (latitude[0], latitude[-1]) == (90, -90)
    assert (longitude.min(), longitude.max()) == (-180, 179.75)
    np.testing.assert_array_equal(dim_coords["ensemble_member"], np.arange(64))
    assert (dim_coords["pressure_level"] == PRESSURE_LEVELS).all()


def test_every_var_covers_the_full_archive() -> None:
    for var in CONFIG.data_vars:
        assert var.internal_attrs.date_available is None, var.path


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
    """The proxy's raw plane is decoded and receives the read-time unit conversion.

    The pressure-level case pins that the trailing size-1 pressure_level dim leaves the
    chunk's memory layout identical to the proxy's (lat, lon) buffer."""
    var = get_var(path)
    chunks = var.encoding.chunks
    assert isinstance(chunks, tuple)
    kelvin = (200 + np.arange(721 * 1440, dtype=np.float32) % 130).reshape(721, 1440)
    assert var.encoding.serializer is not None
    source_bytes = kelvin.tobytes(order="C")

    array = zarr.create_array(
        zarr.storage.LocalStore(str(tmp_path)),
        name=var.name,
        shape=chunks,
        chunks=chunks,
        dtype=var.encoding.dtype,
        fill_value=var.encoding.fill_value,
        filters=var.encoding.filters,
        serializer=var.encoding.serializer,
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
    assert attrs.license == "CC-BY-4.0"
    assert attrs.attribution == (
        "Google requires this attribution: © 2025 DeepMind Technologies Limited's "
        "machine learning models used to "
        "create the experimental data made available at "
        "https://developers.google.com/earth-engine/datasets/catalog/"
        "projects_gcp-public-data-weathernext_assets_weathernext_2_0_0 under CC BY "
        "4.0 licence terms. This data is intended for experimental modelling only "
        "and is not intended, validated, or approved for real world use. Use of the "
        "third-party materials referred to in the Acknowledgements section may be "
        "governed by separate terms and conditions or license provisions. Your use of "
        "the third-party materials is subject to any such terms and you should check "
        "that you can comply with any applicable restrictions or terms and conditions "
        "before use."
    )


def test_spatial_reference_does_not_claim_an_unpublished_datum() -> None:
    spatial_ref = next(coord for coord in CONFIG.coords if coord.name == "spatial_ref")
    assert spatial_ref.attrs.grid_mapping_name == "latitude_longitude"
    assert spatial_ref.attrs.crs_wkt is None
    assert spatial_ref.attrs.semi_major_axis is None
    assert spatial_ref.attrs.spatial_ref is None
