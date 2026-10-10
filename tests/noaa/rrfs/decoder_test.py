import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from gribberish.zarr import GribberishCodec

from reformatters.noaa.rrfs.forecast_18_hour_virtual.template_config import (
    NoaaRrfsForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.template_config import (
    NoaaRrfsForecast84HourVirtualTemplateConfig,
)
from tests.noaa.rrfs.decoder_helpers import write_single_grib_chunk

FIXTURES = Path(__file__).parent / "fixtures"
CASES = (
    ("spfh", "SPFH"),
    ("pevpr", "PEVPR"),
    ("pevap", "PEVAP"),
    ("vegmin", "VEGMIN"),
    ("vegmax", "VEGMAX"),
    ("aotk", "AOTK"),
    ("wildfire", "var discipline=2 master_table=2 parmcat=4 parm=26"),
)

SEMANTIC_MARKERS = (
    ("pressure_cloud_top", -50000.0),
    ("pressure_grid_scale_cloud_top_level", -50000.0),
    ("pressure_grid_scale_cloud_bottom_level", -50000.0),
    ("temperature_cloud_top", -500.0),
    ("geopotential_height_supercooled_liquid_water_base", -5000.0),
    ("geopotential_height_supercooled_liquid_water_top", -5000.0),
    ("soil_porosity_surface", 0.0),
    ("maximum_snow_albedo_surface", 6.0),
    ("direct_evaporation_cease_soil_moisture_surface", 0.0),
    ("albedo_surface", 0.0),
    ("wilting_point_surface", 0.0),
    ("transpiration_stress_onset_soil_moisture_surface", 0.0),
    ("minimal_stomatal_resistance_surface", 0.0),
)


@pytest.mark.parametrize(
    ("fixture_name", "variable_name", "element", "marker"),
    [
        (
            "percent-frozen-marker.grib2",
            "percent_frozen_precipitation_surface",
            "CPOFP",
            -50.0,
        ),
        ("cloud-top-marker.grib2", "geopotential_height_cloud_top", "HGT", -5000.0),
    ]
    + [
        (f"semantic-markers/{sample}-{variable}.grib2", variable, None, marker)
        for sample in ("A", "C")
        for variable, marker in SEMANTIC_MARKERS
    ],
)
def test_semantic_markers_read_as_nan(
    fixture_name: str,
    variable_name: str,
    element: str | None,
    marker: float,
    tmp_path: Path,
) -> None:
    content = (FIXTURES / fixture_name).read_bytes()
    provenance = json.loads((FIXTURES / "semantic_marker_sources.json").read_text())[
        fixture_name
    ]
    assert hashlib.sha256(content).hexdigest() == provenance["sha256"]
    with (
        rasterio.Env(GRIB_NORMALIZE_UNITS="NO"),
        rasterio.MemoryFile(content) as file,
        file.open() as source,
    ):
        expected = source.read(1, out_dtype="float64")
    missing = expected == marker
    assert missing.any()
    assert not missing.all()
    config = NoaaRrfsForecast84HourVirtualTemplateConfig()
    variable = next(v for v in config.data_vars if v.name == variable_name)
    assert variable.internal_attrs.grib_element == (element or provenance["element"])
    temperature_offset = -273.15 if variable_name == "temperature_cloud_top" else 0
    assert variable.encoding.fill_value == marker + temperature_offset
    metadata = json.loads(
        (config.template_path() / variable_name / "zarr.json").read_text()
    )
    store = tmp_path / "decoded.zarr"
    write_single_grib_chunk(store, variable_name, metadata, content)
    with xr.open_zarr(store, consolidated=False, chunks=None, decode_cf=False) as ds:
        raw = ds[variable_name].values.squeeze()
    np.testing.assert_array_equal(raw[missing], variable.encoding.fill_value)
    if temperature_offset:
        assert metadata["codecs"][0] == {
            "name": "scale_offset",
            "configuration": {"offset": -273.15},
        }
    with xr.open_zarr(store, consolidated=False, chunks=None) as dataset:
        actual = dataset[variable_name].values.squeeze()
    expected += temperature_offset
    expected[missing] = np.nan
    np.testing.assert_array_equal(np.isnan(actual), missing)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=float(np.finfo(np.float32).eps),
        atol=3e-5 if temperature_offset else 0,
    )


@pytest.mark.parametrize(
    "config",
    [
        NoaaRrfsForecast84HourVirtualTemplateConfig(),
        NoaaRrfsForecast18HourVirtualTemplateConfig(),
    ],
    ids=lambda c: c.dataset_id,
)
@pytest.mark.parametrize("sample", ["A", "C"])
def test_column_soil_water_scaled_to_mass_per_area(
    config: NoaaRrfsForecast84HourVirtualTemplateConfig
    | NoaaRrfsForecast18HourVirtualTemplateConfig,
    sample: str,
    tmp_path: Path,
) -> None:
    name = "column_integrated_soil_moisture_0m_underground"
    filename = f"semantic-markers/{sample}-{name}.grib2"
    content = (FIXTURES / filename).read_bytes()
    provenance = json.loads((FIXTURES / "semantic_marker_sources.json").read_text())[
        filename
    ]
    assert hashlib.sha256(content).hexdigest() == provenance["sha256"]
    assert len(content) == provenance["end"] - provenance["start"] + 1
    variable = next(v for v in config.data_vars if v.name == name)
    assert variable.internal_attrs.grib_element == provenance["element"] == "CISOILM"
    assert variable.internal_attrs.grib_index_level == provenance["level"]
    assert variable.attrs.units == "kg m-2"
    assert variable.attrs.standard_name is None
    assert variable.attrs.comment == "NaN over water."
    assert np.isnan(variable.encoding.fill_value)
    with (
        rasterio.Env(GRIB_NORMALIZE_UNITS="NO"),
        rasterio.MemoryFile(content) as file,
        file.open() as source,
    ):
        missing = source.read_masks(1) == 0
        expected = source.read(1, out_dtype="float64") * 1000
    assert missing.any()
    assert not missing.all()
    expected[missing] = np.nan
    metadata = json.loads((config.template_path() / name / "zarr.json").read_text())
    assert metadata["codecs"][0] == {
        "name": "scale_offset",
        "configuration": {"scale": 0.001},
    }
    store = tmp_path / "soil-water.zarr"
    write_single_grib_chunk(store, name, metadata, content)
    with xr.open_zarr(store, consolidated=False, chunks=None) as dataset:
        actual = dataset[name].values.squeeze()
    np.testing.assert_array_equal(np.isnan(actual), missing)
    np.testing.assert_allclose(
        actual, expected, rtol=float(np.finfo(np.float32).eps), atol=0
    )
    unscaled_metadata = json.loads(json.dumps(metadata))
    unscaled_metadata["codecs"].pop(0)
    unscaled_store = tmp_path / "unscaled-soil-water.zarr"
    write_single_grib_chunk(unscaled_store, name, unscaled_metadata, content)
    with xr.open_zarr(unscaled_store, consolidated=False, chunks=None) as dataset:
        unscaled = dataset[name].values.squeeze()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(unscaled))
    np.testing.assert_allclose(actual, unscaled / 0.001, rtol=0, atol=0)


@pytest.mark.parametrize("sample", ["A", "C"])
def test_semantic_marker_physical_controls(sample: str, tmp_path: Path) -> None:
    config = NoaaRrfsForecast84HourVirtualTemplateConfig()
    variables = {v.name: v for v in config.data_vars}
    provenance = json.loads((FIXTURES / "semantic_marker_sources.json").read_text())

    def content(name: str) -> bytes:
        filename = f"semantic-markers/{sample}-{name}.grib2"
        value = (FIXTURES / filename).read_bytes()
        assert hashlib.sha256(value).hexdigest() == provenance[filename]["sha256"]
        return value

    def read_gdal(name: str) -> np.ndarray:
        with (
            rasterio.Env(GRIB_NORMALIZE_UNITS="NO"),
            rasterio.MemoryFile(content(name)) as file,
            file.open() as source,
        ):
            return source.read(1)

    water = read_gdal("land_mask") == 0
    vegetation_type = read_gdal("vegetation_type")
    water_point = vegetation_type == 0
    assert water_point.any()
    assert np.all(water[water_point])
    inland_water = vegetation_type == 17
    assert water.any()
    assert (~water).any()
    assert (inland_water & ~water).any()
    layer_name = "number_of_soil_layers_in_root_zone_surface"
    layer_metadata = json.loads(
        (config.template_path() / layer_name / "zarr.json").read_text()
    )
    count_store = tmp_path / "root-zone-layers.zarr"
    write_single_grib_chunk(
        count_store, layer_name, layer_metadata, content(layer_name)
    )
    with xr.open_zarr(count_store, consolidated=False, chunks=None) as dataset:
        actual = dataset[layer_name].values.squeeze()
    np.testing.assert_array_equal(actual, read_gdal(layer_name))
    assert np.all(actual[water | inland_water] == 0)
    assert np.isnan(variables[layer_name].encoding.fill_value)
    for variable_name, marker in SEMANTIC_MARKERS[6:]:
        missing = read_gdal(variable_name) == marker
        expected = (
            water | inland_water
            if variable_name
            in {
                "wilting_point_surface",
                "transpiration_stress_onset_soil_moisture_surface",
                "minimal_stomatal_resistance_surface",
            }
            else water
        )
        np.testing.assert_array_equal(missing, expected)

    with rasterio.MemoryFile(content("cloud_base")) as file, file.open() as source:
        missing_base = source.read_masks(1) == 0
    base_metadata = json.loads(
        (
            config.template_path() / "geopotential_height_cloud_base/zarr.json"
        ).read_text()
    )
    store = tmp_path / "cloud-base.zarr"
    write_single_grib_chunk(
        store, "geopotential_height_cloud_base", base_metadata, content("cloud_base")
    )
    with xr.open_zarr(store, consolidated=False, chunks=None) as dataset:
        np.testing.assert_array_equal(
            np.isnan(dataset.geopotential_height_cloud_base.values.squeeze()),
            missing_base,
        )
    for variable_name, marker in SEMANTIC_MARKERS[:4]:
        np.testing.assert_array_equal(read_gdal(variable_name) == marker, missing_base)
    for variable_name, marker in SEMANTIC_MARKERS[4:6]:
        layer = read_gdal(variable_name)
        valid = layer != marker
        assert np.all(layer[valid] > 0)
        assert (valid & missing_base).any()
        assert variables[variable_name].encoding.fill_value == marker


@pytest.mark.parametrize("fixture_name", [name for name, _ in CASES])
def test_missing_management_provenance(fixture_name: str) -> None:
    filename = f"{fixture_name}-all-missing.grib2"
    content = (FIXTURES / filename).read_bytes()
    provenance = json.loads((FIXTURES / "missing_management_sources.json").read_text())[
        filename
    ]
    assert len(content) == provenance["length"]
    assert hashlib.sha256(content).hexdigest() == provenance["sha256"]
    position = 16
    while content[position + 4] != 5:
        position += int.from_bytes(content[position : position + 4], "big")
    assert (
        int.from_bytes(content[position + 9 : position + 11], "big")
        == provenance["drt"]
        == 2
    )
    assert content[position + 22] == provenance["missing_management"] == 1


@pytest.mark.parametrize(
    "config",
    [
        NoaaRrfsForecast84HourVirtualTemplateConfig(),
        NoaaRrfsForecast18HourVirtualTemplateConfig(),
    ],
    ids=lambda c: c.dataset_id,
)
@pytest.mark.parametrize(
    "variable_name",
    ["aerosol_optical_thickness_atmosphere", "wildfire_potential_surface"],
)
def test_missing_management_fill_metadata(
    config: NoaaRrfsForecast84HourVirtualTemplateConfig
    | NoaaRrfsForecast18HourVirtualTemplateConfig,
    variable_name: str,
) -> None:
    metadata = json.loads(
        (config.template_path() / variable_name / "zarr.json").read_text()
    )
    assert metadata["fill_value"] == "NaN"
    cf_fill = np.frombuffer(
        base64.b64decode(metadata["attributes"]["_FillValue"]),
        dtype=metadata["data_type"],
    ).item()
    assert np.isnan(cf_fill)


@pytest.mark.parametrize(("fixture_name", "element"), CASES)
def test_real_noaa_missing_management_values_agree_with_gdal(
    fixture_name: str, element: str, tmp_path: Path
) -> None:
    content = (FIXTURES / f"{fixture_name}-all-missing.grib2").read_bytes()
    with rasterio.MemoryFile(content) as file, file.open() as source:
        missing = source.read_masks(1) == 0
    assert missing.all()
    store = tmp_path / "decoded.zarr"
    array = zarr.create_array(
        store,
        shape=missing.shape,
        chunks=missing.shape,
        dtype="float64",
        fill_value=np.nan,
        serializer=GribberishCodec(
            var=element, adjust_longitude_range=True, north_up=True
        ),
        compressors=None,
    )
    chunk_path = store / "c/0/0"
    chunk_path.parent.mkdir(parents=True)
    chunk_path.write_bytes(content)
    actual = np.asarray(array[:])
    assert actual.shape == missing.shape
    assert np.isnan(actual).all()
