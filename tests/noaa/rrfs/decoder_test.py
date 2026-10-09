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
    ],
)
def test_semantic_markers_read_as_nan(
    fixture_name: str,
    variable_name: str,
    element: str,
    marker: float,
    tmp_path: Path,
) -> None:
    content = (FIXTURES / fixture_name).read_bytes()
    provenance = json.loads((FIXTURES / "semantic_marker_sources.json").read_text())[
        fixture_name
    ]
    assert hashlib.sha256(content).hexdigest() == provenance["sha256"]
    with rasterio.MemoryFile(content) as file, file.open() as source:
        expected = source.read(1).astype(np.float64)
    missing = expected == marker
    assert missing.any()
    assert not missing.all()
    expected[missing] = np.nan
    config = NoaaRrfsForecast84HourVirtualTemplateConfig()
    variable = next(v for v in config.data_vars if v.name == variable_name)
    assert variable.encoding.fill_value == marker
    metadata = json.loads(
        (config.template_path() / variable_name / "zarr.json").read_text()
    )
    store = tmp_path / "decoded.zarr"
    group = zarr.create_group(store)
    group.create_array(
        variable_name,
        shape=expected.shape,
        chunks=expected.shape,
        dtype="float64",
        fill_value=variable.encoding.fill_value,
        serializer=GribberishCodec(
            var=element, adjust_longitude_range=True, north_up=True
        ),
        compressors=None,
        dimension_names=("y", "x"),
        attributes={"_FillValue": metadata["attributes"]["_FillValue"]},
    )
    chunk_path = store / variable_name / "c/0/0"
    chunk_path.parent.mkdir(parents=True)
    chunk_path.write_bytes(content)
    with xr.open_zarr(store, consolidated=False, chunks=None) as dataset:
        actual = np.asarray(dataset[variable_name].values)
    np.testing.assert_array_equal(np.isnan(actual), missing)
    np.testing.assert_allclose(
        actual, expected, rtol=float(np.finfo(np.float32).eps), atol=0
    )


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
