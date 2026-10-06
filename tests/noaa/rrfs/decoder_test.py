import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

from reformatters.noaa.rrfs.forecast_84_hour_virtual.template_config import (
    NoaaRrfsForecast84HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs_ens.forecast_virtual.template_config import (
    NoaaRrfsEnsForecastVirtualTemplateConfig,
)
from tests.noaa.rrfs.decoder_helpers import write_single_grib_chunk

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(
    params=[
        ("spfh", "specific_humidity_surface", False),
        ("pevpr", "potential_evaporation_rate_surface", False),
        ("pevap", "potential_evaporation_surface", False),
        ("vegmin", "minimum_vegetation_surface", False),
        ("vegmax", "maximum_vegetation_surface", False),
        ("aotk", "aerosol_optical_thickness_atmosphere", True),
        ("wildfire", "wildfire_potential_surface", True),
    ],
    ids=lambda p: p[1],
)
def decoded_values(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    fixture_name, variable_name, members = request.param
    filename = f"{fixture_name}-all-missing.grib2"
    content = (FIXTURES / filename).read_bytes()
    provenance = json.loads((FIXTURES / "missing_management_sources.json").read_text())[
        filename
    ]
    assert len(content) == provenance["length"]
    assert hashlib.sha256(content).hexdigest() == provenance["sha256"]
    assert provenance["drt"] == 2
    assert provenance["missing_management"] == 1
    with rasterio.MemoryFile(content) as file, file.open() as source:
        expected = source.read(1).astype(np.float64)
        missing = source.read_masks(1) == 0
    assert missing.all()
    expected[missing] = np.nan

    config = (
        NoaaRrfsEnsForecastVirtualTemplateConfig()
        if members
        else NoaaRrfsForecast84HourVirtualTemplateConfig()
    )
    metadata = json.loads(
        (config.template_path() / variable_name / "zarr.json").read_text()
    )
    assert metadata["fill_value"] == "NaN"
    store = tmp_path / "decoded.zarr"
    write_single_grib_chunk(store, variable_name, metadata, content)
    with xr.open_zarr(
        store, consolidated=False, chunks=None, decode_cf=False
    ) as dataset:
        assert np.isnan(dataset[variable_name].attrs["_FillValue"])
    with xr.open_zarr(store, consolidated=False, chunks=None) as dataset:
        actual = dataset[variable_name].values.squeeze()
    assert actual.shape == expected.shape
    return actual, expected


def test_real_noaa_missing_management_values_agree_with_gdal(
    decoded_values: tuple[np.ndarray, np.ndarray],
) -> None:
    actual, expected = decoded_values
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
