import base64
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
CASES = (
    ("spfh", "specific_humidity_surface", False),
    ("pevpr", "potential_evaporation_rate_surface", False),
    ("pevap", "potential_evaporation_surface", False),
    ("vegmin", "minimum_vegetation_surface", False),
    ("vegmax", "maximum_vegetation_surface", False),
    ("aotk", "aerosol_optical_thickness_atmosphere", True),
    ("wildfire", "wildfire_potential_surface", True),
)


@pytest.mark.parametrize(("fixture_name", "variable_name", "members"), CASES)
def test_missing_management_provenance_and_fill_metadata(
    fixture_name: str, variable_name: str, members: bool
) -> None:
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
    config = (
        NoaaRrfsEnsForecastVirtualTemplateConfig()
        if members
        else NoaaRrfsForecast84HourVirtualTemplateConfig()
    )
    metadata = json.loads(
        (config.template_path() / variable_name / "zarr.json").read_text()
    )
    assert metadata["fill_value"] == "NaN"
    cf_fill = np.frombuffer(
        base64.b64decode(metadata["attributes"]["_FillValue"]),
        dtype=metadata["data_type"],
    ).item()
    assert np.isnan(cf_fill)


@pytest.mark.parametrize("fixture_name", ["spfh", "aotk"])
def test_real_noaa_missing_management_values_agree_with_gdal(
    fixture_name: str, tmp_path: Path
) -> None:
    content = (FIXTURES / f"{fixture_name}-all-missing.grib2").read_bytes()
    with rasterio.MemoryFile(content) as file, file.open() as source:
        missing = source.read_masks(1) == 0
    assert missing.all()
    config = (
        NoaaRrfsForecast84HourVirtualTemplateConfig()
        if fixture_name == "spfh"
        else NoaaRrfsEnsForecastVirtualTemplateConfig()
    )
    variable_name = (
        "specific_humidity_surface"
        if fixture_name == "spfh"
        else "aerosol_optical_thickness_atmosphere"
    )
    metadata = json.loads(
        (config.template_path() / variable_name / "zarr.json").read_text()
    )
    store = tmp_path / "decoded.zarr"
    write_single_grib_chunk(store, variable_name, metadata, content)
    with xr.open_zarr(store, consolidated=False, chunks=None) as dataset:
        actual = dataset[variable_name].values.squeeze()
    assert actual.shape == missing.shape
    assert np.isnan(actual).all()
