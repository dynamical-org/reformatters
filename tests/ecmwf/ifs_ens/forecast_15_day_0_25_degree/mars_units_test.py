import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from reformatters.common.iterating import item
from reformatters.common.pydantic import replace
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job import (
    EcmwfIfsEnsForecast15Day025DegreeRegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    MarsSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)

FIXTURES = Path(__file__).parent / "fixtures" / "mars"
VARIABLES = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig().data_vars


@pytest.mark.parametrize("data_var", VARIABLES, ids=lambda v: v.name)
def test_mars_raw_units_and_conversion(data_var: EcmwfDataVar, tmp_path: Path) -> None:
    evidence = item(
        entry
        for entry in json.loads((FIXTURES / "audit.json").read_text())
        if entry["name"] == data_var.name
    )
    path = tmp_path / "source.grib"
    path.write_bytes(
        gzip.decompress((FIXTURES / f"{data_var.name}.grib.gz").read_bytes())
    )
    expected = np.full((721, 1440), evidence["background_value"], dtype=np.float64)
    for sample in evidence["samples"]:
        expected[sample["row"], sample["column"]] = sample["value"]
    with rasterio.Env(GRIB_NORMALIZE_UNITS="NO"), rasterio.open(path) as reader:
        assert reader.tags(1)["GRIB_UNIT"] == evidence["NO"]["unit"]
        np.testing.assert_array_equal(reader.read(1), expected)

    if data_var.attrs.units == "degree_Celsius":
        expected -= 273.15
    elif data_var.name.startswith("geopotential_height"):
        expected *= 1 / 9.80665

    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-04"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=0,
        request_type="cf_" + data_var.internal_attrs.grib_index_level_type,
        data_var_group=[data_var],
        downloaded_path=path,
    ).resolve_data_vars()
    job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct()
    # The outer setting must be restored after each MARS read, including its offset.
    with rasterio.Env(GRIB_NORMALIZE_UNITS="YES"):
        actual = job.read_data(coord, data_var)
        assert rasterio.env.get_gdal_config("GRIB_NORMALIZE_UNITS") == "YES"
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_fixture_audit_covers_every_mars_output() -> None:
    evidence = json.loads((FIXTURES / "audit.json").read_text())
    assert {entry["name"] for entry in evidence} == {var.name for var in VARIABLES}
    assert len(evidence) == 19
    assert all(entry["NO"]["max_difference_from_eccodes"] == 0 for entry in evidence)


@pytest.mark.parametrize("name", ["temperature_2m", "dew_point_temperature_2m"])
def test_grib1_surface_temperature_normalization_reproducer(
    name: str, tmp_path: Path
) -> None:
    path = tmp_path / "temperature.grib"
    path.write_bytes(gzip.decompress((FIXTURES / f"{name}.grib.gz").read_bytes()))
    with rasterio.Env(GRIB_NORMALIZE_UNITS="YES"), rasterio.open(path) as reader:
        assert reader.tags(1)["GRIB_UNIT"] == "[C]"
        normalized = reader.read(1)
    with rasterio.Env(GRIB_NORMALIZE_UNITS="NO"), rasterio.open(path) as reader:
        assert reader.tags(1)["GRIB_UNIT"] == "[K]"
        raw = reader.read(1)
    np.testing.assert_array_equal(normalized, raw)


def test_mars_scale_precedes_offset_and_rejects_wrong_units(tmp_path: Path) -> None:

    var = item(v for v in VARIABLES if v.name == "temperature_2m")
    assert var.internal_attrs.mars is not None
    var = replace(
        var,
        internal_attrs=replace(
            var.internal_attrs, mars=replace(var.internal_attrs.mars, scale_factor=2)
        ),
    )
    path = tmp_path / "temperature.grib"
    path.write_bytes(gzip.decompress((FIXTURES / f"{var.name}.grib.gz").read_bytes()))
    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-04"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=0,
        request_type="cf_sfc",
        data_var_group=[var],
        downloaded_path=path,
    ).resolve_data_vars()
    job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct()
    with rasterio.Env(GRIB_NORMALIZE_UNITS="NO"), rasterio.open(path) as reader:
        expected = (reader.read(1) * 2 - 273.15).astype(np.float32)
    np.testing.assert_array_equal(job.read_data(coord, var), expected)
    resolved = item(coord.data_var_group)
    coord = replace(
        coord,
        data_var_group=[
            replace(
                resolved,
                internal_attrs=replace(
                    resolved.internal_attrs, grib_comment="Temperature [C]"
                ),
            )
        ],
    )
    with rasterio.Env(GRIB_NORMALIZE_UNITS="YES"):
        with pytest.raises(AssertionError, match="Unit mismatch"):
            job.read_data(coord, var)
        assert rasterio.env.get_gdal_config("GRIB_NORMALIZE_UNITS") == "YES"
