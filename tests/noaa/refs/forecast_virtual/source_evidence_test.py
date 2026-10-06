import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
import zarr
from gribberish.zarr import GribberishCodec

from reformatters.noaa.noaa_grib_index import parse_grib_index_lines
from reformatters.noaa.refs.forecast_virtual.models import ProductFamily
from reformatters.noaa.refs.forecast_virtual.template_config import (
    NoaaRefsForecastVirtualTemplateConfig,
)
from tests.noaa.refs.forecast_virtual.region_job_test import file_coord, make_job
from tests.noaa.rrfs.decoder_helpers import write_single_grib_chunk

FIXTURES = Path(__file__).parent / "fixtures/zero_width"
INVENTORY = json.loads((FIXTURES / "inventory.json").read_text())


def _sections(raw: bytes) -> dict[int, bytes]:
    assert raw[:4] == b"GRIB"
    assert raw[-4:] == b"7777"
    assert int.from_bytes(raw[8:16], "big") == len(raw)
    position = 16
    sections = {}
    while position < len(raw) - 4:
        size = int.from_bytes(raw[position : position + 4], "big")
        assert size >= 5
        sections[raw[position + 4]] = raw[position : position + size]
        position += size
    assert position == len(raw) - 4
    return sections


def _reference_value(raw: bytes) -> float:
    section = _sections(raw)[5]
    return struct.unpack(">f", section[11:15])[0]


def test_retained_zero_width_messages_account_for_the_affected_schema() -> None:
    assert len(INVENTORY) == 276
    affected = set()
    job = make_job()
    for record in INVENTORY.values():
        key = record["key"]
        family: ProductFamily = key.split(".")[3]
        lead = pd.Timedelta(hours=int(key.split(".")[4][1:]))
        _, element, level, window, selectors = parse_grib_index_lines(
            FIXTURES.parent / (key + ".idx")
        )[record["message"] - 1]
        coord = file_coord(pd.Timestamp("2026-08-13"), family, lead)
        lookup = job._message_lookup(coord.data_vars, lead.total_seconds() / 3600)
        names = {
            v.name
            for v, _ in lookup[element, level, window, coord.index_selectors(selectors)]
        }
        assert names == set(record["variables"]), key
        affected.update(names)
    declared = {v.name for v in NoaaRefsForecastVirtualTemplateConfig().data_vars}
    assert len(affected) == 60
    assert affected <= declared
    for filename, record in INVENTORY.items():
        raw = (FIXTURES / filename).read_bytes()
        sections = _sections(raw)
        section = sections[5]
        assert section.hex() == record["s5"]
        assert int.from_bytes(section[9:11], "big") == record["drt"] == 3
        assert section[22] == record["missing_management"] == 0
        assert int.from_bytes(section[31:35], "big") == 0
        assert section[48] == 0
        assert sections[6][5] == 255

    assert {_reference_value((FIXTURES / name).read_bytes()) for name in INVENTORY} == {
        0.0,
        100.0,
        100.00001525878906,
    }


@pytest.mark.parametrize("filename", list(INVENTORY))
def test_zero_group_messages_decode_to_section5_reference(
    filename: str, tmp_path: Path
) -> None:
    raw = (FIXTURES / filename).read_bytes()
    expected = _reference_value(raw)
    store = tmp_path / "constant.zarr"
    array = zarr.create_array(
        store=store,
        shape=(1059, 1799),
        chunks=(1059, 1799),
        dtype="float64",
        fill_value=np.nan,
        serializer=GribberishCodec(var=None),
        compressors=None,
    )
    chunk = store / "c/0/0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(raw)
    values = array[:]
    assert isinstance(values, np.ndarray)
    assert values.shape == (1059, 1799)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values, expected)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("filename", "value"),
    [
        ("20260813-00-mean-f01-58.grib2", 0.0),
        ("20260813-00-sprd-f01-64.grib2", 0.0),
        ("20260813-00-prob-f01-22.grib2", 100.00001525878906),
        ("20260813-00-prob-f01-51.grib2", 0.0),
    ],
)
def test_independent_gdal_decodes_representative_zero_width_constants(
    filename: str, value: float, tmp_path: Path
) -> None:
    path = FIXTURES / filename
    with rasterio.Env(GDAL_CACHEMAX=16 * 1024 * 1024), rasterio.open(path) as source:
        assert source.shape == (1059, 1799)
        assert source.nodatavals == (None,)
        values = source.read(1)
    assert np.isfinite(values).all()
    assert np.min(values) == np.max(values) == value

    config = NoaaRefsForecastVirtualTemplateConfig()
    for name in INVENTORY[filename]["variables"]:
        metadata = json.loads((config.template_path() / name / "zarr.json").read_text())
        store = tmp_path / name
        write_single_grib_chunk(store, name, metadata, path.read_bytes())
        with xr.open_zarr(store, consolidated=False, chunks=None) as dataset:
            actual = dataset[name].values.squeeze()
        assert actual.shape == values.shape
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, values)
