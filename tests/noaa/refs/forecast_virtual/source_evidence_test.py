import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr

from reformatters.noaa.noaa_grib_index import parse_grib_index_lines
from reformatters.noaa.refs.forecast_virtual.template_config import (
    NoaaRefsForecastVirtualTemplateConfig,
)
from tests.noaa.refs.forecast_virtual.region_job_test import (
    FAMILIES,
    file_coord,
    make_job,
)
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


def test_zero_group_provenance_and_distinct_header_cases() -> None:
    job = make_job()
    references = set()
    decimal_scales = set()
    families = set()
    for filename, record in INVENTORY.items():
        key = record["key"]
        family = next(f for f in FAMILIES if f == key.split(".")[3])
        families.add(family)
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
        raw = (FIXTURES / filename).read_bytes()
        sections = _sections(raw)
        section = sections[5]
        assert section.hex() == record["s5"]
        assert int.from_bytes(section[9:11], "big") == record["drt"] == 3
        assert section[22] == record["missing_management"] == 0
        assert int.from_bytes(section[5:9], "big") == 1059 * 1799
        assert int.from_bytes(section[31:35], "big") == 0
        assert section[48] == 0
        assert sections[6][5] == 255
        references.add(_reference_value(raw))
        scale = int.from_bytes(section[17:19], "big")
        decimal_scales.add(-(scale & 0x7FFF) if scale & 0x8000 else scale)
    assert references == {0.0, 100.0, 100.00001525878906}
    assert decimal_scales == {0, 5}
    assert families == {"mean", "sprd", "prob"}


@pytest.mark.parametrize("filename", list(INVENTORY))
def test_zero_group_constants_match_independent_gdal_through_public_reader(
    filename: str, tmp_path: Path
) -> None:
    path = FIXTURES / filename
    raw = path.read_bytes()
    reference = _reference_value(raw)
    with rasterio.Env(GDAL_CACHEMAX=16 * 1024 * 1024), rasterio.open(path) as source:
        assert source.shape == (1059, 1799)
        assert source.nodatavals == (None,)
        expected = source.read(1)
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(expected, reference)

    (name,) = INVENTORY[filename]["variables"]
    config = NoaaRefsForecastVirtualTemplateConfig()
    metadata = json.loads((config.template_path() / name / "zarr.json").read_text())
    if name == "fog_liquid_water_content_0m":
        assert {"name": "scale_offset", "configuration": {"scale": 1000.0}} in metadata[
            "codecs"
        ]
        expected = expected / 1000.0
    store = tmp_path / "constant.zarr"
    write_single_grib_chunk(store, name, metadata, raw)
    with xr.open_zarr(store, consolidated=False, chunks=None) as dataset:
        actual = dataset[name].values.squeeze()
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)
