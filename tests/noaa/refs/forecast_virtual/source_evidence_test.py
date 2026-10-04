import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from reformatters.noaa.noaa_grib_index import parse_grib_index_lines
from reformatters.noaa.refs.forecast_virtual.models import ProductFamily
from reformatters.noaa.refs.forecast_virtual.template_config import (
    NoaaRefsForecastVirtualTemplateConfig,
)
from tests.noaa.refs.forecast_virtual.region_job_test import file_coord, make_job


def test_retained_zero_width_messages_account_for_the_affected_schema() -> None:
    root = Path(__file__).parent / "fixtures/zero_width"
    inventory = json.loads((root / "inventory.json").read_text())
    assert len(inventory) == 276
    affected = set()
    job = make_job()
    for record in inventory.values():
        key = record["key"]
        family: ProductFamily = key.split(".")[3]
        lead = pd.Timedelta(hours=int(key.split(".")[4][1:]))
        _, element, level, window, selectors = parse_grib_index_lines(
            root.parent / (key + ".idx")
        )[record["message"] - 1]
        coord = file_coord(pd.Timestamp("2026-08-13"), family, lead)
        lookup = job._message_lookup(coord.data_vars, lead.total_seconds() / 3600)
        names = {
            v.name
            for v, _ in lookup[element, level, window, coord.index_selectors(selectors)]
        }
        expected = set()
        for name in record["variables"]:
            var = next(v for v in job.data_vars if v.name == name)
            if var.has_statistic and var.internal_attrs.source_families == ("mean",):
                expected.update(
                    (name, f"{name}_mean")
                    if family == "mean"
                    else (f"{name}_standard_deviation",)
                )
            else:
                expected.add(name)
        assert names == expected, key
        affected.update(names)
    declared = {v.name for v in NoaaRefsForecastVirtualTemplateConfig().data_vars}
    assert len(affected) == 60
    assert affected <= declared
    for filename, record in inventory.items():
        raw = (root / filename).read_bytes()
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
        section = sections[5]
        assert section.hex() == record["s5"]
        assert int.from_bytes(section[9:11], "big") == record["drt"] == 3
        assert section[22] == record["missing_management"] == 0
        assert int.from_bytes(section[31:35], "big") == 0
        assert section[48] == 0
        assert sections[6][5] == 255
        assert (
            record["error"]
            == "PanicException('cannot load 32 bits from a 0-bit region')"
        )


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
    filename: str, value: float
) -> None:
    path = Path(__file__).parent / "fixtures/zero_width" / filename
    with rasterio.Env(GDAL_CACHEMAX=16 * 1024 * 1024), rasterio.open(path) as source:
        assert source.shape == (1059, 1799)
        assert source.nodatavals == (None,)
        values = source.read(1)
    assert np.isfinite(values).all()
    assert np.min(values) == np.max(values) == value
