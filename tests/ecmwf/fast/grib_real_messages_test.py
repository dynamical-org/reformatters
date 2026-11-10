import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gribberish.zarr import GribberishCodec
from numpy.typing import NDArray
from zarr.api.asynchronous import create_array
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

from reformatters.common.config_models import Encoding
from reformatters.ecmwf.fast.grib import Message, StreamReader, inventory
from reformatters.ecmwf.fast.source import EcmwfFastSourceFileCoord
from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig, FastModel

FIXTURES = Path(__file__).parent / "fixtures" / "resilience"


async def _decode(payload: bytes, encoding: Encoding) -> NDArray[np.float64]:
    assert encoding.serializer is not None
    array = await create_array(
        store=MemoryStore(),
        shape=encoding.chunks,
        chunks=encoding.chunks,
        dtype=encoding.dtype,
        fill_value=encoding.fill_value,
        serializer=GribberishCodec.from_dict(encoding.serializer),
        filters=encoding.filters,
        compressors=encoding.compressors,
    )
    assert any(isinstance(codec, GribberishCodec) for codec in array.metadata.codecs)
    await array.store.set(
        "c/0/0/0/0/0", default_buffer_prototype().buffer.from_bytes(payload)
    )
    return np.asarray(await array.getitem(Ellipsis)).reshape(721, 1440)


@pytest.mark.parametrize("model", ["ifs", "aifs"])
@pytest.mark.parametrize("param", ["tp", "ssrd", "10u"])
def test_native_message_inventory_and_template_codec(
    model: FastModel, param: str, caplog: pytest.LogCaptureFixture
) -> None:
    row = next(
        row
        for row in json.loads((FIXTURES / "metadata.json").read_text())
        if row["model"] == model and row["param"] == param
    )
    payload = (FIXTURES / row["file"]).read_bytes()
    assert len(payload) == row["length"]
    assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    variable = next(
        var
        for var in EcmwfFastTemplateConfig(model=model).data_vars
        if var.internal_attrs.grib_index_param == param
    )
    coord = EcmwfFastSourceFileCoord(
        model=model,
        family="cf",
        init_time=pd.Timestamp(row["init_time"]),
        lead_time=pd.Timedelta(hours=row["lead_hours"]),
        data_vars=[variable],
        source_params=frozenset({param}),
        source_members=(0,),
    )
    assert coord.key == row["source_key"]
    assert variable.name == row["variable"]
    assert variable.internal_attrs.source_units == row["native_units"]
    reader = StreamReader(
        iter(payload[start : start + 4093] for start in range(0, len(payload), 4093))
    )
    assert inventory(reader, len(payload), coord) == [
        Message(param, 0, 0, len(payload))
    ]
    assert not caplog.records

    assert variable.encoding.filters == row["filters"]
    values = asyncio.run(_decode(payload, variable.encoding))
    assert values.shape == tuple(row["shape"]) == (721, 1440)
    assert np.isfinite(values).all()
    assert values.min() == row["minimum"]
    assert values.max() == row["maximum"]
    assert values.min() < values.max()
    assert np.count_nonzero(values == 0) == row["zero_count"]
    np.testing.assert_array_equal(
        [values[point["row"], point["column"]] for point in row["snapshots"]],
        [point["value"] for point in row["snapshots"]],
    )


@pytest.mark.parametrize(
    "row",
    json.loads((FIXTURES.parent / "nine_variable" / "metadata.json").read_text()),
    ids=lambda row: (
        f"{row['model']}-{row['family']}-{row['param']}-{row['lead_hours']}h-{row['bits']}bits"
    ),
)
def test_nine_variable_delivery_native_accuracy_and_filters(
    row: dict, caplog: pytest.LogCaptureFixture
) -> None:
    payload = (FIXTURES.parent / "nine_variable" / row["file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    assert len(payload) == row["length"]
    assert payload[7] == row["edition"]
    if row["edition"] == 1:
        pds_length = int.from_bytes(payload[8:11], "big")
        assert pds_length == row["pds_length"]
        assert payload[8 + 40] == (1 if row["family"] == "cf" else 30)
        assert payload[8 + 49] == row["member"]
        if row["family"] == "pf":
            assert pds_length == 106
            assert payload[8 + 50] == 51
        grid_start = 8 + pds_length
        grid_length = int.from_bytes(payload[grid_start : grid_start + 3], "big")
        assert payload[grid_start + grid_length + 10] == row["bits"]
    else:
        sections: dict[int, bytes] = {}
        offset = 16
        while offset < len(payload) - 4:
            size = int.from_bytes(payload[offset : offset + 4], "big")
            sections[payload[offset + 4]] = payload[offset : offset + size]
            offset += size
        assert len(sections[4]) == row["pds_length"]
        assert sections[4][35] == row["member"]
        assert int.from_bytes(sections[5][9:11], "big") == 42
        assert sections[5][19] == row["bits"]
    variable = next(
        v
        for v in EcmwfFastTemplateConfig(model=row["model"]).data_vars
        if v.attrs.short_name == row["param"]
    )
    source = EcmwfFastSourceFileCoord(
        model=row["model"],
        family=row["family"],
        init_time=pd.Timestamp(row["init_time"]),
        lead_time=pd.Timedelta(hours=row["lead_hours"]),
        data_vars=[variable],
        source_params=frozenset({row["param"]}),
        source_members=(row["member"],),
    )
    assert inventory(StreamReader(iter([payload])), len(payload), source) == [
        Message(row["param"], row["member"], 0, len(payload))
    ]
    assert not caplog.records
    assert variable.internal_attrs.source_units == row["native_units"]
    assert variable.encoding.filters == row["filters"]
    values = asyncio.run(_decode(payload, variable.encoding))
    assert np.isfinite(values).all()
    np.testing.assert_allclose(
        [values.min(), values.max()],
        [row["minimum"], row["maximum"]],
        rtol=0,
        atol=row["atol"],
    )
    assert np.count_nonzero(values == 0) == row["zero_count"]
    np.testing.assert_allclose(
        [values[p["row"], p["column"]] for p in row["snapshots"]],
        [p["value"] for p in row["snapshots"]],
        rtol=0,
        atol=row["atol"],
    )
