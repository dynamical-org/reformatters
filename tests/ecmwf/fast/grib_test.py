from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from reformatters.ecmwf.fast import grib
from reformatters.ecmwf.fast.grib import (
    InterruptedGribStreamError,
    SourceContractError,
    StreamReader,
    inventory,
)
from reformatters.ecmwf.fast.source import EcmwfFastSourceFileCoord, Family
from reformatters.ecmwf.fast.template_config import FastModel

PARAMS = frozenset({"2t", "10u", "10v", "100u", "100v", "tp", "ssrd"})

INIT = pd.Timestamp("2026-11-04")


def coord(
    model: FastModel, family: Family = "cf", hours: int = 6
) -> EcmwfFastSourceFileCoord:
    return EcmwfFastSourceFileCoord(
        model=model,
        family=family,
        init_time=INIT,
        lead_time=pd.Timedelta(hours=hours),
        data_vars=[],
    )


def signed(value: int, size: int) -> bytes:
    return (abs(value) | (1 << (size * 8 - 1) if value < 0 else 0)).to_bytes(
        size, "big"
    )


def date(value: pd.Timestamp) -> bytes:
    return value.year.to_bytes(2, "big") + bytes(
        (value.month, value.day, value.hour, value.minute, value.second)
    )


def grib1(
    param: str, member: int, hours: int = 6, payload: bytes = b"payloadGRIB7777"
) -> bytes:
    pds = bytearray(52 if member == 0 else 106)
    pds[:3] = len(pds).to_bytes(3, "big")
    table, parameter = {
        "2t": (128, 167),
        "10u": (128, 165),
        "10v": (128, 166),
        "100u": (228, 246),
        "100v": (228, 247),
        "tp": (128, 228),
        "ssrd": (128, 169),
    }[param]
    pds[3:5] = bytes((table, 98))
    pds[7:10] = bytes((128, parameter, 1))
    pds[12:18] = bytes((26, 11, 4, 0, 0, 1))
    pds[24] = 21
    if hours > 255:
        pds[18:20] = hours.to_bytes(2, "big")
        pds[20] = 10
    else:
        pds[18] = hours
        pds[20] = 0 if hours else 1
    pds[40] = 1 if member == 0 else 30
    pds[49] = member
    gds = bytearray(32)
    gds[:3] = (32).to_bytes(3, "big")
    gds[6:10] = (1440).to_bytes(2, "big") + (721).to_bytes(2, "big")
    gds[10:16] = signed(90000, 3) + signed(-180000, 3)
    gds[17:23] = signed(-90000, 3) + signed(179750, 3)
    gds[23:27] = (250).to_bytes(2, "big") * 2
    bds = (
        (11 + len(payload)).to_bytes(3, "big")
        + b"\x00" * 7
        + bytes((0 if hours == 0 and param in ("tp", "ssrd") else 16,))
        + payload
    )
    body = bytes(pds) + bytes(gds) + bds + b"7777"
    return b"GRIB" + (len(body) + 8).to_bytes(3, "big") + b"\x01" + body


def section(number: int, size: int) -> bytearray:
    return bytearray(size.to_bytes(4, "big") + bytes((number,)) + b"\x00" * (size - 5))


def grib2(
    param: str, member: int, hours: int = 6, payload: bytes = b"payloadGRIB7777"
) -> bytes:
    ref = section(1, 21)
    ref[5:7] = (98).to_bytes(2, "big")
    ref[12:19] = date(INIT)
    grid = section(3, 72)
    grid[6:10] = (721 * 1440).to_bytes(4, "big")
    grid[14] = 6
    grid[30:38] = (1440).to_bytes(4, "big") + (721).to_bytes(4, "big")
    grid[46:54] = signed(90000000, 4) + (180000000).to_bytes(4, "big")
    grid[55:63] = signed(-90000000, 4) + (179750000).to_bytes(4, "big")
    grid[63:71] = (250000).to_bytes(4, "big") * 2
    accum = param in ("tp", "ssrd")
    product = section(4, 61 if accum else 37)
    product[7:9] = (11 if accum else 1).to_bytes(2, "big")
    category, number, height = {
        "2t": (0, 0, 2),
        "10u": (2, 2, 10),
        "10v": (2, 3, 10),
        "100u": (2, 2, 100),
        "100v": (2, 3, 100),
        "tp": (1, 52, 0),
        "ssrd": (4, 7, 0),
    }[param]
    product[9:11] = bytes((category, number))
    product[17] = 1
    product[18:22] = (0 if accum else hours).to_bytes(4, "big")
    product[22] = 1 if accum else 103
    product[24:28] = height.to_bytes(4, "big")
    product[28:34] = b"\xff" * 6
    product[34:37] = bytes((5 if member == 0 else 6, member, 51))
    if accum:
        product[37:44] = date(INIT + pd.Timedelta(hours=hours))
        product[44] = 1
        product[49:52] = bytes((1, 2, 1))
        product[52:56] = hours.to_bytes(4, "big")
    packing = section(5, 25)
    packing[5:9] = (721 * 1440).to_bytes(4, "big")
    packing[9:11] = (42).to_bytes(2, "big")
    bitmap = section(6, 6)
    bitmap[5] = 255
    data = section(7, 5) + payload
    data[:4] = len(data).to_bytes(4, "big")
    body = b"".join((ref, grid, product, packing, bitmap, data)) + b"7777"
    return b"GRIB\x00\x00\x00\x02" + (16 + len(body)).to_bytes(8, "big") + body


def chunks(data: bytes, size: int) -> Iterator[bytes]:
    return (data[i : i + size] for i in range(0, len(data), size))


def source_bytes(model: FastModel, family: Family = "cf", hours: int = 6) -> bytes:
    build = grib1 if model == "ifs" else grib2
    messages = [
        build(param, member, hours)
        for param in sorted(PARAMS)
        for member in (range(1) if family == "cf" else range(1, 51))
    ]
    np.random.default_rng(42).shuffle(messages)
    return b"".join(messages)


@pytest.mark.parametrize("model", ["ifs", "aifs"])
@pytest.mark.parametrize("family", ["cf", "pf"])
@pytest.mark.parametrize("hours", [0, 6, 360])
@pytest.mark.parametrize("chunk_size", [1, 31, 65536])
def test_complete_shuffled_inventory_and_chunk_boundaries(
    model: FastModel, family: Family, hours: int, chunk_size: int
) -> None:
    data = source_bytes(model, family, hours)
    reader = StreamReader(chunks(data, chunk_size))
    messages = inventory(reader, len(data), coord(model, family, hours))
    assert {(m.param, m.member) for m in messages} == {
        (p, m) for p in PARAMS for m in coord(model, family, hours).members
    }
    assert sum(m.length for m in messages) == len(data)
    assert reader.position == len(data)
    assert [m.offset for m in messages] == [
        sum(x.length for x in messages[:i]) for i in range(len(messages))
    ]


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_truncated_duplicate_and_wrong_family_fail_closed(model: FastModel) -> None:
    data = source_bytes(model)
    build = grib1 if model == "ifs" else grib2
    for damaged in (
        data + build("2t", 0),
        b"".join(build(p, 1) for p in PARAMS),
    ):
        with pytest.raises(SourceContractError):
            inventory(StreamReader(chunks(damaged, 31)), len(damaged), coord(model))
    with pytest.raises(InterruptedGribStreamError):
        inventory(StreamReader(chunks(data[:-1], 31)), len(data), coord(model))
    with pytest.raises(SourceContractError, match="trailer"):
        inventory(
            StreamReader(chunks(data[:-4] + b"bad!", 31)), len(data), coord(model)
        )
    with pytest.raises(SourceContractError, match="beyond"):
        inventory(StreamReader(chunks(data + b"extra", 31)), len(data), coord(model))


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_wrong_instantaneous_lead_rejected(model: FastModel) -> None:
    data = source_bytes(model, hours=360)
    with pytest.raises(SourceContractError, match="differs from filename"):
        inventory(StreamReader(chunks(data, 31)), len(data), coord(model, hours=6))


def test_grib1_large_message_extension_rejected() -> None:
    data = b"GRIB\x80\x00\x01\x01"
    with pytest.raises(SourceContractError, match="large-message"):
        inventory(StreamReader(iter([data])), len(data), coord("ifs"))


def test_grib2_second_surface_and_ensemble_type_rejected() -> None:
    for change in (28, 34):
        data = bytearray(grib2("2t", 0))
        product = 16 + 21 + 72
        data[product + change] = 0
        with pytest.raises(
            SourceContractError, match=r"second fixed|ensemble forecast"
        ):
            inventory(StreamReader(iter([bytes(data)])), len(data), coord("aifs"))


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_declared_source_inventory_allows_missing_but_not_extra_fields(
    model: FastModel,
) -> None:
    build = grib1 if model == "ifs" else grib2
    data = build("2t", 0) + build("tp", 0)
    selected = coord(model).model_copy(
        update={"source_params": frozenset({"2t", "tp"}), "source_members": (0,)}
    )
    assert len(inventory(StreamReader(chunks(data, 31)), len(data), selected)) == 2
    assert len(inventory(StreamReader(chunks(data, 31)), len(data), coord(model))) == 2
    full = source_bytes(model)
    with pytest.raises(SourceContractError, match="Unexpected GRIB fields"):
        inventory(StreamReader(chunks(full, 31)), len(full), selected)


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_invalid_source_dates_are_file_scoped(model: FastModel) -> None:
    data = bytearray(grib1("2t", 0) if model == "ifs" else grib2("2t", 0))
    data[8 + 13 if model == "ifs" else 16 + 14] = 13
    with pytest.raises(SourceContractError, match="date"):
        inventory(StreamReader(iter([bytes(data)])), len(data), coord(model))


@pytest.mark.parametrize("size", [-1, 513])
def test_reader_header_size_is_a_programming_invariant(size: int) -> None:
    with pytest.raises(AssertionError, match="Only GRIB headers"):
        StreamReader(iter([])).read(size)


def test_reader_negative_skip_is_a_programming_invariant() -> None:
    with pytest.raises(AssertionError):
        StreamReader(iter([])).skip(-1)


def test_impossible_scanner_position_is_a_programming_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grib, "_grib1", lambda *args: ("2t", 0))
    data = grib1("2t", 0)
    with pytest.raises(AssertionError, match="scanner did not reach trailer"):
        inventory(StreamReader(iter([data])), len(data), coord("ifs"))


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_missing_messages_alert_but_truncated_messages_fail(
    model: FastModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    build = grib1 if model == "ifs" else grib2
    data = build("2t", 0) + build("tp", 0)
    messages = inventory(StreamReader(chunks(data, 31)), len(data), coord(model))
    assert [(m.param, m.member) for m in messages] == [("2t", 0), ("tp", 0)]
    assert len(caplog.records) == 1
    assert coord(model).key in caplog.text
    assert "missing fields=" in caplog.text
    assert "('ssrd', 0)" in caplog.text
    caplog.clear()
    for damaged in (data[:-1], data + b"bad!bad!", data[:-4] + b"bad!"):
        with pytest.raises(SourceContractError):
            inventory(StreamReader(chunks(damaged, 31)), len(damaged), coord(model))
    assert not caplog.records
