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
from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig, FastModel

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
        data_vars=EcmwfFastTemplateConfig(model=model).data_vars,
        source_members=(0,) if family == "cf" else tuple(range(1, 51)),
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
    param: str,
    member: int,
    hours: int = 6,
    payload: bytes = b"",
    *,
    bits: int = 0,
    unused_bits: int = 0,
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
        + bytes((unused_bits,))
        + b"\x00" * 6
        + bytes((bits,))
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


def unknown_message(model: FastModel, parameter: int = 255) -> bytes:
    data = bytearray(grib1("2t", 0) if model == "ifs" else grib2("2t", 0))
    product = 8 if model == "ifs" else 16 + 21 + 72
    data[product + (8 if model == "ifs" else 10)] = parameter
    data[product + (49 if model == "ifs" else 35)] = 255
    return bytes(data)


def resize_product(model: FastModel, data: bytes, size: int) -> bytes:
    product = 8 if model == "ifs" else 16 + 21 + 72
    width = 3 if model == "ifs" else 4
    old_size = int.from_bytes(data[product : product + width], "big")
    resized_product = data[product : product + old_size][:size].ljust(size, b"\x00")
    resized = bytearray(data[:product] + resized_product + data[product + old_size :])
    resized[product : product + width] = size.to_bytes(width, "big")
    if model == "ifs":
        resized[4:7] = len(resized).to_bytes(3, "big")
    else:
        resized[8:16] = len(resized).to_bytes(8, "big")
    return bytes(resized)


def resize_grib2_section(data: bytes, number: int, size: int) -> bytes:
    sections: dict[int, bytes] = {}
    offset = 16
    while offset < len(data) - 4:
        length = int.from_bytes(data[offset : offset + 4], "big")
        sections[data[offset + 4]] = data[offset : offset + length]
        offset += length
    resized = section(number, size)
    original = sections.get(number, b"")
    resized[: len(original)] = original
    resized[:4] = size.to_bytes(4, "big")
    if number == 6:
        resized[5] = 0
    sections[number] = bytes(resized)
    body = b"".join(value for _, value in sorted(sections.items())) + b"7777"
    return data[:8] + (16 + len(body)).to_bytes(8, "big") + body


@pytest.mark.parametrize(
    ("number", "size"),
    [(1, 1024), (2, 1024), (3, 1024), (4, 1024), (5, 1024), (6, 129786)],
)
def test_unknown_oversized_grib2_sections_drop_but_known_headers_stay_hard(
    number: int, size: int, caplog: pytest.LogCaptureFixture
) -> None:
    unknown = resize_grib2_section(unknown_message("aifs"), number, size)
    supported = grib2("tp", 0)
    data = unknown + supported
    messages = inventory(StreamReader(chunks(data, 31)), len(data), coord("aifs"))
    assert messages == [grib.Message("tp", 0, len(unknown), len(supported))]
    assert len(caplog.records) == 1
    assert "dropped unsupported fields={(0, 0, 255): 1}" in caplog.text
    known = resize_grib2_section(grib2("2t", 0), number, size)
    with pytest.raises(SourceContractError, match="Unsupported GRIB2 header size"):
        inventory(StreamReader(chunks(known, 31)), len(known), coord("aifs"))


@pytest.mark.parametrize("number", [2, 3, 4, 6])
def test_truncation_inside_skipped_oversized_grib2_section_stays_hard(
    number: int, caplog: pytest.LogCaptureFixture
) -> None:
    data = resize_grib2_section(unknown_message("aifs"), number, 129786)
    offset = 16
    while data[offset + 4] != number:
        offset += int.from_bytes(data[offset : offset + 4], "big")
    truncated = data[: offset + 513]
    with pytest.raises(InterruptedGribStreamError):
        inventory(StreamReader(chunks(truncated, 31)), len(data), coord("aifs"))
    assert not caplog.records


def test_unknown_oversized_grib1_product_drops_but_known_header_stays_hard(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unknown = resize_product("ifs", unknown_message("ifs"), 1024)
    supported = grib1("tp", 0)
    data = unknown + supported
    messages = inventory(StreamReader(chunks(data, 31)), len(data), coord("ifs"))
    assert messages == [grib.Message("tp", 0, len(unknown), len(supported))]
    assert len(caplog.records) == 1
    assert "dropped unsupported fields={(128, 255): 1}" in caplog.text
    known = resize_product("ifs", grib1("2t", 0), 1024)
    with pytest.raises(SourceContractError, match="Unsupported GRIB1 product section"):
        inventory(StreamReader(chunks(known, 31)), len(known), coord("ifs"))
    caplog.clear()
    with pytest.raises(InterruptedGribStreamError):
        inventory(
            StreamReader(chunks(unknown[: 8 + 513], 31)), len(unknown), coord("ifs")
        )
    assert not caplog.records


@pytest.mark.parametrize("model", ["ifs", "aifs"])
@pytest.mark.parametrize("chunk_size", [1, 31, 65536])
def test_unknown_parameters_drop_and_log_counts_without_changing_supported_refs(
    model: FastModel, chunk_size: int, caplog: pytest.LogCaptureFixture
) -> None:
    supported = source_bytes(model)
    baseline = inventory(
        StreamReader(chunks(supported, 31)), len(supported), coord(model)
    )
    assert not caplog.records
    split = baseline[0].length
    unknown = unknown_message(model)
    other_unknown = unknown_message(model, 254)
    data = unknown + supported[:split] + other_unknown + supported[split:] + unknown
    reader = StreamReader(chunks(data, chunk_size))
    actual = inventory(reader, len(data), coord(model))
    assert actual == [
        m._replace(
            offset=m.offset
            + len(unknown)
            + (len(other_unknown) if m.offset >= split else 0)
        )
        for m in baseline
    ]
    assert reader.position == len(data)
    for original, kept in zip(baseline, actual, strict=True):
        assert (
            data[kept.offset : kept.offset + kept.length]
            == supported[original.offset : original.offset + original.length]
        )
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert coord(model).key in caplog.text
    identities = (
        {(128, 254): 1, (128, 255): 2}
        if model == "ifs"
        else {(0, 0, 254): 1, (0, 0, 255): 2}
    )
    assert f"dropped unsupported fields={identities}" in caplog.text
    assert "missing fields=" not in caplog.text


@pytest.mark.parametrize(
    ("model", "offset", "identity"),
    [
        ("ifs", 8 + 3, (255, 167)),
        ("aifs", 6, (255, 0, 0)),
        ("aifs", 16 + 21 + 72 + 9, (0, 255, 0)),
    ],
)
def test_raw_identity_components_and_missing_fields_share_one_error(
    model: FastModel,
    offset: int,
    identity: tuple[int, ...],
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = bytearray(grib1("2t", 0) if model == "ifs" else grib2("2t", 0))
    data[offset] = 255
    assert (
        inventory(StreamReader(chunks(bytes(data), 1)), len(data), coord(model)) == []
    )
    assert len(caplog.records) == 1
    assert f"dropped unsupported fields={{{identity}: 1}}" in caplog.text
    assert "missing fields=" in caplog.text
    assert "('2t', 0)" in caplog.text


@pytest.mark.parametrize(
    ("model", "offset", "value", "error"),
    [
        ("ifs", 8 + 7, 0, "grid definition"),
        ("ifs", 8 + 13, 13, "date"),
        ("ifs", 8 + 40, 0, "local definition"),
        ("ifs", 8 + 52 + 5, 1, "grid"),
        ("aifs", 16 + 14, 13, "date"),
        ("aifs", 16 + 21 + 13, 1, "grid template"),
        ("aifs", 16 + 21 + 72 + 34, 0, "ensemble forecast"),
        ("aifs", 16 + 21 + 72 + 37 + 8, 0, "grid dimensions"),
        ("aifs", 16 + 21 + 72 + 37 + 25 + 5, 0, "bitmap"),
    ],
)
def test_unknown_identity_precedes_semantic_checks_but_known_gates_remain_hard(
    model: FastModel, offset: int, value: int, error: str
) -> None:
    unknown = bytearray(unknown_message(model))
    unknown[offset] = value
    assert (
        inventory(StreamReader(chunks(bytes(unknown), 1)), len(unknown), coord(model))
        == []
    )
    known = bytearray(grib1("2t", 0) if model == "ifs" else grib2("2t", 0))
    known[offset] = value
    with pytest.raises(SourceContractError, match=error):
        inventory(StreamReader(chunks(bytes(known), 31)), len(known), coord(model))


@pytest.mark.parametrize(("model", "size"), [("ifs", 28), ("aifs", 11)])
def test_unknown_short_product_drops_but_known_short_product_fails(
    model: FastModel, size: int
) -> None:
    unknown = resize_product(model, unknown_message(model), size)
    assert inventory(StreamReader(chunks(unknown, 1)), len(unknown), coord(model)) == []
    build = grib1 if model == "ifs" else grib2
    known = resize_product(model, build("2t", 0), size)
    with pytest.raises(SourceContractError, match="product section"):
        inventory(StreamReader(chunks(known, 31)), len(known), coord(model))


@pytest.mark.parametrize(("model", "size"), [("ifs", 27), ("aifs", 10)])
def test_unknown_product_must_be_safely_framed(model: FastModel, size: int) -> None:
    data = resize_product(model, unknown_message(model), size)
    with pytest.raises(SourceContractError, match="product section"):
        inventory(StreamReader(chunks(data, 31)), len(data), coord(model))


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_unknown_product_cannot_extend_beyond_message(model: FastModel) -> None:
    build = grib1 if model == "ifs" else grib2
    data = bytearray(build("2t", 0, payload=b"x" * 1024))
    product = 8 if model == "ifs" else 16 + 21 + 72
    width = 3 if model == "ifs" else 4
    data[product + (8 if model == "ifs" else 10)] = 255
    data[product : product + width] = (4096).to_bytes(width, "big")
    with pytest.raises(
        SourceContractError, match=r"product section|header size|outside message"
    ):
        inventory(StreamReader(chunks(bytes(data), 31)), len(data), coord(model))


@pytest.mark.parametrize("number", [1, 2])
def test_unknown_grib2_section_sequence_stays_hard(number: int) -> None:
    data = bytearray(unknown_message("aifs"))
    data[16 + 21 + 4] = number
    with pytest.raises(SourceContractError, match=r"Multiple fields|section sequence"):
        inventory(StreamReader(chunks(bytes(data), 31)), len(data), coord("aifs"))


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_unknown_message_framing_and_truncation_remain_hard(
    model: FastModel, caplog: pytest.LogCaptureFixture
) -> None:
    build = grib1 if model == "ifs" else grib2
    data = build("2t", 0) + unknown_message(model)
    for cut in (1, 8, 25, 80):
        with pytest.raises(InterruptedGribStreamError):
            inventory(StreamReader(chunks(data[:-cut], 31)), len(data), coord(model))
    with pytest.raises(SourceContractError, match="trailer"):
        inventory(
            StreamReader(chunks(data[:-4] + b"bad!", 31)), len(data), coord(model)
        )
    with pytest.raises(SourceContractError, match="beyond"):
        inventory(StreamReader(chunks(data + b"extra", 31)), len(data), coord(model))
    with pytest.raises(SourceContractError, match="message length"):
        inventory(StreamReader(chunks(data, 31)), len(data) - 1, coord(model))
    assert not caplog.records


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_dropped_messages_do_not_relax_kept_member_checks(
    model: FastModel,
) -> None:
    build = grib1 if model == "ifs" else grib2
    unknown = unknown_message(model)
    wrong_member = bytearray(build("2t", 0))
    wrong_member[8 + 49 if model == "ifs" else 16 + 21 + 72 + 35] = 1
    data = unknown + bytes(wrong_member)
    with pytest.raises(SourceContractError, match="Unexpected member"):
        inventory(StreamReader(chunks(data, 31)), len(data), coord(model))


@pytest.mark.parametrize("model", ["ifs", "aifs"])
@pytest.mark.parametrize(("family", "member"), [("cf", 0), ("pf", 1)])
def test_duplicate_logs_error_and_keeps_first_offset_and_valid_siblings(
    model: FastModel, family: Family, member: int, caplog: pytest.LogCaptureFixture
) -> None:
    build = grib1 if model == "ifs" else grib2
    unknown = unknown_message(model)
    first, sibling = build("2t", member), build("10u", member)
    duplicate = build("2t", member)
    data = unknown + first + duplicate + sibling
    assert inventory(
        StreamReader(chunks(data, 31)), len(data), coord(model, family)
    ) == [
        grib.Message("2t", member, len(unknown), len(first)),
        grib.Message("10u", member, len(unknown + first + duplicate), len(sibling)),
    ]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "Duplicate GRIB field" in caplog.text


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
def test_truncated_and_wrong_family_fail_closed(model: FastModel) -> None:
    data = source_bytes(model)
    build = grib1 if model == "ifs" else grib2
    wrong_family = b"".join(build(p, 1) for p in PARAMS)
    with pytest.raises(SourceContractError):
        inventory(
            StreamReader(chunks(wrong_family, 31)), len(wrong_family), coord(model)
        )
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
    assert (
        inventory(StreamReader(chunks(data, 31)), len(data), coord(model, hours=6))
        == []
    )


def test_grib1_large_message_extension_rejected() -> None:
    data = b"GRIB\x80\x00\x01\x01"
    with pytest.raises(SourceContractError, match="large-message"):
        inventory(StreamReader(iter([data])), len(data), coord("ifs"))


def test_grib2_ensemble_role_contradiction_is_file_scoped() -> None:
    data = bytearray(grib2("2t", 0))
    product = 16 + 21 + 72
    data[product + 34] = 0
    with pytest.raises(SourceContractError, match="ensemble forecast"):
        inventory(StreamReader(iter([bytes(data)])), len(data), coord("aifs"))


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_declared_source_inventory_keeps_missing_and_skips_unconfigured_fields(
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
    assert {
        m.param for m in inventory(StreamReader(chunks(full, 31)), len(full), selected)
    } == {"2t", "tp"}


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
