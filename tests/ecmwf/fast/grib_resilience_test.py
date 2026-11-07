import struct
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import zarr
from gribberish.zarr import GribberishCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from reformatters.common.pydantic import replace
from reformatters.ecmwf.fast import grib
from reformatters.ecmwf.fast.grib import (
    Message,
    SourceContractError,
    StreamReader,
    inventory,
)
from reformatters.ecmwf.fast.source import EcmwfFastSourceFileCoord
from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig, FastModel
from tests.ecmwf.fast.grib_test import PARAMS, chunks, coord, grib1, grib2, signed

PRODUCT2 = 16 + 21 + 72


def change(data: bytes, offset: int, value: bytes) -> bytes:
    return data[:offset] + value + data[offset + len(value) :]


def grib2_sections(
    data: bytes, transform: Callable[[dict[int, bytearray]], None]
) -> bytes:
    sections = {}
    position = 16
    while position < len(data) - 4:
        size = int.from_bytes(data[position : position + 4], "big")
        sections[data[position + 4]] = bytearray(data[position : position + size])
        position += size
    transform(sections)
    for section in sections.values():
        section[:4] = len(section).to_bytes(4, "big")
    body = b"".join(sections.values()) + b"7777"
    return data[:8] + (16 + len(body)).to_bytes(8, "big") + body


def simple2(data: bytes, bits: int = 0, reference: float = 0.0) -> bytes:
    def packing(sections: dict[int, bytearray]) -> None:
        section = sections[5]
        del section[21:]
        section[9:11] = b"\x00\x00"
        section[11:15] = struct.pack(">f", reference)
        section[19] = bits

    return grib2_sections(data, packing)


def deterministic2(data: bytes) -> bytes:
    def product(sections: dict[int, bytearray]) -> None:
        section = sections[4]
        template = 8 if int.from_bytes(section[7:9], "big") == 11 else 0
        section[7:9] = template.to_bytes(2, "big")
        del section[34:37]

    return grib2_sections(data, product)


def read(data: bytes, source: EcmwfFastSourceFileCoord) -> list[Message]:
    reader = StreamReader(chunks(data, 31))
    result = inventory(reader, len(data), source)
    assert reader.position == len(data)
    return result


@pytest.mark.parametrize("model", ["ifs", "aifs"])
@pytest.mark.parametrize("unsupported", ["grib1", "grib2"])
def test_mixed_editions_preserve_good_unsupported_good_byte_ranges(
    model: FastModel,
    unsupported: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = grib1("2t", 0)
    last = simple2(grib2("10u", 0, payload=b""))
    middle = (
        change(grib1("10v", 0), 8 + 52 + 32 + 3, b"\x40")
        if unsupported == "grib1"
        else change(grib2("10v", 0), PRODUCT2 + 37 + 9, b"\x00\x02")
    )
    data = first + middle + last
    assert read(data, coord(model)) == [
        Message("2t", 0, 0, len(first)),
        Message("10u", 0, len(first) + len(middle), len(last)),
    ]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "packing" in caplog.text
    assert coord(model).key in caplog.text
    with pytest.raises(SourceContractError, match="trailer"):
        read(first + middle[:-4] + b"bad!" + last, coord(model))
    reader = StreamReader(chunks(first + middle[:-8], 31))
    with pytest.raises(grib.InterruptedGribStreamError):
        inventory(reader, len(first + middle), coord(model))


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_cross_edition_precipitation_uses_configured_source_scale(
    model: FastModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    metres = grib1("tp", 0)
    mass = grib2("tp", 0)
    data = grib1("2t", 0) + metres + mass + grib2("10u", 0)
    result = read(data, coord(model))
    rain = next(message for message in result if message.param == "tp")
    expected = metres if model == "ifs" else mass
    assert data[rain.offset : rain.offset + rain.length] == expected
    assert [message.param for message in result] == ["2t", "tp", "10u"]
    assert len(caplog.records) == 1
    assert "source units=" in caplog.text
    assert (
        "expected m" in caplog.text
        if model == "ifs"
        else "expected kg m-2" in caplog.text
    )


def test_equivalent_source_unit_spelling_is_kept_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    variable = next(
        v
        for v in EcmwfFastTemplateConfig(model="aifs").data_vars
        if v.internal_attrs.grib_index_param == "tp"
    )
    variable = replace(
        variable, internal_attrs=replace(variable.internal_attrs, source_units="mm")
    )
    source = coord("aifs").model_copy(
        update={"data_vars": [variable], "source_params": frozenset({"tp"})}
    )
    data = grib2("tp", 0)
    assert read(data, source) == [Message("tp", 0, 0, len(data))]
    assert len(caplog.records) == 1
    assert "equivalent source units=" in caplog.text


def test_unselected_parameters_still_count_in_full_source_inventory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = coord("aifs")
    source = source.model_copy(update={"data_vars": source.data_vars[:1]})
    data = b"".join(
        grib1(param, 0) if param == "tp" else grib2(param, 0)
        for param in sorted(PARAMS)
    )
    assert {message.param for message in read(data, source)} == PARAMS
    assert not caplog.records


def test_selected_parameter_requires_configured_source_units() -> None:
    source = coord("ifs")
    variable = source.data_vars[0]
    variable = replace(
        variable, internal_attrs=replace(variable.internal_attrs, source_units=None)
    )
    source = source.model_copy(update={"data_vars": [variable]})
    with pytest.raises(AssertionError, match="Missing configured source units for 2t"):
        read(grib1("2t", 0), source)


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_all_native_identity_units_match_config_without_equivalence_entries(
    model: FastModel,
) -> None:
    source = coord(model)
    mapping = grib._GRIB1_PARAMS if model == "ifs" else grib._GRIB2_PARAMS
    assert all(source.source_unit(param) == units for param, units in mapping.values())
    assert not any("K" in group for group in grib._EQUIVALENT_UNITS)
    for build in (grib1, grib2):
        for param in ("2t", "10u", "10v", "100u", "100v", "ssrd"):
            data = build(param, 0)
            assert read(data, coord(model))[0].param == param


@pytest.mark.parametrize(
    ("param", "height"),
    [("2t", 2), ("10u", 10), ("10v", 10), ("100u", 100), ("100v", 100)],
)
def test_grib1_native_parameters_accept_explicit_equivalent_heights(
    param: str, height: int
) -> None:
    data = change(grib1(param, 0), 8 + 9, bytes((105,)) + height.to_bytes(2, "big"))
    assert read(data, coord("ifs")) == [Message(param, 0, 0, len(data))]


@pytest.mark.parametrize("number", [130, 131, 132])
@pytest.mark.parametrize("extra_encoding", ["native", "other_grid", "bitmap"])
def test_grib1_unconfigured_generic_fields_do_not_reject_siblings(
    number: int, extra_encoding: str, caplog: pytest.LogCaptureFixture
) -> None:
    first, last = grib1("2t", 0), grib1("10u", 0)
    extra = change(
        grib1("10v", 0), 8 + 8, bytes((number, 100)) + (850).to_bytes(2, "big")
    )
    if extra_encoding == "other_grid":
        extra = change(extra, 8 + 52 + 6, (720).to_bytes(2, "big"))
    elif extra_encoding == "bitmap":
        extra = change(extra, 8 + 7, bytes((192,)))
    assert read(first + extra + last, coord("ifs")) == [
        Message("2t", 0, 0, len(first)),
        Message("10u", 0, len(first) + len(extra), len(last)),
    ]
    assert "dropped unsupported fields" in caplog.text


@pytest.mark.parametrize(
    ("param", "scale", "height"), [("2t", 1, 20), ("10u", -1, 1), ("100v", 2, 10000)]
)
def test_grib2_equivalent_scaled_height_is_not_a_new_level(
    param: str, scale: int, height: int
) -> None:
    data = change(
        grib2(param, 0), PRODUCT2 + 23, signed(scale, 1) + height.to_bytes(4, "big")
    )
    assert read(data, coord("aifs")) == [Message(param, 0, 0, len(data))]


@pytest.mark.parametrize("edition", [1, 2])
def test_extra_height_and_unconfigured_quantity_skip_without_guessing(
    edition: int, caplog: pytest.LogCaptureFixture
) -> None:
    build = grib1 if edition == 1 else grib2
    extra = (
        change(build("10u", 0), 8 + 9, b"\x69\x00\xc8")
        if edition == 1
        else change(build("10u", 0), PRODUCT2 + 24, (200).to_bytes(4, "big"))
    )
    first, last = build("2t", 0), build("10u", 0)
    source = coord("ifs" if edition == 1 else "aifs").model_copy(
        update={"source_params": frozenset({"2t", "10u"})}
    )
    unconfigured = build("10v", 0)
    assert read(first + extra + unconfigured + last, source) == [
        Message("2t", 0, 0, len(first)),
        Message("10u", 0, len(first + extra + unconfigured), len(last)),
    ]
    assert len(caplog.records) == 1
    assert "unconfigured parameter=10v" in caplog.text


@pytest.mark.parametrize("edition", [1, 2])
def test_template_members_admit_only_explicit_labels_and_keep_gaps_visible(
    edition: int, caplog: pytest.LogCaptureFixture
) -> None:
    build = grib1 if edition == 1 else grib2
    data = b"".join(build("2t", member) for member in (1, 2, 3, 51))
    source = coord("ifs" if edition == 1 else "aifs", "pf").model_copy(
        update={"source_members": (1, 3), "source_params": frozenset({"2t"})}
    )
    assert [message.member for message in read(data, source)] == [1, 3]
    assert len(caplog.records) == 1
    assert "unconfigured member=2" in caplog.text
    assert "unconfigured member=51" in caplog.text
    assert "missing fields" not in caplog.text
    wrong_role = change(
        build("2t", 1), 8 + 49 if edition == 1 else PRODUCT2 + 35, b"\x00"
    )
    with pytest.raises(SourceContractError, match="Unexpected member"):
        read(wrong_role, source)


@pytest.mark.parametrize(
    ("edition", "unit", "value", "hours"),
    [
        (1, 0, 360, 6),
        (1, 2, 1, 24),
        (1, 10, 2, 6),
        (1, 11, 1, 6),
        (1, 12, 1, 12),
        (1, 13, 24, 6),
        (1, 14, 12, 6),
        (1, 254, 21600, 6),
        (2, 0, 360, 6),
        (2, 2, 1, 24),
        (2, 10, 2, 6),
        (2, 11, 1, 6),
        (2, 12, 1, 12),
        (2, 13, 21600, 6),
    ],
)
def test_equivalent_time_units_cross_check_filename(
    edition: int, unit: int, value: int, hours: int, caplog: pytest.LogCaptureFixture
) -> None:
    if edition == 1:
        data = change(
            grib1("2t", 0, hours),
            8 + 17,
            bytes((unit,)) + value.to_bytes(2, "big") + b"\x0a",
        )
    else:
        data = change(
            grib2("2t", 0, hours),
            PRODUCT2 + 17,
            bytes((unit,)) + value.to_bytes(4, "big"),
        )
    source = coord("ifs" if edition == 1 else "aifs", hours=hours)
    assert read(data, source) == [Message("2t", 0, 0, len(data))]
    assert len(caplog.records) == 1
    assert "time" in caplog.text
    caplog.clear()
    assert (
        read(
            data,
            source.model_copy(
                update={"lead_time": source.lead_time + pd.Timedelta(hours=1)}
            ),
        )
        == []
    )
    assert len(caplog.records) == 1
    assert "differs from filename" in caplog.text


@pytest.mark.parametrize("param", ["tp", "ssrd"])
@pytest.mark.parametrize("tri", [0, 1, 2, 3, 4, 5, 10])
def test_grib1_run_total_meaning_and_native_time_range_conventions(
    param: str, tri: int
) -> None:
    first, last = grib1("2t", 0), grib1("10u", 0)
    p1, p2 = (0, 6) if tri in (2, 3, 4, 5, 10) else (6, 0)
    middle = change(grib1(param, 0), 8 + 18, bytes((p1, p2, tri)))
    result = read(first + middle + last, coord("ifs"))
    assert [m.param for m in result] == (
        ["2t", param, "10u"] if tri in (0, 4, 10) else ["2t", "10u"]
    )


@pytest.mark.parametrize("param", ["tp", "ssrd"])
def test_grib1_interval_total_and_instantaneous_interval_cannot_replace_configured_meaning(
    param: str,
) -> None:
    total = change(grib1(param, 0), 8 + 18, bytes((3, 6, 4)))
    assert read(total, coord("ifs")) == []
    interval = change(grib1("2t", 0), 8 + 18, bytes((0, 6, 2)))
    assert read(interval, coord("ifs")) == []


@pytest.mark.parametrize("param", ["2t", "tp", "ssrd"])
def test_deterministic_grib2_is_control_only_with_exact_descriptor_layout(
    param: str,
) -> None:
    data = deterministic2(grib2(param, 0))
    assert read(data, coord("aifs")) == [Message(param, 0, 0, len(data))]
    with pytest.raises(SourceContractError, match="no member identity"):
        read(data, coord("aifs", "pf"))


@pytest.mark.parametrize("deterministic", [False, True])
def test_grib2_run_total_accepts_independent_time_units(deterministic: bool) -> None:
    data = change(grib2("tp", 0), PRODUCT2 + 17, b"\x0d")
    data = change(data, PRODUCT2 + 51, b"\x00" + (360).to_bytes(4, "big"))
    if deterministic:
        data = deterministic2(data)
    assert read(data, coord("aifs")) == [Message("tp", 0, 0, len(data))]


@pytest.mark.parametrize(
    ("offset", "value"),
    [
        (17, b"\x03"),
        (18, (3).to_bytes(4, "big")),
        (39, b"\x0d"),
        (44, b"\x02"),
        (49, b"\x00"),
        (50, b"\x01"),
        (51, b"\x03"),
        (52, (3).to_bytes(4, "big")),
        (37, (2025).to_bytes(2, "big")),
    ],
)
def test_unplaceable_grib2_times_or_changed_accumulation_skip_only_message(
    offset: int, value: bytes
) -> None:
    first, last = grib2("2t", 0), grib2("10u", 0)
    bad = change(grib2("tp", 0), PRODUCT2 + offset, value)
    result = read(first + bad + last, coord("aifs"))
    assert result == [
        Message("2t", 0, 0, len(first)),
        Message("10u", 0, len(first + bad), len(last)),
    ]


@pytest.mark.parametrize("edition", [1, 2])
def test_unsupported_packing_or_time_cannot_hide_known_grid_failures(
    edition: int,
) -> None:
    if edition == 1:
        data = change(grib1("2t", 0), 8 + 52 + 32 + 3, b"\x40")
        data = change(data, 8 + 52 + 27, b"\x80")
    else:
        data = change(grib2("2t", 0), PRODUCT2 + 17, b"\xff")
        data = change(data, 16 + 21 + 71, b"\x80")
    with pytest.raises(SourceContractError, match="scanning"):
        read(data, coord("ifs" if edition == 1 else "aifs"))


@pytest.mark.parametrize("edition", [1, 2])
def test_initialization_mismatch_and_unknown_edition_remain_file_failures(
    edition: int,
) -> None:
    if edition == 1:
        data = change(grib1("2t", 0), 8 + 14, b"\x05")
    else:
        data = change(grib2("2t", 0), 16 + 15, b"\x05")
    with pytest.raises(SourceContractError, match="initialization differs"):
        read(data, coord("ifs" if edition == 1 else "aifs"))
    with pytest.raises(SourceContractError, match="Unknown GRIB edition"):
        read(change(data, 7, b"\x03"), coord("ifs"))


@pytest.mark.parametrize("edition", [1, 2])
@pytest.mark.parametrize("reference", [0.0, 42.0])
def test_zero_bit_simple_constant_decodes_on_native_full_grid(
    edition: int, reference: float
) -> None:
    if edition == 1:
        data = change(
            grib1("tp", 0),
            8 + 52 + 32 + 6,
            b"\x42\x2a\x00\x00" if reference else b"\x00" * 4,
        )
    else:
        data = simple2(grib2("tp", 0, payload=b""), reference=reference)
    source = coord("ifs" if edition == 1 else "aifs")
    assert read(data, source) == [Message("tp", 0, 0, len(data))]
    array = zarr.create_array(
        store=MemoryStore(),
        shape=(721, 1440),
        chunks=(721, 1440),
        dtype="float64",
        serializer=GribberishCodec(
            var="tp", adjust_longitude_range=True, north_up=True
        ),
        compressors=(),
    )
    sync(array.store.set("c/0/0", default_buffer_prototype().buffer.from_bytes(data)))
    values = np.asarray(array[:])
    assert values.size == 721 * 1440
    np.testing.assert_array_equal(values, np.full(values.shape, reference))


@pytest.mark.parametrize(
    ("bits", "unused", "size", "accepted"),
    [
        (0, 0, 0, True),
        (0, 8, 1, True),
        (0, 0, 1, False),
        (1, 0, 129780, True),
        (1, 8, 129781, True),
        (1, 0, 129779, False),
        (1, 1, 129780, False),
        (16, 0, 2076480, True),
        (16, 0, 2076479, False),
        (16, 0, 0, False),
    ],
)
def test_grib1_capacity_counts_unused_bits_and_rejects_underfilled_streams(
    bits: int, unused: int, size: int, accepted: bool
) -> None:
    data = grib1("2t", 0, payload=b"\x00" * size, bits=bits, unused_bits=unused)
    if accepted:
        assert read(data, coord("ifs")) == [Message("2t", 0, 0, len(data))]
    else:
        with pytest.raises(SourceContractError, match="packed-bit capacity"):
            read(data, coord("ifs"))


def test_one_bit_ccsds_admission_has_no_precision_gate() -> None:
    data = change(grib2("2t", 0), PRODUCT2 + 37 + 19, b"\x01")
    assert read(data, coord("aifs")) == [Message("2t", 0, 0, len(data))]


@pytest.mark.parametrize("edition", [1, 2])
def test_unknown_raw_identity_keeps_outer_framing_without_known_grid_or_capacity_gates(
    edition: int,
) -> None:
    if edition == 1:
        unknown = grib1("2t", 0, bits=16)
        unknown = change(unknown, 8 + 8, b"\xff")
        unknown = change(unknown, 8 + 52 + 6, b"\x00\x01")
    else:
        unknown = change(grib2("2t", 0), PRODUCT2 + 10, b"\xff")
        unknown = change(unknown, 16 + 21 + 30, b"\x00\x00\x00\x01")
        unknown = change(unknown, PRODUCT2 + 37 + 5, b"\x00\x00\x00\x01")
    first, last = grib1("2t", 0), grib2("10u", 0)
    assert read(first + unknown + last, coord("ifs")) == [
        Message("2t", 0, 0, len(first)),
        Message("10u", 0, len(first + unknown), len(last)),
    ]
    with pytest.raises(SourceContractError, match="trailer"):
        read(first + unknown[:-4] + b"bad!" + last, coord("ifs"))


@pytest.mark.parametrize("edition", [1, 2])
def test_simple_packing_reads_each_messages_scales_and_reference(edition: int) -> None:
    payload = bytes((3,)) * (721 * 1440)
    if edition == 1:
        data = grib1("10u", 0, payload=payload, bits=8)
        data = change(data, 8 + 52 + 32 + 4, signed(1, 2) + b"\x42\x2a\x00\x00")
        data = change(data, 8 + 26, signed(1, 2))
    else:
        data = simple2(grib2("10u", 0, payload=payload), bits=8, reference=42.0)
        data = change(data, PRODUCT2 + 37 + 15, signed(1, 2) + signed(1, 2))
    source = coord("ifs" if edition == 1 else "aifs")
    assert read(data, source) == [Message("10u", 0, 0, len(data))]
    array = zarr.create_array(
        store=MemoryStore(),
        shape=(721, 1440),
        chunks=(721, 1440),
        dtype="float64",
        serializer=GribberishCodec(
            var="10u", adjust_longitude_range=True, north_up=True
        ),
        compressors=(),
    )
    sync(array.store.set("c/0/0", default_buffer_prototype().buffer.from_bytes(data)))
    np.testing.assert_allclose(np.asarray(array[:]), 4.8, rtol=0, atol=1e-14)


@pytest.mark.parametrize(
    ("offset", "value"), [(22, b"\x64"), (28, b"\x00"), (7, b"\x00\x02")]
)
def test_unconfigured_grib2_surface_or_product_does_not_relabel_siblings(
    offset: int, value: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    first, last = grib2("2t", 0), grib2("10u", 0)
    extra = change(grib2("10v", 0), PRODUCT2 + offset, value)
    assert read(first + extra + last, coord("aifs")) == [
        Message("2t", 0, 0, len(first)),
        Message("10u", 0, len(first + extra), len(last)),
    ]
    assert len(caplog.records) == 1
    assert "unsupported GRIB2" in caplog.text
