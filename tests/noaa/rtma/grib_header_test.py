from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from reformatters.noaa.rtma.analysis_virtual.template_config import (
    NoaaRtmaAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.rtma.grib_header import scan_analysis_messages

CONFIG = NoaaRtmaAnalysisVirtualTemplateConfig()
SELECTED = {var.internal_attrs.grib_identity for var in CONFIG.data_vars}


def sparse_source(fixture: dict[str, Any]) -> Callable[[int, int], bytes]:
    def read_range(start: int, end: int) -> bytes:
        result = bytearray(end - start)
        for message in fixture["messages"]:
            offset = message["offset"]
            segments = (
                (offset, bytes.fromhex(message["header"])),
                (offset + message["length"] - 4, b"7777"),
            )
            for pos, content in segments:
                left, right = max(start, pos), min(end, pos + len(content))
                if left < right:
                    result[left - start : right - start] = content[
                        left - pos : right - pos
                    ]
        return bytes(result)

    return read_range


def edit_header(
    fixture: dict[str, Any], message: int, offset: int, data: bytes
) -> None:
    raw = bytearray.fromhex(fixture["messages"][message]["header"])
    raw[offset : offset + len(data)] = data
    fixture["messages"][message]["header"] = raw.hex()


def append_message(fixture: dict[str, Any], message: int) -> int:
    copy = {**fixture["messages"][message], "offset": fixture["size"]}
    fixture["messages"].append(copy)
    fixture["size"] += copy["length"]
    return len(fixture["messages"]) - 1


def test_real_headers_tile_exactly_and_select_by_identity(
    source_fixture: dict[str, Any],
) -> None:
    calls = []
    read = sparse_source(source_fixture)

    def read_range(start: int, end: int) -> bytes:
        calls.append((start, end))
        return read(start, end)

    messages = list(
        scan_analysis_messages(
            read_range,
            source_fixture["size"],
            pd.Timestamp(source_fixture["time"]),
            SELECTED,
        )
    )
    assert len(messages) == 13
    assert {m.identity for m in messages} == SELECTED
    assert [(m.offset, m.length) for m in messages] == [
        (m["offset"], m["length"]) for m in source_fixture["messages"]
    ]
    assert messages[-1].offset + messages[-1].length == source_fixture["size"]
    assert len(calls) == 14
    assert max(end - start for start, end in calls) <= 260


@pytest.mark.parametrize(
    ("offset", "data", "error"),
    [
        (0, b"FAIL", "Invalid GRIB2"),
        (7, b"\x01", "Invalid GRIB2"),
        (8, (0).to_bytes(8), "Invalid GRIB2"),
        (16, (0).to_bytes(4), "Invalid GRIB section length"),
        (16, (10000).to_bytes(4), "bounded range"),
        (37 + 12, b"\0\0", "Lambert"),
        (37 + 16, (6371229).to_bytes(4), "earth radius"),
        (37 + 30, (2145).to_bytes(4), "grid geometry"),
        (37 + 64, b"\0", "scanning"),
        (16 + 17, b"\x01", "reference time"),
        (16 + 14, b"\x0d", "reference time"),
        (118 + 18, (1).to_bytes(4), "zero-lead"),
        (118 + 17, b"\x02", "forecast time unit"),
        (152 + 9, b"\0\x02", "simple packing"),
        (173 + 5, b"\xfe", "without bitmap"),
        (179, (5).to_bytes(4), "Packed data size"),
    ],
)
def test_invalid_selected_headers_raise(
    source_fixture: dict[str, Any], offset: int, data: bytes, error: str
) -> None:
    edit_header(source_fixture, 0, offset, data)
    with pytest.raises(AssertionError, match=error):
        list(
            scan_analysis_messages(
                sparse_source(source_fixture),
                source_fixture["size"],
                pd.Timestamp(source_fixture["time"]),
                SELECTED,
            )
        )


@pytest.mark.parametrize("adjustment", [-1, 1, 100])
def test_file_size_mismatch_raises(
    source_fixture: dict[str, Any], adjustment: int
) -> None:
    with pytest.raises(AssertionError):
        list(
            scan_analysis_messages(
                sparse_source(source_fixture),
                source_fixture["size"] + adjustment,
                pd.Timestamp(source_fixture["time"]),
                SELECTED,
            )
        )


def test_truncated_range_raises(source_fixture: dict[str, Any]) -> None:
    with pytest.raises(AssertionError, match="Truncated GRIB header"):
        list(
            scan_analysis_messages(
                lambda start, end: b"",
                source_fixture["size"],
                pd.Timestamp(source_fixture["time"]),
                SELECTED,
            )
        )


def test_unselected_new_field_can_have_other_grid_time_and_packing(
    source_fixture: dict[str, Any],
) -> None:
    edit_header(source_fixture, 0, 118 + 10, b"\xff")
    edit_header(source_fixture, 0, 37 + 30, (2145).to_bytes(4))
    edit_header(source_fixture, 0, 16 + 17, b"\x01")
    edit_header(source_fixture, 0, 152 + 9, b"\0\x02")
    messages = list(
        scan_analysis_messages(
            sparse_source(source_fixture),
            source_fixture["size"],
            pd.Timestamp(source_fixture["time"]),
            SELECTED,
        )
    )
    assert len(messages) == 12


def test_unselected_new_product_template_is_not_parsed_as_pdt_zero(
    source_fixture: dict[str, Any],
) -> None:
    edit_header(source_fixture, 0, 118 + 10, b"\xff")
    edit_header(source_fixture, 0, 118 + 7, b"\0\x08")
    messages = list(
        scan_analysis_messages(
            sparse_source(source_fixture),
            source_fixture["size"],
            pd.Timestamp(source_fixture["time"]),
            SELECTED,
        )
    )
    assert len(messages) == 12


def test_duplicate_unselected_fields_do_not_reject_selected_fields(
    source_fixture: dict[str, Any],
) -> None:
    for _ in range(2):
        message = append_message(source_fixture, 0)
        edit_header(source_fixture, message, 118 + 10, b"\xff")
    messages = list(
        scan_analysis_messages(
            sparse_source(source_fixture),
            source_fixture["size"],
            pd.Timestamp(source_fixture["time"]),
            SELECTED,
        )
    )
    assert {message.identity for message in messages} == SELECTED
    assert len(messages) == 13


def test_duplicate_field_raises(source_fixture: dict[str, Any]) -> None:
    first = bytes.fromhex(source_fixture["messages"][0]["header"])
    edit_header(source_fixture, 1, 118, first[118:152])
    with pytest.raises(AssertionError, match="Duplicate GRIB field"):
        list(
            scan_analysis_messages(
                sparse_source(source_fixture),
                source_fixture["size"],
                pd.Timestamp(source_fixture["time"]),
                SELECTED,
            )
        )


def test_missing_terminator_raises(source_fixture: dict[str, Any]) -> None:
    read = sparse_source(source_fixture)

    def broken(start: int, end: int) -> bytes:
        return b"0000" if end == source_fixture["size"] else read(start, end)

    with pytest.raises(AssertionError, match="final GRIB terminator"):
        list(
            scan_analysis_messages(
                broken,
                source_fixture["size"],
                pd.Timestamp(source_fixture["time"]),
                SELECTED,
            )
        )
