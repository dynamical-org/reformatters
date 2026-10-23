import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from reformatters.noaa.rtma.grib_header import (
    GRID_NX,
    GRID_NY,
    AnalysisMessage,
    scan_analysis_messages,
)
from tests.noaa.rtma.grib_header_test import (
    SELECTED,
    assert_empty_bitmap_scans_equal,
    edit_header,
    sparse_source,
)

WAVE = (10, 0, 3, 1, 1.0)


def scan(fixture: dict[str, Any]) -> list[AnalysisMessage]:
    return list(
        scan_analysis_messages(
            sparse_source(fixture),
            fixture["size"],
            pd.Timestamp(fixture["time"]),
            SELECTED | {WAVE},
            bitmapped=frozenset({WAVE}),
        )
    )


def bitmap_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    if not fixture["extra_ranges"]:
        pytest.skip("Source precedes bitmap onset")
    return fixture


def test_real_bitmap_headers_use_only_one_extra_five_byte_read(
    source_fixture: dict[str, Any],
) -> None:
    fixture = bitmap_fixture(source_fixture)
    calls = []
    read = sparse_source(fixture)

    def read_range(start: int, end: int) -> bytes:
        calls.append((start, end))
        return read(start, end)

    messages = list(
        scan_analysis_messages(
            read_range,
            fixture["size"],
            pd.Timestamp(fixture["time"]),
            SELECTED | {WAVE},
            bitmapped=frozenset({WAVE}),
        )
    )
    assert len(messages) == 14
    assert {m.identity for m in messages} == SELECTED | {WAVE}
    assert max(b - a for a, b in calls) <= 260
    extra = fixture["extra_ranges"][0]
    wave = next(m for m in messages if m.identity == WAVE)
    assert wave.offset > 0
    assert extra["start"] > wave.offset
    assert (extra["start"], extra["start"] + 5) in calls
    assert len(calls) == len(fixture["messages"]) + 2


@pytest.mark.parametrize(
    ("offset", "data", "error"),
    [
        (152 + 9, b"\0\x02", "simple packing"),
        (152 + 5, (0).to_bytes(4), "All-missing bitmap"),
        (152 + 5, (GRID_NX * GRID_NY + 1).to_bytes(4), "packed point count"),
        (173, (5).to_bytes(4), "bitmap section length"),
        (173, (468132).to_bytes(4), "bitmap section length"),
        (173, (2**32 - 1).to_bytes(4), "bitmap section length"),
        (173 + 4, b"\x07", "section 6"),
        (173 + 5, b"\xff", "Bitmap required"),
        (173 + 5, b"\xfe", "Bitmap required"),
        (173 + 5, b"\x01", "Bitmap required"),
        (173 + 5, b"\xfd", "Bitmap required"),
        (118 + 18, (1).to_bytes(4), "zero-lead"),
        (16 + 17, b"\x0d", "reference time"),
    ],
)
def test_invalid_bitmap_headers_raise(
    source_fixture: dict[str, Any], offset: int, data: bytes, error: str
) -> None:
    fixture = bitmap_fixture(source_fixture)
    edit_header(
        fixture,
        next(i for i, m in enumerate(fixture["messages"]) if m["element"] == "HTSGW"),
        offset,
        data,
    )
    with pytest.raises(AssertionError, match=error):
        scan(fixture)


@pytest.mark.parametrize("raw", [b"", b"\0" * 4, b"\0\0\0\x05\x06", b"\0\0\0\x05\x07"])
def test_invalid_or_short_section_seven_raises(
    source_fixture: dict[str, Any], raw: bytes
) -> None:
    fixture = bitmap_fixture(source_fixture)
    extra = fixture["extra_ranges"][0]
    read = sparse_source(fixture)

    def read_range(start: int, end: int) -> bytes:
        return raw if start == extra["start"] else read(start, end)

    with pytest.raises(AssertionError):
        list(
            scan_analysis_messages(
                read_range,
                fixture["size"],
                pd.Timestamp(fixture["time"]),
                {WAVE},
                bitmapped=frozenset({WAVE}),
            )
        )


@pytest.mark.parametrize(("points", "bits"), [(GRID_NX * GRID_NY, 8), (100, 0)])
def test_full_bitmap_and_zero_bit_constant_field(
    source_fixture: dict[str, Any], points: int, bits: int
) -> None:
    fixture = bitmap_fixture(source_fixture)
    message = deepcopy(next(m for m in fixture["messages"] if m["element"] == "HTSGW"))
    extra = deepcopy(fixture["extra_ranges"][0])
    extra["start"] -= message["offset"]
    message["offset"] = 0
    data_size = (points * bits + 7) // 8 + 5
    message["length"] = extra["start"] + data_size + 4
    extra["bytes"] = (data_size.to_bytes(4) + b"\x07").hex()
    fixture.update(messages=[message], extra_ranges=[extra], size=message["length"])
    edit_header(fixture, 0, 8, message["length"].to_bytes(8))
    edit_header(fixture, 0, 152 + 5, points.to_bytes(4))
    edit_header(fixture, 0, 152 + 19, bytes([bits]))
    assert [m.identity for m in scan(fixture)] == [WAVE]


def test_early_zero_fill_wave_requires_bitmap(source_fixture: dict[str, Any]) -> None:
    if source_fixture["time"] != "2017-12-12T12:00:00":
        pytest.skip("Requires no-bitmap wave era")
    with pytest.raises(AssertionError, match="Bitmap required"):
        scan(source_fixture)


@pytest.mark.parametrize("adjustment", [-1, 1])
def test_bitmap_sections_must_tile_message(
    source_fixture: dict[str, Any], adjustment: int
) -> None:
    fixture = bitmap_fixture(source_fixture)
    index = next(
        i for i, m in enumerate(fixture["messages"]) if m["element"] == "HTSGW"
    )
    message = fixture["messages"][index]
    edit_header(fixture, index, 8, (message["length"] + adjustment).to_bytes(8))
    with pytest.raises(AssertionError, match="tile message"):
        scan(fixture)


def test_bitmap_bounds_rejected_before_extra_read(
    source_fixture: dict[str, Any],
) -> None:
    fixture = bitmap_fixture(source_fixture)
    index = next(
        i for i, m in enumerate(fixture["messages"]) if m["element"] == "HTSGW"
    )
    edit_header(fixture, index, 8, (200).to_bytes(8))
    read = sparse_source(fixture)
    extra = fixture["extra_ranges"][0]

    def read_range(start: int, end: int) -> bytes:
        assert start != extra["start"], "Must reject bounds before reading section 7"
        return read(start, end)

    with pytest.raises(AssertionError, match="bitmap section length"):
        list(
            scan_analysis_messages(
                read_range,
                fixture["size"],
                pd.Timestamp(fixture["time"]),
                {WAVE},
                bitmapped=frozenset({WAVE}),
            )
        )


def test_unselected_bitmap_does_not_read_section_seven(
    source_fixture: dict[str, Any],
) -> None:
    fixture = bitmap_fixture(source_fixture)
    read = sparse_source(fixture)
    extra = fixture["extra_ranges"][0]

    def read_range(start: int, end: int) -> bytes:
        assert start != extra["start"]
        return read(start, end)

    assert (
        len(
            list(
                scan_analysis_messages(
                    read_range, fixture["size"], pd.Timestamp(fixture["time"]), SELECTED
                )
            )
        )
        == 13
    )


def test_bitmap_zero_rejected_for_undeclared_field(
    source_fixture: dict[str, Any],
) -> None:
    fixture = bitmap_fixture(source_fixture)
    with pytest.raises(AssertionError, match="Wrong packed point count"):
        list(
            scan_analysis_messages(
                sparse_source(fixture),
                fixture["size"],
                pd.Timestamp(fixture["time"]),
                SELECTED | {WAVE},
            )
        )


@pytest.mark.parametrize("adjustment", [-1, 1])
def test_source_size_mismatch_rejected(
    source_fixture: dict[str, Any], adjustment: int
) -> None:
    fixture = bitmap_fixture(source_fixture)
    fixture["size"] += adjustment
    with pytest.raises(AssertionError):
        scan(fixture)


@pytest.mark.parametrize(
    "fixture_path",
    sorted((Path(__file__).parents[1] / "rtma_ru/fixtures").glob("*.json")),
)
def test_explicit_empty_bitmap_selection_preserves_ru_reads(fixture_path: Path) -> None:
    assert_empty_bitmap_scans_equal(json.loads(fixture_path.read_text()))
