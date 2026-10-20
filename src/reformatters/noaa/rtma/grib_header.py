from collections.abc import Callable, Iterator
from typing import NamedTuple

from reformatters.common.grib import grib2_message_length
from reformatters.common.types import Timestamp

GRID_NX = 2345
GRID_NY = 1597
GRID_SPACING = 2539.703
GRID_LA1 = 19.228976
GRID_LO1 = 233.723448
GRID_PROJ = (
    "+proj=lcc +lat_0=25 +lon_0=-95 +lat_1=25 +lat_2=25 +R=6371200 +units=m +no_defs"
)


type FieldIdentity = tuple[int, int, int, int, float]


class AnalysisMessage(NamedTuple):
    identity: FieldIdentity
    offset: int
    length: int


def scan_analysis_messages(
    read_range: Callable[[int, int], bytes],
    file_size: int,
    expected_time: Timestamp,
    selected: set[FieldIdentity],
) -> Iterator[AnalysisMessage]:
    """Validate a GRIB chain and yield selected, independently decodable analyses."""
    assert file_size > 0, "Empty GRIB file"
    seen: set[FieldIdentity] = set()
    offset = 0
    while offset < file_size:
        start = max(0, offset - 4)
        end = min(file_size, offset + 256)
        raw = read_range(start, end)
        assert len(raw) == end - start, "Truncated GRIB header range"
        if offset:
            assert raw[:4] == b"7777", "Missing GRIB message terminator"
        header = raw[offset - start :]
        length = grib2_message_length(header)
        assert length is not None, "Invalid GRIB2 message length"
        assert length >= 20, "Invalid GRIB2 message length"
        assert offset + length <= file_size, "GRIB message exceeds file size"
        sections: dict[int, bytes] = {}
        pos = 16
        for number in (1, 3, 4):
            section, pos = _section(header, pos, number, length)
            sections[number] = section
        product = sections[4]
        assert len(product) >= 11, "Truncated product definition"
        if len(product) != 34 or product[5:9] != b"\0\0\0\0":
            offset += length
            continue
        identity = (
            header[6],
            product[9],
            product[10],
            product[22],
            _signed(product[24:28]) * 10.0 ** -_signed(product[23:24]),
        )
        if identity in selected:
            assert identity not in seen, f"Duplicate GRIB field {identity}"
            seen.add(identity)
            _validate_identification(sections[1], expected_time)
            _validate_grid(sections[3])
            _validate_payload(header, pos, length, product)
            yield AnalysisMessage(identity, offset, length)
        offset += length
    assert read_range(file_size - 4, file_size) == b"7777", (
        "Missing final GRIB terminator"
    )


def _validate_payload(header: bytes, pos: int, length: int, product: bytes) -> None:
    assert product[11] == 0, "Expected zero-lead analysis"
    assert product[17] in (0, 1), "Expected minute or hour forecast time unit"
    assert int.from_bytes(product[18:22]) == 0, "Expected zero-lead analysis"
    assert product[28] == 255, "Unexpected second fixed surface"
    representation, pos = _section(header, pos, 5, length)
    assert len(representation) == 21, "Expected simple packing DRT 5.0"
    assert representation[9:11] == b"\x00\x00", "Expected simple packing DRT 5.0"
    assert int.from_bytes(representation[5:9]) == GRID_NX * GRID_NY, (
        "Wrong packed point count"
    )
    bitmap, pos = _section(header, pos, 6, length)
    assert bitmap == b"\0\0\0\x06\x06\xff", (
        "Expected independent message without bitmap"
    )
    assert pos + 5 <= len(header), "Expected GRIB section 7"
    assert header[pos + 4] == 7, "Expected GRIB section 7"
    data_size = int.from_bytes(header[pos : pos + 4])
    packed_bytes = (GRID_NX * GRID_NY * representation[19] + 7) // 8
    assert data_size == packed_bytes + 5, "Packed data size does not match grid"
    assert pos + data_size == length - 4, "GRIB sections do not tile message"


def _section(header: bytes, pos: int, number: int, length: int) -> tuple[bytes, int]:
    assert pos + 5 <= len(header), "Truncated GRIB section header"
    size = int.from_bytes(header[pos : pos + 4])
    assert header[pos + 4] == number, f"Expected GRIB section {number}"
    assert size >= 5, "Invalid GRIB section length"
    assert pos + size <= length - 4, "Invalid GRIB section length"
    assert pos + size <= len(header), "GRIB header exceeds bounded range"
    return header[pos : pos + size], pos + size


def _validate_identification(section: bytes, expected_time: Timestamp) -> None:
    assert len(section) == 21, "Expected identification section length 21"
    assert int.from_bytes(section[5:7]) == 7, "Expected NCEP analysis reference time"
    assert section[11] == 0, "Expected NCEP analysis reference time"
    expected_utc = (
        expected_time.tz_localize("UTC")
        if expected_time.tzinfo is None
        else expected_time.tz_convert("UTC")
    )
    reference_parts = (int.from_bytes(section[12:14]), *section[14:19])
    assert reference_parts == (
        expected_utc.year,
        expected_utc.month,
        expected_utc.day,
        expected_utc.hour,
        expected_utc.minute,
        expected_utc.second,
    ), f"Unexpected GRIB reference time {reference_parts} != {expected_utc}"


def _validate_grid(section: bytes) -> None:
    assert len(section) == 81, "Expected Lambert grid template 3.30"
    assert int.from_bytes(section[12:14]) == 30, "Expected Lambert grid template 3.30"
    assert section[5] == 0, "Wrong grid point count"
    assert int.from_bytes(section[6:10]) == GRID_NX * GRID_NY, "Wrong grid point count"
    assert section[10:12] == b"\0\0", "Unexpected optional grid list"
    assert section[14] == 1, "Wrong earth radius"
    assert section[15] == 0, "Wrong earth radius"
    assert int.from_bytes(section[16:20]) == 6371200, "Wrong earth radius"
    expected = {
        (30, 34): GRID_NX,
        (34, 38): GRID_NY,
        (38, 42): round(GRID_LA1 * 1_000_000),
        (42, 46): round(GRID_LO1 * 1_000_000),
        (47, 51): 25000000,
        (51, 55): 265000000,
        (55, 59): 2539703,
        (59, 63): 2539703,
        (65, 69): 25000000,
        (69, 73): 25000000,
    }
    assert all(
        int.from_bytes(section[a:b]) == value for (a, b), value in expected.items()
    ), "Unexpected RTMA grid geometry"
    assert section[46] & 8, "Unexpected wind frame or grid scanning mode"
    assert section[63:65] == b"\x00@", "Unexpected wind frame or grid scanning mode"
    assert _signed(section[73:77]) == -90000000, "Unexpected southern pole"
    assert int.from_bytes(section[77:81]) == 0, "Unexpected southern pole"


def _signed(raw: bytes) -> int:
    value = int.from_bytes(raw)
    sign = 1 << (8 * len(raw) - 1)
    return -(value & ~sign) if value & sign else value
