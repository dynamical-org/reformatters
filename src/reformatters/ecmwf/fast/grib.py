from collections import Counter
from collections.abc import Iterator
from time import monotonic
from typing import NamedTuple, assert_never

import pandas as pd

from reformatters.common.logging import get_logger
from reformatters.common.time_utils import whole_hours

from .source import EcmwfFastSourceFileCoord

log = get_logger(__name__)

_GRIB1_PARAMS = {
    (128, 167): "2t",
    (128, 165): "10u",
    (128, 166): "10v",
    (228, 246): "100u",
    (228, 247): "100v",
    (128, 228): "tp",
    (128, 169): "ssrd",
}


class InterruptedGribStreamError(EOFError):
    pass


class SourceContractError(ValueError):
    pass


def _require_source(condition: bool, message: str) -> None:
    if not condition:
        raise SourceContractError(message)


class Message(NamedTuple):
    param: str
    member: int
    offset: int
    length: int


class StreamReader:
    def __init__(self, chunks: Iterator[bytes]) -> None:
        self.chunks = chunks
        self.buffer = memoryview(b"")
        self.position = 0
        self.stream_wait_seconds = 0.0

    def _available(self) -> bool:
        while not self.buffer:
            started = monotonic()
            chunk = next(self.chunks, None)
            self.stream_wait_seconds += monotonic() - started
            if chunk is None:
                return False
            self.buffer = memoryview(chunk)
        return True

    def read(self, size: int) -> bytes:
        assert 0 <= size <= 512, "Only GRIB headers may be buffered"
        parts = []
        while size and self._available():
            count = min(size, len(self.buffer))
            parts.append(bytes(self.buffer[:count]))
            self.buffer = self.buffer[count:]
            self.position += count
            size -= count
        if size:
            raise InterruptedGribStreamError(
                "Source stream ended before the declared GRIB boundary"
            )
        return b"".join(parts)

    def skip(self, size: int) -> None:
        assert size >= 0
        while size and self._available():
            count = min(size, len(self.buffer))
            self.buffer = self.buffer[count:]
            self.position += count
            size -= count
        if size:
            raise InterruptedGribStreamError(
                "Source stream ended inside a GRIB payload"
            )

    def assert_eof(self) -> None:
        _require_source(
            not self._available(), "Source contains bytes beyond its listed size"
        )


def _uint(data: bytes) -> int:
    return int.from_bytes(data, "big")


def _signed(data: bytes) -> int:
    value = _uint(data)
    sign = 1 << len(data) * 8 - 1
    return -(value & sign - 1) if value & sign else value


def _date(data: bytes) -> pd.Timestamp:
    try:
        return pd.Timestamp(
            year=_uint(data[:2]),
            month=data[2],
            day=data[3],
            hour=data[4],
            minute=data[5],
            second=data[6],
        )
    except ValueError as exc:
        raise SourceContractError("Invalid GRIB2 date") from exc


def _grib1(
    reader: StreamReader,
    end: int,
    coord: EcmwfFastSourceFileCoord,
    deviations: set[str],
    dropped: Counter[tuple[int, ...]],
) -> tuple[str, int] | None:
    length = reader.read(3)
    pds_size = _uint(length)
    _require_source(pds_size >= 28, "Unsupported GRIB1 product section")
    _require_source(
        reader.position - 3 + pds_size < end - 4,
        "GRIB1 product section outside message",
    )
    pds = length + reader.read(min(pds_size, 512) - 3)
    identity = (pds[3], pds[8])
    if identity not in _GRIB1_PARAMS:
        dropped[identity] += 1
        reader.skip(end - 4 - reader.position)
        return None
    _require_source(52 <= pds_size <= 512, "Unsupported GRIB1 product section")
    if pds[4] != 98:
        deviations.add(f"centre={pds[4]} (expected 98)")
    _require_source(pds[7] == 128, "Expected GRIB1 grid definition without bitmap")
    _require_source(pds[9:12] == b"\x01\x00\x00", "Unexpected GRIB1 level")
    _require_source(
        pds[40] == (1 if coord.family == "cf" else 30),
        "Unsupported ECMWF local definition",
    )
    try:
        init = pd.Timestamp(
            year=(pds[24] - 1) * 100 + pds[12],
            month=pds[13],
            day=pds[14],
            hour=pds[15],
            minute=pds[16],
        )
    except ValueError as exc:
        raise SourceContractError("Invalid GRIB1 date") from exc
    _require_source(
        init == coord.init_time, "GRIB1 initialization differs from filename"
    )
    _require_source(pds[17] == 1, "Unsupported GRIB1 forecast time encoding")
    _require_source(pds[20] in (0, 1, 10), "Unsupported GRIB1 forecast time encoding")
    hours = _uint(pds[18:20]) if pds[20] == 10 else pds[18]
    _require_source(pds[20] == 10 or pds[19] == 0, "Unexpected GRIB1 time range")
    _require_source(
        hours == whole_hours(coord.lead_time),
        "GRIB1 forecast time differs from filename",
    )
    gds_size = _uint(reader.read(3))
    _require_source(gds_size == 32, "Unsupported GRIB1 grid section")
    gds = gds_size.to_bytes(3, "big") + reader.read(gds_size - 3)
    _require_source(gds[5] == 0, "Unsupported GRIB1 grid")
    _require_source(_uint(gds[6:8]) == 1440, "Unsupported GRIB1 grid")
    _require_source(_uint(gds[8:10]) == 721, "Unsupported GRIB1 grid")
    _require_source(
        (
            _signed(gds[10:13]),
            _signed(gds[13:16]),
            _signed(gds[17:20]),
            _signed(gds[20:23]),
        )
        == (90000, -180000, -90000, 179750),
        "Unexpected GRIB1 grid endpoints",
    )
    _require_source(
        _uint(gds[23:25]) == _uint(gds[25:27]) == 250, "Unsupported GRIB1 scanning"
    )
    _require_source(gds[27] == 0, "Unsupported GRIB1 scanning")
    bds_size = _uint(reader.read(3))
    _require_source(bds_size >= 11, "GRIB1 sections do not tile message")
    _require_source(
        reader.position - 3 + bds_size == end - 4, "GRIB1 sections do not tile message"
    )
    bds = reader.read(8)
    _require_source(bds[0] & 240 == 0, "Only simple GRIB1 packing is supported")
    reader.skip(bds_size - 11)
    return (_GRIB1_PARAMS[identity], pds[49])


def _grib2(
    reader: StreamReader,
    end: int,
    discipline: int,
    coord: EcmwfFastSourceFileCoord,
    deviations: set[str],
    dropped: Counter[tuple[int, ...]],
) -> tuple[str, int] | None:
    sections: dict[int, bytes] = {}
    order = []
    oversized_header = False
    while reader.position < end - 4:
        header = reader.read(5)
        size, number = (_uint(header[:4]), header[4])
        _require_source(size >= 5, "GRIB2 section outside message")
        _require_source(
            reader.position + size - 5 <= end - 4, "GRIB2 section outside message"
        )
        _require_source(
            number not in order, "Multiple fields in one GRIB2 message are unsupported"
        )
        order.append(number)
        if number == 7:
            reader.skip(size - 5)
        else:
            prefix_size = min(size, 512)
            sections[number] = header + reader.read(prefix_size - 5)
            reader.skip(size - prefix_size)
            oversized_header |= size > 512
    _require_source(
        order in ([1, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]),
        "Unexpected GRIB2 section sequence",
    )
    ref, grid, product, packing, bitmap = (sections[n] for n in (1, 3, 4, 5, 6))
    _require_source(len(product) >= 11, "Unsupported GRIB2 product section")
    raw_identity = (discipline, product[9], product[10])
    if raw_identity not in ((0, 2, 2), (0, 2, 3), (0, 0, 0), (0, 1, 52), (0, 4, 7)):
        dropped[raw_identity] += 1
        return None
    _require_source(not oversized_header, "Unsupported GRIB2 header size")
    identity = _grib2_product(product, discipline, coord, deviations)
    _require_source(len(ref) == 21, "Unsupported GRIB2 identification")
    if (centre := _uint(ref[5:7])) != 98:
        deviations.add(f"centre={centre} (expected 98)")
    _require_source(
        _date(ref[12:19]) == coord.init_time,
        "GRIB2 initialization differs from filename",
    )
    _require_source(len(grid) == 72, "Unsupported GRIB2 grid template")
    _require_source(_uint(grid[12:14]) == 0, "Unsupported GRIB2 grid template")
    if grid[14] != 6:
        deviations.add(f"earth shape={grid[14]} (expected 6)")
    _require_source(_uint(grid[30:34]) == 1440, "Unsupported GRIB2 grid dimensions")
    _require_source(_uint(grid[34:38]) == 721, "Unsupported GRIB2 grid dimensions")
    _require_source(
        _uint(grid[6:10]) == 721 * 1440, "Unsupported GRIB2 grid dimensions"
    )
    _require_source(
        (
            _signed(grid[46:50]),
            _uint(grid[50:54]),
            _signed(grid[55:59]),
            _uint(grid[59:63]),
        )
        == (90000000, 180000000, -90000000, 179750000),
        "Unexpected GRIB2 grid endpoints",
    )
    _require_source(_uint(grid[38:42]) == 0, "Unsupported GRIB2 grid units or scanning")
    _require_source(
        _uint(grid[63:67]) == _uint(grid[67:71]) == 250000,
        "Unsupported GRIB2 grid units or scanning",
    )
    _require_source(grid[71] == 0, "Unsupported GRIB2 grid units or scanning")
    _require_source(len(packing) == 25, "Only native CCSDS GRIB2 fields are supported")
    _require_source(
        _uint(packing[9:11]) == 42, "Only native CCSDS GRIB2 fields are supported"
    )
    _require_source(
        _uint(packing[5:9]) == 721 * 1440,
        "Only native CCSDS GRIB2 fields are supported",
    )
    _require_source(bitmap == b"\x00\x00\x00\x06\x06\xff", "Unsupported GRIB2 bitmap")
    return identity


def _grib2_product(
    product: bytes,
    discipline: int,
    coord: EcmwfFastSourceFileCoord,
    deviations: set[str],
) -> tuple[str, int]:
    _require_source(len(product) in (37, 61), "Unsupported GRIB2 product section")
    _require_source(_uint(product[5:7]) == 0, "Unsupported GRIB2 product section")
    template = _uint(product[7:9])
    _require_source(template in (1, 11), "Unsupported GRIB2 product template")
    _require_source(
        len(product) == (37 if template == 1 else 61),
        "Unsupported GRIB2 product template",
    )
    if product[36] != 51:
        deviations.add(f"ensemble size={product[36]} (expected 51)")
    _require_source(
        product[34] == (5 if coord.family == "cf" else 6),
        "Unexpected ensemble forecast type",
    )
    _require_source(product[28:34] == b"\xff" * 6, "Unexpected second fixed surface")
    _require_source(product[17] == 1, "Unsupported GRIB2 forecast time unit")
    identity = (discipline, product[9], product[10])
    if identity in ((0, 2, 2), (0, 2, 3), (0, 0, 0)):
        height = _uint(product[24:28])
        _require_source(product[22] == 103, "Unexpected fixed-height surface")
        _require_source(product[23] == 0, "Unexpected fixed-height surface")
        if identity == (0, 0, 0):
            _require_source(height == 2, "Unexpected temperature height")
            param = "2t"
        else:
            _require_source(height in (10, 100), "Unexpected wind height")
            param = f"{height}{('u' if product[10] == 2 else 'v')}"
        _require_source(template == 1, "Expected instantaneous field")
        _require_source(
            _uint(product[18:22]) == whole_hours(coord.lead_time),
            "GRIB2 forecast time differs from filename",
        )
    else:
        _require_source(product[22] == 1, "Expected accumulated surface field")
        _require_source(template == 11, "Expected accumulated surface field")
        _require_source(
            _uint(product[18:22]) == 0, "Accumulation does not start at initialization"
        )
        _require_source(product[44] == 1, "Unexpected accumulation interval")
        _require_source(
            product[49:52] == bytes((1, 2, 1)), "Unexpected accumulation interval"
        )
        _require_source(
            _uint(product[52:56]) == whole_hours(coord.lead_time),
            "Accumulation length differs from filename",
        )
        _require_source(
            _date(product[37:44]) == coord.init_time + coord.lead_time,
            "Accumulation end differs from filename",
        )
        param = "tp" if product[9] == 1 else "ssrd"
    return param, product[35]


def inventory(
    reader: StreamReader, file_size: int, coord: EcmwfFastSourceFileCoord
) -> list[Message]:
    match coord.model:
        case "ifs":
            expected_edition = 1
        case "aifs":
            expected_edition = 2
        case unexpected:
            assert_never(unexpected)
    _require_source(file_size > 0, "Invalid source file size")
    messages = []
    deviations: set[str] = set()
    dropped: Counter[tuple[int, ...]] = Counter()
    seen: set[tuple[str, int]] = set()
    while reader.position < file_size:
        offset = reader.position
        header = reader.read(8)
        _require_source(header[:4] == b"GRIB", "Missing GRIB message magic")
        edition = header[7]
        _require_source(edition == expected_edition, "Unexpected GRIB edition")
        if edition == 1:
            length = _uint(header[4:7])
            _require_source(
                length < 8388608, "Unsupported GRIB1 large-message extension"
            )
        else:
            length = _uint(reader.read(8))
        end = offset + length
        _require_source(
            reader.position + 4 < end <= file_size, "Invalid GRIB message length"
        )
        field = (
            _grib1(reader, end, coord, deviations, dropped)
            if edition == 1
            else _grib2(reader, end, header[6], coord, deviations, dropped)
        )
        assert reader.position == end - 4, "GRIB scanner did not reach trailer boundary"
        _require_source(reader.read(4) == b"7777", "Missing GRIB trailer")
        if field is None:
            continue
        param, member = field
        identity = (param, member)
        _require_source(
            member in coord.members, f"Unexpected member {member} for {coord.family}"
        )
        _require_source(identity not in seen, f"Duplicate GRIB field {identity}")
        seen.add(identity)
        messages.append(Message(param, member, offset, length))
    reader.assert_eof()
    expected = {
        (param, member) for param in coord.source_params for member in coord.members
    }
    _require_source(
        seen <= expected,
        f"Unexpected GRIB fields: {sorted(seen - expected)}",
    )
    if missing := expected - seen:
        deviations.add(f"missing fields={sorted(missing)}")
    if dropped:
        deviations.add(f"dropped unsupported fields={dict(sorted(dropped.items()))}")
    if deviations:
        log.error(
            f"Source discrepancies in {coord.key}: {'; '.join(sorted(deviations))}"
        )
    return messages
