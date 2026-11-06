from collections import Counter
from collections.abc import Buffer, Iterator
from time import monotonic
from typing import NamedTuple

import pandas as pd

from reformatters.common.logging import get_logger

from .source import EcmwfFastSourceFileCoord

log = get_logger(__name__)

# Units are defined by parameter code tables, not stored as strings in GRIB messages.
_GRIB1_PARAMS = (
    {
        (128, 167, level_type, level): ("2t", "K")
        for level_type, level in ((1, 0), (105, 2))
    }
    | {
        (table, number, level_type, level): (param, "m s-1")
        for table, number, param, height in (
            (128, 165, "10u", 10),
            (128, 166, "10v", 10),
            (228, 246, "100u", 100),
            (228, 247, "100v", 100),
        )
        for level_type, level in ((1, 0), (105, height))
    }
    | {
        (128, 228, 1, 0): ("tp", "m"),
        (128, 169, 1, 0): ("ssrd", "J m-2"),
    }
)
_GRIB2_PARAMS: dict[tuple[int, int, int, int, float], tuple[str, str]] = {
    (0, 0, 0, 103, 2): ("2t", "K"),
    (0, 2, 2, 103, 10): ("10u", "m s-1"),
    (0, 2, 3, 103, 10): ("10v", "m s-1"),
    (0, 2, 2, 103, 100): ("100u", "m s-1"),
    (0, 2, 3, 103, 100): ("100v", "m s-1"),
    (0, 1, 52, 1, 0): ("tp", "kg m-2"),
    (0, 4, 7, 1, 0): ("ssrd", "J m-2"),
}
_EQUIVALENT_UNITS = (frozenset({"kg m-2", "mm"}), frozenset({"J m-2", "W s m-2"}))
_TIME_SECONDS = {0: 60, 1: 3600, 2: 86400, 10: 10800, 11: 21600, 12: 43200}
_GRIB1_TIME_SECONDS = _TIME_SECONDS | {13: 900, 14: 1800, 254: 1}
_GRIB2_TIME_SECONDS = _TIME_SECONDS | {13: 1}
_GRID_POINTS = 721 * 1440
_GRIB1_RAW_PARAMS = frozenset(key[:2] for key in _GRIB1_PARAMS)
_GRIB2_RAW_PARAMS = frozenset(key[:3] for key in _GRIB2_PARAMS)


class _MessageSkipError(ValueError):
    pass


def _admit(condition: bool, message: str) -> None:
    if not condition:
        raise _MessageSkipError(message)


def _source_units(
    param: str, units: str, coord: EcmwfFastSourceFileCoord, deviations: set[str]
) -> None:
    _admit(param in coord.source_params, f"unconfigured parameter={param}")
    expected = coord.source_units[param]
    if units == expected:
        return
    _admit(
        any({units, expected} <= group for group in _EQUIVALENT_UNITS),
        f"{param} source units={units} (expected {expected})",
    )
    deviations.add(f"{param} equivalent source units={units} (expected {expected})")


def _time_seconds(value: int, unit: int, edition: int) -> int:
    units = _GRIB1_TIME_SECONDS if edition == 1 else _GRIB2_TIME_SECONDS
    _admit(unit in units, f"unsupported GRIB{edition} time unit={unit}")
    return value * units[unit]


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
    def __init__(self, chunks: Iterator[Buffer]) -> None:
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


def _date(data: bytes, *, interval: bool = False) -> pd.Timestamp:
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
        raise (_MessageSkipError if interval else SourceContractError)(
            "Invalid GRIB2 date"
        ) from exc


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
    reader.skip(pds_size - len(pds))
    raw_identity = (pds[3], pds[8])
    if raw_identity not in _GRIB1_RAW_PARAMS:
        dropped[raw_identity] += 1
        reader.skip(end - 4 - reader.position)
        return None
    _require_source(pds[7] == 128, "Expected GRIB1 grid definition without bitmap")
    _require_source(
        reader.position + 3 <= end - 4, "GRIB1 grid section outside message"
    )
    gds_size = _uint(reader.read(3))
    _require_source(gds_size == 32, "Unsupported GRIB1 grid section")
    _require_source(
        reader.position + gds_size - 3 <= end - 4, "GRIB1 grid section outside message"
    )
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
    _require_source(
        reader.position + 3 <= end - 4, "GRIB1 data section outside message"
    )
    bds_size = _uint(reader.read(3))
    _require_source(bds_size >= 11, "GRIB1 sections do not tile message")
    _require_source(
        reader.position - 3 + bds_size == end - 4, "GRIB1 sections do not tile message"
    )
    bds = reader.read(8)
    if bds[0] & 240 == 0:
        _require_source(
            (bds_size - 11) * 8 - (bds[0] & 15) == bds[7] * _GRID_POINTS,
            "GRIB1 packed-bit capacity differs from grid",
        )
    reader.skip(bds_size - 11)
    identity = (*raw_identity, pds[9], _uint(pds[10:12]))
    if identity not in _GRIB1_PARAMS:
        dropped[identity] += 1
        return None
    _require_source(52 <= pds_size <= 512, "Unsupported GRIB1 product section")
    if pds[4] != 98:
        deviations.add(f"centre={pds[4]} (expected 98)")
    _require_source(
        pds[40] == (1 if coord.family == "cf" else 30),
        "Unsupported ECMWF local definition",
    )
    member = pds[49]
    _require_source(
        (member == 0) == (coord.family == "cf"),
        f"Unexpected member {member} for {coord.family}",
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
    _admit(bds[0] & 240 == 0, "unsupported GRIB1 packing")
    param, units = _GRIB1_PARAMS[identity]
    _source_units(param, units, coord, deviations)
    _grib1_time(pds, param, coord, deviations)
    return param, member


def _grib1_time(
    pds: bytes, param: str, coord: EcmwfFastSourceFileCoord, deviations: set[str]
) -> None:
    unit, p1, p2, tri = pds[17:21]
    _admit(tri in (0, 1, 2, 3, 4, 5, 10), f"unsupported GRIB1 time range={tri}")
    accumulated = param in ("tp", "ssrd")
    if tri in (2, 3, 4, 5):
        _admit(p1 <= p2, "reversed GRIB1 time interval")
        lead = _time_seconds(p2, unit, 1)
        _admit(
            accumulated and tri == 4 and p1 == 0, "changed GRIB1 accumulation meaning"
        )
    else:
        _admit(tri == 10 or p2 == 0, "unexpected GRIB1 time range")
        _admit(tri != 1 or p1 == 0, "GRIB1 analysis has a nonzero forecast time")
        lead = _time_seconds(_uint(pds[18:20]) if tri == 10 else p1, unit, 1)
        # ECMWF's native accumulation parameters use TRI 0/10 for totals since initialization.
        _admit(
            not accumulated or tri in (0, 10) or (tri == 1 and lead == 0),
            "changed GRIB1 accumulation meaning",
        )
    _admit(
        lead == coord.lead_time.total_seconds(),
        "GRIB1 forecast time differs from filename",
    )
    if unit != 1 or tri in (2, 3, 4, 5):
        deviations.add(f"GRIB1 time encoding unit={unit}, range={tri}")


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
        _require_source(reader.position + 5 <= end - 4, "GRIB2 section outside message")
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
    if raw_identity not in _GRIB2_RAW_PARAMS:
        dropped[raw_identity] += 1
        return None
    _require_source(not oversized_header, "Unsupported GRIB2 header size")
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
    _require_source(len(packing) >= 11, "Invalid GRIB2 packing section")
    _require_source(
        _uint(packing[5:9]) == _GRID_POINTS, "Unsupported GRIB2 packed grid dimensions"
    )
    _require_source(bitmap == b"\x00\x00\x00\x06\x06\xff", "Unsupported GRIB2 bitmap")
    identity = _grib2_product(product, discipline, coord, deviations)
    packing_template = _uint(packing[9:11])
    _admit(packing_template in (0, 42), f"unsupported GRIB2 packing={packing_template}")
    _require_source(
        len(packing) == (21 if packing_template == 0 else 25),
        "Invalid GRIB2 packing section",
    )
    return identity


def _grib2_product(
    product: bytes,
    discipline: int,
    coord: EcmwfFastSourceFileCoord,
    deviations: set[str],
) -> tuple[str, int]:
    _require_source(len(product) >= 34, "Unsupported GRIB2 product section")
    template = _uint(product[7:9])
    _admit(template in (0, 1, 8, 11), f"unsupported GRIB2 product template={template}")
    _admit(_uint(product[5:7]) == 0, "unsupported GRIB2 vertical coordinates")
    ensemble = template in (1, 11)
    base = 37 if ensemble else 34
    _require_source(len(product) >= base, "Unsupported GRIB2 product section")
    scale = _signed(product[23:24])
    height = _uint(product[24:28]) * 10.0**-scale if product[23] != 255 else -1
    if product[22] == 1:
        height = 0
    identity = (discipline, product[9], product[10], product[22], height)
    _admit(identity in _GRIB2_PARAMS, f"unsupported GRIB2 physical identity={identity}")
    _admit(product[28:34] == b"\xff" * 6, "unsupported GRIB2 second fixed surface")
    if ensemble:
        if product[36] != 51:
            deviations.add(f"ensemble size={product[36]} (expected 51)")
        _require_source(
            product[34] == (5 if coord.family == "cf" else 6),
            "Unexpected ensemble forecast type",
        )
        member = product[35]
        _require_source(
            (member == 0) == (coord.family == "cf"),
            f"Unexpected member {member} for {coord.family}",
        )
    else:
        _require_source(
            coord.family == "cf", "GRIB2 perturbed field has no member identity"
        )
        member = 0
        deviations.add(
            f"GRIB2 deterministic product template={template} in control file"
        )
    param, units = _GRIB2_PARAMS[identity]
    _source_units(param, units, coord, deviations)
    lead = _time_seconds(_signed(product[18:22]), product[17], 2)
    if template in (8, 11):
        _require_source(len(product) >= base + 12, "Unsupported GRIB2 interval section")
        count = product[base + 7]
        _admit(count == 1, f"unsupported GRIB2 statistical time ranges={count}")
        _require_source(
            len(product) == base + 12 + 12 * count, "Unsupported GRIB2 interval section"
        )
        stat = product[base + 12 : base + 24]
        _admit(
            param in ("tp", "ssrd") and lead == 0 and stat[0] == 1 and stat[1] == 2,
            "changed GRIB2 accumulation meaning",
        )
        lead = _time_seconds(_uint(stat[3:7]), stat[2], 2)
        _admit(
            _date(product[base : base + 7], interval=True)
            == coord.init_time + coord.lead_time,
            "GRIB2 accumulation end differs from filename",
        )
        _admit(
            lead == coord.lead_time.total_seconds(),
            "GRIB2 accumulation length differs from filename",
        )
        if _uint(product[base + 8 : base + 12]) or _uint(stat[8:12]):
            deviations.add(
                f"GRIB2 accumulation missing inputs={_uint(product[base + 8 : base + 12])}, increment unit={stat[7]}, value={_uint(stat[8:12])}"
            )
        if stat[2] != 1:
            deviations.add(f"GRIB2 accumulation time unit={stat[2]}")
        if _uint(stat[8:12]):
            _time_seconds(_uint(stat[8:12]), stat[7], 2)
    else:
        _require_source(len(product) == base, "Unsupported GRIB2 product section")
        _admit(param not in ("tp", "ssrd"), "changed GRIB2 accumulation meaning")
        _admit(
            lead == coord.lead_time.total_seconds(),
            "GRIB2 forecast time differs from filename",
        )
    if product[17] != 1:
        deviations.add(f"GRIB2 forecast time unit={product[17]}")
    return param, member


def inventory(
    reader: StreamReader, file_size: int, coord: EcmwfFastSourceFileCoord
) -> list[Message]:
    _require_source(file_size > 0, "Invalid source file size")
    messages = []
    deviations: set[str] = set()
    dropped: Counter[tuple[int, ...]] = Counter()
    skipped: Counter[str] = Counter()
    seen: set[tuple[str, int]] = set()
    while reader.position < file_size:
        offset = reader.position
        header = reader.read(8)
        _require_source(header[:4] == b"GRIB", "Missing GRIB message magic")
        edition = header[7]
        _require_source(edition in (1, 2), "Unknown GRIB edition")
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
        try:
            field = (
                _grib1(reader, end, coord, deviations, dropped)
                if edition == 1
                else _grib2(reader, end, header[6], coord, deviations, dropped)
            )
        except _MessageSkipError as exc:
            skipped[str(exc)] += 1
            field = None
        assert reader.position == end - 4, "GRIB scanner did not reach trailer boundary"
        _require_source(reader.read(4) == b"7777", "Missing GRIB trailer")
        if field is None:
            continue
        param, member = field
        identity = (param, member)
        if member not in coord.members:
            skipped[f"unconfigured member={member}"] += 1
            continue
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
    if skipped:
        deviations.add(f"skipped messages={dict(sorted(skipped.items()))}")
    if deviations:
        log.error(
            f"Source discrepancies in {coord.key}: {'; '.join(sorted(deviations))}"
        )
    return messages
