"""Structural scan and strict inventory validation of a concatenated GRIB2 blob.

ECDS returns one blob per request with no index sidecar, so the only proof that a
retrieval is complete is the blob's own message inventory, and the only way to read
one message back out is a byte-range index built while scanning it.
"""

import json
from collections.abc import Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from gribberish import parse_grib_message_metadata  # ty: ignore[unresolved-import]

from reformatters.common.logging import get_logger

from .field_identity import VARIABLES_BY_FIELD_KEY, field_key

log = get_logger(__name__)

GRIB_HEADER_BYTES = 16
GRIB_TRAILER = b"7777"
INDEX_SUFFIX = ".index"


class GribMetadata(Protocol):
    statistical_process: str | None
    perturbation_number: int | None
    reference_date: datetime
    forecast_date: datetime
    forecast_date_end: datetime | None
    grid_shape: tuple[int, int]


@dataclass(frozen=True)
class MessageRecord:
    """One message's identity and where it sits in the blob."""

    variable: str
    level: str
    ensemble_member: int
    lead_hours: int
    offset: int
    length: int


def iter_grib_messages(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield each GRIB2 message and its byte offset, following the header's length.

    Scanning for the `GRIB` byte string instead would split messages whose packed
    payload happens to contain it.
    """
    with path.open("rb") as source:
        while header := source.read(GRIB_HEADER_BYTES):
            offset = source.tell() - GRIB_HEADER_BYTES
            assert len(header) == GRIB_HEADER_BYTES, f"Truncated header in {path}"
            assert header[:4] == b"GRIB", f"Expected a GRIB message in {path}"
            message_size = int.from_bytes(header[8:16], byteorder="big")
            assert message_size > GRIB_HEADER_BYTES + len(GRIB_TRAILER)
            message = header + source.read(message_size - GRIB_HEADER_BYTES)
            assert len(message) == message_size, f"Truncated message in {path}"
            assert message.endswith(GRIB_TRAILER), f"Unterminated message in {path}"
            yield offset, message
        assert source.tell() == path.stat().st_size


def count_grib_messages(path: Path) -> int:
    count = sum(1 for _ in iter_grib_messages(path))
    assert count > 0, f"No GRIB messages in {path}"
    return count


def read_message_records(path: Path) -> list[MessageRecord]:
    records = [
        _message_record(offset, message) for offset, message in iter_grib_messages(path)
    ]
    assert len(records) > 0, f"No GRIB messages in {path}"
    return records


def ensemble_member(metadata: GribMetadata) -> int:
    """The control forecast carries no perturbation number and is ensemble member 0."""
    return metadata.perturbation_number or 0


def lead_hours(metadata: GribMetadata) -> int:
    """Hours after initialization that this message's value is labelled at.

    A field aggregated over an interval (an accumulation, a 24-hour mean, a 6-hour
    extremum) is labelled at the end of its interval, so `forecast_date_end` and not
    `forecast_date` is the lead time of every interval field.
    """
    moment = metadata.forecast_date_end or metadata.forecast_date
    return _whole_hours_after_reference(metadata, moment)


def check_inventory(
    records: Iterable[MessageRecord],
    *,
    variables: AbstractSet[str],
    levels: AbstractSet[str],
    ensemble_members: AbstractSet[int],
    lead_time_labels: AbstractSet[str],
) -> None:
    """Assert the records are exactly the requested variable x level x member x lead product."""
    found_by_field: dict[tuple[str, str], set[tuple[int, int]]] = {}
    duplicates = []
    for record in records:
        field = (record.variable, record.level)
        member_and_lead = (record.ensemble_member, record.lead_hours)
        if member_and_lead in found_by_field.setdefault(field, set()):
            duplicates.append((field, member_and_lead))
        found_by_field[field].add(member_and_lead)
    assert not duplicates, f"Duplicate messages: {sorted(duplicates)}"

    expected_fields = {
        (variable, level) for variable in variables for level in levels or {""}
    }
    found_fields = set(found_by_field)
    assert not expected_fields - found_fields, (
        f"Missing fields: {sorted(expected_fields - found_fields)}"
    )
    assert not found_fields - expected_fields, (
        f"Unexpected fields: {sorted(found_fields - expected_fields)}"
    )

    for field, found in found_by_field.items():
        expected = {
            (member, _label_lead_hours(label))
            for member in ensemble_members
            for label in lead_time_labels
        }
        assert not expected - found, f"{field} is missing {sorted(expected - found)}"
        assert not found - expected, (
            f"{field} has unexpected {sorted(found - expected)}"
        )


def write_index(records: Sequence[MessageRecord], index_path: Path) -> None:
    """Write the byte-range index the reformatter reads single messages through."""
    index_path.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)
    )


def read_index(index_path: Path) -> list[MessageRecord]:
    return [
        MessageRecord(**json.loads(line))
        for line in index_path.read_text().splitlines()
    ]


def check_and_index_staged_blob(
    path: Path,
    *,
    variables: AbstractSet[str],
    levels: AbstractSet[str],
    ensemble_members: AbstractSet[int],
    lead_time_labels: AbstractSet[str],
) -> Path:
    """Validate a downloaded blob's inventory and write its index beside it."""
    records = read_message_records(path)
    log.info("Validating %d GRIB messages in %s", len(records), path)
    check_inventory(
        records,
        variables=variables,
        levels=levels,
        ensemble_members=ensemble_members,
        lead_time_labels=lead_time_labels,
    )
    index_path = path.with_name(path.name + INDEX_SUFFIX)
    write_index(records, index_path)
    return index_path


def _message_record(offset: int, message: bytes) -> MessageRecord:
    metadata = parse_grib_message_metadata(message, 0)
    key = field_key(message, metadata.statistical_process)
    variable_and_level = VARIABLES_BY_FIELD_KEY.get(key)
    assert variable_and_level is not None, (
        f"No ECMWF-origin S2S variable has GRIB identity {key}"
    )
    variable, level = variable_and_level
    return MessageRecord(
        variable=variable,
        level=level,
        ensemble_member=ensemble_member(metadata),
        lead_hours=lead_hours(metadata),
        offset=offset,
        length=len(message),
    )


def _label_lead_hours(lead_time_label: str) -> int:
    """The lead time an ECDS `leadtime_hour` label resolves to, `24_48` giving 48."""
    return int(lead_time_label.rpartition("_")[2])


def _whole_hours_after_reference(metadata: GribMetadata, moment: datetime) -> int:
    seconds = (moment - metadata.reference_date).total_seconds()
    assert seconds % 3600 == 0, f"Lead time {seconds}s is not a whole number of hours"
    return int(seconds // 3600)
