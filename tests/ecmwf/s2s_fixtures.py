from pathlib import Path

from reformatters.ecmwf.archive_gribs.grib_inventory import MessageRecord, read_index

FIXTURES = Path(__file__).parent / "fixtures"
# One control forecast message per (variable, level, lead time) of the ECMWF S2S
# initialization of 2026-08-10T00Z, retrieved from ECDS.
CONTROL_BLOB = FIXTURES / "s2s_control_2026-08-10T00Z.grib2"


def blob_records() -> list[MessageRecord]:
    return read_index(CONTROL_BLOB.with_name(CONTROL_BLOB.name + ".index"))


def blob_record(variable: str, level: str, lead_hours: int) -> MessageRecord:
    return next(
        record
        for record in blob_records()
        if (record.variable, record.level, record.lead_hours)
        == (variable, level, lead_hours)
    )


def extract_messages(destination: Path, *records: MessageRecord) -> Path:
    """Write the byte ranges of `records`, as `download_file` would fetch them."""
    contents = CONTROL_BLOB.read_bytes()
    destination.write_bytes(
        b"".join(
            contents[record.offset : record.offset + record.length]
            for record in records
        )
    )
    return destination
