from datetime import datetime
from pathlib import Path

import pytest
from gribberish import parse_grib_message_metadata  # ty: ignore[unresolved-import]

from reformatters.ecmwf.archive_gribs.field_identity import (
    FIELD_KEYS,
    VARIABLES_BY_FIELD_KEY,
    field_key,
)
from reformatters.ecmwf.archive_gribs.grib_inventory import (
    GribMetadata,
    MessageRecord,
    check_and_index_staged_blob,
    check_inventory,
    count_grib_messages,
    iter_grib_messages,
    lead_hours,
    read_index,
    read_message_records,
    write_index,
)
from tests.ecmwf.s2s_fixtures import CONTROL_BLOB, blob_record, extract_messages


def grib_message(payload: bytes = b"contents") -> bytes:
    message_size = 16 + len(payload) + 4
    return (
        b"GRIB\x00\x00\x00\x02"
        + message_size.to_bytes(8, byteorder="big")
        + payload
        + b"7777"
    )


def blob(tmp_path: Path, *messages: bytes) -> Path:
    path = tmp_path / "blob.grib2"
    path.write_bytes(b"".join(messages))
    return path


def test_counts_structural_messages_not_embedded_grib_bytes(tmp_path: Path) -> None:
    path = blob(tmp_path, grib_message(b"packed GRIB payload"), grib_message())

    assert count_grib_messages(path) == 2


def test_rejects_a_truncated_final_message(tmp_path: Path) -> None:
    path = blob(tmp_path, grib_message(), grib_message()[:-6])

    with pytest.raises(AssertionError, match="Truncated message"):
        count_grib_messages(path)


def test_rejects_bytes_that_do_not_start_a_message(tmp_path: Path) -> None:
    path = blob(tmp_path, grib_message(), b"trailing bytes--")

    with pytest.raises(AssertionError, match="Expected a GRIB message"):
        list(iter_grib_messages(path))


def test_every_field_key_identifies_one_variable() -> None:
    assert len(VARIABLES_BY_FIELD_KEY) == len(FIELD_KEYS)


def test_names_every_message_of_a_real_blob() -> None:
    records = read_message_records(CONTROL_BLOB)

    assert [(r.variable, r.level, r.lead_hours) for r in records] == [
        ("2_m_temperature", "", 24),
        ("2_m_temperature", "", 48),
        ("soil_moisture_top_20_cm", "", 24),
        ("temperature", "500_hpa", 24),
        ("total_cloud_cover", "", 24),
        ("total_precipitation", "", 0),
        ("total_precipitation", "", 6),
        ("total_precipitation", "", 12),
        ("convective_precipitation", "", 0),
        ("convective_precipitation", "", 24),
        ("convective_precipitation", "", 48),
    ]


def test_a_24_hour_mean_is_labelled_at_the_end_of_its_interval() -> None:
    """The `0_24` mean covers hours 0 to 24 and belongs at lead time 24, not 0."""
    means = [
        (message, metadata)
        for _offset, message in iter_grib_messages(CONTROL_BLOB)
        if (metadata := _metadata(message)).statistical_process == "average"
    ]
    mean_2m_temperature = [
        (message, metadata)
        for message, metadata in means
        if VARIABLES_BY_FIELD_KEY[field_key(message, "average")][0] == "2_m_temperature"
    ]
    first, second = (metadata for _message, metadata in mean_2m_temperature)

    assert _hours(first, first.forecast_date) == 0
    assert _hours(first, first.forecast_date_end) == 24
    assert lead_hours(first) == 24
    assert lead_hours(second) == 48


def test_an_accumulation_is_at_the_end_of_the_interval_it_spans() -> None:
    """ECDS asks for `leadtime_hour=6` and answers with an interval of 0 to 6 hours."""
    accumulations = [
        metadata
        for _offset, message in iter_grib_messages(CONTROL_BLOB)
        if (metadata := _metadata(message)).statistical_process == "accumulation"
    ]
    six_hour = next(
        metadata
        for metadata in accumulations
        if _hours(metadata, metadata.forecast_date_end) == 6
    )

    assert _hours(six_hour, six_hour.forecast_date) == 0
    assert lead_hours(six_hour) == 6


def test_writes_an_index_that_locates_every_message(tmp_path: Path) -> None:
    index_path = tmp_path / "blob.index"
    write_index(read_message_records(CONTROL_BLOB), index_path)

    contents = CONTROL_BLOB.read_bytes()
    records = read_index(index_path)
    assert len(records) == 11
    for record in records:
        message = contents[record.offset : record.offset + record.length]
        assert message.startswith(b"GRIB")
        assert message.endswith(b"7777")


def complete_inventory() -> list[MessageRecord]:
    return [
        MessageRecord(
            variable=variable,
            level="",
            ensemble_member=member,
            lead_hours=lead_hours,
            offset=0,
            length=1,
        )
        for variable in ("2_m_temperature", "2_m_dewpoint_temperature")
        for member in (1, 2)
        for lead_hours in (24, 48)
    ]


def check_complete(records: list[MessageRecord]) -> None:
    check_inventory(
        records,
        variables={"2_m_temperature", "2_m_dewpoint_temperature"},
        levels=set(),
        ensemble_members={1, 2},
        lead_time_labels={"0_24", "24_48"},
    )


def test_accepts_the_exact_requested_product() -> None:
    check_complete(complete_inventory())


def test_rejects_a_missing_member_lead_pair() -> None:
    with pytest.raises(AssertionError, match="is missing"):
        check_complete(complete_inventory()[:-1])


def test_rejects_a_missing_variable() -> None:
    records = [r for r in complete_inventory() if r.variable == "2_m_temperature"]

    with pytest.raises(AssertionError, match="Missing fields"):
        check_complete(records)


def test_rejects_a_variable_that_was_not_requested() -> None:
    records = complete_inventory()
    unrequested = MessageRecord("skin_temperature", "", 1, 24, 0, 1)

    with pytest.raises(AssertionError, match="Unexpected fields"):
        check_complete([*records, unrequested])


def test_rejects_a_duplicated_message() -> None:
    records = complete_inventory()

    with pytest.raises(AssertionError, match="Duplicate messages"):
        check_complete([*records, records[0]])


def test_rejects_an_unrequested_lead_time() -> None:
    records = complete_inventory()
    unrequested = MessageRecord("2_m_temperature", "", 1, 72, 0, 1)

    with pytest.raises(AssertionError, match="has unexpected"):
        check_complete([*records, unrequested])


def stage_precipitation(tmp_path: Path, *lead_hours: int) -> Path:
    return extract_messages(
        tmp_path / CONTROL_BLOB.name,
        *(blob_record("total_precipitation", "", hours) for hours in lead_hours),
    )


def test_indexes_a_complete_blob(tmp_path: Path) -> None:
    staged = stage_precipitation(tmp_path, 0, 6, 12)

    index_path = check_and_index_staged_blob(
        staged,
        variables={"total_precipitation"},
        levels=set(),
        ensemble_members={0},
        lead_time_labels={"0", "6", "12"},
    )

    assert [record.lead_hours for record in read_index(index_path)] == [0, 6, 12]


def test_rejects_a_blob_missing_one_of_its_messages(tmp_path: Path) -> None:
    staged = stage_precipitation(tmp_path, 0, 6)

    with pytest.raises(AssertionError, match="is missing"):
        check_and_index_staged_blob(
            staged,
            variables={"total_precipitation"},
            levels=set(),
            ensemble_members={0},
            lead_time_labels={"0", "6", "12"},
        )


def _metadata(message: bytes) -> GribMetadata:
    return parse_grib_message_metadata(message, 0)


def _hours(metadata: GribMetadata, moment: datetime | None) -> int:
    assert moment is not None
    return int((moment - metadata.reference_date).total_seconds() // 3600)
