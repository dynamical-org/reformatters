from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from reformatters.ecmwf.archive_gribs.archive import (
    Publication,
    archive_initialization,
    check_available,
    format_init_time,
)
from reformatters.ecmwf.archive_gribs.ecds_client import EcdsJobFailedError
from reformatters.ecmwf.archive_gribs.grib_inventory import INDEX_SUFFIX
from reformatters.ecmwf.archive_gribs.request_shards import (
    EcdsSelection,
    initialization_selections,
)

INIT_TIME = pd.Timestamp("2026-08-10T00:00", tz="UTC")
DST_ROOT = ":s3:bucket/ecmwf-s2s-grib/"
SELECTIONS = initialization_selections(["2_m_temperature", "temperature"])


def valid_constraints(selection: EcdsSelection) -> dict[str, list[str]]:
    return {
        "variable": list(selection.variables),
        "leadtime_hour": list(selection.lead_time_labels),
        "level_value": list(selection.level_values),
    }


@pytest.fixture
def archive_bucket() -> Iterator[dict[str, MagicMock]]:
    with (
        patch("reformatters.ecmwf.archive_gribs.archive.list_files") as list_files,
        patch("reformatters.ecmwf.archive_gribs.archive.constraints") as constraints,
        patch("reformatters.ecmwf.archive_gribs.archive.costing") as costing,
        patch(
            "reformatters.ecmwf.archive_gribs.archive.copy_local_file"
        ) as copy_local_file,
        patch(
            "reformatters.ecmwf.archive_gribs.archive.check_and_index_archived_blob"
        ) as check_and_index_archived_blob,
        patch("reformatters.ecmwf.archive_gribs.archive.EcdsRequest") as request,
    ):
        list_files.return_value = []
        constraints.side_effect = lambda inputs, **_: {
            "variable": inputs["variable"],
            "leadtime_hour": _lead_times_of(inputs),
            "level_value": _level_values_of(inputs),
        }
        costing.side_effect = lambda inputs, **_: (_cost_of(inputs), 1_000_000.0)
        request.return_value.retrieve.side_effect = _write_blob
        check_and_index_archived_blob.side_effect = lambda path, **_: path.with_name(
            path.name + INDEX_SUFFIX
        )
        yield {
            "list_files": list_files,
            "constraints": constraints,
            "costing": costing,
            "copy_local_file": copy_local_file,
            "check_and_index_archived_blob": check_and_index_archived_blob,
            "request": request,
        }


def _selection_for(inputs: dict[str, Any]) -> EcdsSelection:
    return next(
        selection
        for selection in SELECTIONS
        if selection.inputs(INIT_TIME)["variable"] == inputs["variable"]
        and selection.forecast_type == inputs["forecast_type"]
    )


def _lead_times_of(inputs: dict[str, Any]) -> list[str]:
    return list(_selection_for(inputs).lead_time_labels)


def _level_values_of(inputs: dict[str, Any]) -> list[str]:
    return list(_selection_for(inputs).level_values)


def _cost_of(inputs: dict[str, Any]) -> float:
    return float(_selection_for(inputs).cost)


def _write_blob(inputs: dict[str, Any], target: Path, **_: object) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"GRIB")
    return target


def archive(tmp_path: Path, selections: Sequence[EcdsSelection] = SELECTIONS) -> bool:
    return archive_initialization(
        INIT_TIME, selections, DST_ROOT, work_dir=tmp_path, poll_seconds=0
    )


def test_init_time_names_the_archived_directory() -> None:
    assert format_init_time(INIT_TIME) == "2026-08-10"


def test_every_selection_is_retrieved_validated_then_uploaded(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    assert archive(tmp_path)

    assert archive_bucket["request"].return_value.retrieve.call_count == len(SELECTIONS)
    assert archive_bucket["check_and_index_archived_blob"].call_count == len(SELECTIONS)
    uploaded = {
        call.args[1] for call in archive_bucket["copy_local_file"].call_args_list
    }
    assert uploaded == {
        f"{DST_ROOT.rstrip('/')}/2026-08-10/{selection.file_name}{suffix}"
        for selection in SELECTIONS
        for suffix in ("", INDEX_SUFFIX)
    }
    assert archive_bucket["list_files"].call_args.args == (
        ":s3:bucket/ecmwf-s2s-grib/2026-08-10",
    )


def test_only_the_missing_delta_is_transferred(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    archive_bucket["list_files"].return_value = [
        PurePosixPath(selection.file_name) for selection in SELECTIONS[:-1]
    ]

    archive(tmp_path)

    uploaded = [
        call.args[1] for call in archive_bucket["copy_local_file"].call_args_list
    ]
    assert [Path(path).name for path in uploaded] == [
        SELECTIONS[-1].file_name + INDEX_SUFFIX,
        SELECTIONS[-1].file_name,
    ]


def test_a_fully_archived_initialization_makes_no_ecds_calls(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    archive_bucket["list_files"].return_value = [
        PurePosixPath(selection.file_name) for selection in SELECTIONS
    ]

    assert archive(tmp_path)

    archive_bucket["constraints"].assert_not_called()
    archive_bucket["costing"].assert_not_called()
    archive_bucket["request"].return_value.retrieve.assert_not_called()


def test_an_unpublished_initialization_is_skipped(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    archive_bucket["constraints"].side_effect = lambda inputs, **_: {
        "variable": [],
        "leadtime_hour": [],
        "level_value": [],
    }

    assert not archive(tmp_path)

    archive_bucket["costing"].assert_not_called()
    archive_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_partially_published_initialization_is_not_requested(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    published = archive_bucket["constraints"].side_effect
    archive_bucket["constraints"].side_effect = lambda inputs, **_: (
        {"variable": [], "leadtime_hour": [], "level_value": []}
        if inputs["level_type"] == "pressure"
        else published(inputs)
    )

    with pytest.raises(AssertionError, match="ECDS has no ecmwf variable"):
        archive(tmp_path)

    archive_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_selection_missing_from_a_partly_archived_initialization_fails_loudly(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    """Availability is judged over every selection: one empty response is not an unpublished init."""
    archive_bucket["list_files"].return_value = [
        PurePosixPath(selection.file_name) for selection in SELECTIONS[:-1]
    ]
    published = archive_bucket["constraints"].side_effect
    archive_bucket["constraints"].side_effect = lambda inputs, **_: (
        {"variable": [], "leadtime_hour": [], "level_value": []}
        if inputs["variable"] == list(SELECTIONS[-1].variables)
        else published(inputs)
    )

    with pytest.raises(AssertionError, match="ECDS has no ecmwf variable"):
        archive(tmp_path)

    archive_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_missing_lead_time_is_not_requested(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    archive_bucket["constraints"].side_effect = lambda inputs, **_: {
        "variable": inputs["variable"],
        "leadtime_hour": _lead_times_of(inputs)[:-1],
        "level_value": _level_values_of(inputs),
    }

    with pytest.raises(AssertionError, match="ECDS has no ecmwf leadtime_hour"):
        archive(tmp_path)


def test_a_changed_cost_model_is_not_requested(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    archive_bucket["costing"].side_effect = lambda inputs, **_: (1.0, 1_000_000.0)

    with pytest.raises(AssertionError, match="the request cost model has changed"):
        archive(tmp_path)


def test_an_oversized_request_is_not_submitted(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    archive_bucket["costing"].side_effect = lambda inputs, **_: (
        _cost_of(inputs),
        1.0,
    )

    with pytest.raises(AssertionError, match="above the ECDS limit"):
        archive(tmp_path)


def test_an_incomplete_blob_is_not_uploaded_and_its_work_is_kept(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    archive_bucket["check_and_index_archived_blob"].side_effect = AssertionError(
        "is missing"
    )

    with pytest.raises(AssertionError, match="is missing"):
        archive(tmp_path, SELECTIONS[:1])

    archive_bucket["copy_local_file"].assert_not_called()
    assert (
        tmp_path / "2026-08-10" / SELECTIONS[0].file_name / SELECTIONS[0].file_name
    ).exists()


def test_a_selection_whose_jobs_keep_failing_is_not_retrieved_again(
    tmp_path: Path, archive_bucket: dict[str, MagicMock]
) -> None:
    """`EcdsRequest.retrieve` owns the resubmission budget, so the outer retry must not multiply it."""
    archive_bucket["request"].return_value.retrieve.side_effect = EcdsJobFailedError(
        "ended with status failed"
    )

    with pytest.raises(EcdsJobFailedError, match="ended with status failed"):
        archive(tmp_path, SELECTIONS[:1])

    archive_bucket["request"].return_value.retrieve.assert_called_once()
    archive_bucket["copy_local_file"].assert_not_called()


def test_check_available_queries_constraints_without_the_keys_it_checks() -> None:
    with (
        patch("reformatters.ecmwf.archive_gribs.archive.constraints") as constraints,
        patch("reformatters.ecmwf.archive_gribs.archive.costing") as costing,
    ):
        selection = SELECTIONS[0]
        constraints.return_value = valid_constraints(selection)
        costing.return_value = (float(selection.cost), 1_000_000.0)

        check_available(INIT_TIME, [selection])

    queried = constraints.call_args.args[0]
    assert "leadtime_hour" not in queried
    assert "level_value" not in queried
    assert queried["variable"] == list(selection.variables)
    assert queried["day"] == ["10"]


def test_check_available_reports_an_unpublished_initialization() -> None:
    with (
        patch("reformatters.ecmwf.archive_gribs.archive.constraints") as constraints,
        patch("reformatters.ecmwf.archive_gribs.archive.costing") as costing,
    ):
        constraints.return_value = {"variable": [], "leadtime_hour": []}

        availability = check_available(INIT_TIME, SELECTIONS)

    assert availability.publication == "unpublished"
    costing.assert_not_called()


def test_check_available_reports_a_partial_initialization_with_what_is_missing() -> (
    None
):
    with (
        patch("reformatters.ecmwf.archive_gribs.archive.constraints") as constraints,
        patch("reformatters.ecmwf.archive_gribs.archive.costing") as costing,
    ):
        constraints.side_effect = [
            valid_constraints(SELECTIONS[0]),
            {"variable": [], "leadtime_hour": [], "level_value": []},
        ]

        availability = check_available(INIT_TIME, SELECTIONS[:2])

    assert availability.publication == "partial"
    assert availability.missing[0].startswith("ECDS has no ecmwf variable")
    assert all(SELECTIONS[1].file_name in message for message in availability.missing)
    costing.assert_not_called()


def test_check_available_reports_a_published_initialization() -> None:
    with (
        patch("reformatters.ecmwf.archive_gribs.archive.constraints") as constraints,
        patch("reformatters.ecmwf.archive_gribs.archive.costing") as costing,
    ):
        constraints.side_effect = [valid_constraints(s) for s in SELECTIONS]
        costing.side_effect = [(float(s.cost), 1_000_000.0) for s in SELECTIONS]

        availability = check_available(INIT_TIME, SELECTIONS)

    assert availability.publication == "published"
    assert availability.missing == ()
    assert costing.call_count == len(SELECTIONS)


@pytest.mark.parametrize(
    ("cost", "limit", "match"),
    [
        (1.0, 1_000_000.0, "the request cost model has changed"),
        (None, 1.0, "above the ECDS limit"),
    ],
)
def test_check_available_raises_on_costing_problems_once_everything_is_present(
    cost: float | None, limit: float, match: str
) -> None:
    with (
        patch("reformatters.ecmwf.archive_gribs.archive.constraints") as constraints,
        patch("reformatters.ecmwf.archive_gribs.archive.costing") as costing,
    ):
        selection = SELECTIONS[0]
        constraints.return_value = valid_constraints(selection)
        costing.return_value = (
            float(selection.cost) if cost is None else cost,
            limit,
        )

        with pytest.raises(AssertionError, match=match):
            check_available(INIT_TIME, [selection])


class FakeClock:
    def __init__(self, now: pd.Timestamp) -> None:
        self.now = now
        self.sleeps: list[float] = []
        self.oversleep = pd.Timedelta(0)

    def utc_now(self) -> pd.Timestamp:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += pd.Timedelta(seconds=seconds) + self.oversleep


POLL_INTERVAL = pd.Timedelta(minutes=2)
PROBE_DURATION = pd.Timedelta(seconds=30)
WAIT_START = pd.Timestamp("2026-08-12T03:00", tz="UTC")


@pytest.fixture
def clock() -> Iterator[FakeClock]:
    fake = FakeClock(WAIT_START)
    with (
        patch(
            "reformatters.ecmwf.archive_gribs.archive._utc_now",
            side_effect=fake.utc_now,
        ),
        patch(
            "reformatters.ecmwf.archive_gribs.archive.time.sleep",
            side_effect=fake.sleep,
        ),
    ):
        yield fake


def ecds_publishes(
    archive_bucket: dict[str, MagicMock],
    clock: FakeClock,
    states: Sequence[Publication],
    probe_duration: pd.Timedelta = PROBE_DURATION,
) -> list[pd.Timestamp]:
    """Make successive probes observe `states`, the last repeating, and return probe start times.

    Each probe takes `probe_duration` of fake time.
    """
    published = archive_bucket["constraints"].side_effect
    probe_starts: list[pd.Timestamp] = []
    calls = 0

    def constraints(inputs: dict[str, Any], **_: object) -> dict[str, list[str]]:
        nonlocal calls
        probe, position = divmod(calls, len(SELECTIONS))
        calls += 1
        if position == 0:
            probe_starts.append(clock.now)
        if position == len(SELECTIONS) - 1:
            clock.now += probe_duration
        state = states[min(probe, len(states) - 1)]
        if state == "published" or (state == "partial" and position > 0):
            return published(inputs)
        return {"variable": [], "leadtime_hour": [], "level_value": []}

    archive_bucket["constraints"].side_effect = constraints
    return probe_starts


def archive_with_deadline(tmp_path: Path, deadline: pd.Timestamp | None) -> bool:
    return archive_initialization(
        INIT_TIME,
        SELECTIONS,
        DST_ROOT,
        work_dir=tmp_path,
        poll_seconds=0,
        publication_deadline=deadline,
        publication_poll_interval=POLL_INTERVAL,
    )


def minutes_after_start(probe_starts: Sequence[pd.Timestamp]) -> list[float]:
    return [(start - WAIT_START) / pd.Timedelta(minutes=1) for start in probe_starts]


def assert_retrieved(archive_bucket: dict[str, MagicMock]) -> None:
    assert archive_bucket["request"].return_value.retrieve.call_count == len(SELECTIONS)


def test_waits_for_publication_on_the_interval_grid_then_retrieves(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    probe_starts = ecds_publishes(
        archive_bucket, clock, ["unpublished"] * 3 + ["published"]
    )

    assert archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(hours=1))

    assert minutes_after_start(probe_starts) == [0, 2, 4, 6]
    assert clock.sleeps == [90.0] * 3
    assert_retrieved(archive_bucket)


def test_a_probe_slower_than_the_interval_is_followed_immediately(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    probe_starts = ecds_publishes(
        archive_bucket,
        clock,
        ["unpublished", "unpublished", "published"],
        probe_duration=pd.Timedelta(minutes=3),
    )

    assert archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(hours=1))

    assert minutes_after_start(probe_starts) == [0, 3, 6]
    assert clock.sleeps == [0.0, 0.0]
    assert_retrieved(archive_bucket)


def test_never_published_ends_with_a_probe_exactly_at_the_deadline(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    probe_starts = ecds_publishes(archive_bucket, clock, ["unpublished"])

    assert not archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(minutes=9))

    assert minutes_after_start(probe_starts) == [0, 2, 4, 6, 8, 9]
    archive_bucket["costing"].assert_not_called()
    archive_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_probe_overrunning_the_deadline_is_not_followed_by_another(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    probe_starts = ecds_publishes(
        archive_bucket,
        clock,
        ["unpublished", "unpublished", "published"],
        probe_duration=pd.Timedelta(minutes=3),
    )

    assert not archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(minutes=5))

    assert minutes_after_start(probe_starts) == [0, 3]
    archive_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_sleep_overshooting_the_deadline_is_not_followed_by_a_probe(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    clock.oversleep = pd.Timedelta(seconds=10)
    probe_starts = ecds_publishes(archive_bucket, clock, ["unpublished"])

    assert not archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(minutes=5))

    assert probe_starts == [
        WAIT_START + pd.Timedelta(seconds=seconds) for seconds in (0, 130, 260)
    ]
    assert clock.now > WAIT_START + pd.Timedelta(minutes=5)


def test_a_publication_seen_by_the_deadline_probe_is_retrieved_after_the_deadline(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    deadline = WAIT_START + pd.Timedelta(minutes=5)
    probe_starts = ecds_publishes(
        archive_bucket, clock, ["unpublished"] * 3 + ["published"]
    )

    assert archive_with_deadline(tmp_path, deadline)

    assert minutes_after_start(probe_starts) == [0, 2, 4, 5]
    assert clock.now > deadline
    assert_retrieved(archive_bucket)


@pytest.mark.parametrize(
    "deadline", [None, WAIT_START - pd.Timedelta(minutes=1), WAIT_START]
)
def test_without_time_left_to_wait_ecds_is_probed_once(
    tmp_path: Path,
    archive_bucket: dict[str, MagicMock],
    clock: FakeClock,
    deadline: pd.Timestamp | None,
) -> None:
    probe_starts = ecds_publishes(archive_bucket, clock, ["unpublished", "published"])

    assert not archive_with_deadline(tmp_path, deadline)

    assert probe_starts == [WAIT_START]
    assert clock.sleeps == []


def test_an_initialization_published_before_a_passed_deadline_is_retrieved(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    probe_starts = ecds_publishes(archive_bucket, clock, ["published"])

    assert archive_with_deadline(tmp_path, WAIT_START - pd.Timedelta(hours=1))

    assert probe_starts == [WAIT_START]
    assert_retrieved(archive_bucket)


def test_a_partial_publication_is_waited_on_until_published(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    probe_starts = ecds_publishes(
        archive_bucket, clock, ["partial", "partial", "published"]
    )

    assert archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(hours=1))

    assert minutes_after_start(probe_starts) == [0, 2, 4]
    assert_retrieved(archive_bucket)


def test_a_partial_then_unpublished_initialization_is_skipped(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    ecds_publishes(archive_bucket, clock, ["partial", "unpublished"])

    with patch("reformatters.ecmwf.archive_gribs.archive.log") as log:
        assert not archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(minutes=5))

    warning = log.warning.call_args.args
    assert "partially published" in warning[0] % warning[1:]
    archive_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_publication_still_partial_at_the_deadline_fails_loudly(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    probe_starts = ecds_publishes(archive_bucket, clock, ["partial"])

    with pytest.raises(
        AssertionError,
        match=f"ECDS has no ecmwf variable .* for {SELECTIONS[0].file_name}",
    ):
        archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(minutes=5))

    assert minutes_after_start(probe_starts) == [0, 2, 4, 5]
    archive_bucket["request"].return_value.retrieve.assert_not_called()


def test_costing_is_checked_only_on_the_first_fully_published_probe(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    probe_starts = ecds_publishes(
        archive_bucket, clock, ["partial", "partial", "published"]
    )
    archive_bucket["costing"].side_effect = lambda inputs, **_: (1.0, 1_000_000.0)

    with pytest.raises(AssertionError, match="the request cost model has changed"):
        archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(hours=1))

    assert minutes_after_start(probe_starts) == [0, 2, 4]
    archive_bucket["costing"].assert_called_once()
    archive_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_fully_archived_initialization_does_not_wait(
    tmp_path: Path, archive_bucket: dict[str, MagicMock], clock: FakeClock
) -> None:
    archive_bucket["list_files"].return_value = [
        PurePosixPath(selection.file_name) for selection in SELECTIONS
    ]

    assert archive_with_deadline(tmp_path, WAIT_START + pd.Timedelta(hours=1))

    archive_bucket["constraints"].assert_not_called()
    assert clock.sleeps == []
