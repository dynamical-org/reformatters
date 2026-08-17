from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from reformatters.ecmwf.archive_gribs.grib_inventory import INDEX_SUFFIX
from reformatters.ecmwf.archive_gribs.request_shards import (
    EcdsSelection,
    initialization_selections,
)
from reformatters.ecmwf.archive_gribs.stage_gribs import (
    check_available,
    format_init_time,
    stage_initialization,
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
def staged_bucket() -> Iterator[dict[str, MagicMock]]:
    with (
        patch("reformatters.ecmwf.archive_gribs.stage_gribs.list_files") as list_files,
        patch(
            "reformatters.ecmwf.archive_gribs.stage_gribs.constraints"
        ) as constraints,
        patch("reformatters.ecmwf.archive_gribs.stage_gribs.costing") as costing,
        patch(
            "reformatters.ecmwf.archive_gribs.stage_gribs.copy_local_file"
        ) as copy_local_file,
        patch(
            "reformatters.ecmwf.archive_gribs.stage_gribs.check_and_index_staged_blob"
        ) as check_and_index_staged_blob,
        patch("reformatters.ecmwf.archive_gribs.stage_gribs.EcdsRequest") as request,
    ):
        list_files.return_value = []
        constraints.side_effect = lambda inputs, **_: {
            "variable": inputs["variable"],
            "leadtime_hour": _lead_times_of(inputs),
            "level_value": _level_values_of(inputs),
        }
        costing.side_effect = lambda inputs, **_: (_cost_of(inputs), 1_000_000.0)
        request.return_value.retrieve.side_effect = _write_blob
        check_and_index_staged_blob.side_effect = lambda path, **_: path.with_name(
            path.name + INDEX_SUFFIX
        )
        yield {
            "list_files": list_files,
            "constraints": constraints,
            "costing": costing,
            "copy_local_file": copy_local_file,
            "check_and_index_staged_blob": check_and_index_staged_blob,
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


def stage(tmp_path: Path, selections: Sequence[EcdsSelection] = SELECTIONS) -> None:
    stage_initialization(
        INIT_TIME, selections, DST_ROOT, work_dir=tmp_path, poll_seconds=0
    )


def test_init_time_names_the_staged_directory() -> None:
    assert format_init_time(INIT_TIME) == "2026-08-10"


def test_every_selection_is_retrieved_validated_then_uploaded(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    stage(tmp_path)

    assert staged_bucket["request"].return_value.retrieve.call_count == len(SELECTIONS)
    assert staged_bucket["check_and_index_staged_blob"].call_count == len(SELECTIONS)
    uploaded = {
        call.args[1] for call in staged_bucket["copy_local_file"].call_args_list
    }
    assert uploaded == {
        f"{DST_ROOT.rstrip('/')}/2026-08-10/{selection.file_name}{suffix}"
        for selection in SELECTIONS
        for suffix in ("", INDEX_SUFFIX)
    }
    assert staged_bucket["list_files"].call_args.args == (
        ":s3:bucket/ecmwf-s2s-grib/2026-08-10",
    )


def test_only_the_missing_delta_is_transferred(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    staged_bucket["list_files"].return_value = [
        PurePosixPath(selection.file_name) for selection in SELECTIONS[:-1]
    ]

    stage(tmp_path)

    uploaded = [
        call.args[1] for call in staged_bucket["copy_local_file"].call_args_list
    ]
    assert [Path(path).name for path in uploaded] == [
        SELECTIONS[-1].file_name + INDEX_SUFFIX,
        SELECTIONS[-1].file_name,
    ]


def test_a_fully_staged_initialization_makes_no_ecds_calls(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    staged_bucket["list_files"].return_value = [
        PurePosixPath(selection.file_name) for selection in SELECTIONS
    ]

    stage(tmp_path)

    staged_bucket["constraints"].assert_not_called()
    staged_bucket["costing"].assert_not_called()
    staged_bucket["request"].return_value.retrieve.assert_not_called()


def test_an_unpublished_initialization_is_skipped(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    staged_bucket["constraints"].side_effect = lambda inputs, **_: {
        "variable": [],
        "leadtime_hour": [],
        "level_value": [],
    }

    stage(tmp_path)

    staged_bucket["costing"].assert_not_called()
    staged_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_partially_published_initialization_is_not_requested(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    published = staged_bucket["constraints"].side_effect
    staged_bucket["constraints"].side_effect = lambda inputs, **_: (
        {"variable": [], "leadtime_hour": [], "level_value": []}
        if inputs["level_type"] == "pressure"
        else published(inputs)
    )

    with pytest.raises(AssertionError, match="ECDS has no ecmwf variable"):
        stage(tmp_path)

    staged_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_selection_missing_from_a_partly_staged_initialization_fails_loudly(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    """Availability is judged over every selection: one empty response is not an unpublished init."""
    staged_bucket["list_files"].return_value = [
        PurePosixPath(selection.file_name) for selection in SELECTIONS[:-1]
    ]
    published = staged_bucket["constraints"].side_effect
    staged_bucket["constraints"].side_effect = lambda inputs, **_: (
        {"variable": [], "leadtime_hour": [], "level_value": []}
        if inputs["variable"] == list(SELECTIONS[-1].variables)
        else published(inputs)
    )

    with pytest.raises(AssertionError, match="ECDS has no ecmwf variable"):
        stage(tmp_path)

    staged_bucket["request"].return_value.retrieve.assert_not_called()


def test_a_missing_lead_time_is_not_requested(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    staged_bucket["constraints"].side_effect = lambda inputs, **_: {
        "variable": inputs["variable"],
        "leadtime_hour": _lead_times_of(inputs)[:-1],
        "level_value": _level_values_of(inputs),
    }

    with pytest.raises(AssertionError, match="ECDS has no ecmwf leadtime_hour"):
        stage(tmp_path)


def test_a_changed_cost_model_is_not_requested(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    staged_bucket["costing"].side_effect = lambda inputs, **_: (1.0, 1_000_000.0)

    with pytest.raises(AssertionError, match="the request cost model has changed"):
        stage(tmp_path)


def test_an_oversized_request_is_not_submitted(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    staged_bucket["costing"].side_effect = lambda inputs, **_: (
        _cost_of(inputs),
        1.0,
    )

    with pytest.raises(AssertionError, match="above the ECDS limit"):
        stage(tmp_path)


def test_an_incomplete_blob_is_not_uploaded_and_its_work_is_kept(
    tmp_path: Path, staged_bucket: dict[str, MagicMock]
) -> None:
    staged_bucket["check_and_index_staged_blob"].side_effect = AssertionError(
        "is missing"
    )

    with pytest.raises(AssertionError, match="is missing"):
        stage(tmp_path, SELECTIONS[:1])

    staged_bucket["copy_local_file"].assert_not_called()
    assert (
        tmp_path / "2026-08-10" / SELECTIONS[0].file_name / SELECTIONS[0].file_name
    ).exists()


def test_check_available_queries_constraints_without_the_keys_it_checks() -> None:
    with (
        patch(
            "reformatters.ecmwf.archive_gribs.stage_gribs.constraints"
        ) as constraints,
        patch("reformatters.ecmwf.archive_gribs.stage_gribs.costing") as costing,
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
