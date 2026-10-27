import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import typer
import xarray as xr

from reformatters.common import kubernetes, template_utils, validation
from reformatters.common.config import Config, Env
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast18HourVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.dynamical_dataset import (
    NoaaRrfsForecastSubHourlyVirtualDataset,
)
from tests.common.virtual_region_job_test import (
    BLOCK_NBYTES,
    VirtualTestDataset,
    VirtualTestRegionJob,
    _assert_all_values,
    _create_template_ds,
    _make_dataset,
    _make_region_job,
    _NothingPublishedJob,
    _primary_repo,
)

PILOTS = (
    NoaaHrrrForecast18HourVirtualDataset,
    NoaaRrfsForecastSubHourlyVirtualDataset,
)
type PilotDataset = (
    NoaaHrrrForecast18HourVirtualDataset | NoaaRrfsForecastSubHourlyVirtualDataset
)


@pytest.mark.parametrize("dataset_class", PILOTS)
def test_job_deadline_bounds_an_evicted_predecessor_after_the_next_fire(
    dataset_class: type[PilotDataset],
    tmp_path: Path,
) -> None:
    dataset = dataset_class(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    (cron,) = dataset.operational_kubernetes_resources("test")
    spec = cron.as_kubernetes_object()["spec"]
    assert spec["concurrencyPolicy"] == "Allow"
    job = spec["jobTemplate"]["spec"]
    pod = job["template"]["spec"]
    fire = cron.previous_fire_time(pd.Timestamp("2026-10-02T12:59"))
    next_fire = cron.next_fire_time(fire)
    replacement_start = next_fire + timedelta(seconds=1)
    job_end = fire + timedelta(seconds=job["activeDeadlineSeconds"])
    pod_end = replacement_start + timedelta(seconds=pod["activeDeadlineSeconds"])

    assert job["activeDeadlineSeconds"] == pod["activeDeadlineSeconds"]
    assert next_fire < job_end < pod_end
    assert job_end == (
        dataset._virtual_poll_deadline(fire) + dataset.virtual_poll_deadline_grace
    )
    assert job_end < cron.next_fire_time(next_fire)


@pytest.mark.parametrize("dataset_class", [*PILOTS, VirtualTestDataset])
@pytest.mark.parametrize("validation_crashes", [False, True])
def test_virtual_update_validates_once_after_polling_and_retries_only_without_allow(
    dataset_class: type[PilotDataset | VirtualTestDataset],
    validation_crashes: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = (
        _make_dataset(tmp_path)
        if dataset_class is VirtualTestDataset
        else dataset_class(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
            )
        )
    )
    events: list[str] = []
    failure = (
        RuntimeError("store read failed")
        if validation_crashes
        else validation.OperationalValidationError("missing data")
    )
    monkeypatch.setattr(
        dataset_class,
        "_operational_update_jobs",
        Mock(return_value=([], xr.DataTree())),
    )
    monkeypatch.setattr(dataset_class, "_assert_no_structural_drift", Mock())
    monkeypatch.setattr(
        dataset_class,
        "_run_virtual_operational_update",
        Mock(side_effect=lambda *args: events.append("polling ended")),
    )

    def validate(*args: object) -> None:
        events.append("validate")
        raise failure

    checks = Mock(side_effect=validate)
    monkeypatch.setattr(dataset_class, "_validate_dataset", checks)
    monkeypatch.setattr(Config, "env", Env.prod)
    monkeypatch.setenv("CRON_JOB_NAME", f"{dataset.dataset_id}-update")
    submit = Mock()
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", submit)

    with pytest.raises(typer.Exit) as exc_info:
        dataset.update("update-123")

    assert exc_info.value.exit_code == kubernetes.VALIDATION_FAILURE_EXIT_CODE
    assert exc_info.value.__cause__ is failure
    assert events == ["polling ended", "validate"]
    checks.assert_called_once_with("update-123")
    if dataset_class is VirtualTestDataset and not validation_crashes:
        submit.assert_called_once_with(
            f"{dataset.dataset_id}-update",
            "update-123-r1",
            skip_if_next_run_within_deadline=True,
        )
    else:
        submit.assert_not_called()


@pytest.mark.parametrize("dataset_class", PILOTS)
def test_predecessor_ingests_a_file_landing_exactly_at_the_next_fire(
    dataset_class: type[PilotDataset],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = dataset_class(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    (cron,) = pilot.operational_kubernetes_resources("test")
    fire = cron.previous_fire_time(pd.Timestamp("2026-10-02T12:59"))
    next_fire = cron.next_fire_time(fire)
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    job = _NothingPublishedJob(
        **_make_region_job(
            _create_template_ds(1), region=slice(0, 1), processing_mode="update"
        ).model_dump(exclude={"poll_deadline"}),
        poll_deadline=pilot._virtual_poll_deadline(fire),
    )
    coords = job.source_file_coords()
    refs = next(VirtualTestRegionJob.process_virtual_refs(job, coords))
    clock = [next_fire - timedelta(seconds=1)]

    def advance(seconds: float) -> None:
        assert 0 <= seconds <= 1
        clock[0] += timedelta(seconds=1)

    discover = Mock(
        side_effect=lambda pending: (
            [(coord, BLOCK_NBYTES) for coord in pending]
            if clock[0] >= next_fire
            else []
        )
    )
    monkeypatch.setattr(_NothingPublishedJob, "discover_available", discover)
    refs_by_lead = {coord.lead_time: file_refs for coord, file_refs in refs}
    monkeypatch.setattr(
        _NothingPublishedJob,
        "file_refs",
        Mock(side_effect=lambda coord, size: refs_by_lead[coord.lead_time]),
    )
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: clock[0])
    )
    monkeypatch.setattr(time, "sleep", advance)
    job.process_virtual(_primary_repo(dataset.store_factory), [], "main", coords)

    assert discover.call_count == 2
    assert clock[0] == next_fire
    _assert_all_values(dataset, 1)


def test_unreadable_file_is_skipped_until_the_next_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _NothingPublishedJob(
        **_make_region_job(
            _create_template_ds(1), region=slice(0, 1), processing_mode="update"
        ).model_dump()
    )
    broken, healthy = job.source_file_coords()
    refs = next(VirtualTestRegionJob.process_virtual_refs(job, [broken, healthy]))
    discover = Mock(
        side_effect=[[(broken, BLOCK_NBYTES)], [], [(healthy, BLOCK_NBYTES)]]
    )
    build = Mock(side_effect=[ValueError("unreadable index"), refs[1][1]])
    sleep = Mock()
    monkeypatch.setattr(_NothingPublishedJob, "discover_available", discover)
    monkeypatch.setattr(_NothingPublishedJob, "file_refs", build)
    monkeypatch.setattr(time, "sleep", sleep)

    batches = list(job.process_virtual_refs([broken, healthy]))

    assert [[coord for coord, _ in batch] for batch in batches] == [[healthy]]
    assert [call.args[0] for call in discover.call_args_list] == [
        [broken, healthy],
        [healthy],
        [healthy],
    ]
    assert [call.args[0] for call in build.call_args_list] == [broken, healthy]
    assert sleep.call_count == 2

    discover.side_effect = [[(broken, BLOCK_NBYTES)]]
    build.side_effect = [refs[0][1]]
    replacement = job.model_copy()
    assert list(replacement.process_virtual_refs([broken])) == [[refs[0]]]
    assert build.call_args.args[0] == broken
    assert sleep.call_count == 2


def test_complete_window_exits_without_discovery_or_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _NothingPublishedJob(
        **_make_region_job(
            _create_template_ds(1), region=slice(0, 1), processing_mode="update"
        ).model_dump()
    )
    discover = Mock()
    sleep = Mock()
    monkeypatch.setattr(_NothingPublishedJob, "discover_available", discover)
    monkeypatch.setattr(time, "sleep", sleep)

    assert list(job.process_virtual_refs([])) == []

    discover.assert_not_called()
    sleep.assert_not_called()
