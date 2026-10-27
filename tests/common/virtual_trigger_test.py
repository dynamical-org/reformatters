from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest
import typer
from kubernetes.client.exceptions import ApiException

from reformatters.common import kubernetes, template_utils, validation
from reformatters.common.config import Config, Env
from reformatters.common.materialized_update_trigger import MaterializedUpdateTrigger
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.noaa.hrrr.forecast_48_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualDataset,
)
from tests.common.virtual_region_job_test import (
    BLOCK_NBYTES,
    VirtualTestDataset,
    VirtualTestRegionJob,
    VirtualTestSourceFileCoord,
    _create_template_ds,
    _make_dataset,
    _make_region_job,
    _primary_repo,
    _queue_stale_sessions,
    _snapshot_count,
)


@pytest.fixture
def dataset(tmp_path: Path) -> VirtualTestDataset:
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    return dataset


class PollingJob(VirtualTestRegionJob):
    process_virtual_refs = VirtualRegionJob.process_virtual_refs

    def discover_available(
        self, pending: list[VirtualTestSourceFileCoord]
    ) -> list[tuple[VirtualTestSourceFileCoord, int]]:
        return [(coord, BLOCK_NBYTES) for coord in pending]

    def file_refs(
        self, coord: VirtualTestSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        assert file_size == BLOCK_NBYTES
        (batch,) = VirtualTestRegionJob.process_virtual_refs(self, [coord])
        return list(batch[0][1])


def _job() -> PollingJob:
    return PollingJob(
        **_make_region_job(
            _create_template_ds(1), region=slice(0, 1), processing_mode="update"
        ).model_dump()
    )


def test_complete_at_start_has_no_completion_event(dataset: VirtualTestDataset) -> None:
    job = _job()
    assert job.process_virtual_worker_jobs(
        [job], dataset.store_factory, "main", overwrite_chunks=False
    )
    assert not job.process_virtual_worker_jobs(
        [job], dataset.store_factory, "main", overwrite_chunks=False
    )


@pytest.mark.parametrize("unavailable", [False, True])
def test_deadline_has_no_completion_event(
    dataset: VirtualTestDataset, monkeypatch: pytest.MonkeyPatch, unavailable: bool
) -> None:
    now = pd.Timestamp("2026-10-27")
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(
            lambda *args, **kwargs: (
                now.tz_localize(kwargs["tz"]) if kwargs.get("tz") else now
            )
        ),
    )
    job = _job().model_copy(update={"poll_deadline": now})
    if unavailable:
        monkeypatch.setattr(
            PollingJob,
            "discover_available",
            lambda self, pending: (
                [(pending[0], BLOCK_NBYTES)] if len(pending) == 2 else []
            ),
        )
    assert not job.process_virtual_worker_jobs(
        [job], dataset.store_factory, "main", overwrite_chunks=False
    )
    repo = _primary_repo(dataset.store_factory)
    assert len(
        job.filter_already_present(
            job.source_file_coords(), repo.readonly_session("main").store
        )
    ) == int(unavailable)


def test_unreadable_last_file_has_no_completion_event(
    dataset: VirtualTestDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = PollingJob.file_refs
    monkeypatch.setattr(
        PollingJob,
        "file_refs",
        lambda self, coord, size: (
            [] if coord.lead_time == pd.Timedelta("6h") else original(self, coord, size)
        ),
    )
    job = _job()
    repo = _primary_repo(dataset.store_factory)
    before = _snapshot_count(repo)
    assert not job.process_virtual_worker_jobs(
        [job], dataset.store_factory, "main", overwrite_chunks=False
    )
    assert _snapshot_count(repo) == before + 1


@pytest.mark.parametrize("whole_batch", [False, True])
def test_completion_survives_conflict_with_overlapping_writer(
    dataset: VirtualTestDataset,
    monkeypatch: pytest.MonkeyPatch,
    whole_batch: bool,
) -> None:
    repo = _primary_repo(dataset.store_factory)
    job = _job()
    coords = job.source_file_coords()
    before = _snapshot_count(repo)
    stale = repo.writable_session("main")
    original = PollingJob.process_virtual_refs
    winner_completed: list[bool] = []
    opened: list[icechunk.Session] = []

    def overlapping_batch(
        self: PollingJob, remaining: Sequence[VirtualTestSourceFileCoord]
    ) -> Iterator[Sequence[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]]]:
        (batch,) = original(self, remaining)
        winner = _make_region_job(
            _create_template_ds(1), region=slice(0, 1), processing_mode="update"
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                VirtualTestRegionJob,
                "process_virtual_refs",
                lambda self, remaining: iter([batch if whole_batch else batch[:1]]),
            )
            winner_completed.append(winner.process_virtual(repo, [], "main", coords))
        nonlocal opened
        opened = _queue_stale_sessions(monkeypatch, repo, [stale])
        yield batch

    monkeypatch.setattr(PollingJob, "process_virtual_refs", overlapping_batch)
    assert job.process_virtual(repo, [], "main", coords)
    assert winner_completed == [whole_batch]
    assert len(opened) == 2
    assert not job.filter_already_present(coords, repo.readonly_session("main").store)
    assert _snapshot_count(repo) == before + (1 if whole_batch else 2)


@pytest.mark.parametrize("became_complete", [False, True])
@pytest.mark.parametrize("submission_fails", [False, True])
def test_hrrr48_dispatch_precedes_validation_on_the_selected_fire(
    dataset: VirtualTestDataset,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    became_complete: bool,
    submission_fails: bool,
) -> None:
    producer = NoaaHrrrForecast48HourVirtualDataset(
        primary_storage_config=dataset.primary_storage_config
    )
    job = _make_region_job(
        _create_template_ds(1), region=slice(0, 1), processing_mode="update"
    )
    (cron,) = producer.operational_kubernetes_resources("test")
    fire = pd.Timestamp("2024-01-01T00:44")
    successor = cron.next_fire_time(fire)
    now = [successor - pd.Timedelta("1s")]
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now[0])
    )
    monkeypatch.setattr(VirtualTestRegionJob, "refresh_metadata", Mock())
    monkeypatch.setattr(Config, "env", Env.prod)
    monkeypatch.setenv("CRON_JOB_NAME", cron.name)
    monkeypatch.setattr(
        NoaaHrrrForecast48HourVirtualDataset,
        "_monitor",
        lambda *args, **kwargs: nullcontext(),
    )
    planning = Mock(return_value=([job], job.template_ds))
    monkeypatch.setattr(
        NoaaHrrrForecast48HourVirtualDataset, "_operational_update_jobs", planning
    )
    monkeypatch.setattr(
        NoaaHrrrForecast48HourVirtualDataset, "_assert_no_structural_drift", Mock()
    )
    events: list[str] = []

    def process(
        worker_jobs: Sequence[VirtualTestRegionJob], *args: object, **kwargs: object
    ) -> bool:
        assert worker_jobs[0].poll_deadline == producer._virtual_poll_deadline(fire)
        now[0] = successor + pd.Timedelta("30s")
        events.append("polling")
        return became_complete

    monkeypatch.setattr(
        VirtualTestRegionJob, "process_virtual_worker_jobs", Mock(side_effect=process)
    )

    def submit_trigger(*args: object, **kwargs: object) -> None:
        events.append("trigger")
        if submission_fails:
            raise ApiException(status=403)

    submit = Mock(side_effect=submit_trigger)
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", submit)

    def slow_validation(
        self: NoaaHrrrForecast48HourVirtualDataset,
        name: str,
        *,
        job_fire_time: pd.Timestamp | None = None,
    ) -> None:
        assert job_fire_time == fire
        assert now[0] > successor
        assert events == (["polling", "trigger"] if became_complete else ["polling"])
        events.append("validation")
        raise validation.OperationalValidationError("missing data")

    checks = Mock(side_effect=slow_validation.__get__(producer))
    monkeypatch.setattr(
        NoaaHrrrForecast48HourVirtualDataset, "_validate_dataset", checks
    )
    with pytest.raises(typer.Exit) as raised:
        producer.update("test-update")
    assert raised.value.exit_code == kubernetes.VALIDATION_FAILURE_EXIT_CODE
    assert planning.call_args.kwargs["job_fire_time"] == fire
    checks.assert_called_once_with("test-update", job_fire_time=fire)
    assert events == (
        ["polling", "trigger", "validation"]
        if became_complete
        else ["polling", "validation"]
    )
    if became_complete:
        submit.assert_called_once_with(
            "noaa-hrrr-forecast-48-hour-update",
            "noaa-hrrr-forecast-48-hour-t202401010000z",
            kind="trigger",
        )
    else:
        submit.assert_not_called()
    assert ("Could not submit materialized trigger" in caplog.text) == (
        became_complete and submission_fails
    )


@pytest.mark.parametrize(
    ("env", "source"),
    [
        (Env.test, "test-virtual-dataset-update"),
        (Env.prod, "staging-test-virtual-dataset-update"),
    ],
)
def test_only_canonical_production_producer_dispatches(
    dataset: VirtualTestDataset, monkeypatch: pytest.MonkeyPatch, env: Env, source: str
) -> None:
    job = _make_region_job(
        _create_template_ds(1), region=slice(0, 1), processing_mode="update"
    )
    monkeypatch.setattr(VirtualTestRegionJob, "refresh_metadata", Mock())
    monkeypatch.setattr(Config, "env", env)
    monkeypatch.setenv("CRON_JOB_NAME", source)
    monkeypatch.setattr(
        VirtualTestDataset,
        "materialized_update_triggers",
        (MaterializedUpdateTrigger(target_cronjob="consumer-update"),),
    )
    monkeypatch.setattr(
        VirtualTestRegionJob, "process_virtual_worker_jobs", Mock(return_value=True)
    )
    submit = Mock()
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", submit)
    dataset._run_virtual_operational_update([job], 0, 1)
    submit.assert_not_called()
