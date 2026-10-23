from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest
import typer
import zarr
from zarr.storage import MemoryStore

from reformatters.common import kubernetes, validation
from reformatters.common.config import Config
from reformatters.common.storage import StoreFactory
from reformatters.google.weathernext3.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext3ForecastVirtualDataset,
)
from reformatters.google.weathernext3.forecast_virtual.region_job import (
    GoogleWeathernext3ForecastVirtualRegionJob,
)
from tests.google.weathernext3.forecast_virtual.datasets_test import DATASETS


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
@pytest.mark.parametrize("minutes_after_fire", [0, 15, 30])
def test_update_reserves_time_for_inline_validation(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
    minutes_after_fire: int,
) -> None:
    fire = pd.Timestamp("2026-10-05T12:10:00Z").tz_localize(None)
    now = fire + pd.Timedelta(minutes=minutes_after_fire)
    assert dataset._virtual_poll_deadline(now) == fire + pd.Timedelta(minutes=40)
    (update,) = dataset.operational_kubernetes_resources("test")
    assert update.pod_active_deadline == pd.Timedelta(minutes=55)
    assert update.active_deadline == pd.Timedelta(minutes=59)
    assert dataset.retry_late_sources


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
@pytest.mark.parametrize(
    ("failed_checks", "retries"),
    [
        (["CheckVirtualManifestCompleteness"], True),
        (["CheckCurrentData"], True),
        (["CheckVirtualDecodeHealth"], False),
    ],
)
def test_missing_source_retry_polls_until_next_fire_with_frozen_cycle(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
    failed_checks: list[str],
    retries: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fire = pd.Timestamp("2026-10-20T07:10")
    dataset._retry_timing = kubernetes.RetryTiming(
        fire, fire + pd.Timedelta("1h"), fire + pd.Timedelta(minutes=40)
    )
    now = fire + pd.Timedelta(minutes=46)
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: now))
    monkeypatch.setattr(type(Config), "is_prod", property(lambda self: True))
    monkeypatch.setenv("CRON_JOB_NAME", "wn3-update")
    failure = validation.OperationalValidationError("missing", failed_checks)
    monkeypatch.setattr(type(dataset), "validate_dataset", Mock(side_effect=failure))
    create = Mock(return_value=True)
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", create)

    with pytest.raises(typer.Exit):
        dataset._validate_after_update("wn3-update-123")

    if not retries:
        assert "retry_timing" not in create.call_args.kwargs
        return
    args, kwargs = create.call_args
    assert args == ("wn3-update", "wn3-update-123-r1")
    timing = kwargs["retry_timing"]
    assert timing == kubernetes.RetryTiming(
        fire, fire + pd.Timedelta("1h"), fire + pd.Timedelta(minutes=58)
    )
    assert kwargs["active_deadline"] == pd.Timedelta(minutes=27)


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_retry_after_next_fire_keeps_original_reference_time(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fire = pd.Timestamp("2026-10-20T07:10")
    dataset._retry_timing = kubernetes.RetryTiming(
        fire, fire + pd.Timedelta("1h"), fire + pd.Timedelta(minutes=58)
    )
    now = fire + pd.Timedelta(minutes=63)
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: now))
    monkeypatch.setattr(
        type(dataset.store_factory), "primary_store", lambda self: MemoryStore()
    )
    expected, _ = dataset.region_job_class.operational_update_jobs(
        MemoryStore(),
        Path("unused"),
        dataset.template_config.get_template,
        "init_time",
        dataset.template_config.data_vars,
        "wn3-update-123-r1",
        fire,
    )
    (expected_job,) = expected
    assert isinstance(expected_job, GoogleWeathernext3ForecastVirtualRegionJob)
    jobs, _ = dataset._operational_update_jobs("wn3-update-123-r1", Path("unused"))
    validation_job = dataset._virtual_validation_region_job(
        dataset.validators(), "wn3-update-123-r1"
    )
    assert validation_job is not None
    for job in (*jobs, validation_job):
        assert isinstance(job, GoogleWeathernext3ForecastVirtualRegionJob)
        assert job.reference_time == fire
        assert job.launch_scope == expected_job.launch_scope
        assert job.commit_metadata() == expected_job.commit_metadata()
        assert (
            job.commit_metadata()["reference_time"]
            == fire.tz_localize("UTC").isoformat()
        )
    assert dataset._virtual_poll_deadline(now) == fire + pd.Timedelta(minutes=58)


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
@pytest.mark.parametrize("validation_fails", [False, True])
@pytest.mark.parametrize("validation_minutes_after_fire", [30, 65])
def test_inline_validation_rebuilds_wn3_provenance_without_moving_tip(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
    validation_fails: bool,
    validation_minutes_after_fire: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fire = pd.Timestamp("2026-10-05T12:10:00Z").tz_localize(None)
    clock = [fire + pd.Timedelta(minutes=1)]
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: clock[0])
    )
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    monkeypatch.setattr(
        StoreFactory,
        "primary_store",
        Mock(side_effect=lambda: repo.readonly_session("main").store),
    )
    monkeypatch.setattr(type(dataset), "_tmp_store", Mock(return_value=tmp_path))
    monkeypatch.setattr(type(dataset), "_assert_no_structural_drift", Mock())
    snapshot_ids: list[str] = []
    write_jobs: list[GoogleWeathernext3ForecastVirtualRegionJob] = []

    def publish(
        all_jobs: Sequence[GoogleWeathernext3ForecastVirtualRegionJob],
        worker_index: int,
        workers_total: int,
    ) -> None:
        (job,) = all_jobs
        write_jobs.append(job)
        session = repo.writable_session("main")
        group = zarr.open_group(session.store, mode="w")
        group.attrs["test_publication"] = job.reformat_job_name
        snapshot_ids.append(session.commit("published", metadata=job.commit_metadata()))
        clock[0] = fire + pd.Timedelta(minutes=validation_minutes_after_fire)

    monkeypatch.setattr(
        type(dataset), "_run_virtual_operational_update", Mock(side_effect=publish)
    )
    checks = Mock(
        side_effect=validation.OperationalValidationError("corrupt p90")
        if validation_fails
        else None
    )
    monkeypatch.setattr(validation, "validate_dataset", checks)

    if validation_fails:
        with pytest.raises(typer.Exit) as exc:
            dataset.update("wn3-inline-test")
        assert exc.value.exit_code == kubernetes.VALIDATION_FAILURE_EXIT_CODE
    else:
        dataset.update("wn3-inline-test")

    checks.assert_called_once()
    checked_job = checks.call_args.kwargs["region_job"]
    assert isinstance(checked_job, GoogleWeathernext3ForecastVirtualRegionJob)
    validation_fire = fire + pd.Timedelta(hours=validation_minutes_after_fire // 60)
    assert checked_job.reference_time == validation_fire
    assert checked_job.publication_cutoff == pd.Timestamp.max
    assert checked_job.launch_scope is not None
    if validation_fire == fire:
        assert checked_job.commit_metadata() == write_jobs[0].commit_metadata()
    assert snapshot_ids == [repo.lookup_branch("main")]
    assert (
        repo.lookup_snapshot(snapshot_ids[0]).metadata
        == write_jobs[0].commit_metadata()
    )
