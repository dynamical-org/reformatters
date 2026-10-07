from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest
import typer
import zarr

from reformatters.common import kubernetes, validation
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
    assert dataset._virtual_poll_deadline(now) == fire + pd.Timedelta(minutes=25)
    (update,) = dataset.operational_kubernetes_resources("test")
    assert update.pod_active_deadline == pd.Timedelta(minutes=40)


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
    assert checked_job.publication_cutoff == validation_fire - pd.Timedelta(hours=1)
    assert checked_job.launch_scope is not None
    if validation_fire == fire:
        assert checked_job.commit_metadata() == write_jobs[0].commit_metadata()
    else:
        assert checked_job.commit_metadata() != write_jobs[0].commit_metadata()
    assert snapshot_ids == [repo.lookup_branch("main")]
    assert (
        repo.lookup_snapshot(snapshot_ids[0]).metadata
        == write_jobs[0].commit_metadata()
    )
