import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import typer
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import kubernetes, template_utils, validation
from reformatters.common.config import Config, Env
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.types import DatetimeLike
from reformatters.common.virtual_region_job import VirtualRegionJob
from reformatters.noaa.hrrr.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast18HourVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.dynamical_dataset import (
    NoaaRrfsForecastSubHourlyVirtualDataset,
)
from tests.common.virtual_region_job_test import (
    APPEND_DIM_FREQ,
    APPEND_DIM_START,
    BLOCK_NBYTES,
    N_LAT,
    N_LON,
    VirtualTestDataset,
    VirtualTestRegionJob,
    _assert_all_values,
    _create_template_ds,
    _make_dataset,
    _make_region_job,
    _NothingPublishedJob,
    _primary_repo,
    _process_virtual,
)

PILOTS = (
    NoaaHrrrForecast18HourVirtualDataset,
    NoaaHrrrForecast48HourVirtualDataset,
    NoaaRrfsForecastSubHourlyVirtualDataset,
)
type PilotDataset = (
    NoaaHrrrForecast18HourVirtualDataset
    | NoaaHrrrForecast48HourVirtualDataset
    | NoaaRrfsForecastSubHourlyVirtualDataset
)


ADOPTED_VIRTUAL_IDS = (
    "noaa-gfs-forecast-virtual",
    "noaa-gfs-analysis-virtual",
    "noaa-gefs-analysis-0-25-degree-virtual",
    "noaa-gefs-forecast-10-day-0-25-degree-virtual",
    "noaa-gefs-forecast-16-day-0-5-degree-virtual",
    "noaa-gefs-forecast-35-day-0-5-degree-virtual",
    "noaa-hrrr-analysis-virtual",
    "noaa-hrrr-forecast-18-hour-virtual",
    "noaa-hrrr-forecast-48-hour-virtual",
    "noaa-rtma-analysis-virtual",
    "noaa-rrfs-forecast-84-hour-virtual",
    "noaa-rrfs-forecast-18-hour-virtual",
    "noaa-rrfs-forecast-sub-hourly-virtual",
    "noaa-rrfs-ens-forecast-virtual",
    "noaa-refs-forecast-hourly-virtual",
    "noaa-refs-forecast-3-hourly-virtual",
    "ecmwf-aifs-single-forecast-virtual",
    "google-weathernext2-forecast-operational-virtual",
    "google-weathernext3-forecast-48-hour-0-1-degree-virtual",
    "google-weathernext3-forecast-15-day-0-1-degree-virtual",
    "google-weathernext3-forecast-15-day-0-05-degree-virtual",
    "google-weathernext3-forecast-48-hour-0-05-degree-virtual",
)
LEGACY_VIRTUAL_IDS = (
    "google-weathernext2-forecast-historical-virtual",
    "noaa-rtma-ru-analysis-virtual",
    "noaa-urma-analysis-virtual",
)
DATASETS_BY_ID = {dataset.dataset_id: dataset for dataset in DYNAMICAL_DATASETS}


def test_virtual_lifecycle_cohorts_cover_registered_virtuals() -> None:
    assert not set(ADOPTED_VIRTUAL_IDS) & set(LEGACY_VIRTUAL_IDS)
    assert set(ADOPTED_VIRTUAL_IDS + LEGACY_VIRTUAL_IDS) == {
        dataset.dataset_id
        for dataset in DYNAMICAL_DATASETS
        if issubclass(dataset.region_job_class, VirtualRegionJob)
    }

    for dataset_id in LEGACY_VIRTUAL_IDS:
        cron = DATASETS_BY_ID[dataset_id]._operational_cron_job(
            kubernetes.ReformatCronJob
        )
        assert cron.suspend is (
            dataset_id
            not in {"noaa-rtma-ru-analysis-virtual", "noaa-urma-analysis-virtual"}
        )
        assert cron.concurrency_policy != "Allow"


@pytest.mark.parametrize("dataset_class", PILOTS)
def test_pilot_job_deadline_ends_before_the_second_successor(
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


@pytest.mark.parametrize(
    "dataset",
    [DATASETS_BY_ID[dataset_id] for dataset_id in ADOPTED_VIRTUAL_IDS],
    ids=lambda dataset: dataset.dataset_id,
)
def test_virtual_overlap_reserves_validation_inside_the_job_deadline(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    cron = dataset._operational_cron_job(kubernetes.ReformatCronJob)
    fire = cron.previous_fire_time(pd.Timestamp("2026-10-27T12:59"))
    successor = cron.next_fire_time(fire)
    cutoff = dataset._virtual_poll_deadline(fire)
    spec = cron.as_kubernetes_object()["spec"]
    assert spec["concurrencyPolicy"] == "Allow"
    assert cron.active_deadline is not None
    assert (
        spec["jobTemplate"]["spec"]["activeDeadlineSeconds"]
        == spec["jobTemplate"]["spec"]["template"]["spec"]["activeDeadlineSeconds"]
    )
    assert cron.active_deadline == cron.pod_active_deadline
    assert cutoff < fire + cron.pod_active_deadline
    if dataset.dataset_id == "noaa-gefs-forecast-35-day-0-5-degree-virtual":
        assert cron.schedule == "15 2 * * *"
        assert (
            cron.active_deadline
            == cron.pod_active_deadline
            == timedelta(hours=8, minutes=30)
        )
        assert cutoff == fire.normalize() + timedelta(hours=10, minutes=38)
    else:
        assert successor < cutoff
    assert cutoff + dataset.virtual_poll_deadline_grace == fire + cron.active_deadline
    assert fire + cron.active_deadline < cron.next_fire_time(successor)


@pytest.mark.parametrize(
    ("dataset_class", "retry_expected"),
    [
        *(
            pytest.param(type(DATASETS_BY_ID[dataset_id]), False, id=dataset_id)
            for dataset_id in ADOPTED_VIRTUAL_IDS
        ),
        *(
            pytest.param(type(DATASETS_BY_ID[dataset_id]), True, id=dataset_id)
            for dataset_id in LEGACY_VIRTUAL_IDS
        ),
        pytest.param(VirtualTestDataset, True, id="local-legacy"),
    ],
)
@pytest.mark.parametrize("validation_crashes", [False, True])
def test_virtual_update_validates_once_after_polling_and_retries_only_without_allow(
    dataset_class: type[Any],
    retry_expected: bool,
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
    planning = Mock(return_value=([], xr.DataTree()))
    monkeypatch.setattr(dataset_class, "_operational_update_jobs", planning)
    monkeypatch.setattr(dataset_class, "_assert_no_structural_drift", Mock())
    monkeypatch.setattr(
        dataset_class,
        "_run_virtual_operational_update",
        Mock(side_effect=lambda *args, **kwargs: events.append("polling ended")),
    )

    def validate(*args: object, **kwargs: object) -> None:
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
    checks.assert_called_once_with(
        "update-123",
        job_fire_time=planning.call_args.kwargs["job_fire_time"],
    )
    if retry_expected and not validation_crashes:
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


def test_allow_poll_deadline_keeps_the_selected_fire_when_start_straddles_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    (cron,) = dataset.operational_kubernetes_resources("test")
    cron = cron.model_copy(
        update={
            "schedule": "0 0,6,12,18 * * *",
            "concurrency_policy": "Allow",
            "pod_active_deadline": timedelta(hours=6, minutes=3),
        }
    )
    monkeypatch.setattr(
        VirtualTestDataset,
        "operational_kubernetes_resources",
        Mock(return_value=[cron]),
    )
    fire = APPEND_DIM_START + APPEND_DIM_FREQ
    successor = cron.next_fire_time(fire)
    now = [successor - timedelta(seconds=1)]
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now[0])
    )

    def tmp_store(self: VirtualTestDataset) -> Path:
        now[0] = successor + timedelta(seconds=1)
        return tmp_path / "tmp.zarr"

    monkeypatch.setattr(VirtualTestDataset, "_tmp_store", tmp_store)
    template = Mock(return_value=_create_template_ds(1))
    monkeypatch.setattr(VirtualTestDataset, "_get_template", template)
    monkeypatch.setattr(VirtualTestDataset, "_assert_no_structural_drift", Mock())
    monkeypatch.setattr(VirtualTestRegionJob, "refresh_metadata", Mock())
    process = Mock()
    monkeypatch.setattr(VirtualTestRegionJob, "process_virtual_worker_jobs", process)
    monkeypatch.setattr(VirtualTestDataset, "_validate_after_update", Mock())

    dataset.update("straddling-update")

    template.assert_called_once_with(fire)
    (job,) = process.call_args.args[0]
    assert job.poll_deadline == dataset._virtual_poll_deadline(fire)
    assert job.poll_deadline < dataset._virtual_poll_deadline(now[0])


@pytest.mark.parametrize("currency_passes", [False, True])
def test_allow_validation_keeps_the_selected_window_after_the_successor_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, currency_passes: bool
) -> None:
    dataset = _make_dataset(tmp_path)
    (cron,) = dataset.operational_kubernetes_resources("test")
    cron = cron.model_copy(
        update={
            "schedule": "0 0,6,12,18 * * *",
            "concurrency_policy": "Allow",
            "pod_active_deadline": timedelta(hours=6, minutes=3),
        }
    )
    monkeypatch.setattr(
        VirtualTestDataset,
        "operational_kubernetes_resources",
        Mock(return_value=[cron]),
    )
    fire = APPEND_DIM_START + APPEND_DIM_FREQ
    successor = cron.next_fire_time(fire)
    now = [successor - timedelta(seconds=1)]
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now[0])
    )
    selected_ends: list[pd.Timestamp] = []

    def get_template(self: object, end: DatetimeLike) -> xr.DataTree:
        selected_ends.append(pd.Timestamp(end))
        return _create_template_ds(
            len(
                pd.date_range(
                    APPEND_DIM_START, end, freq=APPEND_DIM_FREQ, inclusive="left"
                )
            )
        )

    monkeypatch.setattr(VirtualTestDataset, "_get_template", get_template)

    def tmp_store(self: VirtualTestDataset) -> Path:
        now[0] = successor + timedelta(minutes=2)
        return tmp_path / "tmp.zarr"

    monkeypatch.setattr(VirtualTestDataset, "_tmp_store", tmp_store)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    with (tmp_path / "messages.bin").open("r+b") as messages:
        messages.seek(2 * BLOCK_NBYTES)
        messages.write(np.full((N_LAT, N_LON), np.nan, dtype="<f8").tobytes())

    def poll(
        self: VirtualTestDataset,
        jobs: Sequence[VirtualTestRegionJob],
        worker_index: int,
        workers_total: int,
        *,
        job_fire_time: pd.Timestamp | None = None,
    ) -> None:
        _process_virtual(jobs[0], repo)
        now[0] = successor + timedelta(minutes=2)
        successor_job = _make_region_job(_create_template_ds(2), region=slice(1, 2))
        successor_job.process_virtual(
            repo, [], "main", successor_job.source_file_coords()[:1]
        )

    monkeypatch.setattr(VirtualTestDataset, "_run_virtual_operational_update", poll)
    manifest = validation.CheckVirtualManifestCompleteness()
    decode = validation.CheckVirtualDecodeHealth()
    currency = validation.CheckCurrentData(
        max_delay=timedelta(hours=7) if currency_passes else timedelta(minutes=1)
    )
    checks = (manifest, decode, currency)
    monkeypatch.setattr(VirtualTestDataset, "validators", Mock(return_value=checks))
    validate = Mock(wraps=validation.validate_dataset)
    monkeypatch.setattr(validation, "validate_dataset", validate)

    if currency_passes:
        dataset.update("overlapping-update")
    else:
        with pytest.raises(typer.Exit) as raised:
            dataset.update("overlapping-update")
        assert "CheckCurrentData" in str(raised.value.__cause__)
    assert selected_ends == [fire, fire]
    store = repo.readonly_session("main").store
    with validation.open_flattened_dataset(store, consolidated=False) as ds:
        context = validation.ValidationContext(
            store=store,
            ds=ds,
            append_dim="init_time",
            append_dim_frequency=APPEND_DIM_FREQ,
            data_vars=dataset.template_config.data_vars,
            region_job=validate.call_args.kwargs["region_job"],
        )
        assert all(check.check(context).passed for check in (manifest, decode))
        assert currency.check(context).passed == currency_passes
        standalone = dataset._virtual_validation_region_job(checks, "standalone")
        assert selected_ends[-1] == successor
        successor_context = replace(context, region_job=standalone)
        assert not manifest.check(successor_context).passed
        assert not decode.check(successor_context).passed
