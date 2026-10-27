from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import pandas as pd
import pytest
import typer
import xarray as xr
from kubernetes import client
from zarr.storage import MemoryStore

from reformatters.common import dynamical_dataset, kubernetes, operational, validation
from reformatters.common import parallel_coordination as pc
from reformatters.common.config import Config, Env
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
)
from tests.common.dynamical_dataset_test import _recording_monitor
from tests.common.parallel_coordination_test import FakeRepo

_CRON = "noaa-hrrr-forecast-48-hour-update"


def _job(
    name: str,
    created: str = "2026-10-26T07:53Z",
    *,
    kind: Literal["scheduled", "trigger", "manual"] = "scheduled",
    terminal: str | None = None,
) -> client.V1Job:
    annotations = {}
    if kind != "scheduled":
        annotations["cronjob.kubernetes.io/instantiate"] = "manual"
    if kind == "trigger":
        annotations["dynamical.org/run-kind"] = "trigger"
    return client.V1Job(
        metadata=client.V1ObjectMeta(
            name=name,
            uid=f"uid-{name}",
            creation_timestamp=pd.Timestamp(created).to_pydatetime(),
            annotations=annotations,
            labels={"dynamical.org/cronjob-name": _CRON},
        ),
        status=client.V1JobStatus(
            conditions=[client.V1JobCondition(type=terminal, status="True")]
            if terminal
            else [],
        ),
    )


@dataclass
class Pilot:
    dataset: NoaaHrrrForecast48HourDataset
    jobs: list[client.V1Job]
    api: Mock
    compute: Mock
    process: Mock
    validate: Mock
    sleep: Mock

    def state(self, job: client.V1Job) -> pc.JobState | None:
        return pc.read_job_state(
            self.dataset.store_factory, job.metadata.name, job.metadata.uid
        )

    def write(self, job: client.V1Job, state: pc.JobState) -> None:
        pc.write_job_state(
            self.dataset.store_factory, job.metadata.name, job.metadata.uid, state
        )

    def prepare(self, job: client.V1Job, *, is_first: bool = True) -> bool:
        return self.dataset._prepare_materialized_update(
            job,
            self.dataset._operational_cron_job(kubernetes.ReformatCronJob),
            is_first=is_first,
        )


@pytest.fixture
def pilot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Pilot:
    dataset = NoaaHrrrForecast48HourDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    jobs: list[client.V1Job] = []
    api = Mock()
    api.list_namespaced_job.side_effect = lambda *args, **kwargs: client.V1JobList(
        items=list(jobs)
    )
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", Mock())
    monkeypatch.setattr(kubernetes.client, "BatchV1Api", Mock(return_value=api))
    monkeypatch.setenv("CRON_JOB_NAME", _CRON)
    monkeypatch.setenv("POD_NAMESPACE", "pilot")
    monkeypatch.setattr(Config, "env", Env.prod)
    compute = Mock(return_value=([], xr.DataTree()))
    process = Mock()
    validate = Mock()
    sleep = Mock(side_effect=AssertionError("unexpected wait"))
    monkeypatch.setattr(
        NoaaHrrrForecast48HourDataset, "_tmp_store", lambda self: tmp_path
    )
    monkeypatch.setattr(
        NoaaHrrrForecast48HourDataset, "_operational_update_jobs", compute
    )
    monkeypatch.setattr(NoaaHrrrForecast48HourDataset, "_process_region_jobs", process)
    monkeypatch.setattr(NoaaHrrrForecast48HourDataset, "_validate_dataset", validate)
    monkeypatch.setattr(pc.time, "sleep", sleep)
    monkeypatch.setattr(StoreFactory, "primary_store", Mock(return_value=MemoryStore()))
    monkeypatch.setattr(StoreFactory, "replica_stores", Mock(return_value=[]))
    return Pilot(dataset, jobs, api, compute, process, validate, sleep)


def test_job_listing_uses_cronjob_label_and_stable_order(pilot: Pilot) -> None:
    older = _job("a")
    newer = _job("b")
    latest = _job("z", "2026-10-26T07:54Z")
    pilot.jobs[:] = [latest, newer, older]

    assert kubernetes.list_cronjob_jobs(_CRON) == [older, newer, latest]
    pilot.api.list_namespaced_job.assert_called_once_with(
        "pilot", label_selector=f"dynamical.org/cronjob-name={_CRON}"
    )
    template = pilot.dataset._operational_cron_job(kubernetes.ReformatCronJob)
    assert template.as_kubernetes_object()["spec"]["jobTemplate"]["metadata"][
        "labels"
    ] == {"dynamical.org/cronjob-name": _CRON}


@pytest.mark.parametrize("kind", ["scheduled", "trigger", "manual"])
def test_job_kinds_use_annotations_including_cronjob_owned_operator_jobs(
    kind: Literal["scheduled", "trigger", "manual"],
) -> None:
    job = _job("operator", kind=kind)
    job.metadata.owner_references = [
        client.V1OwnerReference(
            api_version="batch/v1", kind="CronJob", name=_CRON, uid="cron-uid"
        )
    ]
    assert kubernetes.job_kind(job) == kind


@pytest.mark.parametrize("state", [None, "run", "skip", "finalized"])
@pytest.mark.parametrize("terminal", [None, "Complete", "Failed"])
def test_writing_requires_nonterminal_job_and_absent_or_run_state(
    pilot: Pilot, state: pc.JobState | None, terminal: str | None
) -> None:
    job = _job("job", terminal=terminal)
    if state is not None:
        pilot.write(job, state)
    assert pc.job_is_writing(pilot.dataset.store_factory, job) == (
        terminal is None and state in {None, "run"}
    )


def test_failed_index_with_running_sibling_is_still_writing(pilot: Pilot) -> None:
    job = _job("job")
    newer = _job("newer", "2026-10-26T07:54Z")
    pilot.jobs[:] = [job, newer]
    job.status.failed = 1
    job.status.active = 1
    job.status.failed_indexes = "0"
    pilot.write(job, "run")
    assert pc.job_is_writing(pilot.dataset.store_factory, job)

    def deadline(seconds: int) -> None:
        assert seconds == 60
        assert pilot.state(newer) == "run"
        job.status.conditions = [client.V1JobCondition(type="Failed", status="True")]

    pilot.sleep.side_effect = deadline
    assert pilot.prepare(newer)
    pilot.sleep.assert_called_once_with(60)


@pytest.mark.parametrize("state", ["run", "skip", "finalized"])
def test_stale_uid_is_absent(pilot: Pilot, state: pc.JobState) -> None:
    old = _job("job")
    pilot.write(old, state)
    new = _job("job")
    new.metadata.uid = "replacement-job-uid"
    assert pilot.state(new) is None
    assert pc.job_is_writing(pilot.dataset.store_factory, new)
    pilot.jobs[:] = [new]
    assert pilot.prepare(new)
    assert pilot.state(new) == "run"


@pytest.mark.parametrize(
    "kinds",
    [("scheduled", "scheduled"), ("manual", "scheduled"), ("manual", "trigger")],
)
def test_same_second_jobs_run_in_name_order_and_newer_records_run_before_queue(
    pilot: Pilot,
    kinds: tuple[
        Literal["scheduled", "trigger", "manual"],
        Literal["scheduled", "trigger", "manual"],
    ],
) -> None:
    older, newer = _job("a", kind=kinds[0]), _job("b", kind=kinds[1])
    pilot.jobs[:] = [newer, older]
    assert pilot.prepare(older)
    assert pilot.state(older) == "run"

    def finish_older(seconds: int) -> None:
        assert seconds == 60
        assert pilot.state(newer) == "run"
        assert pilot.state(older) == "run"
        pilot.write(older, "finalized")

    pilot.sleep.side_effect = finish_older
    assert pilot.prepare(newer)
    pilot.sleep.assert_called_once_with(60)


def test_every_index_computes_after_predecessor_finalizes_and_reuses_decision(
    pilot: Pilot,
) -> None:
    older, current = _job("a"), _job("b")
    pilot.jobs[:] = [older, current]
    computed: list[str] = []
    events: list[tuple[str, str]] = []
    operational.register_run_monitor(_recording_monitor(events))
    published = "old"

    def finish_older(seconds: int) -> None:
        nonlocal published
        assert seconds == 60
        assert events[-1] == ("enter", _CRON)
        pilot.compute.assert_not_called()
        published = "new"
        pilot.write(older, "finalized")

    def compute(*args: object, **kwargs: object) -> tuple[list[object], xr.DataTree]:
        assert pilot.state(older) == "finalized"
        computed.append(published)
        return [], xr.DataTree()

    pilot.sleep.side_effect = finish_older
    pilot.compute.side_effect = compute
    pilot.dataset.update("b", worker_index=0, workers_total=3)
    # Replacement pod 0 and a highest index starting late must reuse the decision.
    pilot.dataset.update("b", worker_index=0, workers_total=3)
    pilot.dataset.update("b", worker_index=2, workers_total=3)

    assert computed == ["new", "new", "new"]
    pilot.sleep.assert_called_once_with(60)
    pilot.validate.assert_called_once()
    assert all(
        call.kwargs["job_uid"] == current.metadata.uid
        for call in pilot.process.call_args_list
    )


def test_late_index_waits_for_pod_zero_state(pilot: Pilot) -> None:
    job = _job("job")
    pilot.jobs[:] = [job]

    def pod_zero_decides(seconds: int) -> None:
        assert seconds == 5
        pilot.compute.assert_not_called()
        pilot.write(job, "run")

    pilot.sleep.side_effect = pod_zero_decides
    pilot.dataset.update("job", worker_index=2, workers_total=3)
    pilot.compute.assert_called_once()
    pilot.validate.assert_called_once()


@pytest.mark.parametrize("worker_index", [0, 2])
@pytest.mark.parametrize("state", ["skip", "finalized"])
def test_replacement_in_nonwriting_job_exits_without_writing_or_validating(
    pilot: Pilot, worker_index: int, state: pc.JobState
) -> None:
    job = _job("job")
    pilot.jobs[:] = [job]
    pilot.write(job, state)
    pilot.dataset.update("job", worker_index=worker_index, workers_total=3)
    pilot.compute.assert_not_called()
    pilot.process.assert_not_called()
    pilot.validate.assert_not_called()
    pilot.sleep.assert_not_called()


@pytest.mark.parametrize("manual_name", ["operator", "job-r1"])
def test_manual_jobs_never_take_backstop_check(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch, manual_name: str
) -> None:
    trigger = _job("trigger", "2026-10-26T07:00Z", kind="trigger", terminal="Complete")
    manual = _job(manual_name, kind="manual")
    manual.metadata.owner_references = [
        client.V1OwnerReference(
            api_version="batch/v1", kind="CronJob", name=_CRON, uid="cron-uid"
        )
    ]
    pilot.jobs[:] = [trigger, manual]
    currency = Mock(side_effect=AssertionError("manual Job took backstop check"))
    monkeypatch.setattr(
        NoaaHrrrForecast48HourDataset, "_current_data_is_published", currency
    )
    assert pilot.prepare(manual)
    assert pilot.state(manual) == "run"
    currency.assert_not_called()


def _currency_dataset(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch, latest: str
) -> None:
    monkeypatch.setattr(
        validation,
        "open_flattened_dataset",
        Mock(
            return_value=xr.Dataset(
                coords={"init_time": pd.date_range("2026-10-25", latest, freq="6h")}
            )
        ),
    )
    monkeypatch.setattr(
        pd.Timestamp, "now", lambda **kwargs: pd.Timestamp("2026-10-26T08:03")
    )


@pytest.mark.parametrize(
    ("latest", "expected"), [("2026-10-26T06:00", "skip"), ("2026-10-26T00:00", "run")]
)
def test_successful_trigger_skips_backstop_and_early_empty_trigger_runs_backstop(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch, latest: str, expected: pc.JobState
) -> None:
    trigger = _job("trigger", "2026-10-26T07:00Z", kind="trigger")
    backstop = _job("backstop")
    pilot.jobs[:] = [trigger, backstop]
    pilot.write(trigger, "finalized")
    events: list[tuple[str, str]] = []
    operational.register_run_monitor(_recording_monitor(events))
    _currency_dataset(pilot, monkeypatch, latest)
    pilot.dataset.update("backstop", worker_index=0, workers_total=1)
    assert pilot.state(backstop) == expected
    if expected == "skip":
        pilot.compute.assert_not_called()
        pilot.validate.assert_not_called()
        assert events == [("enter", _CRON), ("ok", _CRON)]
    else:
        pilot.compute.assert_called_once()
        pilot.validate.assert_called_once()
    pilot.sleep.assert_not_called()


@pytest.mark.parametrize("older_state", ["finalized", "run"])
def test_backstop_does_not_record_run_before_currency_check_and_waits_only_for_writers(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch, older_state: pc.JobState
) -> None:
    trigger = _job("trigger", "2026-10-26T07:00Z", kind="trigger", terminal="Complete")
    repass = _job("trigger-r1", "2026-10-26T07:30Z", kind="manual")
    backstop = _job("backstop")
    pilot.jobs[:] = [trigger, repass, backstop]
    pilot.write(repass, older_state)

    def currency() -> bool:
        assert pilot.state(backstop) is None
        assert pilot.state(repass) == "finalized"
        return True

    def finish(seconds: int) -> None:
        assert seconds == 60
        assert pilot.state(backstop) is None
        pilot.write(repass, "finalized")

    monkeypatch.setattr(
        NoaaHrrrForecast48HourDataset,
        "_current_data_is_published",
        lambda self: currency(),
    )
    pilot.sleep.side_effect = finish
    assert not pilot.prepare(backstop)
    assert pilot.state(backstop) == "skip"
    assert pilot.sleep.call_count == (1 if older_state == "run" else 0)


@pytest.mark.parametrize(
    ("trigger_time", "skips"),
    [
        ("2026-10-26T04:53Z", True),
        ("2026-10-26T04:52Z", False),
        ("2026-10-26T07:54Z", False),
    ],
)
def test_replacement_pod_zero_backstop_window_uses_job_creation_not_pod_start(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch, trigger_time: str, skips: bool
) -> None:
    trigger = _job("trigger", trigger_time, kind="trigger", terminal="Complete")
    backstop = _job("backstop")
    pilot.jobs[:] = [trigger, backstop]
    monkeypatch.setattr(
        pd.Timestamp, "now", lambda **kwargs: pd.Timestamp("2026-10-26T12:00")
    )
    currency = Mock(return_value=True)
    monkeypatch.setattr(
        NoaaHrrrForecast48HourDataset, "_current_data_is_published", currency
    )
    assert pilot.prepare(backstop) == (not skips)
    assert pilot.state(backstop) == ("skip" if skips else "run")
    assert currency.call_count == int(skips)


@pytest.mark.parametrize("retry", [1, 2, 3, 4, 16])
def test_repass_backoff_steps_and_final_check(pilot: Pilot, retry: int) -> None:
    current = _job("current")
    pilot.jobs[:] = [current]
    pilot.write(current, "finalized")
    pilot.sleep.side_effect = None
    assert pc.backoff_before_repass(pilot.dataset.store_factory, _CRON, current, retry)
    assert pilot.sleep.call_count == min(5 * 2 ** (retry - 1), 30)
    assert pilot.api.list_namespaced_job.call_count == pilot.sleep.call_count
    assert all(call.args == (60,) for call in pilot.sleep.call_args_list)


@pytest.mark.parametrize("newer_terminal", [None, "Failed", "Complete"])
def test_backoff_yields_only_after_newer_nonterminal_job_records_matching_run(
    pilot: Pilot, newer_terminal: str | None
) -> None:
    current = _job("a")
    newer = _job("b", kind="trigger", terminal=newer_terminal)
    pilot.jobs[:] = [current]
    pilot.write(current, "finalized")
    ticks = 0

    def advance(seconds: int) -> None:
        nonlocal ticks
        assert seconds == 60
        ticks += 1
        if ticks == 1:
            pilot.jobs.append(newer)
            pc.write_job_state(pilot.dataset.store_factory, "b", "stale-uid", "run")
        if ticks == 2:
            pilot.write(newer, "finalized")
        if ticks == 3:
            pilot.write(newer, "run")

    pilot.sleep.side_effect = advance
    assert pc.backoff_before_repass(pilot.dataset.store_factory, _CRON, current, 1) == (
        newer_terminal is not None
    )
    assert ticks == (5 if newer_terminal else 3)


def test_queued_trigger_releases_backing_off_job_within_sixty_seconds(
    pilot: Pilot,
) -> None:
    current = _job("old")
    trigger = _job("trigger", "2026-10-26T07:54Z", kind="trigger")
    pilot.jobs[:] = [current, trigger]
    pilot.write(current, "finalized")

    def start_trigger(seconds: int) -> None:
        assert seconds == 60
        assert pilot.prepare(trigger)

    pilot.sleep.side_effect = start_trigger
    assert not pc.backoff_before_repass(pilot.dataset.store_factory, _CRON, current, 1)
    pilot.sleep.assert_called_once_with(60)


def test_healthy_untriggered_update_validates_without_repass(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job("job")
    pilot.jobs[:] = [job]
    _currency_dataset(pilot, monkeypatch, "2026-10-26T06:00")

    def validate(*args: object) -> None:
        assert pilot.dataset._current_data_is_published()

    pilot.validate.side_effect = validate
    pilot.dataset.update("job")
    pilot.validate.assert_called_once()
    pilot.api.create_namespaced_job.assert_not_called()
    pilot.sleep.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        validation.OperationalValidationError("missing late file"),
        RuntimeError("store read failed"),
    ],
)
def test_validation_pages_then_backs_off_outside_update_checkin_and_repasses(
    pilot: Pilot, failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job("job")
    pilot.jobs[:] = [job]
    events: list[tuple[str, str]] = []
    operational.register_run_monitor(_recording_monitor(events))
    pilot.validate.side_effect = failure
    pilot.process.side_effect = lambda **kwargs: pilot.write(job, "finalized")
    submit = Mock()
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", submit)

    def backoff(seconds: int) -> None:
        assert seconds == 60
        assert pilot.state(job) == "finalized"
        assert events == [
            ("enter", _CRON),
            ("ok", _CRON),
            ("enter", _CRON.removesuffix("-update") + "-validate"),
            ("error", _CRON.removesuffix("-update") + "-validate"),
        ]

    pilot.sleep.side_effect = backoff
    with pytest.raises(typer.Exit) as exc:
        pilot.dataset.update("job")
    assert exc.value.exit_code == 20
    assert exc.value.__cause__ is failure
    submit.assert_called_once_with(
        _CRON, "job-r1", skip_if_next_run_within_deadline=False
    )
    assert pilot.sleep.call_count == 5


def test_late_file_repass_succeeds_and_chain_stops_at_r16(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _job("job")
    pilot.jobs[:] = [first]
    pilot.validate.side_effect = validation.OperationalValidationError("late file")
    pilot.sleep.side_effect = None
    submit = Mock()
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", submit)
    with pytest.raises(typer.Exit):
        pilot.dataset.update("job")
    submit.assert_called_once_with(
        _CRON, "job-r1", skip_if_next_run_within_deadline=False
    )

    first.status.conditions = [client.V1JobCondition(type="Failed", status="True")]
    repass = _job("job-r1", "2026-10-26T07:59Z", kind="manual")
    pilot.jobs.append(repass)
    pilot.validate.side_effect = None
    pilot.dataset.update("job-r1")
    assert pilot.validate.call_count == 2
    assert submit.call_count == 1

    repass.status.conditions = [client.V1JobCondition(type="Complete", status="True")]
    last = _job("job-r16", "2026-10-26T08:30Z", kind="manual")
    pilot.jobs.append(last)
    pilot.validate.side_effect = validation.OperationalValidationError("still missing")
    pilot.sleep.reset_mock()
    with pytest.raises(typer.Exit) as exc:
        pilot.dataset.update("job-r16")
    assert exc.value.exit_code == 20
    pilot.sleep.assert_not_called()
    assert submit.call_count == 1


@pytest.mark.parametrize("phase", ["after-cas", "validation", "backoff"])
def test_eviction_after_finalize_releases_queue_and_replacement_exits(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str
) -> None:
    job = _job("job")
    newer = _job("newer", "2026-10-26T07:54Z")
    pilot.jobs[:] = [job, newer]
    repo = FakeRepo()
    repo.create_branch("_job_job", "snap-initial")
    reset = repo.reset_branch

    def reset_branch(name: str, snapshot: str, from_snapshot_id: str) -> None:
        assert pilot.state(job) == "run"
        reset(name, snapshot, from_snapshot_id)

    monkeypatch.setattr(repo, "reset_branch", reset_branch)
    monkeypatch.setattr(
        StoreFactory, "icechunk_repos", Mock(return_value=[("primary", repo)])
    )
    monkeypatch.setattr(pc.template_utils, "write_metadata", Mock())
    monkeypatch.setattr(pc, "copy_zarr_metadata", Mock())

    def evict(*args: object, **kwargs: object) -> None:
        assert pilot.state(job) == "finalized"
        raise KeyboardInterrupt("pod evicted")

    def finalize(**kwargs: object) -> None:
        assert kwargs["job_uid"] == job.metadata.uid
        pc.finalize(
            pilot.dataset.store_factory,
            all_jobs=[],
            merged_results={},
            reformat_job_name="job",
            branch_name="_job_job",
            template_ds=xr.DataTree(),
            tmp_store=tmp_path,
            setup_info={"repo_snapshots": {"primary": "snap-initial"}},
            workers_total=3,
            update_template_with_results=False,
            consolidated=True,
            job_uid=job.metadata.uid,
        )

    pilot.process.side_effect = finalize
    if phase == "after-cas":
        monkeypatch.setattr(repo, "delete_branch", evict)
    elif phase == "validation":
        pilot.validate.side_effect = evict
    else:
        pilot.validate.side_effect = validation.OperationalValidationError("late file")
        pilot.sleep.side_effect = evict
    pilot.write(job, "run")
    pilot.dataset.store_factory.write_coordination_file(
        "job", "setup/ready.json", b"ready"
    )
    pilot.dataset.store_factory.write_coordination_file(
        "job", "results/worker-0.json", b"data"
    )
    with pytest.raises(KeyboardInterrupt, match="pod evicted"):
        pilot.dataset.update("job", worker_index=2, workers_total=3)

    assert len(repo.reset_calls) == 1
    assert pilot.state(job) == "finalized"
    assert not kubernetes.job_is_terminal(job)
    assert pilot.prepare(newer)
    pilot.compute.reset_mock()
    pilot.process.reset_mock()
    pilot.validate.reset_mock()
    pilot.dataset.update("job", worker_index=2, workers_total=3)
    pilot.compute.assert_not_called()
    pilot.process.assert_not_called()
    pilot.validate.assert_not_called()
    if phase != "after-cas":
        assert (
            pilot.dataset.store_factory.read_all_coordination_files("job", "setup")
            == []
        )
        assert (
            pilot.dataset.store_factory.read_all_coordination_files("job", "results")
            == []
        )


def test_highest_index_with_zero_work_finalizes_and_validates(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job("job")
    pilot.jobs[:] = [job]
    pilot.write(job, "run")
    region_job = Mock()
    tree = xr.DataTree()
    region_job.update_template_with_results.return_value = tree
    pilot.compute.return_value = ([region_job], tree)
    monkeypatch.setattr(
        NoaaHrrrForecast48HourDataset,
        "_process_region_jobs",
        dynamical_dataset.DynamicalDataset._process_region_jobs,
    )
    monkeypatch.setattr(StoreFactory, "icechunk_repos", Mock(return_value=[]))
    monkeypatch.setattr(pc, "parallel_setup", Mock(return_value={}))
    monkeypatch.setattr(pc, "collect_results", Mock(return_value={}))
    monkeypatch.setattr(pc.template_utils, "write_metadata", Mock())
    monkeypatch.setattr(pc.template_utils, "assert_no_append_dim_retraction", Mock())
    monkeypatch.setattr(StoreFactory, "open_primary_datatree", Mock(return_value=tree))
    monkeypatch.setattr(pc, "copy_zarr_metadata", Mock())
    process_worker = Mock(side_effect=AssertionError("highest index has no work"))
    monkeypatch.setattr(
        pilot.dataset.region_job_class, "process_worker_jobs", process_worker
    )

    pilot.dataset.update("job", worker_index=2, workers_total=3)

    process_worker.assert_not_called()
    region_job.update_template_with_results.assert_called_once_with({})
    assert pilot.state(job) == "finalized"
    pilot.validate.assert_called_once()
    assert (
        pilot.dataset.store_factory.read_all_coordination_files("job", "results") == []
    )


def test_next_fire_queues_behind_repass_and_publishes_new_unit(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    repass = _job("job-r1", "2026-10-26T13:50Z", kind="manual")
    scheduled = _job("scheduled", "2026-10-26T13:53Z")
    pilot.jobs[:] = [repass, scheduled]
    pilot.write(repass, "run")
    published: list[str] = []
    currency = Mock(side_effect=AssertionError("no recent trigger: cannot skip"))
    monkeypatch.setattr(
        NoaaHrrrForecast48HourDataset, "_current_data_is_published", currency
    )

    def finish_repass(seconds: int) -> None:
        assert seconds == 60
        pilot.compute.assert_not_called()
        assert pilot.state(scheduled) == "run"
        published.append("06Z")
        pilot.write(repass, "finalized")

    def compute(*args: object, **kwargs: object) -> tuple[list[object], xr.DataTree]:
        assert published == ["06Z"]
        return [], xr.DataTree()

    pilot.sleep.side_effect = finish_repass
    pilot.compute.side_effect = compute
    pilot.process.side_effect = lambda **kwargs: published.append("12Z")
    pilot.dataset.update("scheduled")

    assert published == ["06Z", "12Z"]
    pilot.validate.assert_called_once()
    pilot.sleep.assert_called_once_with(60)


def test_persistent_failure_submits_each_repass_through_r16(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot.validate.side_effect = validation.OperationalValidationError("still missing")
    pilot.sleep.side_effect = None
    submit = Mock()
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", submit)
    for retry in range(17):
        name = f"job-r{retry}" if retry else "job"
        job = _job(name, f"2026-10-26T08:{retry:02d}Z", kind="manual")
        pilot.jobs.append(job)
        with pytest.raises(typer.Exit) as exc:
            pilot.dataset.update(name)
        assert exc.value.exit_code == 20
        job.status.conditions = [client.V1JobCondition(type="Failed", status="True")]

    assert [call.args[1] for call in submit.call_args_list] == [
        f"job-r{retry}" for retry in range(1, 17)
    ]
    assert pilot.sleep.call_count == 5 + 10 + 20 + 13 * 30


@pytest.mark.parametrize("state", [None, "skip", "finalized"])
def test_backoff_does_not_yield_to_undecided_or_nonwriting_newer_jobs(
    pilot: Pilot, state: pc.JobState | None
) -> None:
    current, newer = _job("a"), _job("b")
    pilot.jobs[:] = [current, newer]
    if state is not None:
        pilot.write(newer, state)
    pilot.sleep.side_effect = None
    assert pc.backoff_before_repass(pilot.dataset.store_factory, _CRON, current, 1)
    assert pilot.sleep.call_count == 5


def test_final_backoff_check_prevents_repass_submission(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    current, newer = _job("a"), _job("b", kind="trigger")
    pilot.jobs[:] = [current, newer]
    pilot.validate.side_effect = validation.OperationalValidationError("late file")
    submit = Mock()
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", submit)
    ticks = 0

    def advance(seconds: int) -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 5:
            pilot.write(newer, "run")

    pilot.sleep.side_effect = advance
    with pytest.raises(typer.Exit) as exc:
        pilot.dataset._validate_after_update("a", materialized_job=current)
    assert exc.value.exit_code == 20
    assert ticks == 5
    submit.assert_not_called()


def test_replacement_pod_zero_reuses_run_when_trigger_appears_after_decision(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job("scheduled")
    pilot.jobs[:] = [job]
    assert pilot.prepare(job)
    trigger = _job("trigger", "2026-10-26T07:00Z", kind="trigger", terminal="Complete")
    pilot.jobs.append(trigger)
    currency = Mock(return_value=True)
    monkeypatch.setattr(
        NoaaHrrrForecast48HourDataset, "_current_data_is_published", currency
    )

    assert pilot.prepare(job)
    assert pilot.prepare(job, is_first=False)
    assert pilot.state(job) == "run"
    currency.assert_not_called()
    pilot.sleep.assert_not_called()
