import json
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, current_thread
from threading import enumerate as enumerate_threads
from typing import ClassVar, cast
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.download import SourceAccessError
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.ecmwf.fast import region_job
from reformatters.ecmwf.fast.region_job import EcmwfFastRegionJob
from reformatters.ecmwf.fast.source import EcmwfFastSourceFileCoord
from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig, FastModel
from tests.common.virtual_region_job_test import (
    BLOCK_NBYTES,
    _create_template_ds,
)


class PipelineJob(EcmwfFastRegionJob):
    model: ClassVar[FastModel] = "ifs"
    download_concurrency: ClassVar[int] = 2

    def discover_available(
        self, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        return [(coord, BLOCK_NBYTES) for coord in pending]

    def file_refs(
        self, coord: EcmwfFastSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        return [
            VirtualRef(
                self.data_vars[0], coord.out_loc(), "file://source", 0, file_size
            )
        ]


def make_job(n_inits: int = 1) -> PipelineJob:
    template = _create_template_ds(n_inits).to_dataset()
    template["temperature_2m"] = template.temperature_2m.expand_dims(
        ensemble_member=[0], axis=2
    )
    template.temperature_2m.encoding["chunks"] = (1, 1, 1, 2, 3)
    config = EcmwfFastTemplateConfig(model="ifs")
    return PipelineJob(
        tmp_store=Path("unused"),
        template_ds=xr.DataTree(template),
        data_vars=[config.data_vars[0]],
        append_dim="init_time",
        region=slice(0, n_inits),
        reformat_job_name="pipeline-test",
        processing_mode="update",
    )


class ManualPool:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.tasks: list[
            tuple[Future[list[VirtualRef]], Callable[[], list[VirtualRef]]]
        ] = []
        self.closed = False

    def submit(
        self,
        fn: Callable[[EcmwfFastSourceFileCoord, int], list[VirtualRef]],
        coord: EcmwfFastSourceFileCoord,
        size: int,
    ) -> Future[list[VirtualRef]]:
        future: Future[list[VirtualRef]] = Future()
        self.tasks.append((future, lambda: fn(coord, size)))
        return future

    def complete(self, index: int) -> None:
        future, fn = self.tasks[index]
        if future.done():
            return
        try:
            future.set_result(fn())
        except (AssertionError, SourceAccessError) as exc:
            future.set_exception(exc)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        assert wait
        assert cancel_futures
        for future, _ in self.tasks:
            future.cancel()
        self.closed = True


@pytest.fixture
def manual_pool(monkeypatch: pytest.MonkeyPatch) -> ManualPool:
    pool = ManualPool(64)
    monkeypatch.setattr(region_job, "ThreadPoolExecutor", lambda workers: pool)
    return pool


@pytest.mark.parametrize("past_deadline", [False, True])
def test_cutoff_drains_every_submission_in_one_yield(
    monkeypatch: pytest.MonkeyPatch, manual_pool: ManualPool, past_deadline: bool
) -> None:
    job = make_job(70)
    coords = list(job.source_file_coords())
    now = pd.Timestamp("2026-11-05")
    deadline = now + pd.Timedelta(seconds=-1 if past_deadline else 2)
    job = job.model_copy(update={"poll_deadline": deadline})
    clock = 0.0
    sweeps: list[float] = []

    def discover(
        self: PipelineJob, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        sweeps.append(clock)
        selected = (
            coords if past_deadline else coords[:70] if clock == 0 else coords[70:]
        )
        assert all(any(c is p for p in pending) for c in selected)
        return [(c, BLOCK_NBYTES) for c in selected]

    def wait(
        futures: object, timeout: float | None = None, return_when: str = ""
    ) -> None:
        nonlocal clock, now
        if return_when == FIRST_EXCEPTION:
            assert len(manual_pool.tasks) == len(coords)
            for i in range(len(coords)):
                manual_pool.complete(i)
        else:
            assert timeout is not None
            clock += timeout
            now += pd.Timedelta(seconds=timeout)

    monkeypatch.setattr(PipelineJob, "discover_available", discover)
    monkeypatch.setattr(region_job.time, "monotonic", lambda: clock)
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, *a, **kw: now))
    monkeypatch.setattr(region_job, "wait", wait)
    commit = Mock()
    monkeypatch.setattr(PipelineJob, "_commit_virtual_batch", commit)
    assert job.process_virtual(Mock(), [], "main", coords) is False
    batches = [call.args[2] for call in commit.call_args_list]
    assert sweeps == ([0] if past_deadline else [0, 1])
    assert len(batches) == 1
    assert [id(c) for c, _ in batches[0]] == [id(c) for c in coords]
    assert manual_pool.closed


def test_frequent_completions_do_not_postpone_discovery(
    monkeypatch: pytest.MonkeyPatch, manual_pool: ManualPool
) -> None:
    job = make_job(4)
    coords = list(job.source_file_coords())
    clock = 0.0
    sweeps: list[float] = []
    completed = 0

    def discover(
        self: PipelineJob, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        sweeps.append(clock)
        if clock == 0:
            return [(c, BLOCK_NBYTES) for c in coords[:-1]]
        assert pending == [coords[-1]]
        return [(coords[-1], BLOCK_NBYTES)]

    def wait(
        futures: object, timeout: float | None = None, return_when: str = ""
    ) -> None:
        nonlocal clock, completed
        assert timeout is not None
        completion_at = round((completed + 1) * 0.2, 1)
        if completion_at <= clock + timeout:
            clock = completion_at
            manual_pool.complete(completed)
            completed += 1
        else:
            clock += timeout

    monkeypatch.setattr(PipelineJob, "discover_available", discover)
    monkeypatch.setattr(region_job.time, "monotonic", lambda: clock)
    monkeypatch.setattr(region_job, "wait", wait)
    batches = []
    job._poll_ready_files(coords, batches.append, Event())
    assert sweeps == [0, 1]
    assert [c for b in batches for c, _ in b] == coords
    assert len(batches) == len(coords)


def test_slow_sibling_and_arrivals_during_build_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(2)
    coords = list(job.source_file_coords())
    slow, first, during_build, during_commit = coords
    started, release, commit_arrival, admitted = Event(), Event(), Event(), Event()
    build_arrival = Event()
    sweeps: list[list[EcmwfFastSourceFileCoord]] = []
    file_refs = PipelineJob.file_refs

    def refs(
        self: PipelineJob, coord: EcmwfFastSourceFileCoord, size: int
    ) -> list[VirtualRef]:
        if coord is slow:
            started.set()
            assert release.wait(5)
        if coord is during_commit:
            admitted.set()
        return file_refs(self, coord, size)

    def discover(
        self: PipelineJob, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        sweeps.append(pending)
        visible = [slow, first]
        if build_arrival.is_set():
            visible.append(during_build)
        if commit_arrival.is_set():
            visible.append(during_commit)
        return [(c, BLOCK_NBYTES) for c in pending if any(c is v for v in visible)]

    monkeypatch.setattr(PipelineJob, "file_refs", refs)
    monkeypatch.setattr(PipelineJob, "discover_available", discover)
    monkeypatch.setattr(PipelineJob, "tick_interval", pd.Timedelta("0.02s"))
    iterator = cast(
        "Generator[Sequence[tuple[EcmwfFastSourceFileCoord, Sequence[VirtualRef]]]]",
        job.process_virtual_refs(coords),
    )
    try:
        assert [c for c, _ in next(iterator)] == [first]
        assert started.wait(2)
        build_arrival.set()
        assert [c for c, _ in next(iterator)] == [during_build]
        sweep_count = len(sweeps)
        commit_arrival.set()
        assert admitted.wait(2)
        assert len(sweeps) > sweep_count
        assert [c for c, _ in next(iterator)] == [during_commit]
        assert all(not any(c is slow for c in p) for p in sweeps[1:])
        release.set()
        assert [c for c, _ in next(iterator)] == [slow]
        assert list(iterator) == []
    finally:
        release.set()
        iterator.close()


def test_backfill_delegates_to_shared_sweep_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job().model_copy(update={"processing_mode": "backfill"})
    coords = job.source_file_coords()
    shared_loop = Mock(return_value=iter([]))
    monkeypatch.setattr(VirtualRegionJob, "process_virtual_refs", shared_loop)
    assert list(job.process_virtual_refs(coords)) == []
    shared_loop.assert_called_once_with(coords)


@pytest.mark.parametrize("failed", [False, True])
def test_handoff_coalesces_before_eof_but_failure_discards_buffered_files(
    monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    job = make_job()
    coords = list(job.source_file_coords())

    def poll(
        self: PipelineJob,
        remaining: Sequence[EcmwfFastSourceFileCoord],
        publish: Callable,
        stop: Event,
    ) -> None:
        for coord in coords:
            publish([(coord, self.file_refs(coord, BLOCK_NBYTES))])
        if failed:
            publish(SourceAccessError("auth"))

    monkeypatch.setattr(PipelineJob, "_poll_ready_files", poll)
    monkeypatch.setattr(
        region_job, "Thread", lambda *, target, **kwargs: Mock(start=target)
    )
    iterator = job.process_virtual_refs(coords)
    if failed:
        with pytest.raises(SourceAccessError, match="auth"):
            next(iterator)
    else:
        assert [[c for c, _ in batch] for batch in iterator] == [coords]


def test_discovery_and_collection_continue_during_shared_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(2)
    coords = list(job.source_file_coords())
    committing, collected = Event(), Event()
    batches = []
    poll = PipelineJob._poll_ready_files
    writer = current_thread()

    def discover(
        self: PipelineJob, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        visible = coords if committing.is_set() else coords[:1]
        return [(c, BLOCK_NBYTES) for c in pending if c in visible]

    def observed_poll(
        self: PipelineJob,
        remaining: Sequence[EcmwfFastSourceFileCoord],
        publish: Callable,
        stop: Event,
    ) -> None:
        observed = set()

        def forward(item: object) -> None:
            publish(item)
            if isinstance(item, list):
                observed.update(id(coord) for coord, _ in item)
                if len(observed) == len(coords):
                    collected.set()

        poll(self, remaining, forward, stop)

    def commit(
        self: PipelineJob, repo: object, branch: str, batch: object, **kw: object
    ) -> None:
        assert current_thread() is writer
        batches.append(batch)
        if len(batches) == 1:
            committing.set()
            assert collected.wait(5), "source scheduling stalled behind the writer"

    monkeypatch.setattr(PipelineJob, "discover_available", discover)
    monkeypatch.setattr(PipelineJob, "_poll_ready_files", observed_poll)
    monkeypatch.setattr(PipelineJob, "_commit_virtual_batch", commit)
    monkeypatch.setattr(PipelineJob, "tick_interval", pd.Timedelta("0.01s"))
    assert job.process_virtual(Mock(), [], "main", coords)
    assert len(batches) == 2
    assert [c for c, _ in batches[0]] == coords[:1]
    assert len(batches[1]) == len(coords) - 1
    assert {id(c) for c, _ in batches[1]} == {id(c) for c in coords[1:]}
    assert not any(t.name == "ecmwf-fast-poller" for t in enumerate_threads())


@pytest.mark.parametrize(
    "failure", [AssertionError("invariant"), SourceAccessError("auth")]
)
def test_fatal_failure_cancels_unstarted_work(
    monkeypatch: pytest.MonkeyPatch, manual_pool: ManualPool, failure: Exception
) -> None:
    job = make_job(35).model_copy(update={"poll_deadline": pd.Timestamp.min})
    monkeypatch.setattr(PipelineJob, "file_refs", Mock(side_effect=failure))

    def wait(futures: object, **kwargs: object) -> None:
        manual_pool.complete(0)

    monkeypatch.setattr(region_job, "wait", wait)
    with pytest.raises(type(failure), match=str(failure)):
        list(job.process_virtual_refs(job.source_file_coords()))
    assert manual_pool.closed
    assert len(manual_pool.tasks) == 70
    assert all(f.cancelled() for f, _ in manual_pool.tasks[1:])


@pytest.mark.parametrize("at_cutoff", [False, True])
def test_iterator_close_cancels_queued_and_joins_running_reads(
    monkeypatch: pytest.MonkeyPatch,
    at_cutoff: bool,
) -> None:
    now = pd.Timestamp("2026-11-08")
    job = make_job(35).model_copy(
        update={"poll_deadline": now + pd.Timedelta(seconds=1)}
    )
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, *a, **kw: now))
    coords = list(job.source_file_coords())
    started, release, finished, closing = Event(), Event(), Event(), Event()
    draining = Event()
    original_wait = region_job.wait

    def observed_wait(
        futures: Iterable[Future[list[VirtualRef]]],
        timeout: float | None = None,
        return_when: str = "ALL_COMPLETED",
    ) -> tuple[set[Future[list[VirtualRef]]], set[Future[list[VirtualRef]]]]:
        if return_when == FIRST_EXCEPTION:
            draining.set()
        return original_wait(futures, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(region_job, "wait", observed_wait)
    file_refs = PipelineJob.file_refs
    pool = ThreadPoolExecutor(2)
    shutdown = pool.shutdown

    def closing_shutdown(*, wait: bool, cancel_futures: bool) -> None:
        closing.set()
        shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(pool, "shutdown", closing_shutdown)
    monkeypatch.setattr(region_job, "ThreadPoolExecutor", lambda workers: pool)

    def refs(
        self: PipelineJob, coord: EcmwfFastSourceFileCoord, size: int
    ) -> list[VirtualRef]:
        if coord is not coords[0]:
            started.set()
            assert release.wait(5)
            finished.set()
        return file_refs(self, coord, size)

    monkeypatch.setattr(PipelineJob, "file_refs", refs)
    iterator = cast(
        "Generator[Sequence[tuple[EcmwfFastSourceFileCoord, Sequence[VirtualRef]]]]",
        job.process_virtual_refs(coords),
    )
    try:
        assert [c for c, _ in next(iterator)] == [coords[0]]
        assert started.wait(2)
        if at_cutoff:
            now += pd.Timedelta(seconds=2)
            assert draining.wait(2)
        with ThreadPoolExecutor(1) as closer:

            def close() -> None:
                iterator.close()

            closing_future = closer.submit(close)
            assert closing.wait(2)
            assert not closing_future.done()
            assert not finished.is_set()
            release.set()
            closing_future.result(timeout=5)
        assert finished.is_set()
    finally:
        release.set()
        iterator.close()
        pool.shutdown(wait=True, cancel_futures=True)


def test_empty_and_deferred_sweeps_keep_cadence_and_log_duration(
    monkeypatch: pytest.MonkeyPatch,
    manual_pool: ManualPool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = make_job()
    coords = list(job.source_file_coords())
    clock = 0.0
    starts: list[float] = []

    def discover(
        self: PipelineJob, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        nonlocal clock
        starts.append(clock)
        clock += 0.3
        return [(c, BLOCK_NBYTES) for c in pending] if len(starts) == 24 else []

    def sleep(seconds: float) -> None:
        nonlocal clock
        clock += seconds

    def wait(futures: object, **kwargs: object) -> None:
        for i in range(len(manual_pool.tasks)):
            manual_pool.complete(i)

    monkeypatch.setattr(PipelineJob, "discover_available", discover)
    monkeypatch.setattr(region_job.time, "monotonic", lambda: clock)
    stop = Event()
    monkeypatch.setattr(stop, "wait", sleep)
    monkeypatch.setattr(region_job, "wait", wait)
    batches = []
    with caplog.at_level("INFO"):
        job._poll_ready_files(coords, batches.append, stop)
    events = [
        json.loads(r.getMessage().split("Virtual pipeline ", 1)[1])
        for r in caplog.records
        if '"event": "discovery_sweep"' in r.getMessage()
    ]
    assert starts == list(range(24))
    assert [e["files"] for e in events] == [0, 0, 0, 2]
    assert [e["duration_s"] for e in events] == pytest.approx([0.3] * 4)
    assert len(batches) == 1
    assert [c for c, _ in batches[0]] == coords


@pytest.mark.parametrize("running", [False, True])
def test_sweeps_with_outstanding_files_are_not_thinned(
    monkeypatch: pytest.MonkeyPatch,
    manual_pool: ManualPool,
    caplog: pytest.LogCaptureFixture,
    running: bool,
) -> None:
    job = make_job()
    coords = list(job.source_file_coords())
    clock = 0.0
    stop = Event()

    def discover(
        self: PipelineJob, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        return [(c, BLOCK_NBYTES) for c in pending] if clock == 0 else []

    def wait(futures: object, timeout: float, **kwargs: object) -> None:
        nonlocal clock
        clock += timeout
        if clock == 1 and running:
            for future, _ in manual_pool.tasks:
                future.set_running_or_notify_cancel()
        if clock == 3:
            for i in range(len(manual_pool.tasks)):
                manual_pool.complete(i)

    monkeypatch.setattr(PipelineJob, "discover_available", discover)
    monkeypatch.setattr(region_job.time, "monotonic", lambda: clock)
    monkeypatch.setattr(region_job, "wait", wait)
    batches = []
    with caplog.at_level("INFO"):
        job._poll_ready_files(coords, batches.append, stop)
    events = [
        json.loads(r.getMessage().split("Virtual pipeline ", 1)[1])
        for r in caplog.records
        if '"event": "discovery_sweep"' in r.getMessage()
    ]
    assert [e["files"] for e in events] == [2, 0, 0]
    assert [e["outstanding_files"] for e in events] == [0, 2, 2]
    assert [e["active_streams"] for e in events] == [0, 2 * running, 2 * running]
    assert len(batches) == 1


@pytest.mark.parametrize("close_from_writer", [False, True])
def test_writer_failure_and_iterator_close_cancel_queued_work(
    monkeypatch: pytest.MonkeyPatch, manual_pool: ManualPool, close_from_writer: bool
) -> None:
    job = make_job(35)
    coords = list(job.source_file_coords())

    def wait(futures: object, **kwargs: object) -> None:
        manual_pool.complete(0)

    monkeypatch.setattr(region_job, "wait", wait)
    if close_from_writer:
        monkeypatch.setattr(
            PipelineJob,
            "_commit_virtual_batch",
            Mock(side_effect=RuntimeError("writer failed")),
        )
        with pytest.raises(RuntimeError, match="writer failed"):
            job.process_virtual(Mock(), [], "main", coords)
    else:
        iterator = cast(
            "Generator[Sequence[tuple[EcmwfFastSourceFileCoord, Sequence[VirtualRef]]]]",
            job.process_virtual_refs(coords),
        )
        assert [c for c, _ in next(iterator)] == [coords[0]]
        iterator.close()
    assert manual_pool.closed
    assert len(manual_pool.tasks) == 70
    assert all(f.cancelled() for f, _ in manual_pool.tasks[1:])


def test_file_events_include_skipped_files_but_batches_only_contain_ready_refs(
    monkeypatch: pytest.MonkeyPatch,
    manual_pool: ManualPool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = make_job().model_copy(update={"poll_deadline": pd.Timestamp.min})
    coords = job.source_file_coords()
    file_refs = PipelineJob.file_refs

    def refs(
        self: PipelineJob, coord: EcmwfFastSourceFileCoord, size: int
    ) -> list[VirtualRef]:
        return [] if coord is coords[0] else file_refs(self, coord, size)

    def wait(futures: object, **kwargs: object) -> None:
        for i in range(len(manual_pool.tasks)):
            manual_pool.complete(i)

    monkeypatch.setattr(PipelineJob, "file_refs", refs)
    monkeypatch.setattr(region_job, "wait", wait)
    with caplog.at_level("INFO"):
        batches = list(job.process_virtual_refs(coords))
    assert [[coord for coord, _ in batch] for batch in batches] == [[coords[1]]]
    events = [
        json.loads(record.getMessage().split("Virtual pipeline ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("Virtual pipeline ")
    ]
    assert [event["event"] for event in events] == [
        "discovery_sweep",
        "discovered",
        "discovered",
        "ready",
        "ready",
    ]
    discovered = [event for event in events if event["event"] == "discovered"]
    ready = [event for event in events if event["event"] == "ready"]
    assert [event["file"] for event in discovered] == [c.get_url() for c in coords]
    assert [event["file"] for event in ready] == [c.get_url() for c in coords]
    assert [event["ref_count"] for event in ready] == [0, 1]
    assert all(
        event["ready_at"] >= receipt["received_at"]
        for event, receipt in zip(ready, discovered, strict=True)
    )
    assert manual_pool.closed
