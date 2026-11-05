import json
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import ClassVar, cast
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest

from reformatters.common import template_utils, virtual_region_job
from reformatters.common.download import SourceAccessError
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from tests.common.virtual_region_job_test import (
    BLOCK_NBYTES,
    VirtualTestDataVar,
    VirtualTestRegionJob,
    VirtualTestSourceFileCoord,
    _create_template_ds,
)


class PipelineJob(VirtualTestRegionJob):
    completion_ready_updates: ClassVar[bool] = True
    download_concurrency: ClassVar[int] = 2
    process_virtual_refs = VirtualRegionJob.process_virtual_refs

    def discover_available(
        self, pending: list[VirtualTestSourceFileCoord]
    ) -> list[tuple[VirtualTestSourceFileCoord, int]]:
        return [(coord, BLOCK_NBYTES) for coord in pending]

    def file_refs(
        self, coord: VirtualTestSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        return [
            VirtualRef(
                self.data_vars[0], coord.out_loc(), "file://source", 0, file_size
            )
        ]


def make_job(n_inits: int = 1) -> PipelineJob:
    return PipelineJob(
        tmp_store=Path("unused"),
        template_ds=_create_template_ds(n_inits),
        data_vars=[VirtualTestDataVar(name="temperature_2m")],
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
        fn: Callable[[VirtualTestSourceFileCoord, int], list[VirtualRef]],
        coord: VirtualTestSourceFileCoord,
        size: int,
    ) -> Future[list[VirtualRef]]:
        future: Future[list[VirtualRef]] = Future()
        self.tasks.append((future, lambda: fn(coord, size)))
        return future

    def complete(self, index: int) -> None:
        future, fn = self.tasks[index]
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
    monkeypatch.setattr(virtual_region_job, "ThreadPoolExecutor", lambda workers: pool)
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
        self: PipelineJob, pending: list[VirtualTestSourceFileCoord]
    ) -> list[tuple[VirtualTestSourceFileCoord, int]]:
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
    monkeypatch.setattr(virtual_region_job.time, "monotonic", lambda: clock)
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, *a, **kw: now))
    monkeypatch.setattr(virtual_region_job, "wait", wait)
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
        self: PipelineJob, pending: list[VirtualTestSourceFileCoord]
    ) -> list[tuple[VirtualTestSourceFileCoord, int]]:
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
    monkeypatch.setattr(virtual_region_job.time, "monotonic", lambda: clock)
    monkeypatch.setattr(virtual_region_job, "wait", wait)
    batches = list(job.process_virtual_refs(coords))
    assert sweeps == [0, 1]
    assert [c for b in batches for c, _ in b] == coords
    assert len(batches) == len(coords)


def test_slow_sibling_and_arrivals_during_build_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(2)
    coords = list(job.source_file_coords())
    slow, first, during_build, during_commit = coords
    started, release, commit_arrival = Event(), Event(), Event()
    sweeps: list[list[VirtualTestSourceFileCoord]] = []
    file_refs = PipelineJob.file_refs

    def refs(
        self: PipelineJob, coord: VirtualTestSourceFileCoord, size: int
    ) -> list[VirtualRef]:
        if coord is slow:
            started.set()
            assert release.wait(5)
        return file_refs(self, coord, size)

    def discover(
        self: PipelineJob, pending: list[VirtualTestSourceFileCoord]
    ) -> list[tuple[VirtualTestSourceFileCoord, int]]:
        sweeps.append(pending)
        visible = [slow, first]
        if started.is_set():
            visible.append(during_build)
        if commit_arrival.is_set():
            visible.append(during_commit)
        return [(c, BLOCK_NBYTES) for c in pending if any(c is v for v in visible)]

    monkeypatch.setattr(PipelineJob, "file_refs", refs)
    monkeypatch.setattr(PipelineJob, "discover_available", discover)
    monkeypatch.setattr(PipelineJob, "tick_interval", pd.Timedelta("0.02s"))
    iterator = cast(
        "Generator[Sequence[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]]]",
        job.process_virtual_refs(coords),
    )
    try:
        assert [c for c, _ in next(iterator)] == [first]
        assert started.wait(2)
        assert [c for c, _ in next(iterator)] == [during_build]
        sweep_count = len(sweeps)
        commit_arrival.set()
        time.sleep(0.04)
        assert len(sweeps) == sweep_count
        assert [c for c, _ in next(iterator)] == [during_commit]
        assert all(not any(c is slow for c in p) for p in sweeps[1:])
        release.set()
        assert [c for c, _ in next(iterator)] == [slow]
        assert list(iterator) == []
    finally:
        release.set()
        iterator.close()


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("mode", ["backfill", "update"])
def test_flag_off_and_backfill_keep_the_serial_loop(
    monkeypatch: pytest.MonkeyPatch, pipeline: bool, mode: str
) -> None:
    job = make_job().model_copy(update={"processing_mode": mode})
    monkeypatch.setattr(PipelineJob, "completion_ready_updates", pipeline)
    completion_loop = Mock(return_value=iter([]))
    monkeypatch.setattr(PipelineJob, "_completion_ready_refs", completion_loop)
    batches = list(job.process_virtual_refs(job.source_file_coords()))
    if pipeline and mode == "update":
        completion_loop.assert_called_once()
        assert batches == []
    else:
        completion_loop.assert_not_called()
        assert len(batches) == 1
        assert len(batches[0]) == 2


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

    monkeypatch.setattr(virtual_region_job, "wait", wait)
    with pytest.raises(type(failure), match=str(failure)):
        list(job.process_virtual_refs(job.source_file_coords()))
    assert manual_pool.closed
    assert len(manual_pool.tasks) == 70
    assert all(f.cancelled() for f, _ in manual_pool.tasks[1:])


def test_iterator_close_cancels_queued_and_joins_running_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(35)
    coords = list(job.source_file_coords())
    started, release, finished, closing = Event(), Event(), Event(), Event()
    file_refs = PipelineJob.file_refs
    pool = ThreadPoolExecutor(2)
    shutdown = pool.shutdown

    def closing_shutdown(*, wait: bool, cancel_futures: bool) -> None:
        closing.set()
        shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(pool, "shutdown", closing_shutdown)
    monkeypatch.setattr(virtual_region_job, "ThreadPoolExecutor", lambda workers: pool)

    def refs(
        self: PipelineJob, coord: VirtualTestSourceFileCoord, size: int
    ) -> list[VirtualRef]:
        if coord is not coords[0]:
            started.set()
            assert release.wait(5)
            finished.set()
        return file_refs(self, coord, size)

    monkeypatch.setattr(PipelineJob, "file_refs", refs)
    iterator = cast(
        "Generator[Sequence[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]]]",
        job.process_virtual_refs(coords),
    )
    try:
        assert [c for c, _ in next(iterator)] == [coords[0]]
        assert started.wait(2)
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


@pytest.mark.parametrize("ordinary_iterator", [False, True])
@pytest.mark.parametrize("writer_failure", [False, True])
@pytest.mark.parametrize("pipeline", [False, True])
def test_consumer_closes_iterator_on_success_or_writer_failure(
    monkeypatch: pytest.MonkeyPatch,
    ordinary_iterator: bool,
    writer_failure: bool,
    pipeline: bool,
) -> None:
    monkeypatch.setattr(PipelineJob, "completion_ready_updates", pipeline)
    job = make_job()
    coords = list(job.source_file_coords())
    batch = [(c, job.file_refs(c, BLOCK_NBYTES)) for c in coords]
    closed: list[bool] = []

    def batches() -> Iterator[
        Sequence[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]]
    ]:
        try:
            yield batch
        finally:
            closed.append(True)

    iterator = iter([batch]) if ordinary_iterator else batches()
    monkeypatch.setattr(
        PipelineJob, "process_virtual_refs", lambda self, remaining: iterator
    )
    commit = Mock(side_effect=RuntimeError("writer failed") if writer_failure else None)
    monkeypatch.setattr(PipelineJob, "_commit_virtual_batch", commit)
    if writer_failure:
        with pytest.raises(RuntimeError, match="writer failed"):
            job.process_virtual(Mock(), [], "main", coords)
    else:
        assert job.process_virtual(Mock(), [], "main", coords)
    assert closed == ([] if ordinary_iterator else [True])


def test_pipeline_logs_direct_membership_and_only_successful_primary_commit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    job = make_job()
    coords = list(job.source_file_coords())
    monkeypatch.setattr(
        VirtualTestSourceFileCoord,
        "get_url",
        lambda c: f"file:///source/{c.lead_time.value}",
    )
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    replica = icechunk.Repository.create(icechunk.in_memory_storage())
    for r in (repo, replica):
        template_utils.write_metadata(
            job.template_ds,
            r.writable_session("main").store,
            mode="w-",
            consolidated=False,
        )
    monkeypatch.setattr(PipelineJob, "_emit_refs", lambda self, stores, refs: None)
    commit = Mock()
    monkeypatch.setattr(virtual_region_job.storage, "commit_if_icechunk", commit)
    with caplog.at_level("INFO"):
        assert job.process_virtual(repo, [replica], "main", coords)
    events = [
        json.loads(r.getMessage().split("Virtual pipeline ", 1)[1])
        for r in caplog.records
        if r.getMessage().startswith("Virtual pipeline ")
    ]
    batches = {e["batch_id"]: e for e in events if e["event"] == "batch"}
    commits = [e for e in events if e["event"] == "primary_committed"]
    assert commit.call_count == 2 * len(batches)
    assert len(commits) == len(batches)
    assert {e["file"] for e in events if e["event"] == "discovered"} == {
        c.get_url() for c in coords
    }
    assert {e["file"] for e in events if e["event"] == "ready"} == {
        c.get_url() for c in coords
    }
    for event in commits:
        assert event["files"] == batches[event["batch_id"]]["files"]
        assert event["committed_at"] >= batches[event["batch_id"]]["selected_at"]
        assert event["ref_count"] == len(event["files"])
        assert event["resized"] is False
    caplog.clear()
    commit.side_effect = RuntimeError("failed commit")
    with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="failed commit"):
        job.process_virtual(repo, [], "main", coords)
    assert not any(
        '"event": "primary_committed"' in r.getMessage() for r in caplog.records
    )


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
        self: PipelineJob, pending: list[VirtualTestSourceFileCoord]
    ) -> list[tuple[VirtualTestSourceFileCoord, int]]:
        nonlocal clock
        starts.append(clock)
        clock += 0.3
        return [(c, BLOCK_NBYTES) for c in pending] if len(starts) == 3 else []

    def sleep(seconds: float) -> None:
        nonlocal clock
        clock += seconds

    def wait(futures: object, **kwargs: object) -> None:
        for i in range(len(manual_pool.tasks)):
            manual_pool.complete(i)

    monkeypatch.setattr(PipelineJob, "discover_available", discover)
    monkeypatch.setattr(virtual_region_job.time, "monotonic", lambda: clock)
    monkeypatch.setattr(virtual_region_job.time, "sleep", sleep)
    monkeypatch.setattr(virtual_region_job, "wait", wait)
    with caplog.at_level("INFO"):
        batches = list(job.process_virtual_refs(coords))
    events = [
        json.loads(r.getMessage().split("Virtual pipeline ", 1)[1])
        for r in caplog.records
        if '"event": "discovery_sweep"' in r.getMessage()
    ]
    assert starts == [0, 1, 2]
    assert [e["files"] for e in events] == [0, 0, 2]
    assert [e["duration_s"] for e in events] == pytest.approx([0.3] * 3)
    assert len(batches) == 1
    assert [c for c, _ in batches[0]] == coords


@pytest.mark.parametrize("close_from_writer", [False, True])
def test_writer_failure_and_iterator_close_cancel_queued_work(
    monkeypatch: pytest.MonkeyPatch, manual_pool: ManualPool, close_from_writer: bool
) -> None:
    job = make_job(35)
    coords = list(job.source_file_coords())

    def wait(futures: object, **kwargs: object) -> None:
        manual_pool.complete(0)

    monkeypatch.setattr(virtual_region_job, "wait", wait)
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
            "Generator[Sequence[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]]]",
            job.process_virtual_refs(coords),
        )
        assert [c for c, _ in next(iterator)] == [coords[0]]
        iterator.close()
    assert manual_pool.closed
    assert len(manual_pool.tasks) == 70
    assert all(f.cancelled() for f, _ in manual_pool.tasks[1:])
