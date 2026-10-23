from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Literal
from unittest.mock import Mock, call

import pandas as pd
import pytest

from reformatters.common import virtual_region_job
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from tests.common.virtual_region_job_test import (
    BLOCK_NBYTES,
    VirtualTestDataVar,
    VirtualTestRegionJob,
    VirtualTestSourceFileCoord,
    _create_template_ds,
)


@dataclass
class _Clock:
    elapsed: float = 0.0

    def now(self) -> pd.Timestamp:
        return pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=self.elapsed)

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.elapsed += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(virtual_region_job, "time", clock)
    monkeypatch.setattr(
        virtual_region_job,
        "exponential_backoff_time",
        lambda attempt, *, max_delay_seconds: min(2**attempt, max_delay_seconds),
    )
    monkeypatch.setattr(
        virtual_region_job,
        "pd",
        SimpleNamespace(Timestamp=SimpleNamespace(now=clock.now), Index=pd.Index),
    )
    return clock


class _PendingFileJob(VirtualTestRegionJob):
    download_concurrency: ClassVar[int] = 1
    process_virtual_refs = VirtualRegionJob.process_virtual_refs

    def discover_available(
        self, pending: list[VirtualTestSourceFileCoord]
    ) -> list[tuple[VirtualTestSourceFileCoord, int]]:
        return [(coord, BLOCK_NBYTES) for coord in pending]


def _job(mode: Literal["update", "backfill"], clock: _Clock) -> _PendingFileJob:
    return _PendingFileJob(
        tmp_store=Path("unused.zarr"),
        template_ds=_create_template_ds(1),
        data_vars=[VirtualTestDataVar(name="temperature_2m")],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode=mode,
        poll_deadline=clock.now() + pd.Timedelta("2s"),
    )


def _refs(job: _PendingFileJob, coord: VirtualTestSourceFileCoord) -> list[VirtualRef]:
    return [
        VirtualRef(
            data_var=job.data_vars[0],
            out_loc=coord.out_loc(),
            location="https://example.invalid/source.grib2",
            offset=0,
            length=BLOCK_NBYTES,
        )
    ]


@pytest.mark.parametrize("failure", [[], RuntimeError("index unavailable")])
def test_update_retries_empty_batch_until_refs_recover(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    failure: list[VirtualRef] | Exception,
) -> None:
    job = _job("update", clock)
    coord = job.source_file_coords()[0]
    refs = _refs(job, coord)
    build = Mock(side_effect=[failure, refs])
    monkeypatch.setattr(_PendingFileJob, "file_refs", build)

    assert list(job.process_virtual_refs([coord])) == [[(coord, refs)]]

    assert build.call_args_list == [call(coord, BLOCK_NBYTES)] * 2
    assert clock.elapsed == 1


def test_update_retries_only_files_without_refs(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    job = _job("update", clock)
    ready, retrying = job.source_file_coords()
    ready_refs, retried_refs = _refs(job, ready), _refs(job, retrying)
    build = Mock(side_effect=[ready_refs, [], retried_refs])
    discover = Mock(side_effect=job.discover_available)
    monkeypatch.setattr(_PendingFileJob, "file_refs", build)
    monkeypatch.setattr(_PendingFileJob, "discover_available", discover)

    assert list(job.process_virtual_refs([ready, retrying])) == [
        [(ready, ready_refs)],
        [(retrying, retried_refs)],
    ]

    assert discover.call_args_list == [call([ready, retrying]), call([retrying])]
    assert build.call_args_list == [
        call(ready, BLOCK_NBYTES),
        call(retrying, BLOCK_NBYTES),
        call(retrying, BLOCK_NBYTES),
    ]
    assert clock.elapsed == 1


@pytest.mark.parametrize("failure", [[], RuntimeError("index unavailable")])
def test_update_stops_retrying_failed_refs_at_poll_deadline(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    caplog: pytest.LogCaptureFixture,
    failure: list[VirtualRef] | Exception,
) -> None:
    job = _job("update", clock)
    coord = job.source_file_coords()[0]
    build = (
        Mock(side_effect=failure)
        if isinstance(failure, Exception)
        else Mock(return_value=failure)
    )
    monkeypatch.setattr(_PendingFileJob, "file_refs", build)

    with caplog.at_level("INFO"):
        assert list(job.process_virtual_refs([coord])) == []

    assert build.call_args_list == [call(coord, BLOCK_NBYTES)] * 2
    assert clock.elapsed == 2
    assert "Poll deadline reached with 1 source files" in caplog.text


def test_unreadable_file_backoff_bounds_calls_and_logs(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    job = _job("update", clock).model_copy(
        update={"poll_deadline": clock.now() + pd.Timedelta(minutes=10)}
    )
    coord = job.source_file_coords()[0]
    build = Mock(side_effect=RuntimeError("unreadable source index"))
    monkeypatch.setattr(_PendingFileJob, "file_refs", build)
    with caplog.at_level("INFO"):
        assert list(job.process_virtual_refs([coord])) == []
    assert 10 <= build.call_count <= 20
    assert sum(record.levelname == "ERROR" for record in caplog.records) == 1
    assert clock.elapsed == 600
    assert "Poll deadline reached with 1 source files" in caplog.text


def test_file_recovers_after_backoff_without_blocking_new_files(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    job = _job("update", clock).model_copy(
        update={"poll_deadline": clock.now() + pd.Timedelta(minutes=10)}
    )
    failed, healthy = job.source_file_coords()
    failed_refs, healthy_refs = _refs(job, failed), _refs(job, healthy)
    build_times: dict[int, list[float]] = {id(failed): [], id(healthy): []}

    def build(
        self: _PendingFileJob, coord: VirtualTestSourceFileCoord, size: int
    ) -> list[VirtualRef]:
        build_times[id(coord)].append(clock.elapsed)
        if coord is failed and clock.elapsed < 30:
            raise RuntimeError("index temporarily unreadable")
        return failed_refs if coord is failed else healthy_refs

    monkeypatch.setattr(_PendingFileJob, "file_refs", build)
    assert list(job.process_virtual_refs([failed, healthy])) == [
        [(healthy, healthy_refs)],
        [(failed, failed_refs)],
    ]
    assert build_times[id(healthy)] == [0]
    assert build_times[id(failed)] == [0, 1, 3, 7, 15, 31]


@pytest.mark.parametrize("failure", [[], RuntimeError("index unavailable")])
def test_backfill_skips_failed_refs_after_one_sweep(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    failure: list[VirtualRef] | Exception,
) -> None:
    job = _job("backfill", clock)
    coord = job.source_file_coords()[0]
    build = Mock(side_effect=[failure, _refs(job, coord)])
    monkeypatch.setattr(_PendingFileJob, "file_refs", build)

    assert list(job.process_virtual_refs([coord])) == []

    build.assert_called_once_with(coord, BLOCK_NBYTES)
    assert clock.elapsed == 0
