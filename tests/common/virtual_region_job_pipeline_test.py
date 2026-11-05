import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest
from icechunk.store import IcechunkStore
from zarr.abc.store import Store

from reformatters.common import template_utils, virtual_region_job
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from tests.common.virtual_region_job_test import (
    BLOCK_NBYTES,
    VirtualTestDataVar,
    VirtualTestRegionJob,
    VirtualTestSourceFileCoord,
    _create_template_ds,
    _make_dataset,
    _make_replica_repo,
    _message_offset_length,
    _queue_stale_sessions,
    _snapshot_count,
)


class BatchJob(VirtualTestRegionJob):
    download_concurrency: ClassVar[int] = 2
    process_virtual_refs = VirtualRegionJob.process_virtual_refs

    def discover_available(
        self, pending: list[VirtualTestSourceFileCoord]
    ) -> list[tuple[VirtualTestSourceFileCoord, int]]:
        return [(coord, BLOCK_NBYTES) for coord in pending]

    def file_refs(
        self, coord: VirtualTestSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        root = self.template_ds.to_dataset()
        init_index = root.get_index("init_time").get_indexer(
            pd.Index([coord.init_time])
        )[0]
        lead_index = root.get_index("lead_time").get_indexer(
            pd.Index([coord.lead_time])
        )[0]
        offset, length = _message_offset_length(int(init_index), int(lead_index))
        assert length == file_size
        return [
            VirtualRef(
                self.data_vars[0], coord.out_loc(), coord.get_url(), offset, length
            )
        ]


def make_job(n_inits: int = 1) -> BatchJob:
    return BatchJob(
        tmp_store=Path("unused"),
        template_ds=_create_template_ds(n_inits),
        data_vars=[VirtualTestDataVar(name="temperature_2m")],
        append_dim="init_time",
        region=slice(0, n_inits),
        reformat_job_name="pipeline-test",
        processing_mode="update",
    )


@pytest.mark.parametrize("ordinary_iterator", [False, True])
@pytest.mark.parametrize("writer_failure", [False, True])
@pytest.mark.parametrize("mode", ["backfill", "update"])
def test_consumer_closes_iterator_on_success_or_writer_failure(
    monkeypatch: pytest.MonkeyPatch,
    ordinary_iterator: bool,
    writer_failure: bool,
    mode: str,
) -> None:
    job = make_job().model_copy(update={"processing_mode": mode})
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
        BatchJob, "process_virtual_refs", lambda self, remaining: iterator
    )
    commit = Mock(side_effect=RuntimeError("writer failed") if writer_failure else None)
    monkeypatch.setattr(BatchJob, "_commit_virtual_batch", commit)
    if writer_failure:
        with pytest.raises(RuntimeError, match="writer failed"):
            job.process_virtual(Mock(), [], "main", coords)
    else:
        assert job.process_virtual(Mock(), [], "main", coords) is (mode == "update")
    assert closed == ([] if ordinary_iterator else [True])


def pipeline_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage().split("Virtual pipeline ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("Virtual pipeline ")
    ]


@pytest.mark.parametrize("mode", ["backfill", "update"])
def test_shared_batch_logging_uses_processing_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mode: Literal["backfill", "update"],
) -> None:
    dataset = _make_dataset(tmp_path, n_inits=1)
    job = make_job().model_copy(update={"processing_mode": mode})
    primary, _ = dataset.store_factory.icechunk_primary_and_replica_repos()
    replica = _make_replica_repo(tmp_path, dataset)
    for repo in (primary, replica):
        template_utils.write_metadata(
            _create_template_ds(0),
            repo.writable_session("main").store,
            mode="w-",
            consolidated=False,
        )
    with caplog.at_level("INFO"):
        assert job.process_virtual(
            primary, [replica], "main", job.source_file_coords()
        ) is (mode == "update")
    assert _snapshot_count(primary) == _snapshot_count(replica) == 3
    assert caplog.text.count("Committed 2 refs ") == 2
    events = pipeline_events(caplog)
    if mode == "backfill":
        assert events == []
        return
    assert [event["event"] for event in events] == ["batch", "primary_committed"]
    batch, committed = events
    assert (
        batch["files"]
        == committed["files"]
        == [c.get_url() for c in job.source_file_coords()]
    )
    assert batch["batch_id"] == committed["batch_id"] == "pipeline-test:0"
    assert committed["committed_at"] >= batch["selected_at"]
    assert committed["ref_count"] == 2
    assert committed["resized"] is True
    assert committed["duration_s"] >= committed["commit_s"] >= 0
    assert (
        job.filter_already_present(
            job.source_file_coords(), primary.readonly_session("main").store
        )
        == []
    )
    assert (
        job.filter_already_present(
            job.source_file_coords(), replica.readonly_session("main").store
        )
        == []
    )


@pytest.mark.parametrize("race", ["partial", "all", "failure", "exhausted"])
def test_batch_logs_once_and_primary_logs_only_actual_successful_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    race: str,
) -> None:
    dataset = _make_dataset(tmp_path, n_inits=1)
    job = make_job()
    coords = job.source_file_coords()
    monkeypatch.setattr(
        VirtualTestSourceFileCoord,
        "get_url",
        lambda c: (tmp_path / f"{c.lead_time.value}.bin").as_uri(),
    )
    primary, _ = dataset.store_factory.icechunk_primary_and_replica_repos()
    replica = _make_replica_repo(tmp_path, dataset)
    for repo in (primary, replica):
        template_utils.write_metadata(
            _create_template_ds(0),
            repo.writable_session("main").store,
            mode="w-",
            consolidated=False,
        )
    stale = primary.writable_session("main")
    if race in ("partial", "all"):
        batch = [
            (c, job.file_refs(c, BLOCK_NBYTES))
            for c in (coords[:1] if race == "partial" else coords)
        ]
        job._commit_virtual_batch(primary, "main", batch, overwrite_chunks=False)
    initial_primary = _snapshot_count(primary)
    opened = _queue_stale_sessions(monkeypatch, primary, [stale])
    monkeypatch.setattr(
        "reformatters.common.retry.exponential_backoff_time", lambda attempt: 0.0
    )
    if race in ("failure", "exhausted"):
        original_commit = virtual_region_job.storage.commit_if_icechunk

        def failing_commit(
            message: str,
            store: IcechunkStore,
            replicas: Sequence[Store],
            *,
            allow_rebase: bool = True,
        ) -> None:
            if opened and store.session == opened[-1]:
                if race == "failure":
                    raise RuntimeError("primary failed")
                raise icechunk.ConflictError()
            original_commit(message, store, replicas, allow_rebase=allow_rebase)

        monkeypatch.setattr(
            virtual_region_job.storage, "commit_if_icechunk", failing_commit
        )
    caplog.clear()
    with caplog.at_level("INFO"):
        if race in ("failure", "exhausted"):
            with pytest.raises(
                RuntimeError if race == "failure" else icechunk.ConflictError
            ):
                job.process_virtual(primary, [replica], "main", coords)
        else:
            assert job.process_virtual(primary, [replica], "main", coords)
    events = pipeline_events(caplog)
    assert [event["event"] for event in events] == (
        ["batch", "primary_committed"] if race == "partial" else ["batch"]
    )
    assert events[0]["files"] == [c.get_url() for c in coords]
    assert _snapshot_count(replica) == 3
    if race == "partial":
        assert len(opened) == 2
        assert events[1]["files"] == [coords[1].get_url()]
        assert events[1]["ref_count"] == 1
        assert events[1]["resized"] is False
        assert events[0]["batch_id"] == events[1]["batch_id"]
        assert _snapshot_count(primary) == initial_primary + 1
    else:
        assert _snapshot_count(primary) == initial_primary
        assert len(opened) == {"all": 2, "failure": 1, "exhausted": 5}[race]


@pytest.mark.parametrize("empty", ["remaining", "unavailable", "skipped"])
def test_empty_sweeps_emit_no_batch_or_primary_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    empty: str,
) -> None:
    dataset = _make_dataset(tmp_path, n_inits=1)
    job = make_job().model_copy(update={"poll_deadline": pd.Timestamp.min})
    primary, _ = dataset.store_factory.icechunk_primary_and_replica_repos()
    template_utils.write_metadata(
        _create_template_ds(0),
        primary.writable_session("main").store,
        mode="w-",
        consolidated=False,
    )
    if empty == "unavailable":
        monkeypatch.setattr(BatchJob, "discover_available", lambda self, pending: [])
    elif empty == "skipped":
        monkeypatch.setattr(BatchJob, "file_refs", lambda self, coord, size: [])
    with caplog.at_level("INFO"):
        assert not job.process_virtual(
            primary,
            [],
            "main",
            [] if empty == "remaining" else job.source_file_coords(),
        )
    assert pipeline_events(caplog) == []
    assert _snapshot_count(primary) == 2
