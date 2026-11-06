import asyncio
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import StoreLike

from reformatters.common import storage
from reformatters.common.retry import retry
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.common.virtual_retention import retention_start
from tests.common.virtual_region_job_test import (
    BLOCK_NBYTES,
    VirtualTestDataVar,
    VirtualTestRegionJob,
    VirtualTestSourceFileCoord,
)
from tests.common.virtual_retention_test import FREQ, START, RetentionConfig, _repo


def _job(
    config: RetentionConfig, tmp_path: Path, *, positions: int | None = 3
) -> VirtualTestRegionJob:
    return VirtualTestRegionJob(
        tmp_store=tmp_path / "metadata.zarr",
        template_ds=config.get_template(START + 6 * FREQ),
        data_vars=[VirtualTestDataVar(name="temperature_2m")],
        append_dim="init_time",
        region=slice(0, 6),
        reformat_job_name="rolling-test",
        processing_mode="update",
        rolling_positions=positions,
        retention_template_config=config if positions is not None else None,
    )


def _file(
    job: VirtualTestRegionJob, source: Path, step: int, *, lead: int = 0
) -> tuple[VirtualTestSourceFileCoord, list[VirtualRef]]:
    coord = VirtualTestSourceFileCoord(
        init_time=START + step * FREQ, lead_time=lead * FREQ
    )
    return coord, [
        VirtualRef(
            job.data_vars[0],
            coord.out_loc(),
            source.as_uri(),
            (step * 2 + lead) * BLOCK_NBYTES,
            BLOCK_NBYTES,
        )
    ]


def _extend_source(source: Path) -> None:
    source.write_bytes(
        source.read_bytes()
        + b"".join(
            np.full((2, 3), i * 10 + lead, dtype="float64").tobytes()
            for i in (4, 5)
            for lead in range(2)
        )
    )


def _values(repo: icechunk.Repository) -> xr.Dataset:
    return xr.open_zarr(
        repo.readonly_session("main").store,
        chunks=None,
        decode_timedelta=True,
        consolidated=False,
    )


@pytest.mark.parametrize("positions", [1, 3, 12])
def test_append_does_not_trim_and_explicit_trim_maps_by_committed_origin(
    tmp_path: Path, positions: int
) -> None:
    config = RetentionConfig()
    repo, source = _repo(tmp_path, config)
    _extend_source(source)
    job = _job(config, tmp_path, positions=positions)
    job._commit_virtual_batch(
        repo, "main", [_file(job, source, 4)], overwrite_chunks=False
    )
    assert _values(repo).sizes["init_time"] == 5
    assert retention_start(repo.readonly_session("main").store, config) == START
    job.trim_retention(repo, "main")
    kept = min(positions, 5)
    ds = _values(repo)
    np.testing.assert_array_equal(
        ds.temperature_2m.isel(lead_time=0)[:, 0, 0], np.arange(5 - kept, 5) * 10
    )
    assert ds.sizes["init_time"] == kept
    coord, refs = _file(job, source, 4)
    store = repo.readonly_session("main").store
    assert job.filter_already_present([coord], store) == []
    assert job.for_store(store).chunk_key(coord.out_loc(), job.data_vars[0]) == (
        kept - 1,
        0,
        0,
        0,
    )
    assert job.read_ingested_through(store) == coord.init_time
    snapshot = repo.lookup_branch("main")
    job._commit_virtual_batch(repo, "main", [(coord, refs)], overwrite_chunks=False)
    job.trim_retention(repo, "main")
    assert repo.lookup_branch("main") == snapshot


@pytest.mark.parametrize("overwrite", [False, True])
def test_stale_batch_is_dropped_without_reintroducing_positions(
    tmp_path: Path, overwrite: bool
) -> None:
    config = RetentionConfig()
    repo, source = _repo(tmp_path, config)
    job = _job(config, tmp_path, positions=1)
    job.trim_retention(repo, "main")
    snapshot = repo.lookup_branch("main")
    stale = _file(job, source, 0)
    job._commit_virtual_batch(repo, "main", [stale], overwrite_chunks=overwrite)
    assert repo.lookup_branch("main") == snapshot
    assert (
        job.filter_already_present([stale[0]], repo.readonly_session("main").store)
        == []
    )
    assert not job.process_virtual(
        repo, [], "main", [stale[0]], overwrite_chunks=overwrite
    )
    assert repo.lookup_branch("main") == snapshot


@pytest.mark.parametrize("overwrite", [False, True])
def test_worker_drops_stale_at_entry_and_never_trims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool
) -> None:
    config = RetentionConfig()
    repo, _ = _repo(tmp_path, config)
    job = _job(config, tmp_path, positions=1).model_copy(update={"region": slice(0, 4)})
    job.trim_retention(repo, "main")
    snapshot = repo.lookup_branch("main")
    called = []

    def process(
        self: VirtualTestRegionJob,
        primary_repo: icechunk.Repository,
        replica_repos: Sequence[icechunk.Repository],
        branch: str,
        remaining: Sequence[VirtualTestSourceFileCoord],
        *,
        overwrite_chunks: bool = False,
    ) -> bool:
        called.extend(remaining)
        return True

    monkeypatch.setattr(VirtualTestRegionJob, "process_virtual", process)
    factory = Mock(spec=storage.StoreFactory)
    factory.icechunk_primary_and_replica_repos.return_value = (repo, [])
    completed = job.process_virtual_worker_jobs(
        [job], factory, "main", overwrite_chunks=overwrite
    )
    assert completed is overwrite
    assert all(coord.init_time == START + 3 * FREQ for coord in called)
    assert len(called) == (2 if overwrite else 0)
    assert repo.lookup_branch("main") == snapshot


@pytest.mark.parametrize("append_first", [False, True])
def test_conflicting_append_and_trim_replay_from_fresh_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, append_first: bool
) -> None:
    config = RetentionConfig(grouped=True, ordered_axis=True)
    repo, source = _repo(tmp_path, config)
    _extend_source(source)
    job = _job(config, tmp_path)
    coord, root_refs = _file(job, source, 4)
    group_var = VirtualTestDataVar(name="temperature_2m", group="pressure_level")
    refs = [*root_refs, root_refs[0]._replace(data_var=group_var)]
    original_commit = storage.commit_if_icechunk
    attempts = 0

    def competing_commit(
        message: str,
        primary_store: StoreLike,
        replica_stores: Sequence[Store],
        *,
        allow_rebase: bool = True,
    ) -> None:
        nonlocal attempts
        assert not allow_rebase
        attempts += 1
        if attempts == 1:
            with monkeypatch.context() as patch:
                patch.setattr(storage, "commit_if_icechunk", original_commit)
                if append_first:
                    job._commit_virtual_batch(
                        repo, "main", [(coord, refs)], overwrite_chunks=False
                    )
                else:
                    job.trim_retention(repo, "main")
        original_commit(
            message, primary_store, replica_stores, allow_rebase=allow_rebase
        )

    monkeypatch.setattr(storage, "commit_if_icechunk", competing_commit)
    action = (
        partial(job.trim_retention, repo, "main")
        if append_first
        else partial(
            job._commit_virtual_batch,
            repo,
            "main",
            [(coord, refs)],
            overwrite_chunks=False,
        )
    )
    retry(action, max_attempts=5, retryable_exceptions=(icechunk.ConflictError,))
    assert attempts == 2
    monkeypatch.setattr(storage, "commit_if_icechunk", original_commit)
    job.trim_retention(repo, "main")
    for group in (None, "pressure_level"):
        ds = xr.open_zarr(
            repo.readonly_session("main").store,
            group=group,
            chunks=None,
            decode_timedelta=True,
            consolidated=False,
        )
        np.testing.assert_array_equal(
            ds.init_time, pd.date_range(START + 2 * FREQ, periods=3, freq=FREQ)
        )
        np.testing.assert_array_equal(
            ds.temperature_2m.isel(lead_time=0, latitude=0, longitude=0), [20, 30, 40]
        )
        np.testing.assert_array_equal(ds.ingested_forecast_length, [3, 4, np.nan])


@pytest.mark.parametrize("all_stale", [False, True])
def test_stale_drops_count_as_handled_without_false_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_stale: bool
) -> None:
    config = RetentionConfig()
    repo, source = _repo(tmp_path, config)
    job = _job(config, tmp_path, positions=1)
    stale = _file(job, source, 0)
    current = _file(job, source, 3)
    batches = [stale] if all_stale else [stale, current]
    monkeypatch.setattr(
        VirtualTestRegionJob,
        "process_virtual_refs",
        lambda self, remaining: iter([batches]),
    )
    job.trim_retention(repo, "main")
    assert job.process_virtual(repo, [], "main", [coord for coord, _ in batches]) is (
        not all_stale
    )
    assert _values(repo).sizes["init_time"] == 1


@pytest.mark.parametrize("rolling", [False, True])
def test_backfill_appends_without_cleanup(tmp_path: Path, rolling: bool) -> None:
    config = RetentionConfig()
    repo, source = _repo(tmp_path, config)
    _extend_source(source)
    job = _job(config, tmp_path, positions=1 if rolling else None).model_copy(
        update={"processing_mode": "backfill"}
    )
    repo.create_branch("backfill", repo.lookup_branch("main"))
    job._commit_virtual_batch(
        repo, "backfill", [_file(job, source, 4)], overwrite_chunks=False
    )
    store = repo.readonly_session("backfill").store
    assert retention_start(store, config) == START
    ds = xr.open_zarr(store, chunks=None, decode_timedelta=True, consolidated=False)
    np.testing.assert_array_equal(
        ds.temperature_2m.isel(lead_time=0)[:, 0, 0], [0, 10, 20, 30, 40]
    )


def test_archival_mapper_and_rebase_behavior_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RetentionConfig()
    repo, source = _repo(tmp_path, config)
    job = _job(config, tmp_path, positions=None)
    assert job.for_store(repo.readonly_session("main").store) is job
    original = storage.commit_if_icechunk
    rebases = []

    def record(
        message: str,
        primary_store: StoreLike,
        replica_stores: Sequence[Store],
        *,
        allow_rebase: bool = True,
    ) -> None:
        rebases.append(allow_rebase)
        original(message, primary_store, replica_stores, allow_rebase=allow_rebase)

    monkeypatch.setattr(storage, "commit_if_icechunk", record)
    job._commit_virtual_batch(
        repo, "main", [_file(job, source, 3)], overwrite_chunks=True
    )
    _extend_source(source)
    job._commit_virtual_batch(
        repo, "main", [_file(job, source, 4)], overwrite_chunks=False
    )
    assert rebases == [True, False]
    assert _values(repo).sizes["init_time"] == 5


def test_refresh_uses_committed_origin_and_preserves_runtime_coordinates(
    tmp_path: Path,
) -> None:
    config = RetentionConfig(grouped=True)
    repo, _ = _repo(tmp_path, config)
    job = _job(config, tmp_path)
    job.trim_retention(repo, "main")
    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="r+").attrs["test"] = "metadata fix"
    session.commit("Stale attribute")
    factory = Mock(spec=storage.StoreFactory)
    factory.icechunk_primary_and_replica_repos.return_value = (repo, [])
    job.refresh_metadata(factory, tmp_path / "refresh.zarr")
    for group in (None, "pressure_level"):
        ds = xr.open_zarr(
            repo.readonly_session("main").store,
            group=group,
            chunks=None,
            decode_timedelta=True,
            consolidated=False,
        )
        np.testing.assert_array_equal(
            ds.init_time, pd.date_range(START + FREQ, periods=3, freq=FREQ)
        )
        np.testing.assert_array_equal(ds.ingested_forecast_length, [2, 3, 4])
        np.testing.assert_array_equal(
            ds.valid_time, ds.init_time.values[:, None] + ds.lead_time.values[None, :]
        )
    snapshot = repo.lookup_branch("main")
    job.refresh_metadata(factory, tmp_path / "refresh.zarr")
    assert repo.lookup_branch("main") == snapshot


@pytest.mark.parametrize("stale_source", [False, True])
def test_mismatched_reference_label_fails_before_stale_filter(
    tmp_path: Path, stale_source: bool
) -> None:
    config = RetentionConfig()
    repo, source = _repo(tmp_path, config)
    job = _job(config, tmp_path, positions=1)
    job.trim_retention(repo, "main")
    stale, stale_refs = _file(job, source, 0)
    current, current_refs = _file(job, source, 3)
    batch = [(stale, current_refs)] if stale_source else [(current, stale_refs)]
    snapshot = repo.lookup_branch("main")
    with pytest.raises(AssertionError, match="Reference append labels"):
        job._commit_virtual_batch(repo, "main", batch, overwrite_chunks=True)
    assert repo.lookup_branch("main") == snapshot


def test_archival_noop_and_refresh_preserve_serialized_metadata_and_coordinates(
    tmp_path: Path,
) -> None:
    config = RetentionConfig(grouped=True)
    repo, _ = _repo(tmp_path, config)
    job = _job(config, tmp_path, positions=None).model_copy(
        update={"region": slice(0, 4)}
    )
    factory = Mock(spec=storage.StoreFactory)
    factory.icechunk_primary_and_replica_repos.return_value = (repo, [])
    factory.primary_store.side_effect = lambda **kwargs: (
        repo.writable_session("main").store
    )
    factory.replica_stores.return_value = []

    async def serialized_state() -> dict[str, bytes]:
        store = repo.readonly_session("main").store
        result = {}
        async for key in store.list():
            if "temperature_2m/c/" in key:
                continue
            value = await store.get(key, default_buffer_prototype())
            assert value is not None
            result[key] = value.to_bytes()
        return result

    snapshot = repo.lookup_branch("main")
    before = asyncio.run(serialized_state())
    assert any(key.endswith("/zarr.json") for key in before)
    assert "init_time/c/0" in before
    assert not job.process_virtual_worker_jobs(
        [job], factory, "main", overwrite_chunks=False
    )
    job.refresh_metadata(factory, tmp_path / "refresh.zarr")
    assert repo.lookup_branch("main") == snapshot
    assert asyncio.run(serialized_state()) == before
