from functools import partial
from pathlib import Path

import icechunk
import numpy as np
import pytest
import xarray as xr

from reformatters.common.virtual_region_job import VirtualRef
from tests.common.virtual_region_job_test import (
    APPEND_DIM_FREQ,
    APPEND_DIM_START,
    VirtualTestRegionJob,
    VirtualTestSourceFileCoord,
    _block_values,
    _queue_stale_sessions,
)
from tests.common.virtual_retention_test import _open, _setup


def _batch(
    job: VirtualTestRegionJob, step: int
) -> list[tuple[VirtualTestSourceFileCoord, list[VirtualRef]]]:
    candidate = job.model_copy(update={"region": slice(step, step + 1)})
    return [
        (coord, list(refs))
        for batch in candidate.process_virtual_refs(candidate.source_file_coords())
        for coord, refs in batch
    ]


@pytest.mark.parametrize("positions", [1, 12])
def test_append_after_trim_keeps_label_data_identity(
    tmp_path: Path, positions: int
) -> None:
    size = positions + 2
    repo, job, _ = _setup(tmp_path, positions=positions, size=size)
    job.trim_rolling_window(repo, "main")
    batch = _batch(job, size)
    job._commit_virtual_batch(repo, "main", batch, overwrite_chunks=False)
    ds = _open(repo).to_dataset()
    assert ds.sizes["init_time"] == positions + 1
    np.testing.assert_array_equal(ds.init_time, job.template_ds.init_time[2:])
    for i in range(positions + 1):
        for lead in range(2):
            np.testing.assert_array_equal(
                ds.temperature_2m[i, lead], _block_values(i + 2, lead)
            )
    store = repo.readonly_session("main").store
    mapper = job.for_store(store)
    assert mapper.rolling_positions is None
    assert mapper.region == slice(0, positions + 1)
    assert (
        mapper.template_ds.init_time.values[0] == APPEND_DIM_START + 2 * APPEND_DIM_FREQ
    )
    key = mapper.chunk_key(batch[0][0].out_loc(), job.data_vars[0])
    assert key == (positions, 0, 0, 0)
    assert job.read_ingested_through(store) == APPEND_DIM_START + size * APPEND_DIM_FREQ
    assert job.filter_already_present([coord for coord, _ in batch], store) == []
    snapshot = repo.lookup_branch("main")
    job._commit_virtual_batch(repo, "main", batch, overwrite_chunks=False)
    assert repo.lookup_branch("main") == snapshot


@pytest.mark.parametrize("positions", [1, 12])
@pytest.mark.parametrize("overwrite", [False, True])
def test_stale_candidates_do_not_restore_trimmed_labels(
    tmp_path: Path, positions: int, overwrite: bool
) -> None:
    repo, job, _ = _setup(tmp_path, positions=positions, size=positions + 2)
    stale = _batch(job, 0)
    job.trim_rolling_window(repo, "main")
    snapshot = repo.lookup_branch("main")
    coords = [coord for coord, _ in stale]
    assert (
        job.filter_already_present(coords, repo.readonly_session("main").store)
        == coords
    )
    job._commit_virtual_batch(repo, "main", stale, overwrite_chunks=overwrite)
    assert repo.lookup_branch("main") == snapshot


def test_empty_store_keeps_original_planning_tree(tmp_path: Path) -> None:
    repo, job, _ = _setup(tmp_path, size=0)
    mapper = job.for_store(repo.readonly_session("main").store)
    assert mapper.template_ds is job.template_ds
    assert mapper.region == slice(0, 1)
    assert mapper.rolling_positions is None


@pytest.mark.parametrize("append_first", [False, True])
def test_append_trim_conflict_requires_fresh_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, append_first: bool
) -> None:
    repo, job, _ = _setup(tmp_path)
    append = partial(
        job._commit_virtual_batch, repo, "main", _batch(job, 4), overwrite_chunks=False
    )
    trim = partial(job.trim_rolling_window, repo, "main")
    stale = repo.writable_session("main")
    winner, replay = (append, trim) if append_first else (trim, append)
    winner()
    opened = _queue_stale_sessions(monkeypatch, repo, [stale])
    snapshot = repo.lookup_branch("main")
    with pytest.raises(icechunk.ConflictError):
        replay()
    assert repo.lookup_branch("main") == snapshot
    replay()
    assert len(opened) == 2
    trim()
    ds = _open(repo).to_dataset()
    np.testing.assert_array_equal(ds.init_time, job.template_ds.init_time[2:])
    for i in range(3):
        for lead in range(2):
            np.testing.assert_array_equal(
                ds.temperature_2m[i, lead], _block_values(i + 2, lead)
            )
    np.testing.assert_array_equal(ds.ingested_forecast_length, [3, 4, np.nan])


def test_refresh_after_trim_preserves_coordinates_and_is_noop_when_synced(
    tmp_path: Path,
) -> None:
    repo, job, factory = _setup(tmp_path, grouped=True)
    job.trim_rolling_window(repo, "main")
    before = _open(repo).load()
    job.refresh_metadata(factory, tmp_path / "refresh.zarr")
    xr.testing.assert_equal(_open(repo), before)
    snapshot = repo.lookup_branch("main")
    job.refresh_metadata(factory, tmp_path / "refresh.zarr")
    assert repo.lookup_branch("main") == snapshot
