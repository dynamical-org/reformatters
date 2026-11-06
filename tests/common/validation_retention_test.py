from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pandas as pd
import pytest
import zarr
import zarr.core.sync

from reformatters.common import validation
from reformatters.common.config_models import DataVar
from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim
from tests.common.virtual_region_job_test import (
    VirtualTestRegionJob,
    VirtualTestSourceFileCoord,
    _make_region_job,
)
from tests.common.virtual_retention_test import (
    FREQ,
    START,
    RetentionConfig,
    _repo,
)


def _rolling_job(config: RetentionConfig) -> VirtualTestRegionJob:
    return _make_region_job(
        config.get_template(START + 4 * FREQ), region=slice(3, 4)
    ).model_copy(update={"rolling_positions": 3, "retention_template_config": config})


def _context(
    repo: icechunk.Repository, job: VirtualTestRegionJob
) -> validation.ValidationContext:
    store = repo.readonly_session("main").store
    return validation.ValidationContext(
        store=store,
        ds=validation.open_flattened_dataset(store, consolidated=False),
        append_dim="init_time",
        append_dim_frequency=FREQ,
        data_vars=job.data_vars,
        region_job=job,
    )


def _trim(
    repo: icechunk.Repository,
    config: RetentionConfig,
    tmp_path: Path,
    positions: int = 3,
) -> None:
    job = _rolling_job(config).model_copy(update={"rolling_positions": positions})
    job.trim_retention(repo, "main")


@pytest.mark.parametrize("oldest_missing", [False, True])
def test_completeness_covers_retained_window_beyond_hot_region(
    tmp_path: Path, oldest_missing: bool
) -> None:
    config = RetentionConfig()
    repo, _ = _repo(tmp_path, config)
    _trim(repo, config, tmp_path)
    if oldest_missing:
        session = repo.writable_session("main")
        zarr.core.sync.sync(session.store.delete("temperature_2m/c/0/0/0/0"))
        session.commit("Remove oldest retained reference")
    job = _rolling_job(config)
    context = _context(repo, job)

    mapper = context.virtual_region_job()
    assert mapper.region == slice(0, 3)
    assert mapper.template_ds.to_dataset().get_index("init_time")[0] == START + FREQ
    assert job.region == slice(3, 4)
    result = validation.CheckVirtualManifestCompleteness().check(context)

    assert result.passed is not oldest_missing, result.message
    if oldest_missing:
        assert str(START + FREQ) in result.message
    else:
        assert "All 3 ingested" in result.message


@pytest.mark.parametrize(
    "check",
    [
        validation.CheckVirtualManifestCompleteness(),
        validation.CheckVirtualDecodeHealth(),
    ],
)
def test_candidates_are_clipped_at_both_snapshot_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check: validation.Validator
) -> None:
    config = RetentionConfig()
    repo, _ = _repo(tmp_path, config)
    _trim(repo, config, tmp_path)
    candidates = [
        VirtualTestSourceFileCoord(
            init_time=START + i * FREQ, lead_time=pd.Timedelta(0)
        )
        for i in range(5)
    ]
    probed: list[pd.Timestamp] = []

    def source_file_coords(
        self: VirtualTestRegionJob,
    ) -> Sequence[VirtualTestSourceFileCoord]:
        return candidates

    def filter_already_present(
        self: VirtualTestRegionJob,
        coords: Sequence[VirtualTestSourceFileCoord],
        store: object,
    ) -> list[VirtualTestSourceFileCoord]:
        probed.extend(c.init_time for c in coords)
        return []

    monkeypatch.setattr(VirtualTestRegionJob, "source_file_coords", source_file_coords)
    monkeypatch.setattr(
        VirtualTestRegionJob, "filter_already_present", filter_already_present
    )

    result = check.check(_context(repo, _rolling_job(config)))

    assert result.passed, result.message
    assert probed == [START + i * FREQ for i in range(1, 4)]


def test_oldest_decode_uses_snapshot_chunk_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RetentionConfig()
    repo, source = _repo(tmp_path, config, size=9)
    _trim(repo, config, tmp_path, positions=8)
    source.write_bytes(np.full(9 * 2 * 2 * 3, np.nan, dtype="float64").tobytes())
    keys: list[tuple[int, ...]] = []
    original = VirtualTestRegionJob.chunk_key

    def chunk_key(
        self: VirtualTestRegionJob,
        out_loc: Mapping[Dim, CoordinateValue],
        var: DataVar[Any],
    ) -> tuple[int, ...] | None:
        key = original(self, out_loc, var)
        if key is not None:
            keys.append(key)
        return key

    monkeypatch.setattr(VirtualTestRegionJob, "chunk_key", chunk_key)
    context = _context(repo, _rolling_job(config))
    result = validation.CheckVirtualDecodeHealth(
        max_workers=1, allow_all_nan_vars=("temperature_2m",)
    ).check(context)

    assert result.passed, result.message
    assert result.checked_count == 4
    assert {key[0] for key in keys} == {0, 7}
    for label in context.ds.get_index("init_time")[[0, 7]]:
        assert str(label) in result.message


@pytest.mark.parametrize(
    "check",
    [
        validation.CheckCurrentData(max_delay=pd.Timedelta("2h")),
        validation.CheckRecentNans(),
        validation.CheckVirtualManifestCompleteness(),
        validation.CheckVirtualDecodeHealth(),
    ],
)
def test_empty_retained_store_fails_without_indexing(
    tmp_path: Path, check: validation.Validator
) -> None:
    config = RetentionConfig()
    repo, _ = _repo(tmp_path, config, size=0)

    result = check.check(_context(repo, _rolling_job(config)))

    assert not result.passed
    assert "position" in result.message or "No source files" in result.message


def test_replica_different_origins_fail_coordinate_check(tmp_path: Path) -> None:
    config = RetentionConfig()
    repo, _ = _repo(tmp_path, config)
    job = _rolling_job(config)
    before = _context(repo, job).ds
    _trim(repo, config, tmp_path)
    context = _context(repo, job)
    result = validation.CheckReplicaMatchesPrimary().check(
        validation.ValidationContext(
            store=context.store,
            ds=context.ds,
            append_dim="init_time",
            region_job=job,
            primary_ds=before,
        )
    )
    assert not result.passed
    assert "different for coords" in result.message
