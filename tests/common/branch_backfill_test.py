import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr.errors

from reformatters.common import storage, template_utils
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.zarr import copy_zarr_metadata
from tests.common import dynamical_dataset_test, parallel_writes_test
from tests.common.dynamical_dataset_test import (
    ExampleDataset,
    _existing_store_tree,
)
from tests.common.parallel_writes_test import (
    ParallelDataset,
    _create_template_ds,
)
from tests.common.parallel_writes_test import (
    TestIcechunkParallelWrites as IcechunkFixture,
)

kubernetes_backfill_dataset = dynamical_dataset_test.kubernetes_backfill_dataset
_use_thread_pool_for_shard_writes = (
    parallel_writes_test._use_thread_pool_for_shard_writes
)


@pytest.fixture
def branch_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParallelDataset:
    monkeypatch.setattr(
        ParallelDataset, "_get_template", lambda self, end: _create_template_ds()
    )
    return IcechunkFixture()._make_dataset(tmp_path)


def _run_worker(
    dataset: ParallelDataset, worker_index: int, *, overwrite_chunks: bool = True
) -> None:
    dataset.backfill(
        pd.Timestamp("2025-01-01T04:00"),
        "branch-job",
        worker_index=worker_index,
        workers_total=2,
        overwrite_chunks=overwrite_chunks,
    )


@pytest.mark.slow
def test_branch_workers_and_retries_never_publish_to_main(
    branch_dataset: ParallelDataset,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    IcechunkFixture()._init_store(branch_dataset)
    ((_, repo),) = branch_dataset.store_factory.icechunk_repos(sort="primary-first")
    original = repo.lookup_branch("main")
    repo.create_branch("candidate", original)
    monkeypatch.setenv("REFORMATTERS_ICECHUNK_BRANCH", "candidate")
    # Retain ready/results as if the final worker died before coordination cleanup.
    monkeypatch.setattr(
        StoreFactory, "clear_coordination_files", lambda self, name: None
    )
    for worker in (0, 0):
        _run_worker(branch_dataset, worker)
        assert repo.lookup_branch("main") == original
    session = repo.writable_session("main")
    expanded = _create_template_ds(num_time=6)
    template_utils.write_metadata(expanded, branch_dataset._tmp_store())
    copy_zarr_metadata(expanded, branch_dataset._tmp_store(), session.store)
    advanced_main = session.commit("Independent main update")
    for _ in range(2):
        _run_worker(branch_dataset, 1)
        assert repo.lookup_branch("main") == advanced_main
    assert set(repo.list_branches()) == {"main", "candidate"}
    assert "BACKFILL WORKER 0 TARGET BRANCH: candidate" in caplog.text
    assert "BACKFILL WORKER 1 TARGET BRANCH: candidate" in caplog.text
    existing = branch_dataset._open_existing_store()
    assert existing is not None
    _, end = branch_dataset._resolve_backfill_template(
        existing, None, overwrite_chunks=True, overwrite_metadata=False
    )
    assert end == pd.Timestamp("2025-01-01T04:00")
    target = xr.open_zarr(
        branch_dataset.store_factory.primary_store(), consolidated=False
    )
    published = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    assert target.sizes["time"] == 4
    assert published.sizes["time"] == 6
    assert np.all(target.var0.values == 1)
    assert np.all(np.isnan(published.var0.values))


@pytest.mark.parametrize("worker_index", [0, 1])
@pytest.mark.parametrize("existing_repo", [False, True])
def test_missing_repository_or_branch_is_never_created(
    branch_dataset: ParallelDataset,
    monkeypatch: pytest.MonkeyPatch,
    worker_index: int,
    existing_repo: bool,
) -> None:
    if existing_repo:
        IcechunkFixture()._init_store(branch_dataset)
    ((_, _, location),) = branch_dataset.store_factory._icechunk_storages()
    monkeypatch.setenv("REFORMATTERS_ICECHUNK_BRANCH", "missing")
    with pytest.raises((icechunk.NotFoundError, ValueError)):
        _run_worker(branch_dataset, worker_index)
    with pytest.raises((icechunk.NotFoundError, ValueError)):
        branch_dataset.store_factory.primary_store(writable=True)
    assert icechunk.Repository.exists(location) == existing_repo
    if existing_repo:
        assert set(icechunk.Repository.open(location).list_branches()) == {"main"}


@pytest.mark.parametrize("branch", ["", "main", "_job_other", " candidate "])
def test_invalid_branch_override(monkeypatch: pytest.MonkeyPatch, branch: str) -> None:
    monkeypatch.setenv("REFORMATTERS_ICECHUNK_BRANCH", branch)
    with pytest.raises(AssertionError, match=r"[Bb]ranch override"):
        storage.icechunk_branch()


@pytest.mark.parametrize(
    "name", ["REFORMATTERS_ICECHUNK_BRNACH", "REFORMATTERS_BRANCH"]
)
def test_branch_environment_typo_is_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "candidate")
    with pytest.raises(AssertionError, match="Unknown Icechunk branch"):
        storage.icechunk_branch()


@pytest.mark.parametrize("operation", ["update", "local", "new-store"])
def test_branch_override_rejects_unsupported_operations(
    branch_dataset: ParallelDataset,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setenv("REFORMATTERS_ICECHUNK_BRANCH", "candidate")
    actions = {
        "update": lambda: branch_dataset.update("job"),
        "local": lambda: branch_dataset.backfill_local(
            pd.Timestamp("2025-01-01T04:00")
        ),
        "new-store": lambda: _run_worker(branch_dataset, 0, overwrite_chunks=False),
    }
    with pytest.raises(AssertionError):
        actions[operation]()
    ((_, _, location),) = branch_dataset.store_factory._icechunk_storages()
    assert not icechunk.Repository.exists(location)


@pytest.mark.parametrize("replicas", [False, True])
def test_branch_override_rejects_zarr_and_configured_replicas(
    monkeypatch: pytest.MonkeyPatch,
    replicas: bool,
) -> None:
    monkeypatch.setenv("REFORMATTERS_ICECHUNK_BRANCH", "candidate")
    config = StorageConfig(
        base_path="unused",
        format=DatasetFormat.ICECHUNK if replicas else DatasetFormat.ZARR3,
    )
    factory = StoreFactory(
        primary_storage_config=config,
        replica_storage_configs=[config] if replicas else [],
        dataset_id="test",
        template_config_version="1",
    )
    with pytest.raises(AssertionError, match="Branch backfills"):
        factory.icechunk_repos(sort="primary-first")


@pytest.mark.parametrize("branch", [None, "candidate"])
def test_driver_logs_summarizes_and_propagates_resolved_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kubernetes_backfill_dataset: ExampleDataset,
    branch: str | None,
) -> None:
    if branch is not None:
        monkeypatch.setenv("REFORMATTERS_ICECHUNK_BRANCH", branch)
    monkeypatch.setattr(
        ExampleDataset, "_open_existing_store", lambda self: _existing_store_tree()
    )
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    log = Mock()
    monkeypatch.setattr("reformatters.common.dynamical_dataset.log", log)
    mock_run = Mock()
    monkeypatch.setattr(subprocess, "run", mock_run)
    kubernetes_backfill_dataset.backfill_kubernetes(overwrite_chunks=True)
    manifest = json.loads(mock_run.call_args.kwargs["input"])
    env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    selected = [
        item["value"] for item in env if item["name"] == "REFORMATTERS_ICECHUNK_BRANCH"
    ]
    assert selected == ([branch] if branch else [])
    assert f"`{branch or 'main'}`" in summary.read_text()
    log.info.assert_any_call("BACKFILL DRIVER TARGET BRANCH: %s", branch or "main")


def test_branch_driver_rejects_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
    kubernetes_backfill_dataset: ExampleDataset,
) -> None:
    monkeypatch.setenv("REFORMATTERS_ICECHUNK_BRANCH", "candidate")
    mock_run = Mock()
    monkeypatch.setattr(subprocess, "run", mock_run)
    with pytest.raises(AssertionError, match="require --overwrite-chunks"):
        kubernetes_backfill_dataset.backfill_kubernetes(overwrite_metadata=True)
    mock_run.assert_not_called()


def test_branch_requires_existing_dataset(
    branch_dataset: ParallelDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    ((_, repo),) = branch_dataset.store_factory.icechunk_repos(sort="primary-first")
    original = repo.lookup_branch("main")
    repo.create_branch("candidate", original)
    monkeypatch.setenv("REFORMATTERS_ICECHUNK_BRANCH", "candidate")
    with pytest.raises(zarr.errors.GroupNotFoundError):
        _run_worker(branch_dataset, 0)
    assert repo.lookup_branch("candidate") == original
    assert repo.lookup_branch("main") == original
