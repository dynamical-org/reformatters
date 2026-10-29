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

from reformatters.common import template_utils
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.zarr import copy_zarr_metadata
from tests.common import parallel_writes_test
from tests.common.dynamical_dataset_test import (
    ExampleDatasetStorageConfig,
    ExampleVirtualDataset,
    _existing_store_tree,
)
from tests.common.parallel_writes_test import ParallelDataset, _create_template_ds
from tests.common.parallel_writes_test import (
    TestIcechunkParallelWrites as IcechunkFixture,
)

_use_thread_pool_for_shard_writes = (
    parallel_writes_test._use_thread_pool_for_shard_writes
)


@pytest.fixture
def branch_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParallelDataset:
    monkeypatch.setattr(
        ParallelDataset,
        "_get_template",
        lambda self, end: _create_template_ds(
            num_time=int(
                (pd.Timestamp(end) - pd.Timestamp("2025-01-01")).total_seconds() / 3600
            )
        ),
    )
    return IcechunkFixture()._make_dataset(tmp_path)


@pytest.fixture
def submit_job(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(ParallelDataset, "_can_run_in_kubernetes", lambda self: True)
    monkeypatch.setattr(
        ExampleVirtualDataset, "_can_run_in_kubernetes", lambda self: True
    )
    monkeypatch.setattr(
        ExampleVirtualDataset, "_get_template", lambda self, end: _existing_store_tree()
    )
    submit = Mock()
    monkeypatch.setattr(subprocess, "run", submit)
    return submit


def _run_worker(
    dataset: ParallelDataset | ExampleVirtualDataset,
    worker_index: int,
    *,
    workers_total: int = 2,
    keep_branch: str | None = "candidate",
    overwrite_chunks: bool = True,
    overwrite_metadata: bool = False,
    end_hour: int = 4,
) -> None:
    dataset.backfill(
        pd.Timestamp("2025-01-01") + pd.Timedelta(hours=end_hour),
        "branch-job",
        worker_index=worker_index,
        workers_total=workers_total,
        overwrite_chunks=overwrite_chunks,
        overwrite_metadata=overwrite_metadata,
        keep_branch=keep_branch,
    )


def _assert_complete_candidate(dataset: ParallelDataset, num_time: int = 4) -> None:
    candidate = xr.open_zarr(
        dataset.store_factory.primary_store(branch="candidate"), consolidated=False
    )
    assert candidate.sizes["time"] == num_time
    np.testing.assert_array_equal(
        candidate.time.values,
        pd.date_range("2025-01-01", periods=num_time, freq="h").values,
    )
    assert set(candidate.data_vars) == {"var0", "var1", "var2", "var3"}
    for variable in candidate.data_vars.values():
        np.testing.assert_array_equal(variable.values, np.ones(variable.shape))


def _start_backfill(
    dataset: ParallelDataset | ExampleVirtualDataset,
    *,
    driver: bool,
    keep_branch: str = "candidate",
    overwrite_chunks: bool = True,
    overwrite_metadata: bool = False,
) -> None:
    if driver:
        dataset.backfill_kubernetes(
            overwrite_chunks=overwrite_chunks,
            overwrite_metadata=overwrite_metadata,
            keep_branch=keep_branch,
            docker_image="test-image",
        )
    else:
        _run_worker(
            dataset,
            0,
            keep_branch=keep_branch,
            overwrite_chunks=overwrite_chunks,
            overwrite_metadata=overwrite_metadata,
        )


@pytest.mark.slow
@pytest.mark.parametrize("advance_main", [False, True])
def test_retained_branch_is_complete_and_isolated_from_main(
    branch_dataset: ParallelDataset, advance_main: bool
) -> None:
    IcechunkFixture()._init_store(branch_dataset)
    ((_, repo),) = branch_dataset.store_factory.icechunk_repos(sort="primary-first")
    original = repo.lookup_branch("main")
    main_before = xr.open_zarr(branch_dataset.store_factory.primary_store()).load()

    _run_worker(branch_dataset, 0, workers_total=3, end_hour=6, overwrite_metadata=True)
    assert set(repo.list_branches()) == {"main", "candidate"}
    assert repo.lookup_branch("main") == original
    assert original in {snapshot.id for snapshot in repo.ancestry(branch="candidate")}
    xr.testing.assert_identical(
        xr.open_zarr(branch_dataset.store_factory.primary_store()), main_before
    )

    if advance_main:
        session = repo.writable_session("main")
        expanded = _create_template_ds(num_time=8)
        expanded.attrs["independent_update"] = "published"
        template_utils.write_metadata(expanded, branch_dataset._tmp_store())
        copy_zarr_metadata(expanded, branch_dataset._tmp_store(), session.store)
        original = session.commit("Independent main update")
        main_before = xr.open_zarr(branch_dataset.store_factory.primary_store()).load()

    for worker in (1, 1, 2):
        _run_worker(
            branch_dataset, worker, workers_total=3, end_hour=6, overwrite_metadata=True
        )
        assert repo.lookup_branch("main") == original
        xr.testing.assert_identical(
            xr.open_zarr(branch_dataset.store_factory.primary_store()), main_before
        )

    _assert_complete_candidate(branch_dataset, num_time=6)
    assert set(repo.list_branches()) == {"main", "candidate"}
    assert branch_dataset.store_factory.list_coordination_files("branch-job", "") == []


@pytest.mark.slow
@pytest.mark.parametrize("worker_zero_retry", [False, True])
def test_existing_candidate_fails_without_reusing_or_mutating_it(
    branch_dataset: ParallelDataset, worker_zero_retry: bool
) -> None:
    IcechunkFixture()._init_store(branch_dataset)
    ((_, repo),) = branch_dataset.store_factory.icechunk_repos(sort="primary-first")
    main_snapshot = repo.lookup_branch("main")
    if worker_zero_retry:
        _run_worker(branch_dataset, 0)
    else:
        repo.create_branch("candidate", main_snapshot)
    candidate_snapshot = repo.lookup_branch("candidate")
    coordination = branch_dataset.store_factory.list_coordination_files(
        "branch-job", ""
    )

    with pytest.raises(icechunk.AlreadyExistsError, match="branch already exists"):
        _run_worker(branch_dataset, 0)

    assert repo.lookup_branch("main") == main_snapshot
    assert repo.lookup_branch("candidate") == candidate_snapshot
    assert set(repo.list_branches()) == {"main", "candidate"}
    assert (
        branch_dataset.store_factory.list_coordination_files("branch-job", "")
        == coordination
    )


@pytest.mark.slow
def test_final_worker_retries_after_metadata_commit_before_cleanup(
    branch_dataset: ParallelDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    IcechunkFixture()._init_store(branch_dataset)
    ((_, repo),) = branch_dataset.store_factory.icechunk_repos(sort="primary-first")
    original = repo.lookup_branch("main")
    _run_worker(branch_dataset, 0)

    with monkeypatch.context() as attempt:
        attempt.setattr(
            StoreFactory,
            "clear_coordination_files",
            Mock(side_effect=RuntimeError("cleanup interrupted")),
        )
        with pytest.raises(RuntimeError, match="cleanup interrupted"):
            _run_worker(branch_dataset, 1)

    _assert_complete_candidate(branch_dataset)
    assert repo.lookup_branch("main") == original
    assert branch_dataset.store_factory.list_coordination_files("branch-job", "setup")
    _run_worker(branch_dataset, 1)

    _assert_complete_candidate(branch_dataset)
    assert repo.lookup_branch("main") == original
    assert set(repo.list_branches()) == {"main", "candidate"}
    assert branch_dataset.store_factory.list_coordination_files("branch-job", "") == []


@pytest.mark.parametrize("existing_repo", [False, True])
@pytest.mark.parametrize("driver", [False, True])
def test_missing_repository_or_dataset_is_never_created(
    branch_dataset: ParallelDataset,
    submit_job: Mock,
    existing_repo: bool,
    driver: bool,
) -> None:
    ((_, _, location),) = branch_dataset.store_factory._icechunk_storages()
    repo = icechunk.Repository.create(location) if existing_repo else None
    original = repo.lookup_branch("main") if repo is not None else None

    with pytest.raises(
        (AssertionError, icechunk.NotFoundError, zarr.errors.GroupNotFoundError)
    ):
        _start_backfill(branch_dataset, driver=driver)

    submit_job.assert_not_called()
    assert icechunk.Repository.exists(location) == existing_repo
    if repo is not None:
        assert set(repo.list_branches()) == {"main"}
        assert repo.lookup_branch("main") == original
        with pytest.raises(zarr.errors.GroupNotFoundError):
            xr.open_zarr(repo.readonly_session("main").store)


@pytest.mark.parametrize("driver", [False, True])
@pytest.mark.parametrize(
    "branch", ["", " ", "main", "_job_other", " candidate", "candidate\t"]
)
def test_invalid_branch_names_are_rejected_before_writes(
    branch_dataset: ParallelDataset,
    submit_job: Mock,
    driver: bool,
    branch: str,
) -> None:
    with pytest.raises(AssertionError, match="keep-branch"):
        _start_backfill(branch_dataset, driver=driver, keep_branch=branch)

    submit_job.assert_not_called()
    ((_, _, location),) = branch_dataset.store_factory._icechunk_storages()
    assert not icechunk.Repository.exists(location)


@pytest.mark.parametrize("driver", [False, True])
@pytest.mark.parametrize("overwrite_metadata", [False, True])
def test_retained_branch_requires_chunk_overwrite(
    branch_dataset: ParallelDataset,
    submit_job: Mock,
    driver: bool,
    overwrite_metadata: bool,
) -> None:
    with pytest.raises(AssertionError, match="overwrite-chunks"):
        _start_backfill(
            branch_dataset,
            driver=driver,
            overwrite_chunks=False,
            overwrite_metadata=overwrite_metadata,
        )

    submit_job.assert_not_called()
    ((_, _, location),) = branch_dataset.store_factory._icechunk_storages()
    assert not icechunk.Repository.exists(location)


@pytest.mark.parametrize("driver", [False, True])
@pytest.mark.parametrize(
    "unsupported", ["zarr", "icechunk-replica", "zarr-replica", "virtual"]
)
def test_retained_branch_rejects_unsupported_storage_and_virtual_data(
    branch_dataset: ParallelDataset,
    submit_job: Mock,
    tmp_path: Path,
    driver: bool,
    unsupported: str,
) -> None:
    dataset: ParallelDataset | ExampleVirtualDataset = branch_dataset
    match unsupported:
        case "zarr":
            dataset = branch_dataset.model_copy(
                update={
                    "primary_storage_config": StorageConfig(
                        base_path=str(tmp_path), format=DatasetFormat.ZARR3
                    )
                }
            )
        case "icechunk-replica" | "zarr-replica":
            dataset = branch_dataset.model_copy(
                update={
                    "replica_storage_configs": [
                        StorageConfig(
                            base_path=str(tmp_path / "replica"),
                            format=(
                                DatasetFormat.ICECHUNK
                                if unsupported == "icechunk-replica"
                                else DatasetFormat.ZARR3
                            ),
                        )
                    ]
                }
            )
        case "virtual":
            dataset = ExampleVirtualDataset(
                primary_storage_config=ExampleDatasetStorageConfig(
                    base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
                )
            )

    with pytest.raises(AssertionError, match="keep-branch"):
        _start_backfill(dataset, driver=driver)

    submit_job.assert_not_called()
    for _, _, location in dataset.store_factory._icechunk_storages():
        assert not icechunk.Repository.exists(location)


@pytest.mark.parametrize("keep_branch", [None, "candidate"])
def test_driver_propagates_branch_as_argv_without_changing_storage(
    branch_dataset: ParallelDataset, submit_job: Mock, keep_branch: str | None
) -> None:
    IcechunkFixture()._init_store(branch_dataset)
    ((_, repo),) = branch_dataset.store_factory.icechunk_repos(sort="primary-first")
    original = repo.lookup_branch("main")

    branch_dataset.backfill_kubernetes(
        overwrite_chunks=True, keep_branch=keep_branch, docker_image="test-image"
    )

    submit_job.assert_called_once()
    manifest = json.loads(submit_job.call_args.kwargs["input"])
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    flags = [arg for arg in container["command"] if arg.startswith("--keep-branch")]
    assert flags == (["--keep-branch=candidate"] if keep_branch else [])
    assert "--overwrite-chunks" in container["command"]
    assert not any("BRANCH" in item["name"] for item in container["env"])
    assert set(repo.list_branches()) == {"main"}
    assert repo.lookup_branch("main") == original


@pytest.mark.slow
def test_normal_backfill_still_publishes_and_cleans_temporary_branch(
    branch_dataset: ParallelDataset,
) -> None:
    IcechunkFixture()._init_store(branch_dataset)
    ((_, repo),) = branch_dataset.store_factory.icechunk_repos(sort="primary-first")
    original = repo.lookup_branch("main")

    for worker in (0, 0, 1):
        _run_worker(branch_dataset, worker, keep_branch=None)

    assert repo.lookup_branch("main") != original
    assert set(repo.list_branches()) == {"main"}
    published = xr.open_zarr(branch_dataset.store_factory.primary_store())
    for variable in published.data_vars.values():
        np.testing.assert_array_equal(variable.values, np.ones(variable.shape))
    assert branch_dataset.store_factory.list_coordination_files("branch-job", "") == []
