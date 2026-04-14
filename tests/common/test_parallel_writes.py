"""Integration tests for parallel write coordination across multiple workers."""

import json
import pickle
from collections.abc import Iterable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import dask
import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import computed_field

from reformatters.common import dynamical_dataset as dd_module
from reformatters.common import storage as storage_module
from reformatters.common import template_utils
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.region_job import (
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, ArrayFloat32, Dim, Timedelta, Timestamp


def _pickle_loads(data: bytes) -> dict:
    return pickle.loads(data)  # noqa: S301


_LAT_SIZE = 3
_LON_SIZE = 4


class ParallelDataVar(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=(1, _LAT_SIZE, _LON_SIZE),
        shards=(2, _LAT_SIZE, _LON_SIZE),
        compressors=[],
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="C",
        long_name="Test variable",
        short_name="test",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(keep_mantissa_bits=10)


class ParallelSourceFileCoord(SourceFileCoord):
    time: Timestamp

    def get_url(self) -> str:
        return f"https://test.org/{self.time.isoformat()}"


class ParallelRegionJob(RegionJob[ParallelDataVar, ParallelSourceFileCoord]):
    max_vars_per_job: ClassVar[int] = 2

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[ParallelDataVar],  # noqa: ARG002
    ) -> Sequence[ParallelSourceFileCoord]:
        return [
            ParallelSourceFileCoord(time=t)
            for t in processing_region_ds[self.append_dim].values
        ]

    def download_file(self, coord: ParallelSourceFileCoord) -> Path:  # noqa: ARG002
        return Path("testfile")

    def read_data(
        self,
        coord: ParallelSourceFileCoord,  # noqa: ARG002
        data_var: ParallelDataVar,  # noqa: ARG002
    ) -> ArrayFloat32:
        return np.ones((_LAT_SIZE, _LON_SIZE), dtype=np.float32)


class ParallelTemplateConfig(TemplateConfig[ParallelDataVar]):
    dims: tuple[Dim, ...] = ("time", "latitude", "longitude")
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2025-01-01")
    append_dim_frequency: Timedelta = pd.Timedelta("1h")

    @computed_field
    @property
    def dataset_id(self) -> str:
        return "test-parallel-dataset"

    @computed_field
    @property
    def version(self) -> str:
        return "v1.0"

    @computed_field
    @property
    def data_vars(self) -> list[ParallelDataVar]:
        return [
            ParallelDataVar(name="var0"),
            ParallelDataVar(name="var1"),
            ParallelDataVar(name="var2"),
            ParallelDataVar(name="var3"),
        ]


class ParallelDataset(DynamicalDataset[ParallelDataVar, ParallelSourceFileCoord]):
    template_config: ParallelTemplateConfig = ParallelTemplateConfig()
    region_job_class: type[ParallelRegionJob] = ParallelRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule="0 0 * * *",
                pod_active_deadline=timedelta(minutes=30),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="1G",
                shared_memory="1G",
                ephemeral_storage="1G",
                secret_names=self.store_factory.k8s_secret_names(),
            ),
            ValidationCronJob(
                name=f"{self.dataset_id}-validate",
                schedule="0 1 * * *",
                pod_active_deadline=timedelta(minutes=30),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="1G",
                shared_memory="1G",
                ephemeral_storage="1G",
                secret_names=self.store_factory.k8s_secret_names(),
            ),
        ]


def _create_template_ds(num_time: int = 4) -> xr.Dataset:
    ds = xr.Dataset(
        {
            f"var{i}": xr.Variable(
                data=dask.array.full(
                    (num_time, _LAT_SIZE, _LON_SIZE),
                    np.nan,
                    dtype=np.float32,
                    chunks=(1, _LAT_SIZE, _LON_SIZE),
                ),
                dims=["time", "latitude", "longitude"],
                encoding={
                    "dtype": "float32",
                    "chunks": (1, _LAT_SIZE, _LON_SIZE),
                    "shards": (2, _LAT_SIZE, _LON_SIZE),
                    "fill_value": np.nan,
                },
            )
            for i in range(4)
        },
        coords={
            "time": pd.date_range("2025-01-01", freq="h", periods=num_time),
            "latitude": np.linspace(0, 90, _LAT_SIZE),
            "longitude": np.linspace(0, 140, _LON_SIZE),
        },
        attrs={"dataset_id": "test-parallel-dataset"},
    )
    ds["time"].encoding["fill_value"] = -1
    ds["latitude"].encoding["fill_value"] = np.nan
    ds["longitude"].encoding["fill_value"] = np.nan
    return ds


class TestZarr3ParallelWrites:
    """Test that zarr v3 parallel writes preserve a reader-safe view."""

    def _make_dataset(self, tmp_path: Path) -> ParallelDataset:
        return ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ZARR3
            ),
        )

    def test_single_worker_backfill(self, tmp_path: Path) -> None:
        dataset = self._make_dataset(tmp_path)
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)

        dataset._process_region_jobs(
            all_jobs=ParallelRegionJob.get_jobs(
                tmp_store=dataset._tmp_store(),
                template_ds=template_ds,
                append_dim="time",
                all_data_vars=ParallelTemplateConfig().data_vars,
                reformat_job_name="test",
            ),
            worker_index=0,
            workers_total=1,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=False,
        )

        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert result.sizes["time"] == 4
        for var in ["var0", "var1", "var2", "var3"]:
            assert np.all(result[var].values == 1.0)

    def test_two_worker_backfill(self, tmp_path: Path) -> None:
        """Two workers split variable groups and both write data correctly."""
        dataset = self._make_dataset(tmp_path)
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )
        assert len(all_jobs) >= 2

        for worker_index in range(2):
            dataset._process_region_jobs(
                all_jobs=all_jobs,
                worker_index=worker_index,
                workers_total=2,
                reformat_job_name="test",
                template_ds=template_ds,
                tmp_store=dataset._tmp_store(),
                update_template_with_results=False,
            )

        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert result.sizes["time"] == 4
        for var in ["var0", "var1", "var2", "var3"]:
            assert np.all(result[var].values == 1.0)

    def test_update_defers_metadata_until_last_worker(self, tmp_path: Path) -> None:
        """For updates (update_template_with_results=True), zarr v3 metadata is deferred.
        Readers should not see expanded dimensions until the last worker finalizes."""
        dataset = self._make_dataset(tmp_path)
        template_ds = _create_template_ds()
        # Write initial smaller dataset so we can "update" to the full size
        initial_ds = _create_template_ds(num_time=2)
        template_utils.write_metadata(initial_ds, dataset.store_factory)

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        # Worker 0 — processes its jobs but does NOT expand zarr v3 metadata
        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=0,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=True,
        )

        # Reader should still see original 2 time steps (metadata not yet expanded)
        reader_ds = xr.open_zarr(dataset.store_factory.primary_store())
        assert reader_ds.sizes["time"] == 2

        # Worker 1 (last) — finalizes, writes metadata
        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=1,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=True,
        )

        # Now reader should see expanded dataset
        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert result.sizes["time"] == 4
        for var in ["var0", "var1", "var2", "var3"]:
            assert np.all(result[var].values == 1.0)

    def test_coordination_files_cleaned_up(self, tmp_path: Path) -> None:
        dataset = self._make_dataset(tmp_path)
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        for worker_index in range(2):
            dataset._process_region_jobs(
                all_jobs=all_jobs,
                worker_index=worker_index,
                workers_total=2,
                reformat_job_name="test",
                template_ds=template_ds,
                tmp_store=dataset._tmp_store(),
                update_template_with_results=False,
            )

        # Coordination files should be cleaned up after last worker
        assert (
            dataset.store_factory.read_all_coordination_files("test", "results") == []
        )
        assert dataset.store_factory.read_all_coordination_files("test", "setup") == []


class TestIcechunkParallelWrites:
    """Test that icechunk parallel writes preserve a reader-safe view via temp branch."""

    def _make_dataset(self, tmp_path: Path) -> ParallelDataset:
        return ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
            ),
        )

    def _init_store(
        self, dataset: ParallelDataset, template_ds: xr.Dataset | None = None
    ) -> None:
        """Write metadata to the icechunk store, matching production backfill flow."""
        if template_ds is None:
            template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)

    def test_single_worker_backfill(self, tmp_path: Path) -> None:
        dataset = self._make_dataset(tmp_path)
        self._init_store(dataset)
        template_ds = _create_template_ds()

        dataset._process_region_jobs(
            all_jobs=ParallelRegionJob.get_jobs(
                tmp_store=dataset._tmp_store(),
                template_ds=template_ds,
                append_dim="time",
                all_data_vars=ParallelTemplateConfig().data_vars,
                reformat_job_name="test",
            ),
            worker_index=0,
            workers_total=1,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=False,
        )

        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert result.sizes["time"] == 4
        for var in ["var0", "var1", "var2", "var3"]:
            assert np.all(result[var].values == 1.0)

    def test_two_worker_backfill_reader_safety(self, tmp_path: Path) -> None:
        """While workers write on a temp branch, readers on main see no new data."""
        dataset = self._make_dataset(tmp_path)
        # Init with 0 time steps so main has the arrays but no data
        empty_template = _create_template_ds(num_time=0)
        self._init_store(dataset, template_ds=empty_template)
        template_ds = _create_template_ds()

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        # Worker 0 — creates temp branch, expands metadata, processes its jobs
        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=0,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=False,
        )

        # Reader on main should see no new data yet
        reader_store = dataset.store_factory.primary_store()
        reader_ds = xr.open_zarr(reader_store)
        assert reader_ds.sizes["time"] == 0

        # Worker 1 (last) — processes, merges, resets main
        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=1,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=False,
        )

        # Now reader on main should see all data
        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert result.sizes["time"] == 4
        for var in ["var0", "var1", "var2", "var3"]:
            assert np.all(result[var].values == 1.0)

    def test_temp_branch_cleaned_up(self, tmp_path: Path) -> None:
        dataset = self._make_dataset(tmp_path)
        self._init_store(dataset)
        template_ds = _create_template_ds()

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        for worker_index in range(2):
            dataset._process_region_jobs(
                all_jobs=all_jobs,
                worker_index=worker_index,
                workers_total=2,
                reformat_job_name="test",
                template_ds=template_ds,
                tmp_store=dataset._tmp_store(),
                update_template_with_results=False,
            )

        # Temp branch should be cleaned up
        repos = dataset.store_factory.icechunk_repos(sort="primary-first")
        assert len(repos) == 1
        _, repo = repos[0]
        branches = list(repo.list_branches())
        assert branches == ["main"]


class TestReplicaParallelWrites:
    """Test parallel writes with primary + replica icechunk stores."""

    def test_two_worker_backfill_with_replica(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Use config base_path directly so primary and replica get different local paths
        monkeypatch.setattr(
            storage_module,
            "_get_store_path",
            lambda dataset_id, version, config: (
                f"{config.base_path}/{dataset_id}/v{version}"
                f".{'icechunk' if config.format == DatasetFormat.ICECHUNK else 'zarr'}"
            ),
        )
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path / "primary"), format=DatasetFormat.ICECHUNK
            ),
            replica_storage_configs=[
                StorageConfig(
                    base_path=str(tmp_path / "replica"), format=DatasetFormat.ICECHUNK
                ),
            ],
        )
        template_ds = _create_template_ds(num_time=2)
        # Init both stores with metadata matching production backfill flow
        template_utils.write_metadata(template_ds, dataset.store_factory)
        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        for worker_index in range(2):
            dataset._process_region_jobs(
                all_jobs=all_jobs,
                worker_index=worker_index,
                workers_total=2,
                reformat_job_name="test",
                template_ds=template_ds,
                tmp_store=dataset._tmp_store(),
                update_template_with_results=False,
            )

        # Both primary and replica should have all data
        for store in [
            dataset.store_factory.primary_store(),
            *dataset.store_factory.replica_stores(),
        ]:
            result = xr.open_zarr(store)
            assert result.sizes["time"] == 2
            for var in ["var0", "var1", "var2", "var3"]:
                assert np.all(result[var].values == 1.0)

        # Temp branches cleaned up on both repos
        for _role, repo in dataset.store_factory.icechunk_repos(sort="primary-first"):
            assert list(repo.list_branches()) == ["main"]


class TestResultsAggregation:
    """Test that results from all workers are correctly merged."""

    def test_results_contain_all_variables(self, tmp_path: Path) -> None:
        """With 2 workers splitting 4 vars into groups of 2, merged results have all 4 vars."""
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ZARR3
            ),
        )
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        # Run worker 0 (gets some variable groups)
        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=0,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=False,
        )

        # Check worker 0 wrote its results
        result_files = dataset.store_factory.read_all_coordination_files(
            "test", "results"
        )
        assert len(result_files) == 1
        worker_0_results = _pickle_loads(result_files[0])
        assert len(worker_0_results) > 0

        # Run worker 1 (gets remaining variable groups, then finalizes)
        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=1,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=False,
        )

        # After finalization, all data should be present
        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert set(result.data_vars) == {"var0", "var1", "var2", "var3"}


class TestWorkerEdgeCases:
    """Test edge cases like workers with zero jobs."""

    def test_worker_with_zero_jobs(self, tmp_path: Path) -> None:
        """3 workers but only 2 jobs — worker 2 gets nothing but still participates."""
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ZARR3
            ),
        )
        template_ds = _create_template_ds(num_time=2)
        template_utils.write_metadata(template_ds, dataset.store_factory)

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        for worker_index in range(3):
            dataset._process_region_jobs(
                all_jobs=all_jobs,
                worker_index=worker_index,
                workers_total=3,
                reformat_job_name="test",
                template_ds=template_ds,
                tmp_store=dataset._tmp_store(),
                update_template_with_results=False,
            )

        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert result.sizes["time"] == 2


def _run_workers(
    dataset: "ParallelDataset",
    all_jobs: Sequence,
    template_ds: xr.Dataset,
    *,
    workers_total: int,
    reformat_job_name: str = "test",
    worker_indices: Iterable[int] | None = None,
    update_template_with_results: bool = False,
) -> None:
    """Helper: run a set of worker indices sequentially against the same dataset."""
    if worker_indices is None:
        worker_indices = range(workers_total)
    for worker_index in worker_indices:
        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=worker_index,
            workers_total=workers_total,
            reformat_job_name=reformat_job_name,
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=update_template_with_results,
        )


class TestWorkerRestart:
    """Failure mode: a worker pod dies mid-processing and k8s restarts it
    with the same WORKER_INDEX. Re-running `_process_region_jobs` with the
    same args must converge to the correct final state."""

    def test_zarr3_restart_is_idempotent(self, tmp_path: Path) -> None:
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ZARR3
            ),
        )
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)
        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        # Worker 0 runs, "dies", restarts (runs again), then worker 1 finalizes.
        _run_workers(
            dataset,
            all_jobs,
            template_ds,
            workers_total=2,
            worker_indices=[0, 0, 1],
        )

        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert result.sizes["time"] == 4
        for var in ["var0", "var1", "var2", "var3"]:
            assert np.all(result[var].values == 1.0)

    def test_icechunk_restart_is_idempotent(self, tmp_path: Path) -> None:
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
            ),
        )
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)
        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        # Worker 0 runs, "dies", restarts (reuses existing temp branch
        # and ready.json), then worker 1 finalizes.
        _run_workers(
            dataset,
            all_jobs,
            template_ds,
            workers_total=2,
            worker_indices=[0, 0, 1],
        )

        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert result.sizes["time"] == 4
        for var in ["var0", "var1", "var2", "var3"]:
            assert np.all(result[var].values == 1.0)

        # Temp branch cleaned up
        _, repo = dataset.store_factory.icechunk_repos(sort="primary-first")[0]
        assert list(repo.list_branches()) == ["main"]


class TestWorker0SnapshotStability:
    """Failure mode: worker 0 dies after writing ready.json and is restarted.
    On retry, setup_info.repo_snapshots must stay pinned to the original value
    (read back from ready.json), not refreshed from a new lookup_branch("main").
    Otherwise an external write to main between attempts is silently overwritten
    at finalize."""

    def test_retry_reuses_snapshot_from_ready_json(self, tmp_path: Path) -> None:
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
            ),
        )
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)
        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        # First attempt creates ready.json with real snapshot A.
        _run_workers(
            dataset, all_jobs, template_ds, workers_total=2, worker_indices=[0]
        )

        # Overwrite with a sentinel the code could only preserve (not produce).
        sentinel = "SENTINEL_SNAP_00000000"
        dataset.store_factory.write_coordination_file(
            "test",
            "setup/ready.json",
            json.dumps({"repo_snapshots": {"primary": sentinel}}).encode(),
        )

        # Worker 0 retry: setdefault should reuse sentinel, not refresh.
        _run_workers(
            dataset, all_jobs, template_ds, workers_total=2, worker_indices=[0]
        )

        ready_data = dataset.store_factory.read_all_coordination_files("test", "setup")[
            0
        ]
        assert json.loads(ready_data)["repo_snapshots"]["primary"] == sentinel


class TestLastWorkerRetryAfterPartialFinalize:
    """Failure mode: last worker dies mid-finalize after resetting some repos
    but not others. On retry, the skip-if-already-reset check must skip done
    repos and complete the remaining ones."""

    def test_retry_after_replica_reset_completes_primary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            storage_module,
            "_get_store_path",
            lambda dataset_id, version, config: (
                f"{config.base_path}/{dataset_id}/v{version}.icechunk"
            ),
        )
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path / "primary"), format=DatasetFormat.ICECHUNK
            ),
            replica_storage_configs=[
                StorageConfig(
                    base_path=str(tmp_path / "replica"),
                    format=DatasetFormat.ICECHUNK,
                ),
            ],
        )
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)
        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        # Patch copy_zarr_metadata to raise on the 2nd icechunk_only call,
        # simulating a pod death between the replica and primary reset_branch.
        # icechunk_only=True is only passed by _finalize, not _parallel_setup.
        original_copy = dd_module.copy_zarr_metadata
        finalize_count = [0]
        should_raise = [True]

        def wrapped_copy(*args: object, **kwargs: object) -> None:
            if kwargs.get("icechunk_only") and should_raise[0]:
                finalize_count[0] += 1
                if finalize_count[0] == 2:
                    raise RuntimeError("simulated pod death between resets")
            return original_copy(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(dd_module, "copy_zarr_metadata", wrapped_copy)

        _run_workers(
            dataset, all_jobs, template_ds, workers_total=2, worker_indices=[0]
        )
        with pytest.raises(RuntimeError, match="simulated pod death"):
            _run_workers(
                dataset, all_jobs, template_ds, workers_total=2, worker_indices=[1]
            )

        # Replica was reset; primary was not.
        primary_repo = dataset.store_factory.icechunk_repos(sort="primary-first")[0][1]
        replica_repo = dataset.store_factory.icechunk_repos(sort="primary-last")[0][1]
        primary_pre_retry = primary_repo.lookup_branch("main")
        replica_pre_retry = replica_repo.lookup_branch("main")

        # Disable the simulated failure and retry the last worker.
        should_raise[0] = False
        _run_workers(
            dataset, all_jobs, template_ds, workers_total=2, worker_indices=[1]
        )

        # Primary advanced, replica stayed at its already-reset value.
        assert primary_repo.lookup_branch("main") != primary_pre_retry
        assert replica_repo.lookup_branch("main") == replica_pre_retry

        # Both stores have the full data.
        for store in [
            dataset.store_factory.primary_store(),
            *dataset.store_factory.replica_stores(),
        ]:
            result = xr.open_zarr(store)
            assert result.sizes["time"] == 4
            for var in ["var0", "var1", "var2", "var3"]:
                assert np.all(result[var].values == 1.0)

        # Temp branches cleaned up on both repos.
        for _role, repo in dataset.store_factory.icechunk_repos(sort="primary-first"):
            assert list(repo.list_branches()) == ["main"]


class TestReplicaOrdering:
    """Design invariant: replicas are written before the primary in both the
    parallel-setup and finalize phases, so that if a failure occurs mid-way
    the primary (which drives future work) still reflects the pre-update state."""

    def test_primary_last_sort_order(self, tmp_path: Path) -> None:
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path / "primary"), format=DatasetFormat.ICECHUNK
            ),
            replica_storage_configs=[
                StorageConfig(
                    base_path=str(tmp_path / "replica-0"),
                    format=DatasetFormat.ICECHUNK,
                ),
                StorageConfig(
                    base_path=str(tmp_path / "replica-1"),
                    format=DatasetFormat.ICECHUNK,
                ),
            ],
        )
        # Avoid collision from shared _get_store_path base
        roles_first = [
            role
            for role, _ in dataset.store_factory.icechunk_repos(sort="primary-first")
        ]
        roles_last = [
            role
            for role, _ in dataset.store_factory.icechunk_repos(sort="primary-last")
        ]
        assert roles_first[0] == "primary"
        assert roles_last[-1] == "primary"
