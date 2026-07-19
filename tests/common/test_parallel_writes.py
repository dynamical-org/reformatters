"""Integration tests for parallel write coordination across multiple workers.

Marked `slow` because each test runs `_process_region_jobs` end-to-end,
which spins up executors, writes zarr metadata, and (for icechunk tests)
opens a repo session.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar

import dask
import dask.array
import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from pydantic import computed_field

from reformatters.common import (
    materialized_region_job as materialized_region_job_module,
)
from reformatters.common import parallel_coordination as pc_module
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
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import SourceFileCoord
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, ArrayFloat32, Dim, Timedelta, Timestamp
from reformatters.common.zarr import copy_zarr_metadata

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _use_thread_pool_for_shard_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Substitute ThreadPoolExecutor for ProcessPoolExecutor in materialized_region_job
    so shard writes skip spawn overhead. SharedMemory works in-process, and the
    zarr writes are I/O-bound — threads are fine for these tests."""
    monkeypatch.setattr(
        materialized_region_job_module, "ProcessPoolExecutor", ThreadPoolExecutor
    )


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


class ParallelRegionJob(
    MaterializedRegionJob[ParallelDataVar, ParallelSourceFileCoord]
):
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

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
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


def _create_template_ds(num_time: int = 4) -> xr.DataTree:
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
    return xr.DataTree.from_dict({"/": ds})


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

    def test_update_rejects_structural_drift(self, tmp_path: Path) -> None:
        """An operational update whose template changed a non-append structural field
        (here, the longitude chunk size) is rejected by worker 0 before any writes."""
        dataset = self._make_dataset(tmp_path)
        initial_ds = _create_template_ds(num_time=2)
        template_utils.write_metadata(initial_ds, dataset.store_factory)

        # Drift the longitude (non-append) chunk on every data var (kept uniform so
        # get_jobs works). An append-axis change alone would be allowed by the guard.
        drifted_ds = _create_template_ds(num_time=4)
        drifted_root = drifted_ds.to_dataset()
        for var in drifted_root.data_vars:
            drifted_root[var].encoding["chunks"] = (1, _LAT_SIZE, _LON_SIZE // 2)

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=drifted_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )

        with pytest.raises(ValueError, match="does not match the existing store"):
            dataset._process_region_jobs(
                all_jobs=all_jobs,
                worker_index=0,
                workers_total=1,
                reformat_job_name="test",
                template_ds=drifted_ds,
                tmp_store=dataset._tmp_store(),
                update_template_with_results=True,
            )

        # Guard ran before any writes: the store is untouched (still 2 time steps).
        reader_ds = xr.open_zarr(dataset.store_factory.primary_store())
        assert reader_ds.sizes["time"] == 2

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
        self, dataset: ParallelDataset, template_ds: xr.DataTree | None = None
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
    """Last worker reads every worker's results file and merges them before
    handing the dict to update_template_with_results."""

    def test_last_worker_merges_results_across_workers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With workers_total=2 and max_vars_per_job=2, worker 0 processes
        var0/var1 while worker 1 processes var2/var3. The dict passed to
        update_template_with_results must include keys from both."""
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

        captured: list[Mapping[str, Sequence[Any]]] = []
        original_update = ParallelRegionJob.update_template_with_results

        def capturing_update(
            self: ParallelRegionJob,
            process_results: Mapping[str, Sequence[Any]],
        ) -> xr.DataTree:
            captured.append(dict(process_results))
            return original_update(self, process_results)

        monkeypatch.setattr(
            ParallelRegionJob, "update_template_with_results", capturing_update
        )

        _run_workers(
            dataset,
            all_jobs,
            template_ds,
            workers_total=2,
            update_template_with_results=True,
        )

        # update_template_with_results is invoked exactly once, on the last
        # worker's finalize, with the merged dict from every worker's results file.
        assert len(captured) == 1
        merged = captured[0]
        assert set(merged.keys()) == {"var0", "var1", "var2", "var3"}
        # Each var spans every time step in the template (2 shards, 2 times each).
        for var_name in ("var0", "var1", "var2", "var3"):
            times = sorted(c.out_loc["time"] for c in merged[var_name])
            assert times == list(template_ds.to_dataset()["time"].values)

    def test_partial_read_failure_trims_template_to_last_successful_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reads that fail after a cutoff produce ReadFailed coords. The
        default update_template_with_results trims the template to the max
        successful append-dim value, so the finalized store stops there."""
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ZARR3
            ),
        )
        # Initial state: empty time dim (simulating an update extending from nothing).
        template_utils.write_metadata(
            _create_template_ds(num_time=0), dataset.store_factory
        )
        template_ds = _create_template_ds(num_time=4)

        # Fail reads for the second shard (t=2, t=3). First shard (t=0, t=1) succeeds.
        failure_cutoff = pd.Timestamp("2025-01-01 02:00")

        def flaky_read(
            self: ParallelRegionJob,
            coord: ParallelSourceFileCoord,
            data_var: ParallelDataVar,
        ) -> ArrayFloat32:
            if coord.time >= failure_cutoff:
                raise RuntimeError("simulated read failure")
            return np.ones((_LAT_SIZE, _LON_SIZE), dtype=np.float32)

        monkeypatch.setattr(ParallelRegionJob, "read_data", flaky_read)

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )
        _run_workers(
            dataset,
            all_jobs,
            template_ds,
            workers_total=2,
            update_template_with_results=True,
        )

        # Template trimmed to the latest successful time (t=1 inclusive → 2 steps).
        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert result.sizes["time"] == 2
        for var in ("var0", "var1", "var2", "var3"):
            assert np.all(result[var].values == 1.0)


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
    dataset: ParallelDataset,
    all_jobs: Sequence,
    template_ds: xr.DataTree,
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
        original_copy = pc_module.copy_zarr_metadata
        finalize_count = [0]
        should_raise = [True]

        def wrapped_copy(*args: object, **kwargs: object) -> None:
            if kwargs.get("icechunk_only") and should_raise[0]:
                finalize_count[0] += 1
                if finalize_count[0] == 2:
                    raise RuntimeError("simulated pod death between resets")
            return original_copy(*args, **kwargs)  # ty: ignore[invalid-argument-type]

        monkeypatch.setattr(pc_module, "copy_zarr_metadata", wrapped_copy)

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

    def test_retry_before_any_reset_completes_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mirror of the above: last worker dies during finalize BEFORE any
        repo's main has been reset (failure on the first copy_zarr_metadata
        call, which is for the replica). On retry, neither repo's skip fires
        and both get fully reset."""
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

        primary_repo = dataset.store_factory.icechunk_repos(sort="primary-first")[0][1]
        replica_repo = dataset.store_factory.icechunk_repos(sort="primary-last")[0][1]
        initial_primary = primary_repo.lookup_branch("main")
        initial_replica = replica_repo.lookup_branch("main")

        # Fail on the 1st icechunk_only copy (the replica in replicas-first order)
        # — pod dies before any reset_branch. icechunk_only=True is only passed
        # from _finalize, not _parallel_setup.
        original_copy = pc_module.copy_zarr_metadata
        finalize_count = [0]
        should_raise = [True]

        def wrapped_copy(*args: object, **kwargs: object) -> None:
            if kwargs.get("icechunk_only") and should_raise[0]:
                finalize_count[0] += 1
                if finalize_count[0] == 1:
                    raise RuntimeError("simulated pod death before first reset")
            return original_copy(*args, **kwargs)  # ty: ignore[invalid-argument-type]

        monkeypatch.setattr(pc_module, "copy_zarr_metadata", wrapped_copy)

        _run_workers(
            dataset, all_jobs, template_ds, workers_total=2, worker_indices=[0]
        )
        with pytest.raises(RuntimeError, match="simulated pod death"):
            _run_workers(
                dataset, all_jobs, template_ds, workers_total=2, worker_indices=[1]
            )

        # Neither repo was reset.
        assert primary_repo.lookup_branch("main") == initial_primary
        assert replica_repo.lookup_branch("main") == initial_replica

        # Retry with the failure disabled — both repos now advance.
        should_raise[0] = False
        _run_workers(
            dataset, all_jobs, template_ds, workers_total=2, worker_indices=[1]
        )

        assert primary_repo.lookup_branch("main") != initial_primary
        assert replica_repo.lookup_branch("main") != initial_replica

        for store in [
            dataset.store_factory.primary_store(),
            *dataset.store_factory.replica_stores(),
        ]:
            result = xr.open_zarr(store)
            assert result.sizes["time"] == 4
            for var in ["var0", "var1", "var2", "var3"]:
                assert np.all(result[var].values == 1.0)

        for _role, repo in dataset.store_factory.icechunk_repos(sort="primary-first"):
            assert list(repo.list_branches()) == ["main"]


class TestReplicaOrdering:
    """Replicas are written before the primary in both parallel-setup and
    finalize, so that if a failure occurs mid-way the primary (which drives
    future work) still reflects the pre-update state.

    Setup ordering: _parallel_setup passes ic_stores[0] (primary, per
    icechunk_repos(sort="primary-first")) as primary_store to
    commit_if_icechunk. commit_if_icechunk is separately verified to commit
    replicas first in tests/common/test_storage.py.

    Finalize ordering: _finalize iterates icechunk_repos(sort="primary-last")
    and does the per-repo commit + reset in that order. The test below
    observes the reset_branch calls end-to-end. Ordering under a partial
    failure between replica and primary resets is additionally verified by
    TestLastWorkerRetryAfterPartialFinalize."""

    def test_icechunk_repos_sort_order(self, tmp_path: Path) -> None:
        """icechunk_repos returns the primary first or last per the sort kwarg."""
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
        # In test mode _get_store_path ignores config.base_path so the three
        # configs resolve to the same on-disk repo. We only inspect role
        # labels here, which are unaffected.
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

    def test_finalize_resets_replicas_before_primary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: a 2-worker backfill with 2 replicas calls reset_branch
        on replica-0, replica-1, then primary — in that order — during finalize."""
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
                    base_path=str(tmp_path / "replica-0"),
                    format=DatasetFormat.ICECHUNK,
                ),
                StorageConfig(
                    base_path=str(tmp_path / "replica-1"),
                    format=DatasetFormat.ICECHUNK,
                ),
            ],
        )
        template_ds = _create_template_ds(num_time=2)
        template_utils.write_metadata(template_ds, dataset.store_factory)

        def _role_from_storage(storage_str: str) -> str:
            for role in ("primary", "replica-0", "replica-1"):
                if f"/{role}/" in storage_str:
                    return role
            raise AssertionError(f"cannot identify role in {storage_str!r}")

        reset_order: list[str] = []
        original_reset = icechunk.Repository.reset_branch

        def recording_reset(
            self: icechunk.Repository, *args: object, **kwargs: object
        ) -> object:
            reset_order.append(_role_from_storage(str(self.storage)))
            return original_reset(self, *args, **kwargs)  # ty: ignore[invalid-argument-type]

        monkeypatch.setattr(icechunk.Repository, "reset_branch", recording_reset)

        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
        )
        _run_workers(dataset, all_jobs, template_ds, workers_total=2)

        assert reset_order == ["replica-0", "replica-1", "primary"]


class TestConcurrentJobs:
    """Two jobs with overlapping lifetimes both capture the same initial
    snapshot during setup. When they race to finalize, the second to arrive
    finds main has moved past its original snapshot and cannot publish —
    preserving the first job's work. It fails loudly, leaving its temp branch
    and coordination files in place for inspection."""

    def test_second_job_finalize_raises_when_main_already_moved(
        self, tmp_path: Path
    ) -> None:
        dataset = ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
            ),
        )
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)

        primary_repo = dataset.store_factory.icechunk_repos(sort="primary-first")[0][1]
        initial_snapshot = primary_repo.lookup_branch("main")

        jobs_a = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="job-a",
        )
        jobs_b = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="job-b",
        )

        # Both jobs' worker 0 set up on the *same* main snapshot.
        _run_workers(
            dataset,
            jobs_a,
            template_ds,
            workers_total=2,
            reformat_job_name="job-a",
            worker_indices=[0],
        )
        _run_workers(
            dataset,
            jobs_b,
            template_ds,
            workers_total=2,
            reformat_job_name="job-b",
            worker_indices=[0],
        )

        setup_a = json.loads(
            dataset.store_factory.read_all_coordination_files("job-a", "setup")[0]
        )
        setup_b = json.loads(
            dataset.store_factory.read_all_coordination_files("job-b", "setup")[0]
        )
        assert (
            setup_a["repo_snapshots"]["primary"]
            == setup_b["repo_snapshots"]["primary"]
            == initial_snapshot
        )

        # Job A last worker finalizes: main advances from initial → S_A.
        _run_workers(
            dataset,
            jobs_a,
            template_ds,
            workers_total=2,
            reformat_job_name="job-a",
            worker_indices=[1],
        )
        after_a = primary_repo.lookup_branch("main")
        assert after_a != initial_snapshot

        # Job B last worker finalizes: current main (S_A) != job-b's
        # original_snapshot (initial), so it cannot publish and raises.
        with pytest.raises(RuntimeError, match="main moved during this backfill"):
            _run_workers(
                dataset,
                jobs_b,
                template_ds,
                workers_total=2,
                reformat_job_name="job-b",
                worker_indices=[1],
            )
        assert primary_repo.lookup_branch("main") == after_a

        # Job A cleaned up; job B's temp branch and coordination files are left
        # in place for inspection.
        assert set(primary_repo.list_branches()) == {"_job_job-b", "main"}
        assert dataset.store_factory.read_all_coordination_files("job-a", "setup") == []
        assert dataset.store_factory.read_all_coordination_files("job-b", "setup") != []


class TestOverwriteBackfillConcurrency:
    """Overwrite backfills commit straight to main and coexist with operational
    updates; an update whose finalize finds main moved replays onto it.
    See "Concurrent jobs writing to the same dataset" in docs/parallel_processing.md.
    """

    def _make_dataset(self, tmp_path: Path) -> ParallelDataset:
        return ParallelDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
            ),
        )

    def _repo(self, dataset: ParallelDataset) -> icechunk.Repository:
        return dataset.store_factory.icechunk_repos(sort="primary-first")[0][1]

    @staticmethod
    def _array(store: object, name: str) -> zarr.Array:
        array = zarr.open_group(store=store)[name]  # ty: ignore[invalid-argument-type]
        assert isinstance(array, zarr.Array)
        return array

    def test_overwrite_backfill_commits_to_main_and_preserves_concurrent_update(
        self, tmp_path: Path
    ) -> None:
        dataset = self._make_dataset(tmp_path)
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)
        repo = self._repo(dataset)

        # Backfill only var0/var1 so a concurrent write to var2 can't be
        # legitimately rewritten by the backfill itself.
        all_jobs = ParallelRegionJob.get_jobs(
            tmp_store=dataset._tmp_store(),
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=ParallelTemplateConfig().data_vars,
            reformat_job_name="test",
            filter_variable_names=["var0", "var1"],
        )

        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=0,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=False,
            overwrite_chunks=True,
        )

        # An operational update commits to main between the backfill's workers.
        update_session = repo.writable_session("main")
        self._array(update_session.store, "var2")[0:2] = np.float32(7.0)
        update_session.commit("simulated operational update")

        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=1,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=dataset._tmp_store(),
            update_template_with_results=False,
            overwrite_chunks=True,
        )

        # No temp branch was ever created, the backfilled vars are written, and
        # the update's concurrent commit survived the backfill's rebases.
        assert list(repo.list_branches()) == ["main"]
        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert np.all(result["var0"].values == 1.0)
        assert np.all(result["var1"].values == 1.0)
        assert np.all(result["var2"].values[0:2] == 7.0)

    def test_replay_publishes_update_onto_moved_main(self, tmp_path: Path) -> None:
        dataset = self._make_dataset(tmp_path)
        template_ds = _create_template_ds()
        template_utils.write_metadata(template_ds, dataset.store_factory)
        repo = self._repo(dataset)
        original_snapshot = repo.lookup_branch("main")

        # An update's template may add a new array (a template merged mid-flight);
        # replay must carry the new array's chunks too.
        updated_ds = template_ds["/"].to_dataset()
        updated_ds["var_new"] = updated_ds["var0"].copy()
        updated_ds["var_new"].encoding = dict(updated_ds["var0"].encoding)
        updated_template = xr.DataTree.from_dict({"/": updated_ds})
        tmp_store = tmp_path / "replay-tmp.zarr"
        template_utils.write_metadata(updated_template, tmp_store)

        # The update runs on a temp branch: expand metadata (creates var_new),
        # then write var0 and var_new data, mirroring setup + worker commits.
        repo.create_branch("_job_update", original_snapshot)
        setup_session = repo.writable_session("_job_update")
        copy_zarr_metadata(
            updated_template, tmp_store, setup_session.store, icechunk_only=True
        )
        setup_session.commit("Expand dataset")
        worker_session = repo.writable_session("_job_update")
        self._array(worker_session.store, "var0")[:] = np.float32(2.0)
        self._array(worker_session.store, "var_new")[:] = np.float32(3.0)
        worker_session.commit("update worker")

        # Meanwhile an overwrite backfill commits var1 data straight to main.
        backfill_session = repo.writable_session("main")
        self._array(backfill_session.store, "var1")[:] = np.float32(9.0)
        backfill_session.commit("backfill chunks")

        pc_module._replay_branch_onto_main(
            repo,
            "primary",
            "_job_update",
            original_snapshot,
            updated_template,
            tmp_store,
            "Update at 2025-01-01T00:00:00Z",
        )

        result = xr.open_zarr(dataset.store_factory.primary_store())
        assert np.all(result["var0"].values == 2.0)
        assert np.all(result["var_new"].values == 3.0)
        assert np.all(result["var1"].values == 9.0), (
            "the backfill's concurrent commit must survive the replay"
        )

        # The replay commit reads as a normal update to consumers; the machinery
        # marker lives in snapshot metadata, where the idempotence check finds it.
        tip = next(iter(repo.ancestry(branch="main")))
        assert tip.message == "Update at 2025-01-01T00:00:00Z"
        assert tip.metadata["replayed_branch"] == "_job_update"

        # A crash-retry of finalize replays again; the metadata marker makes it a
        # no-op instead of stacking duplicate commits.
        commits_after_replay = len(list(repo.ancestry(branch="main")))
        pc_module._replay_branch_onto_main(
            repo,
            "primary",
            "_job_update",
            original_snapshot,
            updated_template,
            tmp_store,
            "Update at 2025-01-01T00:05:00Z",
        )
        assert len(list(repo.ancestry(branch="main"))) == commits_after_replay
