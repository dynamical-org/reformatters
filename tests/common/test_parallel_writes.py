"""Integration tests for parallel write coordination across multiple workers."""

import pickle
from collections.abc import Iterable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import dask
import dask.array
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from icechunk.store import IcechunkStore
from pydantic import computed_field

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

    def _init_store(self, dataset: ParallelDataset) -> None:
        """Create the icechunk store with initial empty dataset commit."""
        store = dataset.store_factory.primary_store(writable=True)
        assert isinstance(store, IcechunkStore)
        zarr.open_group(store, mode="w", attributes={"initialized": True})
        store.session.commit(message="init empty store")

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
        self._init_store(dataset)
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

        # Reader on main should see no data yet (only the empty init commit)
        reader_store = dataset.store_factory.primary_store()
        reader_ds = xr.open_zarr(reader_store)
        assert reader_ds.sizes.get("time", 0) == 0

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
        repos = dataset.store_factory.icechunk_repos()
        assert len(repos) == 1
        _, repo = repos[0]
        branches = list(repo.list_branches())
        assert branches == ["main"]


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
