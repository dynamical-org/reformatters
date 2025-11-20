from collections.abc import Sequence
from itertools import batched, pairwise
from pathlib import Path
from typing import ClassVar

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils, validation
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.region_job import (
    RegionJob,
    SourceFileCoord,
    SourceFileStatus,
)
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
    StoreFactory,
    get_local_tmp_store,
)
from reformatters.common.types import ArrayFloat32, Timestamp


class ExampleDataVar(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float32", fill_value=np.nan, chunks=(1, 10, 15), shards=None
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="C",
        long_name="Test variable",
        short_name="test",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(keep_mantissa_bits=10)


class ExampleSourceFileCoords(SourceFileCoord):
    time: Timestamp

    def get_url(self) -> str:
        return f"https://test.org/testfile{self.time.strftime('%Y%m%d%H%M')}"


class ExampleRegionJob(RegionJob[ExampleDataVar, ExampleSourceFileCoords]):
    max_vars_per_backfill_job: ClassVar[int] = 2

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[ExampleDataVar],
    ) -> Sequence[Sequence[ExampleDataVar]]:
        return list(batched(data_vars, 3, strict=False))

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        _data_var_group: Sequence[ExampleDataVar],
    ) -> Sequence[ExampleSourceFileCoords]:
        return [
            ExampleSourceFileCoords(time=t)
            for t in processing_region_ds[self.append_dim].values
        ]

    def download_file(self, coord: ExampleSourceFileCoords) -> Path:
        if coord.time == pd.Timestamp("2025-01-01T00"):
            raise FileNotFoundError()  # simulate a missing file
        return Path("testfile")

    def read_data(
        self,
        coord: ExampleSourceFileCoords,
        _data_var: ExampleDataVar,
    ) -> ArrayFloat32:
        if coord.time == pd.Timestamp("2025-01-01T06"):
            raise ValueError("Test error")  # simulate a read error
        return np.ones((10, 15), dtype=np.float32)


@pytest.fixture
def store_factory() -> StoreFactory:
    return StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset-A",
        template_config_version="test-version",
    )


@pytest.fixture
def template_ds() -> xr.Dataset:
    return _create_template_ds()


def _create_template_ds(
    num_vars: int = 4, var_fill_value: float = np.nan
) -> xr.Dataset:
    num_time = 48
    ds = xr.Dataset(
        {
            f"var{i}": xr.Variable(
                data=dask.array.full(  # type: ignore[no-untyped-call]
                    (num_time, 10, 15),
                    var_fill_value,
                    dtype=np.float32,
                    chunks=(num_time // 4, 10, 15),
                ),
                dims=["time", "latitude", "longitude"],
                encoding={
                    "dtype": "float32",
                    "chunks": (num_time // 4, 10, 15),
                    "shards": (num_time // 2, 10, 15),
                    "fill_value": var_fill_value,
                },
            )
            for i in range(num_vars)
        },
        coords={
            "time": pd.date_range("2025-01-01", freq="h", periods=num_time),
            "latitude": np.linspace(0, 90, 10),
            "longitude": np.linspace(0, 140, 15),
        },
        attrs={"dataset_id": "test-dataset-A"},
    )
    ds["time"].encoding["fill_value"] = -1
    ds["latitude"].encoding["fill_value"] = np.nan
    ds["longitude"].encoding["fill_value"] = np.nan
    return ds


def test_region_job(template_ds: xr.Dataset, store_factory: StoreFactory) -> None:
    tmp_store = get_local_tmp_store()

    # Write zarr metadata for this RegionJob to write into
    template_utils.write_metadata(template_ds, store_factory)

    job = ExampleRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=[ExampleDataVar(name=str(name)) for name in template_ds.data_vars],
        append_dim="time",
        region=slice(0, 18),
        reformat_job_name="test-job",
    )

    primary_store = store_factory.primary_store(writable=True)
    replica_stores = store_factory.replica_stores(writable=True)

    template_utils.write_metadata(job.template_ds, tmp_store)
    job.process(primary_store, replica_stores)

    ds = xr.open_zarr(store_factory.primary_store())
    region_template_ds = template_ds.isel({job.append_dim: job.region})
    region_ds = ds.isel({job.append_dim: job.region})
    assert np.array_equal(region_ds.time.values, region_template_ds.time.values)

    expected_values = np.ones((18, 10, 15))
    expected_values[0, :, :] = np.nan
    expected_values[6, :, :] = np.nan
    for data_var in region_ds.data_vars.values():
        np.testing.assert_array_equal(data_var.values, expected_values)


@pytest.mark.slow
@pytest.mark.parametrize("var_fill_value", [np.nan, 0.0])
def test_region_job_empty_chunk_writing(
    store_factory: StoreFactory,
    monkeypatch: pytest.MonkeyPatch,
    var_fill_value: float,
) -> None:
    template_ds = _create_template_ds(num_vars=1, var_fill_value=var_fill_value)

    tmp_store = get_local_tmp_store()

    # Write zarr metadata for this RegionJob to write into
    template_utils.write_metadata(template_ds, store_factory)

    jobs = ExampleRegionJob.get_jobs(
        "backfill",
        tmp_store,
        template_ds,
        "time",
        [ExampleDataVar(name=str(name)) for name in template_ds.data_vars],
        reformat_job_name="test-job",
    )

    def read_data(
        self: ExampleRegionJob,
        coord: ExampleSourceFileCoords,
        data_var: ExampleDataVar,
    ) -> ArrayFloat32:
        # Write only data filled with var_fill_value to the second shard.
        # If write_empty_chunks was False, this would lead us to not write
        # a shard to disk. Our current behavior is to write empty chunks,
        # so we should expect the 1.0.0 shards to be present and they should
        # be read as filled with var_fill_value.
        if coord.time >= pd.Timestamp("2025-01-02T00"):
            return np.full((10, 15), var_fill_value, dtype=np.float32)
        else:
            return np.full((10, 15), 1.0, dtype=np.float32)

    monkeypatch.setattr(ExampleRegionJob, "read_data", read_data)

    primary_store = store_factory.primary_store(writable=True)
    replica_stores = store_factory.replica_stores(writable=True)

    for job in jobs:
        template_utils.write_metadata(job.template_ds, tmp_store)
        job.process(primary_store, replica_stores)

    ds = xr.open_zarr(store_factory.primary_store())

    result = validation.check_for_expected_shards(primary_store, ds)
    assert result.passed, result.message

    # ExampleRegionJob.download_file raises FileNotFoundError for 2025-01-01T00:00
    # so we don't write to this region in the shared array. Therefore we expect the
    # values in this region to be NaN (the initialized value for float shared arrays)
    first_hour = (
        ds.sel(time=slice("2025-01-01T00:00", "2025-01-01T00:00")).to_array().to_numpy()
    )
    assert np.isnan(first_hour).all()

    middle_hours = (
        ds.sel(time=slice("2025-01-01T01:00", "2025-01-01T23:00")).to_array().to_numpy()
    )
    np.testing.assert_array_equal(middle_hours, np.full_like(middle_hours, 1.0))

    # read_data returns var_fill_value for times >= 2025-01-02
    second_day = ds.sel(time=slice("2025-01-02", None)).to_array().to_numpy()
    np.testing.assert_array_equal(second_day, np.full_like(second_day, var_fill_value))


def test_update_template_with_results(template_ds: xr.Dataset) -> None:
    tmp_store = get_local_tmp_store()

    job = ExampleRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=[ExampleDataVar(name=str(name)) for name in template_ds.data_vars],
        append_dim="time",
        region=slice(0, 18),
        reformat_job_name="test-job",
    )

    # Mock process results
    process_results = {
        "var0": [
            ExampleSourceFileCoords(
                time=pd.Timestamp("2025-01-01T12"), status=SourceFileStatus.Succeeded
            ),
            ExampleSourceFileCoords(
                time=pd.Timestamp("2025-01-02T12"),
                status=SourceFileStatus.DownloadFailed,
            ),
        ],
        "var1": [
            ExampleSourceFileCoords(
                time=pd.Timestamp("2025-01-02T00"), status=SourceFileStatus.Succeeded
            )
        ],
    }

    updated_template = job.update_template_with_results(process_results)

    assert updated_template.time.max() == pd.Timestamp("2025-01-02T00")


def test_source_file_coord_out_loc_default_impl() -> None:
    coord = ExampleSourceFileCoords(time=pd.Timestamp("2025-01-01T00"))
    assert coord.out_loc() == {"time": pd.Timestamp("2025-01-01T00")}


def test_source_file_coord_append_dim_coord() -> None:
    coord = ExampleSourceFileCoords(time=pd.Timestamp("2025-01-01T00"))
    assert coord.append_dim_coord == pd.Timestamp("2025-01-01T00")


def test_get_jobs_grouping_no_filters(template_ds: xr.Dataset) -> None:
    data_vars = [ExampleDataVar(name=str(name)) for name in template_ds.data_vars]
    tmp_store = get_local_tmp_store()
    jobs = ExampleRegionJob.get_jobs(
        kind="backfill",
        tmp_store=tmp_store,
        template_ds=template_ds,
        append_dim="time",
        all_data_vars=data_vars,
        reformat_job_name="test-job",
    )
    # RegionJobA groups vars into batches of 3 -> [3,1] and then then max_backfill_jobs of 2 -> [2,1,1]
    # and shards of size 24 -> 2 shards

    # jobs are sorted by region start
    assert all(a.region.start <= b.region.start for a, b in pairwise(jobs))

    assert len(jobs) == 6
    assert [j.data_vars for j in jobs] == [
        (data_vars[0], data_vars[1]),
        (data_vars[2],),
        (data_vars[3],),
        (data_vars[0], data_vars[1]),
        (data_vars[2],),
        (data_vars[3],),
    ]
    assert [j.region for j in jobs] == [
        slice(0, 24),
        slice(0, 24),
        slice(0, 24),
        slice(24, 48),
        slice(24, 48),
        slice(24, 48),
    ]


def test_get_jobs_grouping_filters(template_ds: xr.Dataset) -> None:
    data_vars = [ExampleDataVar(name=str(name)) for name in template_ds.data_vars]
    tmp_store = get_local_tmp_store()
    jobs = ExampleRegionJob.get_jobs(
        kind="backfill",
        tmp_store=tmp_store,
        template_ds=template_ds,
        append_dim="time",
        all_data_vars=data_vars,
        reformat_job_name="test-job",
        filter_variable_names=["var0", "var1", "var2"],
        filter_start=pd.Timestamp("2025-01-02T03"),
        filter_end=pd.Timestamp("2025-01-02T06"),
    )
    # RegionJobA groups vars into batches of 3 -> [3] and then then max_backfill_jobs of 2 -> [2, 1]
    # and shards of size 24 -> 2 shards
    # but filters limit to only second shard

    # jobs are sorted by region start
    assert all(a.region.start <= b.region.start for a, b in pairwise(jobs))

    assert len(jobs) == 2
    assert [j.data_vars for j in jobs] == [
        (data_vars[0], data_vars[1]),
        (data_vars[2],),
    ]
    assert [j.region for j in jobs] == [
        slice(24, 48),
        slice(24, 48),
    ]
    processing_region_ds, output_region_ds = jobs[0]._get_region_datasets()
    np.testing.assert_array_equal(
        processing_region_ds.time.values,
        pd.date_range("2025-01-02T00", freq="h", periods=24),
    )
    np.testing.assert_array_equal(
        output_region_ds.time.values,
        pd.date_range("2025-01-02T00", freq="h", periods=24),
    )


def test_get_jobs_grouping_filters_and_worker_index(template_ds: xr.Dataset) -> None:
    data_vars = [ExampleDataVar(name=str(name)) for name in template_ds.data_vars]
    tmp_store = get_local_tmp_store()
    jobs = ExampleRegionJob.get_jobs(
        kind="backfill",
        tmp_store=tmp_store,
        template_ds=template_ds,
        append_dim="time",
        all_data_vars=data_vars,
        reformat_job_name="test-job",
        filter_variable_names=["var0", "var1", "var2"],
        filter_start=pd.Timestamp("2025-01-02T03"),
        filter_end=pd.Timestamp("2025-01-02T06"),
        worker_index=0,
        workers_total=2,
    )
    # RegionJobA groups vars into batches of 3 -> [3] and then then max_backfill_jobs of 2 -> [2, 1]
    # and shards of size 24 -> 2 shards
    # but filters limit to only second shard
    # and then gets the first worker's single job

    # jobs are sorted by region start
    assert all(a.region.start <= b.region.start for a, b in pairwise(jobs))

    assert len(jobs) == 1
    assert [j.data_vars for j in jobs] == [
        (data_vars[0], data_vars[1]),
    ]
    assert [j.region for j in jobs] == [
        slice(24, 48),
    ]


def test_get_jobs_grouping_filter_contains(template_ds: xr.Dataset) -> None:
    data_vars = [ExampleDataVar(name=str(name)) for name in template_ds.data_vars]
    tmp_store = get_local_tmp_store()
    jobs = ExampleRegionJob.get_jobs(
        kind="backfill",
        tmp_store=tmp_store,
        template_ds=template_ds,
        append_dim="time",
        all_data_vars=data_vars,
        reformat_job_name="test-job",
        filter_contains=[pd.Timestamp("2025-01-01T05")],
    )
    # Only the first shard [0,24)
    assert all(a.region.start <= b.region.start for a, b in pairwise(jobs))
    assert len(jobs) == 3
    assert [j.data_vars for j in jobs] == [
        (data_vars[0], data_vars[1]),
        (data_vars[2],),
        (data_vars[3],),
    ]
    assert [j.region for j in jobs] == [
        slice(0, 24),
        slice(0, 24),
        slice(0, 24),
    ]


def test_get_jobs_grouping_filter_contains_second_shard(
    template_ds: xr.Dataset,
) -> None:
    data_vars = [ExampleDataVar(name=str(name)) for name in template_ds.data_vars]
    tmp_store = get_local_tmp_store()
    jobs = ExampleRegionJob.get_jobs(
        kind="backfill",
        tmp_store=tmp_store,
        template_ds=template_ds,
        append_dim="time",
        all_data_vars=data_vars,
        reformat_job_name="test-job",
        filter_contains=[pd.Timestamp("2025-01-02T00")],
    )
    # Only the second shard [24,48)
    assert all(a.region.start <= b.region.start for a, b in pairwise(jobs))
    assert len(jobs) == 3
    assert [j.data_vars for j in jobs] == [
        (data_vars[0], data_vars[1]),
        (data_vars[2],),
        (data_vars[3],),
    ]
    assert [j.region for j in jobs] == [
        slice(24, 48),
        slice(24, 48),
        slice(24, 48),
    ]


def test_get_jobs_grouping_filter_contains_all_shards(template_ds: xr.Dataset) -> None:
    data_vars = [ExampleDataVar(name=str(name)) for name in template_ds.data_vars]
    tmp_store = get_local_tmp_store()
    jobs = ExampleRegionJob.get_jobs(
        kind="backfill",
        tmp_store=tmp_store,
        template_ds=template_ds,
        append_dim="time",
        all_data_vars=data_vars,
        reformat_job_name="test-job",
        filter_contains=[pd.Timestamp("2025-01-01T12"), pd.Timestamp("2025-01-02T06")],
    )
    # Only the second shard [24,48)
    assert all(a.region.start <= b.region.start for a, b in pairwise(jobs))
    assert len(jobs) == 6
    assert [j.data_vars for j in jobs] == [
        (data_vars[0], data_vars[1]),
        (data_vars[2],),
        (data_vars[3],),
        (data_vars[0], data_vars[1]),
        (data_vars[2],),
        (data_vars[3],),
    ]
    assert [j.region for j in jobs] == [
        slice(0, 24),
        slice(0, 24),
        slice(0, 24),
        slice(24, 48),
        slice(24, 48),
        slice(24, 48),
    ]
