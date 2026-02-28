from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from zarr.abc.store import Store

from reformatters.common import region_job as region_job_module
from reformatters.common import shared_memory_utils, validation
from reformatters.common.iterating import shard_slice_indexers
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.types import AppendDim
from reformatters.noaa.mrms.conus_analysis_hourly.dynamical_dataset import (
    NoaaMrmsConusAnalysisHourlyDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG
from tests.xarray_testing import assert_no_nulls


@pytest.fixture
def dataset() -> NoaaMrmsConusAnalysisHourlyDataset:
    return make_dataset()


def make_dataset() -> NoaaMrmsConusAnalysisHourlyDataset:
    return NoaaMrmsConusAnalysisHourlyDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG,
        replica_storage_configs=[
            StorageConfig(
                base_path="s3://replica-bucket/path", format=DatasetFormat.ICECHUNK
            )
        ],
    )


def _patch_write_first_shard_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only write the first spatial shard (all indexer slices start at 0).

    Replaces write_shards rather than write_shard_to_zarr because the latter
    is pickled for ProcessPoolExecutor and local functions can't be pickled.
    """
    _orig_write_shard_to_zarr = shared_memory_utils.write_shard_to_zarr

    def _first_shard_only_write_shards(
        processing_region_da_template: xr.DataArray,
        shared_buffer: object,
        append_dim: AppendDim,
        output_region_ds: xr.Dataset,
        store: Store | Path,
        cpu_process_executor: object,
    ) -> None:
        all_indexers = shard_slice_indexers(
            output_region_ds[processing_region_da_template.name]
        )
        for si in all_indexers:
            if all(s.start == 0 for s in si):
                _orig_write_shard_to_zarr(
                    processing_region_da_template,
                    shared_buffer.name,  # type: ignore[union-attr]
                    append_dim,
                    output_region_ds,
                    store,
                    si,
                )

    monkeypatch.setattr(
        region_job_module, "write_shards", _first_shard_only_write_shards
    )


def _set_time_shard_size(ds: xr.Dataset, time_shard_size: int) -> xr.Dataset:
    """Override the time shard/chunk encoding on all data vars."""
    for var in ds.data_vars.values():
        time_idx = var.dims.index("time")
        shards = list(var.encoding["shards"])
        chunks = list(var.encoding["chunks"])
        shards[time_idx] = time_shard_size
        chunks[time_idx] = time_shard_size
        var.encoding["shards"] = tuple(shards)
        var.encoding["chunks"] = tuple(chunks)
    return ds


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Trim time dimension and set time shard size to 2 so that:
    #  - backfill fills exactly 1 time shard (2 timesteps)
    #  - operational update crosses a shard boundary (adds 1 timestep in shard 2)
    # This also tests deaccumulation at shard boundaries via get_processing_region.
    test_start = pd.Timestamp("2024-01-15T00:00")

    def _trimmed_get_template(
        dataset: NoaaMrmsConusAnalysisHourlyDataset,
    ) -> None:
        orig = dataset._get_template
        monkeypatch.setattr(
            dataset,
            "_get_template",
            lambda end: _set_time_shard_size(
                orig(end).sel(time=slice(test_start, None)),
                time_shard_size=2,
            ),
        )

    _patch_write_first_shard_only(monkeypatch)

    dataset = make_dataset()
    _trimmed_get_template(dataset)

    filter_variable_names = [
        "precipitation_surface",
        "categorical_precipitation_type_surface",
    ]

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-01-15T02:00"),
        filter_start=test_start,
        filter_variable_names=filter_variable_names,
    )

    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert_array_equal(
        backfill_ds["time"],
        pd.date_range("2024-01-15T00:00", "2024-01-15T01:00", freq="1h"),
    )

    # Only the first spatial shard is written (lat < 700, lon < 1400).
    first_shard = backfill_ds.isel(latitude=slice(0, 700), longitude=slice(0, 1400))

    # categorical_precipitation_type_surface is instant, no deaccumulation NaN
    assert_no_nulls(first_shard["categorical_precipitation_type_surface"])

    # precipitation_surface first timestep is NaN from deaccumulation
    point = backfill_ds.isel(latitude=622, longitude=817)
    assert np.isnan(point["precipitation_surface"].values[0])
    assert np.all(np.isfinite(point["precipitation_surface"].values[1:]))

    # Snapshot: snow (cat=3) with non-zero precipitation at this point
    assert_allclose(
        point["precipitation_surface"].values,
        np.array([np.nan, 0.00125122], dtype=np.float32),
        rtol=1e-4,
    )
    assert_array_equal(
        point["categorical_precipitation_type_surface"].values,
        np.array([3.0, 3.0], dtype=np.float32),
    )

    # Operational update adds one more hour, crossing a time shard boundary.
    # precipitation_surface should still be valid (not NaN) at the shard boundary
    # because get_processing_region buffers one step back for deaccumulation.
    dataset = make_dataset()
    _trimmed_get_template(dataset)
    update_end = pd.Timestamp("2024-01-15T03:00")
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: update_end),
    )
    orig_get_jobs = dataset.region_job_class.get_jobs
    monkeypatch.setattr(
        dataset.region_job_class,
        "get_jobs",
        lambda *args, **kwargs: orig_get_jobs(
            *args, **{**kwargs, "filter_variable_names": filter_variable_names}
        ),
    )

    dataset.update("test-update")

    updated_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )

    assert_array_equal(
        updated_ds["time"],
        pd.date_range("2024-01-15T00:00", "2024-01-15T02:00", freq="1h"),
    )

    updated_point = updated_ds.isel(latitude=622, longitude=817)
    # All non-first timesteps have valid precipitation (including shard boundary)
    assert np.all(np.isfinite(updated_point["precipitation_surface"].values[1:]))
    assert_no_nulls(updated_point["categorical_precipitation_type_surface"])

    assert_allclose(
        updated_point["precipitation_surface"].values,
        np.array([np.nan, 0.00125122, 0.00141907], dtype=np.float32),
        rtol=1e-4,
    )
    assert_array_equal(
        updated_point["categorical_precipitation_type_surface"].values,
        np.array([3.0, 3.0, 3.0], dtype=np.float32),
    )


def test_operational_kubernetes_resources(
    dataset: NoaaMrmsConusAnalysisHourlyDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert len(update_cron_job.secret_names) > 0
    assert update_cron_job.suspend is True

    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert len(validation_cron_job.secret_names) > 0
    assert validation_cron_job.suspend is True


def test_validators(dataset: NoaaMrmsConusAnalysisHourlyDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
