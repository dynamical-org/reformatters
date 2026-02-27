from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from reformatters.common import validation
from reformatters.common.pydantic import replace
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.mrms.conus_analysis_hourly.dynamical_dataset import (
    NoaaMrmsConusAnalysisHourlyDataset,
)
from reformatters.noaa.mrms.conus_analysis_hourly.region_job import (
    NoaaMrmsRegionJob,
    NoaaMrmsSourceFileCoord,
)
from reformatters.noaa.mrms.conus_analysis_hourly.template_config import (
    NoaaMrmsConusAnalysisHourlyTemplateConfig,
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


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset()

    filter_variable_names = [
        "precipitation_surface",
        "categorical_precipitation_type_surface",
    ]

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-01-15T02:00"),
        filter_start=pd.Timestamp("2024-01-15T00:00"),
        filter_variable_names=filter_variable_names,
    )

    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert_array_equal(
        backfill_ds["time"],
        pd.date_range("2024-01-15T00:00", "2024-01-15T01:00", freq="1h"),
    )

    # categorical_precipitation_type_surface is instant, no deaccumulation NaN
    assert_no_nulls(backfill_ds["categorical_precipitation_type_surface"])

    # precipitation_surface first timestep is NaN from deaccumulation
    point_ds = backfill_ds.isel(latitude=1750, longitude=3500)
    assert np.isnan(point_ds["precipitation_surface"].values[0])
    assert np.isfinite(point_ds["precipitation_surface"].values[1])

    # Operational update
    dataset = make_dataset()
    append_dim_end = pd.Timestamp("2024-01-15T05:00")
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: append_dim_end),
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

    # Should extend beyond the backfill
    assert updated_ds["time"].max() >= pd.Timestamp("2024-01-15T02:00")


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


@pytest.mark.slow
def test_precipitation_not_null_at_shard_boundary() -> None:
    """
    Test that precipitation_surface is not NaN at the start of the 2nd shard.
    Deaccumulation needs a previous value, but shard boundaries should have valid data
    due to processing region buffering.
    """
    dataset = make_dataset()
    config = dataset.template_config

    precip_var = next(v for v in config.data_vars if v.name == "precipitation_surface")
    time_dim_index = config.dims.index("time")
    assert isinstance(precip_var.encoding.shards, tuple)
    time_shard_size = precip_var.encoding.shards[time_dim_index]

    shard_2_start = (
        config.append_dim_start + time_shard_size * config.append_dim_frequency
    )

    assert time_shard_size == 2160

    dataset.backfill_local(
        append_dim_end=shard_2_start + pd.Timedelta(hours=3),
        filter_start=shard_2_start,
        filter_variable_names=["precipitation_surface"],
    )

    ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )

    shard_2_ds = ds.sel(time=slice(shard_2_start, None))
    expected_times = pd.date_range(shard_2_start, periods=3, freq="1h")
    assert_array_equal(shard_2_ds["time"].values, expected_times)

    # All timesteps at start of shard 2 should have valid precipitation
    precip = shard_2_ds["precipitation_surface"].isel(latitude=1750, longitude=3500)
    assert_no_nulls(precip)


@pytest.mark.slow
def test_single_file_integration(tmp_path: Path) -> None:
    """Download a real MRMS file from S3, read all template variables, and verify
    that the GRIB lat/lon coordinates exactly match dimension_coordinates."""
    template_config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    time = pd.Timestamp("2024-01-15T12:00")

    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-mrms-conus-analysis-hourly"}
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coords_checked = False
    for group in NoaaMrmsRegionJob.source_groups(template_config.data_vars):
        data_var = group[0]
        internal = data_var.internal_attrs
        coord = NoaaMrmsSourceFileCoord(
            time=time,
            product=internal.mrms_product,
            level=internal.mrms_level,
        )
        coord = replace(coord, downloaded_path=region_job.download_file(coord))

        for data_var in group:
            data = region_job.read_data(coord, data_var)
            assert data.shape == (3500, 7000), f"Wrong shape for {data_var.name}"
            assert not np.all(np.isnan(data)), f"All NaN for {data_var.name}"

        # Verify spatial info from the GRIB matches template (once is sufficient)
        if not coords_checked:
            assert coord.downloaded_path is not None

            ds = xr.open_dataset(coord.downloaded_path, engine="rasterio")

            # Verify lat/lon coordinates match template dimension_coordinates
            dim_coords = template_config.dimension_coordinates()
            np.testing.assert_allclose(
                ds.y.values,
                dim_coords["latitude"],
                atol=1e-6,
                err_msg="GRIB latitudes do not match template dimension_coordinates",
            )
            np.testing.assert_allclose(
                ds.x.values,
                dim_coords["longitude"],
                atol=1e-6,
                err_msg="GRIB longitudes do not match template dimension_coordinates",
            )

            # Verify CRS/spatial_ref attributes match template
            spatial_ref_coord = next(
                c for c in template_config.coords if c.name == "spatial_ref"
            )
            template_attrs = spatial_ref_coord.attrs.model_dump(exclude_none=True)
            file_attrs = dict(ds.spatial_ref.attrs)
            # Compare attributes present in both (template has custom comment,
            # file has GeoTransform generated by rasterio)
            common_keys = set(template_attrs) & set(file_attrs)
            for key in common_keys:
                assert file_attrs[key] == template_attrs[key], (
                    f"spatial_ref.{key}: file={file_attrs[key]!r} != template={template_attrs[key]!r}"
                )

            coords_checked = True

    assert coords_checked
