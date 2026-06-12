from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.gefs.forecast_10_day_spatial.dynamical_dataset import (
    GefsForecast10DaySpatialDataset,
)
from reformatters.noaa.gefs.forecast_10_day_spatial.region_job import (
    _S3_LOCATION_PREFIX,
    GefsForecast10DaySpatialRegionJob,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar


@pytest.fixture
def dataset() -> GefsForecast10DaySpatialDataset:
    return GefsForecast10DaySpatialDataset(
        primary_storage_config=StorageConfig(
            base_path="s3://test-bucket/path", format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = GefsForecast10DaySpatialDataset(
        primary_storage_config=StorageConfig(
            base_path="s3://test-bucket/path", format=DatasetFormat.ICECHUNK
        ),
    )
    filter_variable_names = ["temperature_2m", "total_precipitation_surface"]

    # Trim to 2 ensemble members and 3 lead times to limit network use.
    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: (
            orig_get_template(end_time)
            .isel(ensemble_member=slice(0, 2))
            .sel(lead_time=slice("0h", "6h"))
        ),
    )

    # 1. Backfill a fixed, immutable slice of the archive.
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-01-01T01:00"),
        filter_start=pd.Timestamp("2023-12-31T12:00"),
        filter_variable_names=filter_variable_names,
    )

    ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert ds.init_time.values[-1] == np.datetime64("2024-01-01T00:00")
    point = ds.sel(init_time="2024-01-01T00:00", latitude=0, longitude=0)
    # Raw GRIB values: Kelvin temperatures, window-accumulated precipitation.
    np.testing.assert_allclose(
        point["temperature_2m"].sel(ensemble_member=0, lead_time=["3h", "6h"]).values,
        [300.23886718750003, 300.437578125],
    )
    np.testing.assert_allclose(
        point["temperature_2m"].sel(ensemble_member=1, lead_time=["3h", "6h"]).values,
        [300.52451171875003, 300.636171875],
    )
    np.testing.assert_allclose(
        point["total_precipitation_surface"]
        .sel(ensemble_member=0, lead_time=["3h", "6h"])
        .values,
        [2.7800000000000002, 4.9],
    )
    np.testing.assert_allclose(
        point["total_precipitation_surface"]
        .sel(ensemble_member=1, lead_time=["3h", "6h"])
        .values,
        [1.7000000000000002, 3.4000000000000004],
    )
    # Accumulated precipitation has no hour-0 values; temperature does.
    assert point["total_precipitation_surface"].sel(lead_time="0h").isnull().all()
    assert point["temperature_2m"].sel(lead_time="0h").notnull().all()

    # 2. Operational update: "now" is during the 06z publication window.
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2024-01-01T07:00")),
    )
    # Updates process all variables; filter to the test's two.
    orig_update_jobs = (
        GefsForecast10DaySpatialRegionJob.operational_update_jobs.__func__
    )  # type: ignore[attr-defined]

    def filtered_update_jobs(
        cls: type[GefsForecast10DaySpatialRegionJob],
        *,
        all_data_vars: Sequence[GEFSDataVar],
        **kwargs: Any,  # noqa: ANN401 - passthrough to the wrapped classmethod
    ) -> object:
        return orig_update_jobs(
            cls,
            all_data_vars=[v for v in all_data_vars if v.name in filter_variable_names],
            **kwargs,
        )

    monkeypatch.setattr(
        GefsForecast10DaySpatialRegionJob,
        "operational_update_jobs",
        classmethod(filtered_update_jobs),
    )

    dataset.update("test-update")

    updated_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert updated_ds.init_time.values[-1] == np.datetime64("2024-01-01T06:00")
    updated_point = updated_ds.sel(
        init_time="2024-01-01T06:00",
        ensemble_member=0,
        lead_time="3h",
        latitude=0,
        longitude=0,
    )
    np.testing.assert_allclose(updated_point["temperature_2m"].values, 300.7870703125)
    np.testing.assert_allclose(updated_point["total_precipitation_surface"].values, 0.2)


def test_virtual_container_matches_ref_locations(
    dataset: GefsForecast10DaySpatialDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == _S3_LOCATION_PREFIX


def test_operational_kubernetes_resources(
    dataset: GefsForecast10DaySpatialDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.secret_names == dataset.store_factory.k8s_secret_names()
    # One fire per init, 3 minutes before the earliest observed publication start
    # (init+3:46); the deadline bounds polling on a file that never publishes.
    assert update_cron_job.schedule == "43 3,9,15,21 * * *"
    assert update_cron_job.pod_active_deadline == timedelta(hours=2, minutes=10)
    # Virtual operational updates are strictly single-writer.
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    # Suspended until the operational test launches.
    assert update_cron_job.suspend is True
    assert validation_cron_job.suspend is True
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"


def test_validators(dataset: GefsForecast10DaySpatialDataset) -> None:
    validators = tuple(dataset.validators())
    assert validators == (validation.check_forecast_current_data,)
