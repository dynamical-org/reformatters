import numpy as np
import pandas as pd
import pytest
import xarray as xr

# from reformatters.common import validation
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfIfsEnsForecast15Day025DegreeDataset:
    return EcmwfIfsEnsForecast15Day025DegreeDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch, dataset: EcmwfIfsEnsForecast15Day025DegreeDataset
) -> None:
    existing_ds_append_dim_end = (
        dataset.template_config.append_dim_start + pd.Timedelta(days=1)
    )
    operational_update_append_dim_end = existing_ds_append_dim_end + pd.Timedelta(
        days=1
    )

    # Local backfill reformat
    dataset.backfill_local(append_dim_end=existing_ds_append_dim_end)
    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    # existing_ds_append_dim_end is exclusive, so should have only processed the first forecast
    assert ds.init_time.max() == dataset.template_config.append_dim_start

    # Operational update
    monkeypatch.setattr(
        dataset.region_job_class,
        "_update_append_dim_end",
        lambda: operational_update_append_dim_end,
    )
    # monkeypatch.setattr(
    #     dataset.region_job_class,
    #     "_update_append_dim_start",
    #     lambda existing_ds: pd.Timestamp(existing_ds.init_time.max().item()),
    # ) <- can we remove this one?

    dataset.update("test-update")

    # Check resulting dataset
    updated_ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    breakpoint()
    np.testing.assert_array_equal(
        updated_ds.init_time,
        pd.date_range(
            existing_ds_append_dim_end,
            operational_update_append_dim_end,
            freq=dataset.template_config.append_dim_frequency,
        ),
    )
    subset_ds = updated_ds.sel(latitude=0, longitude=0, method="nearest")
    breakpoint()
    np.testing.assert_array_equal(
        subset_ds["temperature_2m"].values, [190.0, 163.0, 135.0]
    )


# def test_operational_kubernetes_resources(
#     dataset: EcmwfIfsEnsForecast15Day025DegreeDataset,
# ) -> None:
#     cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")

#     assert len(cron_jobs) == 2
#     update_cron_job, validation_cron_job = cron_jobs
#     assert update_cron_job.name == f"{dataset.dataset_id}-operational-update"
#     assert validation_cron_job.name == f"{dataset.dataset_id}-validation"
#     assert update_cron_job.secret_names == [
#         dataset.primary_storage_config.k8s_secret_name
#     ]
#     assert validation_cron_job.secret_names == [
#         dataset.primary_storage_config.k8s_secret_name
#     ]


# def test_validators(dataset: EcmwfIfsEnsForecast15Day025DegreeDataset) -> None:
#     validators = tuple(dataset.validators())
#     assert len(validators) == 2
#     assert all(isinstance(v, validation.DataValidator) for v in validators)
