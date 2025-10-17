from unittest.mock import Mock

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
    orig_get_template = dataset.template_config.get_template

    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).sel(
            lead_time=slice("0h", "6h"), ensemble_member=slice(0, 1)
        ),
    )

    dataset.backfill_local(append_dim_end=pd.Timestamp("2024-02-04T00:00:00"))

    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    # existing_ds_append_dim_end is exclusive, so should have only processed the first forecast
    assert ds.init_time.values == [np.datetime64("2024-02-03T00:00:00")]

    actual_values = ds.sel(
        init_time="2024-02-03T00:00:00", latitude=0, longitude=0
    ).temperature_2m

    expected_values = np.array(
        [
            [27.0, 28.5],  # lead time 0h, ensemble members 0 and 1
            [27.25, 25.75],  # lead time 3h, ensemble members 0 and 1
            [27.25, 26.875],  # lead time 6h, ensemble members 0 and 1
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(actual_values, expected_values)

    # Operational update
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        Mock(return_value=pd.Timestamp("2024-02-05T00:00:00")),
    )
    dataset.update("test-update")

    updated_ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)

    np.testing.assert_array_equal(
        updated_ds.init_time,
        np.array(
            [np.datetime64("2024-02-03T00:00:00"), np.datetime64("2024-02-04T00:00:00")]
        ),
    )
    actual_values = updated_ds.sel(latitude=0, longitude=0).temperature_2m
    expected_values = np.array(
        [
            [  # init time 2024-02-03T00:00:00
                [27.0, 28.5],  # lead time 0h, ensemble members 0 and 1
                [27.25, 25.75],  # lead time 3h, ensemble members 0 and 1
                [27.25, 26.875],  # lead time 6h, ensemble members 0 and 1
            ],
            [  # init time 2024-02-04T00:00:00
                [27.875, 28.375],  # lead time 0h, ensemble members 0 and 1
                [27.75, 27.375],  # lead time 3h, ensemble members 0 and 1
                [27.75, 27.625],  # lead time 6h, ensemble members 0 and 1
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual_values, expected_values)


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
