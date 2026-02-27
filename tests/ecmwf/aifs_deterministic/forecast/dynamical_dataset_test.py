from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.ecmwf.aifs_deterministic.forecast.dynamical_dataset import (
    EcmwfAifsForecastDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfAifsForecastDataset:
    return EcmwfAifsForecastDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch, dataset: EcmwfAifsForecastDataset
) -> None:
    variables_to_check = ["temperature_2m", "precipitation_rate_surface"]
    monkeypatch.setattr(
        type(dataset.template_config),
        "data_vars",
        [
            var
            for var in dataset.template_config.data_vars
            if var.name in variables_to_check
        ],
    )

    orig_get_template = dataset.template_config.get_template

    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).sel(
            lead_time=slice("0h", "6h"),
        )[variables_to_check],
    )
    # Backfill one 6-hour init
    dataset.backfill_local(append_dim_end=pd.Timestamp("2024-04-01T06:00:00"))

    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    np.testing.assert_array_equal(
        ds.init_time.values, [np.datetime64("2024-04-01T00:00:00")]
    )

    point_ds = ds.sel(init_time="2024-04-01T00:00:00", latitude=0, longitude=0)
    assert point_ds["temperature_2m"].shape == (2,)
    assert not np.all(np.isnan(point_ds["temperature_2m"].values))

    # Snapshot values at (init_time=0h, lead_time=6h, lat=0, lon=0) from 2024-04-01
    point_6h = ds.sel(
        init_time="2024-04-01T00:00:00", latitude=0, longitude=0, lead_time="6h"
    )
    np.testing.assert_allclose(float(point_6h["temperature_2m"]), 28.75)
    np.testing.assert_allclose(
        float(point_6h["precipitation_rate_surface"]), 0.00164794921875
    )

    # Operational update
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        Mock(return_value=pd.Timestamp("2024-04-01T12:00:00")),
    )
    dataset.update("test-update")

    updated_ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    np.testing.assert_array_equal(
        updated_ds.init_time,
        np.array(
            [
                np.datetime64("2024-04-01T00:00:00"),
                np.datetime64("2024-04-01T06:00:00"),
            ]
        ),
    )

    # Snapshot complete temperature and precip values at the test point after update.
    # Backfill values from init_time=00Z should be unchanged alongside new 06Z values.
    point_updated = updated_ds.sel(latitude=0, longitude=0, lead_time="6h")
    np.testing.assert_allclose(
        point_updated["temperature_2m"].values,
        [28.75, 29.5],  # [00Z backfill, 06Z update]
    )
    np.testing.assert_allclose(
        point_updated["precipitation_rate_surface"].values,
        [0.00164794921875, 0.000202178955078125],  # [00Z backfill, 06Z update]
    )


def test_operational_kubernetes_resources(
    dataset: EcmwfAifsForecastDataset,
) -> None:
    cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs
    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert update_cron_job.secret_names == [
        dataset.primary_storage_config.k8s_secret_name
    ]
    assert validation_cron_job.secret_names == [
        dataset.primary_storage_config.k8s_secret_name
    ]


def test_validators(dataset: EcmwfAifsForecastDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
