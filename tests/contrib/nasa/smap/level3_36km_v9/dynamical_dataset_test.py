from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.contrib.nasa.smap.level3_36km_v9.dynamical_dataset import (
    NasaSmapLevel336KmV9Dataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> NasaSmapLevel336KmV9Dataset:
    return NasaSmapLevel336KmV9Dataset(primary_storage_config=NOOP_STORAGE_CONFIG)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch, dataset: NasaSmapLevel336KmV9Dataset
) -> None:
    # Mock NASA Earthdata authentication
    mock_session = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_content = lambda chunk_size: [b"fake_hdf5_data"]
    mock_session.get.return_value = mock_response

    monkeypatch.setattr(
        "reformatters.contrib.nasa.smap.level3_36km_v9.region_job.get_authenticated_session",
        lambda: mock_session,
    )

    # Mock rasterio to return fake soil moisture data
    # Create data that varies by date: 0.25, 0.30, 0.35 for the three days
    def mock_rasterio_open(path: str) -> MagicMock:
        mock_reader = MagicMock()
        # Extract date from path to determine which values to return
        if "20150401" in path:
            value = 0.25
        elif "20150402" in path:
            value = 0.30
        elif "20150403" in path:
            value = 0.35
        else:
            value = 0.0

        # Create array with shape (406, 964) matching y_size, x_size
        data = np.full((406, 964), value, dtype=np.float32)
        mock_reader.read.return_value = data
        mock_reader.__enter__ = lambda self: self
        mock_reader.__exit__ = lambda self, *args: None
        return mock_reader

    monkeypatch.setattr("rasterio.open", mock_rasterio_open)

    # Local backfill reformat
    dataset.backfill_local(append_dim_end=pd.Timestamp("2015-04-03"))
    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    assert ds.time.min() == pd.Timestamp("2015-04-01")
    assert ds.time.max() == pd.Timestamp("2015-04-02")

    # Operational update - mock pd.Timestamp.now() to control the update end time
    monkeypatch.setattr(
        "pandas.Timestamp.now",
        lambda tz=None: pd.Timestamp("2015-04-04"),
    )

    dataset.update("test-update")

    # Check resulting dataset
    updated_ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)

    np.testing.assert_array_equal(
        updated_ds.time,
        pd.date_range(
            "2015-04-01",
            "2015-04-03",
            freq=dataset.template_config.append_dim_frequency,
        ),
    )
    subset_ds = updated_ds.sel(x=0, y=0, method="nearest")
    np.testing.assert_array_equal(
        subset_ds["soil_moisture_am"].values, [0.25, 0.30, 0.35]
    )
    np.testing.assert_array_equal(
        subset_ds["soil_moisture_pm"].values, [0.25, 0.30, 0.35]
    )


def test_operational_kubernetes_resources(
    dataset: NasaSmapLevel336KmV9Dataset,
) -> None:
    cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs
    assert update_cron_job.name == f"{dataset.dataset_id}-operational-update"
    assert validation_cron_job.name == f"{dataset.dataset_id}-validation"
    assert update_cron_job.secret_names == [
        dataset.primary_storage_config.k8s_secret_name,
        "nasa-earthdata",
    ]
    assert validation_cron_job.secret_names == [
        dataset.primary_storage_config.k8s_secret_name
    ]


def test_validators(dataset: NasaSmapLevel336KmV9Dataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
