from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.nasa.imerg.analysis_early_v7.dynamical_dataset import (
    NasaImergAnalysisEarlyV7Dataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG

# Synthetic per-variable source values returned by the mocked HDF5 reader.
# precipitation is in mm/hr and is scaled by 1/3600 on read.
_PRECIP_MM_PER_HR = 3.6
_PROBABILITY_PERCENT = 80.0
_QUALITY_INDEX = 0.9


@pytest.fixture
def dataset() -> NasaImergAnalysisEarlyV7Dataset:
    return NasaImergAnalysisEarlyV7Dataset(primary_storage_config=NOOP_STORAGE_CONFIG)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch, dataset: NasaImergAnalysisEarlyV7Dataset
) -> None:
    # Mock NASA Earthdata authenticated download.
    mock_session = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_content = lambda chunk_size: [b"fake_hdf5_data"]
    mock_session.get.return_value = mock_response
    monkeypatch.setattr(
        "reformatters.nasa.imerg.region_job.get_authenticated_session",
        lambda: mock_session,
    )

    # Mock rasterio to return a uniform (lon=3600, lat=1800) array per variable.
    def mock_rasterio_open(path: str) -> MagicMock:
        if "precipitationQualityIndex" in path:
            value = _QUALITY_INDEX
        elif "probabilityLiquidPrecipitation" in path:
            value = _PROBABILITY_PERCENT
        elif "precipitation" in path:
            value = _PRECIP_MM_PER_HR
        else:
            value = 0.0
        mock_reader = MagicMock()
        mock_reader.read.return_value = np.full((3600, 1800), value, dtype=np.float32)
        mock_reader.__enter__ = lambda self: self
        mock_reader.__exit__ = lambda self, *args: None
        return mock_reader

    monkeypatch.setattr("rasterio.open", mock_rasterio_open)

    dataset.backfill_local(append_dim_end=pd.Timestamp("1998-01-01T01:30"))
    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    np.testing.assert_array_equal(
        ds.time,
        pd.date_range("1998-01-01T00:00", "1998-01-01T01:00", freq="30min"),
    )

    # Operational update appends the next granule.
    monkeypatch.setattr(
        "pandas.Timestamp.now", lambda tz=None: pd.Timestamp("1998-01-01T02:00")
    )
    dataset.update("test-update")

    updated_ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    np.testing.assert_array_equal(
        updated_ds.time,
        pd.date_range("1998-01-01T00:00", "1998-01-01T01:30", freq="30min"),
    )

    point = updated_ds.sel(latitude=0.0, longitude=0.0, method="nearest")
    # precipitation and quality use keep_mantissa_bits=7 rounding, so allow ~0.5% tolerance.
    np.testing.assert_allclose(
        point["precipitation_surface"].values,
        np.full(4, _PRECIP_MM_PER_HR / 3600.0, dtype=np.float32),  # mm/hr -> kg m-2 s-1
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        point["precipitation_quality_index_surface"].values,
        np.full(4, _QUALITY_INDEX, dtype=np.float32),
        rtol=1e-2,
    )
    # probability uses no-rounding, so it is stored exactly.
    np.testing.assert_array_equal(
        point["probability_of_liquid_precipitation_surface"].values,
        np.full(4, _PROBABILITY_PERCENT, dtype=np.float32),
    )


def test_operational_kubernetes_resources(
    dataset: NasaImergAnalysisEarlyV7Dataset,
) -> None:
    cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs
    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    # Earthdata credentials are required for the update job.
    assert "nasa-earthdata" in update_cron_job.secret_names
    assert update_cron_job.shared_memory is not None


def test_validators(dataset: NasaImergAnalysisEarlyV7Dataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
