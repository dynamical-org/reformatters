from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from reformatters.common import validation
from reformatters.nasa.imerg.analysis_early import NasaImergAnalysisEarlyDataset
from reformatters.nasa.imerg.analysis_early.region_job import (
    NasaImergAnalysisEarlyRegionJob,
)
from reformatters.nasa.imerg.analysis_late import NasaImergAnalysisLateDataset
from reformatters.nasa.imerg.dynamical_dataset import (
    NasaImergAnalysisMaterializedDataset,
)
from reformatters.nasa.imerg.template_config import (
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    MM_PER_HR_TO_KG_M2_S,
)
from tests.chunk_utils import shrink_chunks_and_shards
from tests.common.dynamical_dataset_test import (
    NOOP_STORAGE_CONFIG,
    assert_configured_validators,
)

# Raw mm/hr precipitation returned by the mocked reader, keyed by granule start.
_PRECIP_MM_HR = {"S000000": 10.0, "S003000": 20.0, "S010000": 30.0}


def _mock_reader(subdataset_path: str) -> MagicMock:
    # Source band is (lon, lat); latitude index increases south -> north.
    lat_index = np.broadcast_to(
        np.arange(GRID_LAT_SIZE, dtype=np.float32), (GRID_LON_SIZE, GRID_LAT_SIZE)
    )
    if "precipitationQualityIndex" in subdataset_path:
        # Encode the source latitude index so the transpose/flip can be checked.
        data = lat_index.copy()
    else:
        start = next(s for s in _PRECIP_MM_HR if s in subdataset_path)
        data = np.full((GRID_LON_SIZE, GRID_LAT_SIZE), _PRECIP_MM_HR[start], np.float32)

    reader = MagicMock()
    reader.read.return_value = data
    reader.__enter__ = lambda self: self
    reader.__exit__ = lambda self, *args: None
    return reader


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = lambda: None
    mock_response.iter_content = lambda chunk_size: [b"fake_hdf5"]
    mock_session = Mock()
    mock_session.get.return_value = mock_response

    for name in ("get_earthdata_session", "get_pps_session"):
        monkeypatch.setattr(
            f"reformatters.nasa.imerg.region_job.{name}", lambda: mock_session
        )
    monkeypatch.setattr("rasterio.open", _mock_reader)


def _shrink_template(
    monkeypatch: pytest.MonkeyPatch, dataset: NasaImergAnalysisMaterializedDataset
) -> None:
    orig = dataset._get_template
    monkeypatch.setattr(
        dataset,
        "_get_template",
        lambda end: shrink_chunks_and_shards(orig(end)),
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pipeline(monkeypatch)
    dataset = NasaImergAnalysisEarlyDataset(primary_storage_config=NOOP_STORAGE_CONFIG)
    _shrink_template(monkeypatch, dataset)

    # Two 30-minute granules: 1998-01-01 00:00 and 00:30.
    dataset.backfill_local(append_dim_end=pd.Timestamp("1998-01-01T01:00"))
    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    assert_array_equal(
        ds["time"],
        pd.date_range("1998-01-01T00:00", "1998-01-01T00:30", freq="30min"),
    )

    # Unit conversion: 10 and 20 mm/hr -> kg m-2 s-1 (divide by 3600).
    point = ds.sel(latitude=0.05, longitude=0.05, method="nearest")
    # rtol accommodates keep_mantissa_bits=7 float rounding.
    assert_allclose(
        point["precipitation_surface"].values,
        np.array([10.0, 20.0], dtype=np.float32) * MM_PER_HR_TO_KG_M2_S,
        rtol=1e-2,
    )

    # Orientation: QI encodes the source latitude index (0 = south). After the
    # transpose + flip, latitude 89.95 (north) must carry the largest index.
    north = ds.sel(latitude=89.95, longitude=0.05, method="nearest")
    south = ds.sel(latitude=-89.95, longitude=0.05, method="nearest")
    assert north["precipitation_quality_index_surface"].values[0] > GRID_LAT_SIZE - 100
    assert south["precipitation_quality_index_surface"].values[0] < 100

    # Operational update adds the 01:00 granule through the jsimpson (PPS) path.
    # `now` is offset ahead by the run's publish latency buffer so the granule
    # arriving at 01:00 falls inside the requested append_dim_end.
    update_end = pd.Timestamp("1998-01-01T01:30")
    mocked_now = update_end + NasaImergAnalysisEarlyRegionJob.publish_latency
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **k: mocked_now))
    dataset.update("test-update")
    updated = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    assert_array_equal(
        updated["time"],
        pd.date_range("1998-01-01T00:00", "1998-01-01T01:00", freq="30min"),
    )
    updated_point = updated.sel(latitude=0.05, longitude=0.05, method="nearest")
    assert_allclose(
        updated_point["precipitation_surface"].values,
        np.array([10.0, 20.0, 30.0], dtype=np.float32) * MM_PER_HR_TO_KG_M2_S,
        rtol=1e-2,
    )

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources() -> None:
    dataset = NasaImergAnalysisEarlyDataset(primary_storage_config=NOOP_STORAGE_CONFIG)
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))
    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs
    assert update_cron_job.name == "nasa-imerg-analysis-early-update"
    assert "nasa-pps" in update_cron_job.secret_names
    assert "nasa-earthdata" in update_cron_job.secret_names
    assert validation_cron_job.name == "nasa-imerg-analysis-early-validate"


@pytest.mark.parametrize(
    "dataset",
    [
        NasaImergAnalysisEarlyDataset(primary_storage_config=NOOP_STORAGE_CONFIG),
        NasaImergAnalysisLateDataset(primary_storage_config=NOOP_STORAGE_CONFIG),
    ],
)
def test_validators(dataset: NasaImergAnalysisMaterializedDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
