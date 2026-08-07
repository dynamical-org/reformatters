"""Real-source integration tests for the Google WeatherNext 2 virtual dataset.

`gs://weathernext/` is private, so these need Google credentials in the environment
(GOOGLE_APPLICATION_CREDENTIALS or the other GCS variables object_store reads):

    WEATHERNEXT_CREDENTIALED_TESTS=1 uv run pytest \
        tests/google/weathernext2/forecast_virtual/real_files_integration_test.py

They confirm what only real bytes can: that a reference built from a real store listing
decodes to the value the source holds, in both source store layouts, with the read-time
unit conversions applied. The codec pipeline itself is proved without credentials by
template_config_test.test_encoding_decodes_bytes_the_source_wrote, and reference routing
by region_job_test.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.google.weathernext2.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastVirtualDataset,
)
from tests.common.dynamical_dataset_test import assert_configured_validators

# The source bucket needs credentials, which CI does not have — CI runs the full suite,
# so gate on an explicit opt-in rather than the `slow` marker alone.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("WEATHERNEXT_CREDENTIALED_TESTS") != "1",
        reason="requires Google credentials; set WEATHERNEXT_CREDENTIALED_TESTS=1 to run",
    ),
]

# 0 degrees north, 0 degrees east: latitude index 360, longitude index 0.
_LATITUDE, _LONGITUDE = 0.0, 0.0
_LEAD_TIME = pd.Timedelta("6h")


def _dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable_names: Sequence[str],
    init_time: pd.Timestamp,
) -> xr.Dataset:
    """Backfill one init of `variable_names` into a local icechunk store and open it.

    Trimmed to the first lead time so the backfill lists a handful of source prefixes;
    chunk shapes are untouched, since a virtual chunk must stay one source chunk.
    """
    dataset = GoogleWeathernext2ForecastVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )
    original_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: original_get_template(end_time).isel(lead_time=[0]),
    )
    dataset.backfill_local(
        append_dim_end=init_time + dataset.template_config.append_dim_frequency,
        filter_start=init_time,
        filter_variable_names=list(variable_names),
    )
    assert_configured_validators(dataset)
    return validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )


def test_per_init_store_era_snapshot_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_time = pd.Timestamp("2025-03-01T06:00")
    ds = _dataset(
        tmp_path,
        monkeypatch,
        ["temperature_2m", "total_precipitation_surface", "temperature"],
        init_time,
    )

    assert ds.init_time.values[-1] == init_time.to_datetime64()
    cell = ds.sel(
        latitude=_LATITUDE,
        longitude=_LONGITUDE,
        init_time=init_time,
        lead_time=_LEAD_TIME,
    )
    # Snapshot values: the source's float32 chunk with the read-time ScaleOffset applied
    # (K -> degree_Celsius, metres -> kg m-2).
    np.testing.assert_allclose(cell["temperature_2m"].values, 28.20932)
    np.testing.assert_allclose(
        cell["total_precipitation_surface"].values, 0.46721694, rtol=1e-6
    )
    # 850 hPa is the source's level index 10; a wrong descending-to-ascending level map
    # would land on another level's values.
    np.testing.assert_allclose(
        cell["pressure_level/temperature"].sel(pressure_level=850).values, 18.29538
    )


def test_yearly_store_era_snapshot_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The 7th 6-hourly init of the 2022_to_2023 store, so a wrong init index within the
    # store would read another init's values.
    init_time = pd.Timestamp("2022-01-02T12:00")
    ds = _dataset(
        tmp_path,
        monkeypatch,
        ["temperature_2m", "wind_speed_10m", "temperature"],
        init_time,
    )

    assert ds.init_time.values[-1] == init_time.to_datetime64()
    cell = ds.sel(
        latitude=_LATITUDE,
        longitude=_LONGITUDE,
        init_time=init_time,
        lead_time=_LEAD_TIME,
    )
    np.testing.assert_allclose(cell["temperature_2m"].values, 26.838043)
    np.testing.assert_allclose(cell["wind_speed_10m"].values, 4.212481)
    # Before the per-init stores all 13 levels share one chunk, which no reference can
    # address, so the pressure_level group has no data in this era.
    assert np.isnan(cell["pressure_level/temperature"].sel(pressure_level=850).values)
