"""Gated end-to-end checks for the WeatherNext native object proxy.

Run with:

    WEATHERNEXT_PROXY_TESTS=1 uv run pytest tests/google/weathernext2/forecast_virtual/real_files_integration_test.py
"""

import os
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.google.weathernext2.forecast_historical_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastHistoricalVirtualDataset,
)
from reformatters.google.weathernext2.forecast_operational_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastOperationalVirtualDataset,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("WEATHERNEXT_PROXY_TESTS") != "1",
        reason="requires the deployed lag-aware WeatherNext native object proxy",
    ),
]

type WeatherNextDataset = (
    GoogleWeathernext2ForecastHistoricalVirtualDataset
    | GoogleWeathernext2ForecastOperationalVirtualDataset
)


def _dataset(
    dataset: WeatherNextDataset,
    monkeypatch: pytest.MonkeyPatch,
    init_time: pd.Timestamp,
    variable_names: Sequence[str],
    lead_indices: Sequence[int] = (1,),
) -> xr.Dataset:
    original_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: original_get_template(end_time).isel(
            lead_time=list(lead_indices)
        ),
    )
    dataset.backfill_local(
        append_dim_end=init_time + dataset.template_config.append_dim_frequency,
        filter_start=init_time,
        filter_variable_names=list(variable_names),
    )
    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    return ds


def _selected_cell(ds: xr.Dataset, init_time: pd.Timestamp) -> xr.Dataset:
    return ds.sel(
        latitude=0,
        longitude=0,
        init_time=init_time,
        ensemble_member=0,
        lead_time=pd.Timedelta("12h"),
    )


def test_historical_native_chunk_values_and_pressure_transpose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_time = pd.Timestamp("2022-01-02T12:00")
    dataset = GoogleWeathernext2ForecastHistoricalVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )

    ds = _dataset(
        dataset,
        monkeypatch,
        init_time,
        [var.path for var in dataset.template_config.data_vars],
        lead_indices=(0, 1, 2, 3),
    )
    cell = _selected_cell(ds, init_time)

    np.testing.assert_allclose(cell["temperature_2m"].values, 26.998407, rtol=1e-6)
    np.testing.assert_allclose(
        cell["pressure_level/temperature"].sel(pressure_level=850).values,
        17.85119,
        rtol=1e-6,
    )
    _assert_all_variable_semantics(ds, dataset, init_time)


def test_operational_native_chunk_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_time = pd.Timestamp("2025-03-01T06:00")
    dataset = GoogleWeathernext2ForecastOperationalVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )

    ds = _dataset(
        dataset,
        monkeypatch,
        init_time,
        [var.path for var in dataset.template_config.data_vars],
        lead_indices=(0, 1, 2, 3),
    )
    cell = _selected_cell(ds, init_time)

    np.testing.assert_allclose(cell["temperature_2m"].values, 27.686975, rtol=1e-6)
    np.testing.assert_allclose(
        cell["pressure_level/temperature"].sel(pressure_level=850).values,
        18.787012,
        rtol=1e-6,
    )
    _assert_all_variable_semantics(ds, dataset, init_time)


def _assert_all_variable_semantics(
    ds: xr.Dataset,
    dataset: WeatherNextDataset,
    init_time: pd.Timestamp,
) -> None:
    selected = ds.sel(
        init_time=init_time,
        ensemble_member=0,
        lead_time=pd.Timedelta("12h"),
    )
    for var in dataset.template_config.data_vars:
        values = selected[var.path].values
        assert not np.isinf(values).any(), var.path
        nan_count = int(np.isnan(values).sum())
        if var.path == "sea_surface_temperature":
            assert nan_count == 351_876
        else:
            assert nan_count == 0, var.path

    precipitation = ds["total_precipitation_surface"].sel(
        init_time=init_time,
        ensemble_member=0,
    )
    planes = [precipitation.isel(lead_time=index).values for index in range(4)]
    for previous, current in pairwise(planes):
        fraction_decreased = float(np.mean(current < previous))
        assert 0.4 < fraction_decreased < 0.6
