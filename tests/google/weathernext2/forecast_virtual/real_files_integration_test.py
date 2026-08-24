"""End-to-end tests for the WeatherNext 2 virtual dataset.

These tests require the deployed authenticated proxy:

    WEATHERNEXT_PROXY_TESTS=1 uv run pytest tests/google/weathernext2/forecast_virtual/real_files_integration_test.py

The snapshot values were verified by selective range-decoding the same source chunks
through the approved WeatherNext service account.
"""

import os
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.config_models import ROOT
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.google.weathernext2.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastVirtualDataset,
)
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    OUTPUT_CHUNK_LENGTH,
    PROXY_LOCATION_PREFIX,
)
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2DataVar,
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)
from tests.common.dynamical_dataset_test import assert_configured_validators

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("WEATHERNEXT_PROXY_TESTS") != "1",
        reason="requires the deployed WeatherNext proxy",
    ),
]

_LATITUDE, _LONGITUDE = 0.0, 0.0
_LEAD_TIME = pd.Timedelta("12h")
_ENSEMBLE_MEMBER = 0


def _dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable_names: Sequence[str],
    init_time: pd.Timestamp,
) -> xr.Dataset:
    dataset = GoogleWeathernext2ForecastVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )
    original_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: original_get_template(end_time).isel(lead_time=[1]),
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


def _selected_cell(ds: xr.Dataset, init_time: pd.Timestamp) -> xr.Dataset:
    return ds.sel(
        latitude=_LATITUDE,
        longitude=_LONGITUDE,
        init_time=init_time,
        ensemble_member=_ENSEMBLE_MEMBER,
        lead_time=_LEAD_TIME,
    )


def test_operational_store_boundary_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_time = pd.Timestamp("2025-03-01T06:00")
    ds = _dataset(
        tmp_path,
        monkeypatch,
        ["temperature_2m", "temperature"],
        init_time,
    )

    cell = _selected_cell(ds, init_time)
    np.testing.assert_allclose(cell["temperature_2m"].values, 27.686975, rtol=1e-6)
    np.testing.assert_allclose(
        cell["pressure_level/temperature"].sel(pressure_level=850).values,
        18.787012,
        rtol=1e-6,
    )


def test_annual_store_boundary_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_time = pd.Timestamp("2022-01-02T12:00")
    ds = _dataset(
        tmp_path,
        monkeypatch,
        ["temperature_2m", "temperature"],
        init_time,
    )

    cell = _selected_cell(ds, init_time)
    np.testing.assert_allclose(cell["temperature_2m"].values, 26.998407, rtol=1e-6)
    np.testing.assert_allclose(
        cell["pressure_level/temperature"].sel(pressure_level=850).values,
        17.85119,
        rtol=1e-6,
    )


_CONFIG = GoogleWeathernext2ForecastVirtualTemplateConfig()


def _proxy_plane(
    var: GoogleWeathernext2DataVar,
    era: Literal["annual", "operational"],
    lead_index: int = 0,
    init_time: pd.Timestamp | None = None,
) -> npt.NDArray[np.float32]:
    pressure_suffix = ".0" if var.group is not ROOT else ""
    if era == "annual":
        init_time = init_time or pd.Timestamp("2022-01-01T00:00")
        store = (
            f"weathernext_2_0_0/zarr/{init_time.year}_to_{init_time.year + 1}/"
            "predictions.zarr"
        )
        year_start = pd.Timestamp(f"{init_time.year}-01-01T00:00")
        init_index = int((init_time - year_start) // pd.Timedelta("6h"))
        chunk_indices = f"{init_index}.0.{lead_index}.0.0{pressure_suffix}"
    else:
        init_time = init_time or pd.Timestamp("2025-01-01T00:00")
        store = (
            "weathernext_2_0_0/zarr/2025_to_present/"
            f"{init_time:%Y%m%d_%H}hr_01_preds/predictions.zarr"
        )
        chunk_indices = f"0.{lead_index}.0.0{pressure_suffix}"
    url = (
        f"{PROXY_LOCATION_PREFIX}plane/0/{store}/"
        f"{var.internal_attrs.source_name}/{chunk_indices}"
    )
    response = httpx.get(url, timeout=120)
    response.raise_for_status()
    assert len(response.content) == OUTPUT_CHUNK_LENGTH
    return np.frombuffer(response.content, dtype="<f4")


@pytest.mark.parametrize(
    ("init_time", "era"),
    [
        (pd.Timestamp("2022-01-02T12:00"), "annual"),
        (pd.Timestamp("2025-03-01T06:00"), "operational"),
    ],
)
def test_all_generated_target_refs_and_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    init_time: pd.Timestamp,
    era: Literal["annual", "operational"],
) -> None:
    ds = _dataset(
        tmp_path, monkeypatch, [var.name for var in _CONFIG.data_vars], init_time
    )
    cell = _selected_cell(ds, init_time).sel(pressure_level=50)
    for var in _CONFIG.data_vars:
        raw = float(
            _proxy_plane(var, era, lead_index=1, init_time=init_time).reshape(
                721, 1440
            )[360, 720]
        )
        if var.attrs.units == "degree_Celsius":
            expected = raw - 273.15
        elif var.path == "pressure_level/geopotential_height":
            expected = raw / 9.80665
        elif var.path == "total_precipitation_surface":
            expected = raw / 0.001
        else:
            expected = raw
        np.testing.assert_allclose(float(cell[var.path].values), expected, rtol=1e-6)


def test_all_variables_have_verified_missing_semantics_in_both_source_eras() -> None:
    for era in ("annual", "operational"):
        for var in _CONFIG.data_vars:
            values = _proxy_plane(var, era)
            assert not np.isinf(values).any(), (era, var.path)
            nan_count = int(np.isnan(values).sum())
            if var.path == "sea_surface_temperature":
                assert nan_count == 351_876, era
            else:
                assert nan_count == 0, (era, var.path)
            if var.path == "total_precipitation_surface":
                assert -0.001 < float(values.min()) < 0, era


def test_precipitation_leads_are_six_hour_intervals_not_run_totals() -> None:
    var = next(
        var for var in _CONFIG.data_vars if var.path == "total_precipitation_surface"
    )
    planes = [_proxy_plane(var, "operational", lead_index) for lead_index in range(4)]
    for previous, current in pairwise(planes):
        fraction_decreased = float(np.mean(current < previous))
        assert 0.4 < fraction_decreased < 0.6
