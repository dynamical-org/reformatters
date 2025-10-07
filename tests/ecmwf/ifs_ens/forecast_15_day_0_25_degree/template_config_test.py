import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)


def test_update_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Ensure that `uv run main <dataset-id> update-template` has been run and
    all changes to EcmwfIfsEnsForecast15Day025DegreeTemplateConfig are reflected in the on-disk Zarr template.
    """
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    with open(template_config.template_path() / "zarr.json") as f:
        existing_template = json.load(f)

    test_template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(
        EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
        "template_path",
        lambda _self: test_template_path,
    )

    template_config.update_template()

    with open(template_config.template_path() / "zarr.json") as f:
        updated_template = json.load(f)

    assert existing_template == updated_template


def test_derive_coordinates() -> None:
    """Test coordinate derivation."""
    # Create a minimal dataset for testing
    ds = xr.Dataset(
        coords={
            "init_time": pd.date_range("2000-01-01", periods=3, freq="3h"),
            "lead_time": pd.timedelta_range(
                "0h", "145h", freq="3h"
            ),  # note: not the full range
            "ensemble_member": np.arange(1, 51),
            "latitude": np.linspace(-90, 90, 721),
            "longitude": np.linspace(-180, 179.75, 1440),
        }
    )

    derived = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig().derive_coordinates(ds)

    assert "spatial_ref" in derived
    assert derived["spatial_ref"][0] == ()  # Empty dimensions tuple

    # 3 init times, all should have 360h expected forecast length
    assert "expected_forecast_length" in derived
    assert list(derived["expected_forecast_length"][1]) == [pd.Timedelta("360h")] * 3

    # 3 init times, all should have NaT ingested forecast length
    assert "ingested_forecast_length" in derived
    assert all(
        np.isnat(ingested) for ingested in derived["ingested_forecast_length"][1]
    )


def test_dimension_coordinates_shapes_and_values() -> None:
    cfg = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    dim_coords = cfg.dimension_coordinates()
    # must have exactly these dims
    assert set(dim_coords) == {
        "init_time",
        "lead_time",
        "latitude",
        "longitude",
        "ensemble_member",
    }

    # init_time: only the start timestamp
    init = dim_coords["init_time"]
    assert isinstance(init, pd.DatetimeIndex)
    assert len(init) == 1
    assert init[0] == cfg.append_dim_start

    # lead_time: goes out to 15 days; switches from 3-hourly to 6-hourly at 144h (6 days)
    lt = dim_coords["lead_time"]
    assert isinstance(lt, pd.TimedeltaIndex)
    assert pd.Timedelta("0h") == lt[0]
    assert pd.Timedelta("141h") in lt  # last 3-hourly timestep
    assert pd.Timedelta("144h") in lt  # 6 days; switch to 6-hourly here
    assert (
        pd.Timedelta("147h") not in lt
    )  # should not be included -- would be 3-hourly during the 6-hourly domain
    assert pd.Timedelta("150h") in lt  # first 6-hourly timestep
    assert pd.Timedelta("360h") == lt[-1]

    # latitude: from +90 to -90, 0.25° steps
    lat = dim_coords["latitude"]
    assert isinstance(lat, np.ndarray)
    assert lat[0] == 90.0
    assert lat[-1] == -90.0
    assert len(lat) == 721

    # longitude: from -180 to +179.75, 0.25° steps
    lon = dim_coords["longitude"]
    assert isinstance(lon, np.ndarray)
    assert lon[0] == -180.0
    assert lon[-1] == 179.75
    assert len(lon) == 1440

    # ensemble_member: contains 1-50 (1 control + 49 perturbed)
    em = dim_coords["ensemble_member"]
    assert isinstance(em, np.ndarray)
    assert all(em == np.arange(1, 51))
