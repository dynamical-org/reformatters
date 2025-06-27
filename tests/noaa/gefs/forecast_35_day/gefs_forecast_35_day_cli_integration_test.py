import json
from datetime import timedelta
from pathlib import Path

import pytest
import xarray as xr
from pytest import MonkeyPatch

from reformatters.noaa.gefs.forecast_35_day import (
    cli,
    reformat,
    template,
    template_config,
)

pytestmark = pytest.mark.slow


def test_update_template(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    with open(template._TEMPLATE_PATH / "zarr.json") as latest_f:
        template_consolidated_metadata = json.load(latest_f)

    test_template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(template, "_TEMPLATE_PATH", test_template_path)

    cli.update_template()

    with open(test_template_path / "zarr.json") as test_f:
        updated_consolidated_metadata = json.load(test_f)

    assert json.dumps(updated_consolidated_metadata) == json.dumps(
        template_consolidated_metadata
    )


def test_backfill_local_and_operational_update(monkeypatch: MonkeyPatch) -> None:
    init_time_start = template_config.INIT_TIME_START
    init_time_end = init_time_start + timedelta(days=1)

    # 1. Backfill archive
    cli.backfill_local(init_time_end=init_time_end.isoformat())
    original_ds = xr.open_zarr(reformat.get_store(), decode_timedelta=True, chunks=None)

    space_subset_ds = original_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))
    # Variables with hour 0 values
    assert (
        (space_subset_ds[["wind_u_100m", "temperature_2m"]].isnull().mean() == 0)
        .all()
        .to_array()
        .all()
    )
    # Variables with no hour 0 values
    assert (
        (
            space_subset_ds[["precipitation_surface", "maximum_temperature_2m"]]
            .sel(lead_time=slice("3h", None))  # trim off hour 0
            .isnull()
            .mean()
            == 0
        )
        .all()
        .to_array()
        .all()
    )

    point_ds = original_ds.sel(
        latitude=0,
        longitude=0,
        init_time=init_time_start,
        lead_time="3h",
        ensemble_member=1,
    )
    assert point_ds["temperature_2m"] == 23.25
    assert point_ds["wind_u_100m"] == 1.515625
    assert point_ds["precipitation_surface"] == 1.2040138244628906e-05
    assert point_ds["maximum_temperature_2m"] == 23.875

    # Check precipitation is deaccumulated
    # Deaccumulated precipitation should have at least 5% of values lower than the previous value
    # 5% may sound like a low threshold, but ~50% of values are 0 (no diff from previous value)
    # and 33% of are nan in this test due to unavailable hour 0 values
    assert (
        original_ds["precipitation_surface"]
        # subset in space just for speed
        .sel(latitude=slice(50, 0), longitude=slice(0, 50), lead_time=slice("24h"))
        .diff(dim="lead_time")
        < 0
    ).mean() > 0.05

    # 2. Update archive
    monkeypatch.setattr(
        reformat,
        "_get_operational_update_init_time_end",
        lambda: init_time_end + timedelta(days=1),
    )

    cli.update(job_name="test")
    updated_ds = xr.open_zarr(reformat.get_store(), decode_timedelta=True, chunks=None)

    assert len(updated_ds.init_time) == 2
    assert updated_ds.init_time.max() == init_time_end
    assert set(original_ds.keys()) == set(updated_ds.keys())

    space_subset_ds = updated_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))
    # Variables with hour 0 values
    assert (
        (space_subset_ds[["wind_u_100m", "temperature_2m"]].isnull().mean() == 0)
        .all()
        .to_array()
        .all()
    )
    # Variables with no hour 0 values
    assert (
        (
            space_subset_ds[["precipitation_surface", "maximum_temperature_2m"]]
            .sel(lead_time=slice("3h", None))  # trim off hour 0
            .isnull()
            .mean()
            == 0
        )
        .all()
        .to_array()
        .all()
    )

    point_ds = updated_ds.sel(
        latitude=0,
        longitude=0,
        lead_time="6h",
        ensemble_member=1,
        init_time=init_time_end,
    )

    assert point_ds["temperature_2m"] == 23.125
    assert point_ds["wind_u_100m"] == 1.625
    assert point_ds["precipitation_surface"] == 0
    assert point_ds["maximum_temperature_2m"] == 23.375

    # Check precipitation is deaccumulated
    assert (
        updated_ds["precipitation_surface"]
        # subset in space just for speed
        .sel(latitude=slice(50, 0), longitude=slice(0, 50), lead_time=slice("24h"))
        .diff(dim="lead_time")
        < 0
    ).mean() > 0.05
