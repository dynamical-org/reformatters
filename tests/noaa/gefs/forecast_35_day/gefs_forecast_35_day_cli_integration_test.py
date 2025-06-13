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


def test_reformat_local_and_operational_update(monkeypatch: MonkeyPatch) -> None:
    init_time_start = template_config.INIT_TIME_START
    init_time_end = init_time_start + timedelta(days=1)

    # 1. Backfill archive
    cli.reformat_local(init_time_end=init_time_end.isoformat())
    original_ds = xr.open_zarr(reformat.get_store(), decode_timedelta=True, chunks=None)

    space_subset_ds = original_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))
    assert (space_subset_ds.isnull().mean() == 0).all().to_array().all()

    point_ds = original_ds.sel(
        latitude=0,
        longitude=0,
        init_time=init_time_start,
        lead_time="0h",
        ensemble_member=1,
    )
    assert point_ds["temperature_2m"] == 23.875
    assert point_ds["wind_u_100m"] == 1.65625

    # 2. Update archive
    monkeypatch.setattr(
        reformat,
        "_get_operational_update_init_time_end",
        lambda: init_time_end + timedelta(days=1),
    )

    cli.reformat_operational_update(job_name="test")
    updated_ds = xr.open_zarr(reformat.get_store(), decode_timedelta=True, chunks=None)

    assert len(updated_ds.init_time) == 2
    assert updated_ds.init_time.max() == init_time_end
    assert set(original_ds.keys()) == set(updated_ds.keys())

    space_subset_ds = updated_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))
    assert (space_subset_ds.isnull().mean() == 0).all().to_array().all()

    point_ds = updated_ds.sel(
        latitude=0,
        longitude=0,
        lead_time="0h",
        ensemble_member=1,
    )

    assert point_ds["temperature_2m"].sel(init_time=init_time_start) == 23.875
    assert point_ds["wind_u_100m"].sel(init_time=init_time_start) == 1.65625

    assert point_ds["temperature_2m"].sel(init_time=init_time_end) == 23.375
    assert point_ds["wind_u_100m"].sel(init_time=init_time_end) == 1.09375
