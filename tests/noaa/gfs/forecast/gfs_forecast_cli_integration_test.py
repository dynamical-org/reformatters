import json
from datetime import timedelta
from pathlib import Path

import pytest
import xarray as xr
from pytest import MonkeyPatch

from reformatters.common import zarr
from reformatters.noaa.gfs.forecast import (
    cli,
    reformat,
)
from reformatters.noaa.gfs.forecast.template_config import GFS_FORECAST_TEMPLATE_CONFIG

pytestmark = pytest.mark.slow


def test_update_template(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    with open(GFS_FORECAST_TEMPLATE_CONFIG.template_path() / "zarr.json") as latest_f:
        template_consolidated_metadata = json.load(latest_f)

    test_template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(
        GFS_FORECAST_TEMPLATE_CONFIG, "template_path", lambda: test_template_path
    )

    cli.update_template()

    with open(test_template_path / "zarr.json") as test_f:
        updated_consolidated_metadata = json.load(test_f)

    assert json.dumps(updated_consolidated_metadata) == json.dumps(
        template_consolidated_metadata
    )


def test_reformat_local(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(zarr, "_LOCAL_ZARR_STORE_BASE_PATH", tmp_path)
    init_time_start = GFS_FORECAST_TEMPLATE_CONFIG.append_dim_start
    init_time_end = init_time_start + timedelta(days=1)

    # 1. Backfill archive
    cli.reformat_local(init_time_end=init_time_end.isoformat())
    original_ds = xr.open_zarr(reformat.get_store(), decode_timedelta=True, chunks=None)

    space_subset_ds = original_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))

    # These variables are present at all lead times
    assert (
        (
            space_subset_ds[["temperature_2m", "precipitable_water_atmosphere"]]
            .isnull()
            .mean()
            == 0
        )
        .all()
        .to_array()
        .all()
    )

    # These variables are not present at hour 0
    assert (
        (
            space_subset_ds[
                [
                    "precipitation_surface",
                    "maximum_temperature_2m",
                    "minimum_temperature_2m",
                ]
            ]
            .sel(lead_time=slice("1h", None))
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
        lead_time="2h",
    )

    assert point_ds["temperature_2m"] == 27.875
    assert point_ds["maximum_temperature_2m"] == 28.125
    assert point_ds["minimum_temperature_2m"] == 27.875
    assert point_ds["precipitation_surface"] == 1.7404556e-05
    assert point_ds["precipitable_water_atmosphere"] == 56.5
