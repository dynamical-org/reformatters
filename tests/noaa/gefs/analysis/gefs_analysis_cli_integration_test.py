import json
import subprocess
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import MonkeyPatch

from reformatters.common import zarr
from reformatters.noaa.gefs.analysis import (
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


def test_reformat_local_reforecast_period(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(zarr, "_LOCAL_ZARR_STORE_BASE_PATH", tmp_path)
    time_start = template_config.TIME_START
    time_end = time_start + timedelta(days=1)

    # 1. Backfill archive
    cli.reformat_local(time_end=time_end.isoformat())
    original_ds = xr.open_zarr(reformat.get_store(), decode_timedelta=True, chunks=None)

    space_subset_ds = original_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))
    # First time step is all null because reforecast starts at hour 3
    assert (
        (space_subset_ds.sel(time="2000-01-01T00").isnull().mean() == 1)
        .all()
        .to_array()
        .all()
    )
    # No values should be null hour 3+
    assert (
        (space_subset_ds.sel(time=slice("2000-01-01T03", None)).isnull().mean() == 0)
        .all()
        .to_array()
        .all()
    )

    point_ds = original_ds.sel(
        latitude=0, longitude=0, time=time_start + timedelta(hours=3)
    )
    assert np.isclose(point_ds["precipitation_surface"], 0.00032806)
    assert np.isclose(point_ds["temperature_2m"], 26.125)
    assert np.isclose(point_ds["wind_u_100m"], 4.0)


def test_reformat_local_reforecast_to_pre_v12_transition_period(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    time_start = pd.Timestamp("2019-12-31T00:00")
    # Add 3 more hours to exclusive end point so we end on a 00Z hour
    # during the pre v12 data which is 6 hourly. This will give us no nans after interpolation.
    time_end = time_start + timedelta(hours=24 * 2 + 3)

    # Update the template so this test starts processing at time_start
    monkeypatch.setattr(template_config, "TIME_START", time_start)
    cli.update_template()

    monkeypatch.setattr(zarr, "_LOCAL_ZARR_STORE_BASE_PATH", tmp_path)

    # 1. Backfill archive
    cli.reformat_local(time_end=time_end.isoformat())
    original_ds = xr.open_zarr(reformat.get_store(), decode_timedelta=True, chunks=None)

    # We're starting in the middle of the available data so no time steps are null
    assert (
        (original_ds.count() / np.prod(list(original_ds.sizes.values())))
        .to_array()
        .all()
    )

    point_ds = original_ds.sel(latitude=0, longitude=0)
    assert np.allclose(
        point_ds["temperature_2m"],
        np.array([28.375, 27.875, 27.875, 28.125, 28.375, 28.625, 28.75, 28.5, 27.5, 27.5, 27.5, 27.75, 28.0, 28.125, 28.375, 27.875, 27.5])
    )  # fmt: skip

    assert np.allclose(
        point_ds["precipitation_surface"],
        np.array([0.0, 9.23871994e-06, 9.23871994e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 9.23871994e-06, 6.70552254e-06, 4.17232513e-06, 1.31726265e-05, 2.21729279e-05, 1.10864639e-05, 0.0, 5.55515289e-05, 1.11103058e-04])
    )  # fmt: skip

    assert np.allclose(
        point_ds["wind_u_100m"],
        np.array([0.75, 1.125, 0.890625, 0.703125, 0.18359375, 0.78125, 0.8515625, 1.484375, 0.359375, 0.5, 0.640625, 1.265625, 1.90625, 2.09375, 2.28125, 1.78125, 1.296875])
    )  # fmt: skip

    # Restore the template
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "checkout",
            template._TEMPLATE_PATH,
        ],
        check=True,
    )


def test_reformat_local_pre_v12_to_v12_transition_period(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    time_start = pd.Timestamp("2020-09-22T00:00")
    time_end = time_start + timedelta(days=2)

    # Update the template so this test starts processing at time_start
    monkeypatch.setattr(template_config, "TIME_START", time_start)
    cli.update_template()

    monkeypatch.setattr(zarr, "_LOCAL_ZARR_STORE_BASE_PATH", tmp_path)

    # 1. Backfill archive
    cli.reformat_local(time_end=time_end.isoformat())
    original_ds = xr.open_zarr(reformat.get_store(), decode_timedelta=True, chunks=None)

    # We're starting in the middle of the available data so no time steps are null
    assert (
        (original_ds.count() / np.prod(list(original_ds.sizes.values())))
        .to_array()
        .all()
    )

    point_ds = original_ds.sel(latitude=0, longitude=0)

    assert np.allclose(
        point_ds["temperature_2m"],
        np.array([22.5, 22.5, 22.5, 22.5, 22.625, 22.75, 22.875, 22.875, 22.875, 22.75, 22.75, 22.875, 23.0, 23.125, 23.375, 23.375])
    )  # fmt: skip

    assert np.allclose(
        point_ds["precipitation_surface"],
        np.array([2.00195312e-01, 1.00097656e-01, 1.06692314e-05, 7.18235970e-06, 3.71038914e-06, 1.85519457e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 4.61935997e-06, 9.23871994e-06, 9.27597284e-07, 0.0, 0.0])
    )  # fmt: skip

    assert np.allclose(
        point_ds["wind_u_100m"],
        np.array([3.34375, 3.875, 4.375, 4.9375, 5.5, 4.9375, 4.3125, 4.375, 4.4375, 3.78125, 3.09375, 1.9375, 0.78125, 1.828125, 2.78125, 3.0625])
    )  # fmt: skip

    # Restore the template
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "checkout",
            template._TEMPLATE_PATH,
        ],
        check=True,
    )
