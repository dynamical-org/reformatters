from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _pytest.monkeypatch import MonkeyPatch

from reformatters.u_arizona.swann import SWANNDataset

pytestmark = pytest.mark.slow


def test_reformat_local(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    dataset = SWANNDataset()
    # Dataset starts at 1981-10-01
    dataset.reformat_local(append_dim_end=pd.Timestamp("1981-10-02"))
    ds = xr.open_zarr(dataset._final_store(), chunks=None)
    assert ds.snow_depth.mean() == 0.23608214
    assert ds.snow_water_equivalent.mean() == 0.0433126

    subset_ds = ds.sel(latitude=48.583335, longitude=-94, method="nearest")
    assert subset_ds.snow_depth.values == 190.0
    assert subset_ds.snow_water_equivalent.values == 35.0


def test_reformat_operational_update(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    dataset = SWANNDataset()
    # Dataset starts at 1981-10-01
    dataset.reformat_local(append_dim_end=pd.Timestamp("1981-10-02"))
    ds = xr.open_zarr(dataset._final_store(), chunks=None)
    assert ds.time.max() == pd.Timestamp("1981-10-01")

    monkeypatch.setattr(
        dataset.region_job_class,
        "_operational_append_dim_end",
        lambda: pd.Timestamp("1981-10-04"),
    )

    monkeypatch.setattr(
        dataset.region_job_class,
        "_operational_append_dim_start",
        lambda existing_ds: pd.Timestamp(existing_ds.time.max().item()),
    )

    dataset.reformat_operational_update("test-reformat-operational-update")
    updated_ds = xr.open_zarr(dataset._final_store(), chunks=None)
    np.testing.assert_array_equal(
        updated_ds.time, pd.date_range("1981-10-01", "1981-10-03")
    )
