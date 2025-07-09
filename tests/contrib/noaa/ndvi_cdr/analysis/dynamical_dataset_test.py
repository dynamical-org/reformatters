from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _pytest.monkeypatch import MonkeyPatch

from reformatters.common.dynamical_dataset import DynamicalDatasetStorageConfig
from reformatters.contrib.noaa.ndvi_cdr.ndvi_cdr.analysis.dynamical_dataset import (
    NoaaNdviCdrAnalysisDataset,
)

pytestmark = pytest.mark.slow


noop_storage_config = DynamicalDatasetStorageConfig(
    base_path="noop",
)


def test_backfill_local(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    dataset = NoaaNdviCdrAnalysisDataset(storage_config=noop_storage_config)
    # Dataset starts at 1981-06-24, test with a couple days after start
    dataset.backfill_local(append_dim_end=pd.Timestamp("1981-06-25"))
    ds = xr.open_zarr(dataset._final_store(), chunks=None)

    # Check ndvi_raw values
    np.testing.assert_allclose(ds.ndvi_raw.mean().item(), 0.1766624, rtol=1e-06)
    np.testing.assert_equal(np.count_nonzero(ds.ndvi_raw.isnull()), 25007972)
    np.testing.assert_allclose(
        ds.sel(latitude=59.98, longitude=105.61, method="nearest").ndvi_raw.item(),
        0.22558,
        rtol=1e-4,
    )

    # Check qa value
    np.testing.assert_allclose(ds.qa.mean().item(), 4820.94356451, rtol=1e-08)
    np.testing.assert_equal(
        ds.sel(latitude=89.93, longitude=-180, method="nearest").qa.item(),
        -24438,
    )
    np.testing.assert_equal(
        ds.sel(latitude=59.98, longitude=105.61, method="nearest").qa.item(),
        24706,
    )

    # Check ndvi_usable values
    # 24706 has a binary representation of 0110000010000010
    # This has the cloud flag set, so it gets masked out in the usable data
    assert np.isnan(
        ds.sel(latitude=59.98, longitude=105.61, method="nearest").ndvi_usable.item()
    )


def test_update(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    dataset = NoaaNdviCdrAnalysisDataset(storage_config=noop_storage_config)
    # Dataset starts at 1981-06-24, test with initial data
    dataset.backfill_local(append_dim_end=pd.Timestamp("1981-06-25"))
    ds = xr.open_zarr(dataset._final_store(), chunks=None)
    assert ds.time.max() == pd.Timestamp("1981-06-24")

    # Mock pd.Timestamp.now() to control the update end date
    monkeypatch.setattr("pandas.Timestamp.now", lambda: pd.Timestamp("1981-06-26"))

    dataset.update("test-update")
    updated_ds = xr.open_zarr(dataset._final_store(), chunks=None)
    np.testing.assert_array_equal(
        updated_ds.time, pd.date_range("1981-06-24", "1981-06-25")
    )

    updated_raw_values = updated_ds.sel(
        latitude=59.98, longitude=105.61, method="nearest"
    ).ndvi_raw.values
    np.testing.assert_array_equal(
        updated_raw_values,
        np.array(
            [
                0.22558594,
                0.15722656,
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        updated_ds.sel(latitude=-37.83, longitude=141.4, method="nearest")
        .sel(time="1981-06-25")
        .ndvi_usable.item(),
        0.927734375,
    )
