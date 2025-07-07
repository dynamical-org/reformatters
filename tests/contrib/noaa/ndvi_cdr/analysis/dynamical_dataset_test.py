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

    assert np.isclose(ds.ndvi_raw.mean().item(), 0.1766624)
    assert np.count_nonzero(ds.ndvi_raw.isnull()) == 25007972

    assert np.isclose(
        ds.sel(latitude=59.98, longitude=105.61, method="nearest").ndvi_raw.item(),
        0.22558,
        rtol=1e-4,
    )
