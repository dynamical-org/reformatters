from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from _pytest.monkeypatch import MonkeyPatch

from reformatters.u_arizona.swann.dataset import SWANNDataset

pytestmark = pytest.mark.slow


def test_reformat_local(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    dataset = SWANNDataset()
    # Dataset starts at 1981-10-01
    dataset.reformat_local(append_dim_end=pd.Timestamp("1981-10-02"))
    ds = xr.open_zarr(dataset._store(), chunks=None)
    assert ds.snow_depth.mean() == 0.23608214
    assert ds.snow_water_equivalent.mean() == 0.0433126

    subset_ds = ds.sel(latitude=48.583335, longitude=-94, method="nearest")
    assert subset_ds.snow_depth.values == 190.0
    assert subset_ds.snow_water_equivalent.values == 35.0
