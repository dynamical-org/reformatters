from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _pytest.monkeypatch import MonkeyPatch

from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.contrib.uarizona.swann.analysis import UarizonaSwannAnalysisDataset
from reformatters.contrib.uarizona.swann.analysis.region_job import (
    UarizonaSwannAnalysisRegionJob,
    UarizonaSwannAnalysisSourceFileCoord,
)

pytestmark = pytest.mark.slow


noop_storage_config = StorageConfig(
    base_path="noop",
    format=DatasetFormat.ZARR3,
)


def test_backfill_local(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    dataset = UarizonaSwannAnalysisDataset(storage_config=noop_storage_config)
    # Dataset starts at 1981-10-01
    dataset.backfill_local(append_dim_end=pd.Timestamp("1981-10-02"))
    ds = xr.open_zarr(dataset.primary_store_factory.primary_store(), chunks=None)
    assert ds.snow_depth.mean() == 0.23608214
    assert ds.snow_water_equivalent.mean() == 0.0433126

    subset_ds = ds.sel(latitude=48.583335, longitude=-94, method="nearest")
    assert subset_ds.snow_depth.values == 190.0
    assert subset_ds.snow_water_equivalent.values == 35.0


def test_update(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    dataset = UarizonaSwannAnalysisDataset(storage_config=noop_storage_config)
    # Dataset starts at 1981-10-01
    dataset.backfill_local(append_dim_end=pd.Timestamp("1981-10-02"))
    ds = xr.open_zarr(dataset.primary_store_factory.primary_store(), chunks=None)
    assert ds.time.max() == pd.Timestamp("1981-10-01")

    monkeypatch.setattr(
        dataset.region_job_class,
        "_update_append_dim_end",
        lambda: pd.Timestamp("1981-10-04"),
    )

    monkeypatch.setattr(
        dataset.region_job_class,
        "_update_append_dim_start",
        lambda existing_ds: pd.Timestamp(existing_ds.time.max().item()),
    )

    dataset.update("test-update")
    updated_ds = xr.open_zarr(
        dataset.primary_store_factory.primary_store(), chunks=None
    )
    np.testing.assert_array_equal(
        updated_ds.time, pd.date_range("1981-10-01", "1981-10-03")
    )
    subset_ds = updated_ds.sel(latitude=48.583335, longitude=-94, method="nearest")
    np.testing.assert_array_equal(subset_ds.snow_depth.values, [190.0, 163.0, 135.0])
    np.testing.assert_array_equal(
        subset_ds.snow_water_equivalent.values, [35.0, 33.0, 29.0]
    )


def test_update_template_trimming(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    dataset = UarizonaSwannAnalysisDataset(storage_config=noop_storage_config)
    # Dataset starts at 1981-10-01
    dataset.backfill_local(append_dim_end=pd.Timestamp("1981-10-02"))
    ds = xr.open_zarr(dataset.primary_store_factory.primary_store(), chunks=None)
    assert ds.time.max() == pd.Timestamp("1981-10-01")

    monkeypatch.setattr(
        dataset.region_job_class,
        "_update_append_dim_end",
        lambda: pd.Timestamp("1981-10-04"),
    )

    monkeypatch.setattr(
        dataset.region_job_class,
        "_update_append_dim_start",
        lambda existing_ds: pd.Timestamp(existing_ds.time.max().item()),
    )

    original_download_file = dataset.region_job_class.download_file

    def mock_download_file(
        self: UarizonaSwannAnalysisRegionJob,
        coord: UarizonaSwannAnalysisSourceFileCoord,
    ) -> Path:
        # Simulate download failure for 1981-10-03 (the last day we're trying to process)
        if coord.time == pd.Timestamp("1981-10-03"):
            raise FileNotFoundError(f"File not found for {coord.time}")
        return original_download_file(self, coord)

    monkeypatch.setattr(dataset.region_job_class, "download_file", mock_download_file)

    dataset.update("test-update")
    updated_ds = xr.open_zarr(
        dataset.primary_store_factory.primary_store(), chunks=None
    )

    # The dataset should only extend to 1981-10-02 because 1981-10-03 failed to download
    np.testing.assert_array_equal(
        updated_ds.time, pd.date_range("1981-10-01", "1981-10-02")
    )
    subset_ds = updated_ds.sel(latitude=48.583335, longitude=-94, method="nearest")
    np.testing.assert_array_equal(subset_ds.snow_depth.values, [190.0, 163.0])
    np.testing.assert_array_equal(subset_ds.snow_water_equivalent.values, [35.0, 33.0])
