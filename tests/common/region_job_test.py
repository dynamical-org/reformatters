from collections.abc import Sequence
from itertools import batched
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from reformatters.common import template_utils
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.region_job import (
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.types import ArrayFloat32, Timestamp
from reformatters.common.zarr import get_zarr_store


class DataVarA(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float32", fill_value=np.nan, chunks=(1, 10, 15), shards=None
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="C",
        long_name="Test variable",
        short_name="test",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(keep_mantissa_bits=10)


class SourceFileCoordA(SourceFileCoord):
    time: Timestamp

    def get_url(self) -> str:
        return f"https://test.org/testfile{self.time.strftime('%Y%m%d%H%M')}"


class RegionJobA(RegionJob[DataVarA, SourceFileCoordA]):
    max_vars_per_backfill_job: ClassVar[int] = 4

    def group_data_vars(
        self,
        processing_region_ds: xr.Dataset,
    ) -> Sequence[Sequence[DataVarA]]:
        return list(batched(self.data_vars, 3))

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[DataVarA],
    ) -> Sequence[SourceFileCoordA]:
        return [
            SourceFileCoordA(time=t)
            for t in processing_region_ds[self.append_dim].values
        ]

    def download_file(self, coord: SourceFileCoordA) -> Path:
        if coord.time == pd.Timestamp("2025-01-01T00"):
            raise FileNotFoundError()  # simulate a missing file
        return Path("testfile")

    def read_data(
        self,
        coord: SourceFileCoordA,
        data_var: DataVarA,
    ) -> ArrayFloat32:
        if coord.time == pd.Timestamp("2025-01-01T06"):
            raise ValueError("Test error")  # simulate a read error
        return np.ones((10, 15), dtype=np.float32)


@pytest.fixture
def template_ds() -> xr.Dataset:
    num_time = 48
    return xr.Dataset(
        {
            f"var{i}": xr.Variable(
                data=np.ones((num_time, 10, 15), dtype=np.float32),
                dims=["time", "latitude", "longitude"],
                encoding={
                    "dtype": "float32",
                    "chunks": (num_time // 2, 10, 15),
                    "shards": (num_time, 10, 15),
                },
            )
            for i in range(4)
        },
        coords={
            "time": pd.date_range("2025-01-01", freq="h", periods=num_time),
            "latitude": np.linspace(0, 90, 10),
            "longitude": np.linspace(0, 140, 15),
        },
    )


@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork.* may lead to deadlocks in the child"
)
def test_region_job(template_ds: xr.Dataset) -> None:
    # Gross test hack: register LocalStore as isinstance(zarr.storage.FsspecStore)
    # to pass pydantic validation
    zarr.storage.FsspecStore.register(zarr.storage.LocalStore)

    store = get_zarr_store("test-dataset", "test-version")

    # Write zarr metadata for this RegionJob to write into
    template_utils.write_metadata(template_ds, store, mode="w")

    job = RegionJobA(
        store=store,
        template_ds=template_ds,
        data_vars=[DataVarA(name=name) for name in template_ds.data_vars.keys()],
        append_dim="time",
        region=slice(0, 18),
    )
    job.process()


def test_source_file_coord_out_loc_default_impl() -> None:
    coord = SourceFileCoordA(time=pd.Timestamp("2025-01-01T00"))
    assert coord.out_loc() == {"time": pd.Timestamp("2025-01-01T00")}
