from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.config_models import DataVar
from reformatters.common.region_job import (
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.template_config import Dim
from reformatters.common.types import ArrayFloat32


# --- a fake DataVar and its internal‐attrs just for typing in our TestRegionJob ---
class FakeInternalAttrs:
    keep_mantissa_bits: int = 23


class FakeDataVar(DataVar[FakeInternalAttrs]):
    pass


# --- minimal SourceFileCoord subclass for testing ---------------
class TestSourceFileCoord(SourceFileCoord):
    # required fields can be added here (e.g. init_time, lead_time, etc.)

    def get_url(self) -> str:
        # For testing, just return a dummy URL string
        return "file:///tmp/testfile"

    def out_loc(
        self,
    ) -> Mapping[Dim, slice | int | float | pd.Timestamp | pd.Timedelta | str]:
        # For testing, just return a fixed location mapping
        return {"time": 0}


# --- minimal RegionJob subclass for testing --------------------
class TestRegionJob(RegionJob[FakeDataVar]):
    # satisfy the ClassVar requirement
    max_vars_per_backfill_job: ClassVar[int] = 4

    def group_data_vars(
        self,
        processing_region_ds: xr.Dataset,
    ) -> Sequence[Sequence[FakeDataVar]]:
        # For testing, just return all data_vars as a single group
        return [self.data_vars]

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[FakeDataVar],
    ) -> Sequence[TestSourceFileCoord]:
        # For testing, return a single TestSourceFileCoord
        return [TestSourceFileCoord()]

    def download_file(self, coord: TestSourceFileCoord) -> Path:
        # For testing, just return a dummy path
        return Path("/tmp/testfile")

    def read_data(
        self,
        coord: TestSourceFileCoord,
        data_var: FakeDataVar,
    ) -> ArrayFloat32:
        # For testing, return a 1D float32 array of ones of length 1
        return np.ones((1,), dtype=np.float32)

    # you can override apply_data_transformations or summarize_processing_state
    # if your tests need to observe or inject custom behavior
