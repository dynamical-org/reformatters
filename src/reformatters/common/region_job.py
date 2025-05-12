from collections.abc import Sequence
from typing import Generic, TypeVar

import pydantic
import xarray as xr
from zarr.storage import FsspecStore

from reformatters.common.config_models import DataVar


class RegionJob(pydantic.BaseModel):
    store: FsspecStore
    template_ds: xr.Dataset
    data_vars: Sequence[DataVar]
    append_dimension: str  # e.g. "time" or "init_time"
    region: slice  # an integer slice along append_dimension
    max_vars_per_backfill_job: int

    def process(self) -> None:
        raise NotImplementedError("Implement your dataset specific logic here")

    def region_template_ds(self) -> xr.Dataset:
        var_names = [v.name for v in self.data_vars]
        # select only those variables and then index the append_dimension
        return self.template_ds[var_names].isel({self.append_dimension: self.region})

    def get_processing_group_size(self) -> int:
        n = len(self.data_vars)
        if n > 6:
            return 4
        elif n > 3:
            return 2
        else:
            return 1
