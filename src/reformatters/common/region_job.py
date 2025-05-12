from collections.abc import Sequence
from typing import Any, Generic, TypeVar

import pydantic
import xarray as xr
from zarr.storage import FsspecStore

from reformatters.common.config_models import DataVar
from reformatters.common.template_config import AppendDim

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])


class RegionJob(pydantic.BaseModel, Generic[DATA_VAR]):
    store: FsspecStore
    template_ds: xr.Dataset
    data_vars: Sequence[DATA_VAR]
    append_dim: AppendDim
    region: slice  # pydantic annotated slice of ints AI!
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

    @pydantic.field_validator("region")
    def _validate_region_is_int_slice(cls, v: slice) -> slice:
        # ensure start, stop, step are ints or None
        for attr in ("start", "stop", "step"):
            val = getattr(v, attr)
            assert val is None or isinstance(val, int), (
                f"region.{attr!r} must be an int or None, got {type(val).__name__}"
            )
        return v
