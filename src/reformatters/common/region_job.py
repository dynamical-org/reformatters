from collections.abc import Sequence
from typing import Annotated, Any, Generic, TypeVar

import pydantic
import xarray as xr
from pydantic.functional_validators import AfterValidator
from zarr.storage import FsspecStore

from reformatters.common.config_models import DataVar
from reformatters.common.template_config import AppendDim

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])


class RegionJob(pydantic.BaseModel, Generic[DATA_VAR]):
    store: FsspecStore
    template_ds: xr.Dataset
    data_vars: Sequence[DATA_VAR]
    append_dim: AppendDim
    region: Annotated[
        slice,
        AfterValidator(
            lambda s: isinstance(s.start, int)
            and isinstance(s.stop, int)
            and s.step is None
        ),
    ]
    max_vars_per_backfill_job: int

    def process(self) -> None:
        raise NotImplementedError("Implement your dataset specific logic here")

    def region_template_ds(self) -> xr.Dataset:
        var_names = [v.name for v in self.data_vars]
        return self.template_ds[var_names].isel({self.append_dim: self.region})

    def get_processing_group_size(self) -> int:
        match len(self.data_vars):
            case n if n > 6:
                return 4
            case n if n > 3:
                return 2
            case _:
                return 1
