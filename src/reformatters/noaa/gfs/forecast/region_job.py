from collections.abc import Mapping, Sequence
from itertools import product

import pandas as pd
import xarray as xr

from reformatters.common.iterating import item
from reformatters.common.region_job import (
    CoordinateValueOrRange,
)
from reformatters.common.types import (
    Dim,
)
from reformatters.noaa.gfs.region_job import (
    NoaaGfsCommonRegionJob,
    NoaaGfsSourceFileCoord,
)
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_utils import has_hour_0_values


class NoaaGfsForecastSourceFileCoord(NoaaGfsSourceFileCoord):
    """Coordinates of a single source file to process for forecast dataset."""

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class NoaaGfsForecastRegionJob(NoaaGfsCommonRegionJob[NoaaGfsForecastSourceFileCoord]):
    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[NoaaDataVar]
    ) -> Sequence[NoaaGfsForecastSourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds."""
        var_has_hour_0_values = item({has_hour_0_values(v) for v in data_var_group})
        if not var_has_hour_0_values:
            processing_region_ds = processing_region_ds.sel(lead_time=slice("1h", None))

        return [
            NoaaGfsForecastSourceFileCoord(
                init_time=pd.Timestamp(init_time),
                lead_time=pd.Timedelta(lead_time),
                data_vars=data_var_group,
            )
            for init_time, lead_time in product(
                processing_region_ds["init_time"].values,
                processing_region_ds["lead_time"].values,
            )
        ]
