from collections.abc import Mapping, Sequence

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


class NoaaGfsAnalysisSourceFileCoord(NoaaGfsSourceFileCoord):
    """Coordinates of a single source file to process for analysis dataset."""

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"time": self.init_time + self.lead_time}


class NoaaGfsAnalysisRegionJob(NoaaGfsCommonRegionJob):
    """Region job for GFS analysis data processing."""

    def get_processing_region(self) -> slice:
        """Buffer start by one step to allow deaccumulation without gaps in resulting output."""
        return slice(max(0, self.region.start - 1), self.region.stop)

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaDataVar],
    ) -> Sequence[NoaaGfsAnalysisSourceFileCoord]:
        times = pd.to_datetime(processing_region_ds["time"].values)
        group_has_hour_0 = item({has_hour_0_values(var) for var in data_var_group})

        if group_has_hour_0:
            init_times = times.floor("6h")
        else:
            init_times = (times - pd.Timedelta("1h")).floor("6h")

        lead_times = times - init_times

        return [
            NoaaGfsAnalysisSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                data_vars=data_var_group,
            )
            for init_time, lead_time in zip(init_times, lead_times, strict=True)
        ]

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[NoaaGfsSourceFileCoord]]
    ) -> xr.Dataset:
        # Remove the last hour. We pull accumulated variables (precipitation) from the 1 hour lead time,
        # but use the 0 hour lead time for other variables. This results in one additional
        # hour of data for accumulated variables. Trim it off so we aren't left with nans for
        # most variables in the final step.
        return (
            super()
            .update_template_with_results(process_results)
            .isel(time=slice(None, -1))
        )
