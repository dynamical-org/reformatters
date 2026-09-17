from collections.abc import Mapping, Sequence

import pandas as pd
import xarray as xr

from reformatters.common.iterating import group_by, item
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValue,
    SourceFileResult,
)
from reformatters.common.types import (
    Dim,
)
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
)
from reformatters.noaa.hrrr.region_job import NoaaHrrrRegionJob, NoaaHrrrSourceFileCoord

log = get_logger(__name__)


class NoaaHrrrAnalysisSourceFileCoord(NoaaHrrrSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.init_time + self.lead_time}


class NoaaHrrrAnalysisRegionJob(NoaaHrrrRegionJob):
    """Region job for HRRR analysis data processing."""

    @classmethod
    def source_file_var_groups(
        cls,
        data_vars: Sequence[NoaaHrrrDataVar],
    ) -> Sequence[Sequence[NoaaHrrrDataVar]]:
        """Group by the inputs of analysis_lead_time, so a group shares one lead at every time of a region."""
        return group_by(
            data_vars,
            lambda v: (
                v.internal_attrs.hrrr_file_type,
                v.has_hour_0_values(),
                v.internal_attrs.analysis_hour_0_unusable_from,
            ),
        )

    def get_processing_region(self) -> slice:
        """Buffer start by one step to allow deaccumulation without gaps in resulting output."""
        return slice(max(0, self.region.start - 1), self.region.stop)

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaHrrrDataVar],
    ) -> Sequence[NoaaHrrrAnalysisSourceFileCoord]:
        times = pd.to_datetime(processing_region_ds["time"].values)
        file_type = item({var.internal_attrs.hrrr_file_type for var in data_var_group})

        coords = []
        for time in times:
            lead_time = item({var.analysis_lead_time(time) for var in data_var_group})
            coords.append(
                NoaaHrrrAnalysisSourceFileCoord(
                    init_time=time - lead_time,
                    lead_time=lead_time,
                    domain="conus",
                    file_type=file_type,
                    data_vars=data_var_group,
                )
            )
        return coords

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[SourceFileResult]]
    ) -> xr.DataTree:
        # Remove the last hour. Variables read from the 1 hour lead time reach one hour
        # further than those read from hour 0. Trim it off so we aren't left with nans for
        # most variables in the final step.
        return (
            super()
            .update_template_with_results(process_results)
            .isel(time=slice(None, -1))
        )
