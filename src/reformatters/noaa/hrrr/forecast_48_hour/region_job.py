from collections.abc import Mapping, Sequence

import pandas as pd
import xarray as xr

from reformatters.common.iterating import item
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValueOrRange,
)
from reformatters.common.types import (
    Dim,
)
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
)
from reformatters.noaa.hrrr.region_job import NoaaHrrrRegionJob, NoaaHrrrSourceFileCoord
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)


class NoaaHrrrForecast48HourSourceFileCoord(NoaaHrrrSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }


class NoaaHrrrForecast48HourRegionJob(NoaaHrrrRegionJob):
    """Region job for HRRR 48-hour forecast data processing."""

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaHrrrDataVar],
    ) -> Sequence[NoaaHrrrForecast48HourSourceFileCoord]:
        """Generate source file coordinates for the processing region."""
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        group_has_hour_0 = item({has_hour_0_values(var) for var in data_var_group})
        if not group_has_hour_0:
            lead_times = lead_times[lead_times > pd.Timedelta(hours=0)]

        file_type = item({var.internal_attrs.hrrr_file_type for var in data_var_group})

        return [
            NoaaHrrrForecast48HourSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                domain="conus",
                file_type=file_type,  # ty: ignore[invalid-argument-type]
                data_vars=data_var_group,
            )
            for init_time in init_times
            for lead_time in lead_times
        ]
