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


class NoaaHrrrAnalysisSourceFileCoord(NoaaHrrrSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        time = self.init_time + self.lead_time
        return {
            "time": time,
        }


class NoaaHrrrAnalysisRegionJob(NoaaHrrrRegionJob):
    """Region job for HRRR analysis data processing."""

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaHrrrDataVar],
    ) -> Sequence[NoaaHrrrAnalysisSourceFileCoord]:
        times = pd.to_datetime(processing_region_ds["time"].values)
        group_has_hour_0 = item({has_hour_0_values(var) for var in data_var_group})

        if group_has_hour_0:
            init_times = times
            lead_time = pd.Timedelta("0h")
        else:
            init_times = times - pd.Timedelta(hours=1)
            lead_time = pd.Timedelta("1h")

        file_type = item({var.internal_attrs.hrrr_file_type for var in data_var_group})

        return [
            NoaaHrrrAnalysisSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                domain="conus",
                file_type=file_type,
                data_vars=data_var_group,
            )
            for init_time in init_times
        ]
