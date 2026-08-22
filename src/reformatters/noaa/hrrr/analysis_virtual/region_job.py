from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd
import xarray as xr

from reformatters.common.iterating import group_by, item
from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim, Timedelta
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrVirtualRegionJob,
    NoaaHrrrVirtualSourceFileCoord,
)


class NoaaHrrrAnalysisVirtualSourceFileCoord(NoaaHrrrVirtualSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.init_time + self.lead_time}


class NoaaHrrrAnalysisVirtualRegionJob(
    NoaaHrrrVirtualRegionJob[NoaaHrrrAnalysisVirtualSourceFileCoord]
):
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("12h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaHrrrDataVar],
    ) -> Sequence[NoaaHrrrAnalysisVirtualSourceFileCoord]:
        """Use the shortest present lead for each variable: f00 if available, f01 otherwise."""
        times = pd.to_datetime(processing_region_ds["time"].values)
        var_groups = group_by(
            data_var_group,
            lambda v: (v.internal_attrs.hrrr_file_type, v.has_hour_0_values()),
        )
        coords = []
        for vars_in_file in var_groups:
            file_type = item({v.internal_attrs.hrrr_file_type for v in vars_in_file})
            has_hour_0_values = item({v.has_hour_0_values() for v in vars_in_file})
            lead_time = pd.Timedelta("0h") if has_hour_0_values else pd.Timedelta("1h")
            coords.extend(
                NoaaHrrrAnalysisVirtualSourceFileCoord(
                    init_time=time - lead_time,
                    lead_time=lead_time,
                    domain="conus",
                    file_type=file_type,
                    data_vars=vars_in_file,
                )
                for time in times
            )
        return coords
