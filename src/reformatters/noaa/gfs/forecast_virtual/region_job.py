from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd
import xarray as xr

from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim, Timedelta
from reformatters.noaa.gfs.virtual_region_job import (
    GFS_FILE_TYPES,
    NoaaGfsVirtualRegionJob,
    NoaaGfsVirtualSourceFileCoord,
    carried_by,
)
from reformatters.noaa.models import NoaaDataVar


class NoaaGfsForecastVirtualSourceFileCoord(NoaaGfsVirtualSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class NoaaGfsForecastVirtualRegionJob(
    NoaaGfsVirtualRegionJob[NoaaGfsForecastVirtualSourceFileCoord]
):
    # The two 6h cycles before the one a run is firing for, plus its own, so a couple
    # of missed runs still self-heal.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("18h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaDataVar],
    ) -> Sequence[NoaaGfsForecastVirtualSourceFileCoord]:
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)

        coords = []
        for init_time in init_times:
            for lead_time in lead_times:
                for file_type in GFS_FILE_TYPES:
                    # GFS publishes no windowed message at f000, and drops five
                    # instantaneous ones there too; a job filtered to variables one
                    # product does not carry reads only the other.
                    file_vars = [
                        var
                        for var in data_var_group
                        if carried_by(var, file_type)
                        and (lead_time > pd.Timedelta(0) or var.has_hour_0_values())
                    ]
                    if not file_vars:
                        continue
                    coords.append(
                        NoaaGfsForecastVirtualSourceFileCoord(
                            init_time=init_time,
                            lead_time=lead_time,
                            file_type=file_type,
                            data_vars=file_vars,
                        )
                    )
        return coords
