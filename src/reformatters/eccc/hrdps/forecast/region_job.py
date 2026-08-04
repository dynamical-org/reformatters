from collections.abc import Mapping, Sequence

import pandas as pd
import xarray as xr

from reformatters.common.iterating import item
from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim
from reformatters.eccc.hrdps.hrdps_config_models import EcccHrdpsDataVar
from reformatters.eccc.hrdps.region_job import (
    EcccHrdpsRegionJob,
    EcccHrdpsSourceFileCoord,
)


class EcccHrdpsForecastSourceFileCoord(EcccHrdpsSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class EcccHrdpsForecastRegionJob(EcccHrdpsRegionJob):
    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcccHrdpsDataVar],
    ) -> Sequence[EcccHrdpsForecastSourceFileCoord]:
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)

        data_var = item(data_var_group)  # each HRDPS grib file contains one variable
        if not data_var.has_hour_0_values():
            lead_times = lead_times[lead_times > pd.Timedelta(hours=0)]

        return [
            EcccHrdpsForecastSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                data_var=data_var,
            )
            for init_time in init_times
            for lead_time in lead_times
        ]
