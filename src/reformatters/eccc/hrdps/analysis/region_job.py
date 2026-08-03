from collections.abc import Mapping, Sequence

import pandas as pd
import xarray as xr

from reformatters.common.iterating import item
from reformatters.common.region_job import CoordinateValue, SourceFileResult
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim
from reformatters.eccc.hrdps.hrdps_config_models import EcccHrdpsDataVar
from reformatters.eccc.hrdps.region_job import (
    HRDPS_INIT_FREQUENCY,
    EcccHrdpsRegionJob,
    EcccHrdpsSourceFileCoord,
)


class EcccHrdpsAnalysisSourceFileCoord(EcccHrdpsSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.init_time + self.lead_time}


class EcccHrdpsAnalysisRegionJob(EcccHrdpsRegionJob):
    def get_processing_region(self) -> slice:
        """Buffer start by one step to allow deaccumulation without gaps in resulting output."""
        return slice(max(0, self.region.start - 1), self.region.stop)

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcccHrdpsDataVar],
    ) -> Sequence[EcccHrdpsAnalysisSourceFileCoord]:
        times = pd.to_datetime(processing_region_ds["time"].values)

        data_var = item(data_var_group)  # each HRDPS grib file contains one variable
        init_freq_hours = f"{whole_hours(HRDPS_INIT_FREQUENCY)}h"
        if data_var.has_hour_0_values():
            init_times = times.floor(init_freq_hours)
        else:
            init_times = (times - pd.Timedelta("1h")).floor(init_freq_hours)

        lead_times = times - init_times

        return [
            EcccHrdpsAnalysisSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                data_var=data_var,
            )
            for init_time, lead_time in zip(init_times, lead_times, strict=True)
        ]

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[SourceFileResult]]
    ) -> xr.DataTree:
        # Remove the last hour. We pull accumulated variables (precipitation, radiation)
        # from lead times 1-6 hours, but use lead times 0-5 hours for other variables.
        # This results in one additional hour of data for accumulated variables. Trim it
        # off so we aren't left with nans for most variables in the final step.
        return (
            super()
            .update_template_with_results(process_results)
            .isel(time=slice(None, -1))
        )
