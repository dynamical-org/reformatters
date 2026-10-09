from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.region_job import RegionJob
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, DatetimeLike, Timedelta, Timestamp
from reformatters.noaa.rrfs.models import NoaaRrfsDataVar
from reformatters.noaa.rrfs.region_job import (
    SOURCE_PREFIX,
    NoaaRrfsRegionJob,
    NoaaRrfsSourceFileCoord,
)


class NoaaRrfsSubHourlySourceFileCoord(NoaaRrfsSourceFileCoord):
    source_family: Literal["subh"] = "subh"

    def get_url(self) -> str:
        hour = whole_hours(self.lead_time)
        cycle = self.init_time.strftime("%Y%m%d/%H")
        prefix = self.init_time.strftime("rrfs.t%Hz")
        return f"{SOURCE_PREFIX}rrfs.{cycle}/{prefix}.2dfld.3km.subh.f{hour:03}.conus.grib2"

    def message_lead_times(self) -> Sequence[Timedelta]:
        return tuple(self.lead_time - pd.Timedelta(minutes=m) for m in (45, 30, 15, 0))


class NoaaRrfsForecastSubHourlyVirtualRegionJob(NoaaRrfsRegionJob):
    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaRrfsDataVar],
        reformat_job_name: str,
        job_fire_time: Timestamp | None = None,
    ) -> tuple[
        Sequence[RegionJob[NoaaRrfsDataVar, NoaaRrfsSourceFileCoord]], xr.DataTree
    ]:
        return super().operational_update_jobs(
            primary_store,
            tmp_store,
            get_template_fn,
            append_dim,
            all_data_vars,
            reformat_job_name,
            job_fire_time=(job_fire_time or pd.Timestamp.now()) - pd.Timedelta("1h"),
        )

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaRrfsDataVar],
    ) -> Sequence[NoaaRrfsSubHourlySourceFileCoord]:
        assert all(v.internal_attrs.source_family == "subh" for v in data_var_group)
        file_leads = sorted(
            {
                lead.ceil("1h")
                for lead in pd.to_timedelta(processing_region_ds["lead_time"].values)
            }
        )
        return [
            NoaaRrfsSubHourlySourceFileCoord(
                init_time=init, lead_time=lead, data_vars=variables
            )
            for init in pd.to_datetime(processing_region_ds["init_time"].values)
            for lead in file_leads
            if (variables := [v for v in data_var_group if v.available_at(lead)])
        ]
