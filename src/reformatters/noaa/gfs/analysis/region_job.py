from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.iterating import item
from reformatters.common.region_job import (
    CoordinateValue,
    RegionJob,
    SourceFileResult,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    DatetimeLike,
    Dim,
    Timestamp,
)
from reformatters.noaa.gfs.region_job import (
    NoaaGfsCommonRegionJob,
    NoaaGfsSourceFileCoord,
)
from reformatters.noaa.models import NoaaDataVar

NOAA_GFS_INIT_FREQUENCY = pd.Timedelta("6h")


class NoaaGfsAnalysisSourceFileCoord(NoaaGfsSourceFileCoord):
    """Coordinates of a single source file to process for analysis dataset."""

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.init_time + self.lead_time}


class NoaaGfsAnalysisRegionJob(NoaaGfsCommonRegionJob):
    """Region job for GFS analysis data processing."""

    max_vars_per_download_group = 3
    max_vars_per_job = 12

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaDataVar],
        reformat_job_name: str,
        *,
        job_reference_time: Timestamp | None = None,
    ) -> tuple[Sequence[RegionJob[NoaaDataVar, NoaaGfsSourceFileCoord]], xr.DataTree]:
        existing_ds = xr.open_zarr(primary_store, decode_timedelta=True, chunks=None)
        append_dim_start = existing_ds[append_dim].max()
        reference = (
            job_reference_time if job_reference_time is not None else pd.Timestamp.now()
        )
        template_ds = get_template_fn(reference.floor("6h") + pd.Timedelta(hours=7))
        jobs = cls.get_jobs(
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds

    def get_processing_region(self) -> slice:
        """Buffer start by one step to allow deaccumulation without gaps in resulting output."""
        return slice(max(0, self.region.start - 1), self.region.stop)

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaDataVar],
    ) -> Sequence[NoaaGfsAnalysisSourceFileCoord]:
        times = pd.to_datetime(processing_region_ds["time"].values)
        group_has_hour_0 = item({var.has_hour_0_values() for var in data_var_group})

        init_freq_hours = f"{whole_hours(NOAA_GFS_INIT_FREQUENCY)}h"
        if group_has_hour_0:
            init_times = times.floor(init_freq_hours)
        else:
            init_times = (times - pd.Timedelta("1h")).floor(init_freq_hours)

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
        self, process_results: Mapping[str, Sequence[SourceFileResult]]
    ) -> xr.DataTree:
        # Remove the last hour. We pull accumulated variables (precipitation) from the 1 hour lead time,
        # but use the 0 hour lead time for other variables. This results in one additional
        # hour of data for accumulated variables. Trim it off so we aren't left with nans for
        # most variables in the final step.
        return (
            super()
            .update_template_with_results(process_results)
            .isel(time=slice(None, -1))
        )
