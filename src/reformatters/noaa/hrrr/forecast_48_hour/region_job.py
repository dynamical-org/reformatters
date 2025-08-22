from collections.abc import Callable, Mapping, Sequence
from itertools import groupby
from pathlib import Path

import pandas as pd
import xarray as xr

from reformatters.common.iterating import item
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.storage import StoreFactory
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timedelta,
    Timestamp,
)
from reformatters.noaa.hrrr.hrrr_config_models import (
    HRRRDataVar,
    HRRRDomain,
    HRRRFileType,
)
from reformatters.noaa.hrrr.read_data import download_hrrr_file, read_hrrr_data
from reformatters.noaa.noaa_utils import has_hour_0_values


class HRRRSourceFileCoord(SourceFileCoord):
    """Source file coordinate for HRRR forecast data."""

    init_time: Timestamp
    lead_time: Timedelta
    domain: HRRRDomain
    file_type: HRRRFileType

    def get_url(self) -> str:
        """Return the URL for this HRRR file."""
        lead_time_hours = whole_hours(self.lead_time)
        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")

        return f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{init_date_str}/{self.domain}/hrrr.t{init_hour_str}z.wrf{self.file_type}f{int(lead_time_hours):02d}.grib2"

    def get_idx_url(self) -> str:
        """Return the URL for the GRIB index file."""
        return f"{self.get_url()}.idx"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        """Return the output location for this file's data in the dataset."""
        # Map to the standard dimension names used in the template
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }


class NoaaHrrrForecast48HourRegionJob(RegionJob[HRRRDataVar, HRRRSourceFileCoord]):
    """Region job for HRRR 48-hour forecast data processing."""

    max_vars_per_download_group = 8  # currently a best guess

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[HRRRDataVar],
    ) -> Sequence[Sequence[HRRRDataVar]]:
        """Group variables by HRRR file type, since each file type comes from a separate source file."""
        # Sort by file type first, then group
        sorted_vars = sorted(data_vars, key=lambda v: v.internal_attrs.hrrr_file_type)
        return tuple(
            tuple(vars_iter)
            for _, vars_iter in groupby(
                sorted_vars,
                key=lambda v: (has_hour_0_values(v), v.internal_attrs.hrrr_file_type),
            )
        )

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store_factory: StoreFactory,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[HRRRDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[HRRRDataVar, HRRRSourceFileCoord]],
        xr.Dataset,
    ]:
        """Generate operational update jobs for HRRR forecast data."""
        # For operational updates, we want to process recent forecast data
        # HRRR provides forecasts every hour, but 48-hour forecasts are only available
        # every 6 hours (00, 06, 12, 18 UTC)

        existing_ds = xr.open_zarr(primary_store_factory.primary_store())
        append_dim_start = cls._update_append_dim_start(existing_ds)

        append_dim_end = cls._update_append_dim_end()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            kind="operational-update",
            primary_store_factory=primary_store_factory,
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[HRRRDataVar],
    ) -> Sequence[HRRRSourceFileCoord]:
        """Generate source file coordinates for the processing region."""
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        group_has_hour_0 = item({has_hour_0_values(var) for var in data_var_group})
        if not group_has_hour_0:
            lead_times = lead_times[lead_times > pd.Timedelta(hours=0)]

        file_type = item({var.internal_attrs.hrrr_file_type for var in data_var_group})

        return [
            HRRRSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                domain="conus",
                file_type=file_type,
            )
            for init_time in init_times
            for lead_time in lead_times
        ]

    def download_file(self, coord: HRRRSourceFileCoord) -> Path:
        """Download a HRRR file and return the local path."""
        # Filter data_vars to only those matching this file type
        relevant_vars = [
            var
            for var in self.data_vars
            if var.internal_attrs.hrrr_file_type == coord.file_type
        ]

        if not relevant_vars:
            # No variables for this file type, which shouldn't happen in normal operation
            raise ValueError(f"No variables found for file type {coord.file_type}")

        # Use the standalone function for consistent download behavior
        return download_hrrr_file(
            init_time=coord.init_time,
            lead_time=coord.lead_time,
            domain=coord.domain,
            file_type=coord.file_type,
            data_vars=relevant_vars,
        )

    def read_data(
        self,
        coord: HRRRSourceFileCoord,
        data_var: HRRRDataVar,
    ) -> ArrayFloat32:
        """Read data from a HRRR file for a specific variable."""
        assert coord.downloaded_path is not None  # for type check, system guarantees it
        return read_hrrr_data(coord.downloaded_path, data_var)

    @classmethod
    def _update_append_dim_end(cls) -> pd.Timestamp:
        """Get the end time for operational updates."""
        return pd.Timestamp.now(tz="UTC")

    @classmethod
    def _update_append_dim_start(cls, existing_ds: xr.Dataset) -> pd.Timestamp:
        """Get the start time for operational updates based on existing data."""
        ds_max_time = existing_ds["init_time"].max().item()
        return pd.Timestamp(ds_max_time) - pd.Timedelta(hours=6)
