"""
RegionJob implementation for NOAA HRRR 48-hour forecast data.

This module implements the new RegionJob abstraction for processing HRRR forecast data,
replacing the old procedural approach in reformat.py and reformat_internals.py with a
class-based, testable, and maintainable implementation.

Key features:
- Handles HRRR-specific file types (sfc, prs, nat, subh) and groups variables by file type
- Downloads GRIB files with byte-range optimization using index files
- Filters out hour 0 data for accumulated/averaged variables
- Supports operational updates with appropriate time window handling
- Integrates with the common RegionJob framework for parallelization and error handling
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import xarray as xr

from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.storage import StoreFactory
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
        lead_time_hours = self.lead_time.total_seconds() / (60 * 60)
        if lead_time_hours != round(lead_time_hours):
            raise ValueError(
                f"Lead time {self.lead_time} must be a whole number of hours"
            )

        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")

        return f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{init_date_str}/{self.domain}/hrrr.t{init_hour_str}z.wrf{self.file_type}f{int(lead_time_hours):02d}.grib2"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        """Return the output location for this file's data in the dataset."""
        # Map to the standard dimension names used in the template
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }

    def get_idx_url(self) -> str:
        """Return the URL for the GRIB index file."""
        return f"{self.get_url()}.idx"


class NoaaHrrrForecast48HourRegionJob(RegionJob[HRRRDataVar, HRRRSourceFileCoord]):
    """Region job for HRRR 48-hour forecast data processing."""

    # HRRR files can be large, so limit download parallelism
    download_parallelism: int = 8

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[HRRRDataVar],
    ) -> Sequence[Sequence[HRRRDataVar]]:
        """
        Group variables by HRRR file type, since each file type comes from a separate source file.
        """
        from itertools import groupby

        # Sort by file type first, then group
        sorted_vars = sorted(data_vars, key=lambda v: v.internal_attrs.hrrr_file_type)
        groups = []
        for _file_type, vars_iter in groupby(
            sorted_vars, key=lambda v: v.internal_attrs.hrrr_file_type
        ):
            groups.append(list(vars_iter))

        return groups

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

        # Check if we have existing data - operational updates require existing data
        try:
            existing_ds = xr.open_zarr(primary_store_factory.store())
            append_dim_start = cls._update_append_dim_start(existing_ds)
        except (FileNotFoundError, ValueError) as e:
            # Operational updates require existing data - backfill must be done first
            raise ValueError(
                "No existing data found for operational update. "
                "Please run a backfill job first to create the initial dataset."
            ) from e

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

        # All data variables in a group should be from the same file type
        file_types = {var.internal_attrs.hrrr_file_type for var in data_var_group}
        if len(file_types) != 1:
            raise ValueError(
                f"All variables in group must be from same file type, got: {file_types}"
            )
        file_type = next(iter(file_types))

        # Check if any variable in the group doesn't have hour 0 values
        group_has_hour_0 = any(has_hour_0_values(var) for var in data_var_group)
        if not group_has_hour_0:
            # Filter out hour 0 for this group
            lead_times = pd.to_timedelta(
                [lt for lt in lead_times if lt > pd.Timedelta(hours=0)]
            )

        # Generate coordinates for all init_time x lead_time combinations
        coords = []
        for init_time in init_times:
            for lead_time in lead_times:
                coords.append(
                    HRRRSourceFileCoord(
                        init_time=init_time,
                        lead_time=lead_time,
                        domain="conus",
                        file_type=file_type,
                    )
                )

        return coords

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
        if coord.downloaded_path is None:
            raise ValueError("File must be downloaded before reading")

        # Use the standalone function for consistent data reading
        return read_hrrr_data(coord.downloaded_path, data_var)

    @classmethod
    def _update_append_dim_end(cls) -> pd.Timestamp:
        """Get the end time for operational updates."""
        # For operational updates, go up to current time
        return pd.Timestamp.now(tz="UTC")

    @classmethod
    def _update_append_dim_start(cls, existing_ds: xr.Dataset) -> pd.Timestamp:
        """Get the start time for operational updates based on existing data."""
        ds_max_time = existing_ds["init_time"].max().item()
        # Go back a few days to ensure we catch any delayed data
        return pd.Timestamp(ds_max_time) - pd.Timedelta(days=3)
