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

import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio  # type: ignore
import xarray as xr
import zarr

from reformatters.common.config import Config
from reformatters.common.download import download_to_disk, http_store
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timestamp,
)
from reformatters.noaa.hrrr.hrrr_config_models import (
    HRRRDataVar,
    HRRRDomain,
    HRRRFileType,
)
from reformatters.noaa.noaa_utils import has_hour_0_values


class HRRRSourceFileCoord(SourceFileCoord):
    """Source file coordinate for HRRR forecast data."""

    init_time: Timestamp
    lead_time: pd.Timedelta
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
        final_store: zarr.abc.store.Store,
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

        # Check if we have existing data
        try:
            existing_ds = xr.open_zarr(final_store)
            append_dim_start = cls._update_append_dim_start(existing_ds)
        except (FileNotFoundError, ValueError):
            # If no existing data, start from the beginning of HRRR v3
            append_dim_start = pd.Timestamp("2018-07-13T12:00")

        append_dim_end = cls._update_append_dim_end()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            kind="operational-update",
            final_store=final_store,
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

        # Filter out hour 0 for variables that don't have hour 0 values
        # (accumulated or last N hours avg values)
        filtered_coords = []
        for data_var in data_var_group:
            if not has_hour_0_values(data_var):
                # Filter out hour 0 for this variable
                var_lead_times = [lt for lt in lead_times if lt > pd.Timedelta(hours=0)]
            else:
                var_lead_times = list(lead_times)

            # All data variables in a group should be from the same file type
            file_types = {var.internal_attrs.hrrr_file_type for var in data_var_group}
            if len(file_types) != 1:
                raise ValueError(
                    f"All variables in group must be from same file type, got: {file_types}"
                )
            file_type = next(iter(file_types))

            # HRRR forecast data is available for CONUS domain
            domains: list[HRRRDomain] = ["conus"]

            coords_for_var = [
                HRRRSourceFileCoord(
                    init_time=init_time,
                    lead_time=lead_time,
                    domain=domain,
                    file_type=file_type,
                )
                for init_time, lead_time, domain in product(
                    init_times, var_lead_times, domains
                )
            ]
            filtered_coords.extend(coords_for_var)

        # Remove duplicates while preserving order
        seen = set()
        unique_coords = []
        for coord in filtered_coords:
            coord_key = (
                coord.init_time,
                coord.lead_time,
                coord.domain,
                coord.file_type,
            )
            if coord_key not in seen:
                seen.add(coord_key)
                unique_coords.append(coord)

        return unique_coords

    def download_file(self, coord: HRRRSourceFileCoord) -> Path:
        """Download a HRRR file and return the local path."""
        store = http_store("https://noaa-hrrr-bdp-pds.s3.amazonaws.com")

        # Generate local file paths
        lead_time_hours = coord.lead_time.total_seconds() / (60 * 60)
        init_date_str = coord.init_time.strftime("%Y%m%d")
        init_hour_str = coord.init_time.strftime("%H")

        remote_path = f"hrrr.{init_date_str}/{coord.domain}/hrrr.t{init_hour_str}z.wrf{coord.file_type}f{int(lead_time_hours):02d}.grib2"
        local_path_filename = remote_path.replace("/", "_")

        # Ensure download directory exists
        download_dir = Path("data/download/")
        download_dir.mkdir(parents=True, exist_ok=True)

        idx_remote_path = f"{remote_path}.idx"
        idx_local_path = download_dir / f"{local_path_filename}.idx"
        local_path = download_dir / local_path_filename

        try:
            # First download the index file
            download_to_disk(
                store,
                idx_remote_path,
                idx_local_path,
                overwrite_existing=not Config.is_dev,  # Cache files during development
            )

            # Parse the index file to get byte ranges for our variables
            # We need to filter data_vars to only those matching this file type
            relevant_vars = [
                var
                for var in self.data_vars
                if var.internal_attrs.hrrr_file_type == coord.file_type
            ]

            if not relevant_vars:
                # No variables for this file type, which shouldn't happen in normal operation
                raise ValueError(f"No variables found for file type {coord.file_type}")

            byte_range_starts, byte_range_ends = self._parse_index_byte_ranges(
                idx_local_path, relevant_vars
            )

            # Download the GRIB file with specific byte ranges
            download_to_disk(
                store,
                remote_path,
                local_path,
                overwrite_existing=not Config.is_dev,
                byte_ranges=(byte_range_starts, byte_range_ends),
            )

            return local_path

        except Exception as e:
            # In the original code, missing files return None and are handled gracefully
            # Rather than failing the entire job, we'll log and re-raise so the base class
            # can handle this appropriately
            raise FileNotFoundError(
                f"Failed to download HRRR file {remote_path}: {e}"
            ) from e

    def read_data(
        self,
        coord: HRRRSourceFileCoord,
        data_var: HRRRDataVar,
    ) -> ArrayFloat32:
        """Read data from a HRRR file for a specific variable."""
        if coord.downloaded_path is None:
            raise ValueError("File must be downloaded before reading")

        grib_element = data_var.internal_attrs.grib_element
        grib_description = data_var.internal_attrs.grib_description

        try:
            with (
                warnings.catch_warnings(),
                rasterio.open(coord.downloaded_path) as reader,
            ):
                # Find the matching band for this variable
                matching_bands = [
                    band_i + 1  # rasterio uses 1-based indexing
                    for band_i in range(reader.count)
                    if grib_description in reader.descriptions[band_i]
                    and reader.tags(band_i + 1)["GRIB_ELEMENT"] == grib_element
                ]

                if len(matching_bands) != 1:
                    raise ValueError(
                        f"Expected exactly 1 matching band for {grib_element}, "
                        f"found {len(matching_bands)} in {coord.downloaded_path}"
                    )

                rasterio_band_index = matching_bands[0]

                # Read the data
                result: ArrayFloat32 = reader.read(
                    rasterio_band_index,
                    out_dtype=np.float32,
                )

                # HRRR data comes in as (y, x) but we need to transpose to (x, y)
                # to match our longitude, latitude dimension order
                return result.T

        except Exception as e:
            raise ValueError(
                f"Failed to read data from {coord.downloaded_path}: {e}"
            ) from e

    def _parse_index_byte_ranges(
        self,
        idx_local_path: Path,
        data_vars: Sequence[HRRRDataVar],
    ) -> tuple[list[int], list[int]]:
        """Parse GRIB index file to get byte ranges for specific variables."""
        with open(idx_local_path) as index_file:
            index_contents = index_file.read()

        byte_range_starts = []
        byte_range_ends = []

        for var_info in data_vars:
            var_match_str = f"{var_info.internal_attrs.grib_element}:{var_info.internal_attrs.grib_index_level}"
            var_match_str = re.escape(var_match_str)

            matches = re.findall(
                f"\\d+:(\\d+):.+:{var_match_str}:.+(\\n\\d+:(\\d+))?",
                index_contents,
            )

            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly 1 match for {var_info.name}, "
                    f"found {len(matches)} in {idx_local_path}"
                )

            match = matches[0]
            start_byte = int(match[0])

            if match[2] != "":
                end_byte = int(match[2])
            else:
                # If no end byte specified, add a large offset
                # (similar to existing logic in read_data.py)
                end_byte = start_byte + (10 * (2**30))  # +10 GiB

            byte_range_starts.append(start_byte)
            byte_range_ends.append(end_byte)

        return byte_range_starts, byte_range_ends

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
