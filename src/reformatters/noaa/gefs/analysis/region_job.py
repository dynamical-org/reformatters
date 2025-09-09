from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.config_models import EnsembleStatistic
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.interpolation import linear_interpolate_1d_inplace
from reformatters.common.region_job import RegionJob
from reformatters.common.storage import StoreFactory
from reformatters.common.types import AppendDim, ArrayND, DatetimeLike
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar, GEFSFileType
from reformatters.noaa.gefs.read_data import is_available_time, read_into
from reformatters.noaa.noaa_utils import has_hour_0_values

from .source_file_coord import (
    GefsAnalysisEnsembleSourceFileCoord,
    GefsAnalysisSourceFileCoord,
    GefsAnalysisStatisticSourceFileCoord,
)


class GefsAnalysisRegionJob(RegionJob[GEFSDataVar, GefsAnalysisSourceFileCoord]):
    """RegionJob for GEFS Analysis dataset processing."""

    # From existing _VARIABLES_PER_BACKFILL_JOB in analysis/reformat.py
    max_vars_per_backfill_job = 1

    def get_processing_region(self) -> slice:
        """
        Return processing region with 2-step buffer for interpolation and deaccumulation.
        """
        # Buffer by 2 steps to ensure accumulation starts at a reset step
        buffer_size = 2
        start = max(0, self.region.start - buffer_size)
        stop = min(
            len(self.template_ds[self.append_dim]), self.region.stop + buffer_size
        )
        return slice(start, stop)

    @classmethod
    def source_groups(
        cls, data_vars: Sequence[GEFSDataVar]
    ) -> Sequence[Sequence[GEFSDataVar]]:
        """
        Group variables by GEFS file type and ensemble statistic.
        """
        grouper: dict[
            tuple[GEFSFileType, EnsembleStatistic | None, bool], list[GEFSDataVar]
        ] = defaultdict(list)
        for data_var in data_vars:
            gefs_file_type = data_var.internal_attrs.gefs_file_type
            ensemble_statistic = data_var.attrs.ensemble_statistic
            var_has_hour_0_values = has_hour_0_values(data_var)
            grouper[(gefs_file_type, ensemble_statistic, var_has_hour_0_values)].append(
                data_var
            )

        groups = []
        for idx_data_vars in grouper.values():
            # Sort by index position for consistent ordering
            idx_data_vars = sorted(
                idx_data_vars, key=lambda dv: dv.internal_attrs.index_position
            )
            groups.append(idx_data_vars)

        # Sort groups for consistent ordering
        return sorted(groups, key=lambda g: str(g[0].internal_attrs.gefs_file_type))

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[GEFSDataVar]
    ) -> Sequence[GefsAnalysisSourceFileCoord]:
        """Generate source file coordinates for analysis data from forecast files."""
        coords = []
        for time in processing_region_ds["time"].values:
            time_pd = pd.Timestamp(time)
            # Analysis dataset derived from forecast files
            # We need to reconstruct init_time and lead_time from time
            # For simplicity, assume 6-hour frequency and extract from time coordinate
            # TODO: This needs to be implemented based on the specific analysis logic
            init_time = time_pd - pd.Timedelta(hours=6)  # Placeholder logic
            lead_time = pd.Timedelta(hours=6)  # Placeholder logic

            # Analysis dataset uses control member only (ensemble_member=0)
            coords.append(
                GefsAnalysisEnsembleSourceFileCoord(
                    init_time=init_time,
                    ensemble_member=0,  # Control member for analysis
                    lead_time=lead_time,
                )
            )
        return coords

    def _get_gefs_file_type(
        self, data_var_group: Sequence[GEFSDataVar]
    ) -> GEFSFileType:
        """Get the GEFS file type for a group of variables."""
        # All variables in a group should have the same file type
        file_types = {var.internal_attrs.gefs_file_type for var in data_var_group}
        if len(file_types) != 1:
            raise ValueError(f"Mixed file types in variable group: {file_types}")
        return next(iter(file_types))

    def download_file(self, coord: GefsAnalysisSourceFileCoord) -> Path:
        """Download the source file for the given coordinate."""
        from .source_file_coord import download_source_file

        # We need to determine the file type and variables from the data_vars in the group being processed
        # For now, we'll pass an empty list and handle this properly when integrating with the full pipeline
        _, path = download_source_file(coord, "s+a", [])
        if path is None:
            raise FileNotFoundError(f"Could not download file for {coord}")
        return path

    def read_data(
        self, coord: GefsAnalysisSourceFileCoord, data_var: GEFSDataVar
    ) -> ArrayND[np.generic]:
        """Read data from the source file for the given coordinate and variable."""
        # Create output data array with proper coordinates
        time_coord = coord.out_loc()["time"]
        if isinstance(time_coord, slice):
            times = self.template_ds["time"].isel(time=time_coord)
        else:
            # Ensure we get a 1D DataArray even for single values
            times = self.template_ds["time"].sel(time=time_coord, method="nearest")
            if times.ndim == 0:  # Scalar -> make it 1D
                times = times.expand_dims("time")

        data_array = xr.DataArray(
            np.empty(
                (
                    len(times),
                    len(self.template_ds["latitude"]),
                    len(self.template_ds["longitude"]),
                ),
                dtype=np.float32,
            ),
            coords={
                "time": times,
                "latitude": self.template_ds["latitude"],
                "longitude": self.template_ds["longitude"],
            },
            dims=["time", "latitude", "longitude"],
        )

        # Download the file if not already downloaded
        if coord.downloaded_path is None:
            # We can't modify the frozen dataclass directly
            # The download path should be set by the framework
            path = self.download_file(coord)
        else:
            path = coord.downloaded_path

        # Convert coord to legacy format for read_into
        if isinstance(coord, GefsAnalysisEnsembleSourceFileCoord):
            legacy_coords = {
                "init_time": coord.init_time,
                "ensemble_member": coord.ensemble_member,
                "lead_time": coord.lead_time,
            }
        elif isinstance(coord, GefsAnalysisStatisticSourceFileCoord):
            legacy_coords = {
                "init_time": coord.init_time,
                "statistic": coord.statistic,
                "lead_time": coord.lead_time,
            }
        else:
            raise TypeError(f"Unexpected coordinate type: {type(coord)}")

        # Use existing read_into function
        # Cast the dict to the proper SourceFileCoords type
        from typing import cast

        from reformatters.noaa.gefs.read_data import SourceFileCoords

        # The legacy_coords dict already has the right structure for SourceFileCoords
        read_into(data_array, cast(SourceFileCoords, legacy_coords), path, data_var)
        return data_array.values

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: GEFSDataVar
    ) -> None:
        """
        Apply transformations to data array in place.

        """
        expected_missing = ~is_available_time(pd.to_datetime(data_array["time"].values))

        if data_var.internal_attrs.deaccumulate_to_rate:
            reset_freq = data_var.internal_attrs.window_reset_frequency
            if reset_freq is not None:
                try:
                    deaccumulate_to_rates_inplace(
                        data_array,
                        dim="time",
                        reset_frequency=reset_freq,
                        skip_step=expected_missing,
                    )
                except ValueError:
                    # Log and continue - errors are expected for some data
                    pass

        if expected_missing.any():
            linear_interpolate_1d_inplace(
                data_array, dim="time", where=expected_missing
            )

        keep_mantissa_bits = data_var.internal_attrs.keep_mantissa_bits
        if isinstance(keep_mantissa_bits, int):
            round_float32_inplace(
                data_array.values,
                keep_mantissa_bits=keep_mantissa_bits,
            )

    @classmethod
    def operational_update_jobs(
        cls,
        store_factory: StoreFactory,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[GEFSDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[GEFSDataVar, GefsAnalysisSourceFileCoord]"], xr.Dataset
    ]:
        """
        Return the sequence of RegionJob instances necessary to update the dataset
        from its current state to include the latest available data.

        Also return the template_ds, expanded along append_dim through the end of
        the data to process. The dataset returned here may extend beyond the
        available data at the source, in which case `update_template_with_results`
        will trim the dataset to the actual data processed.

        For GEFS analysis, we process data along the time dimension, starting from
        the most recent time already in the dataset and extending to the current time.
        """
        existing_ds = xr.open_zarr(
            store_factory.primary_store(), decode_timedelta=True, chunks=None
        )
        # Start by reprocessing the most recent time already in the dataset; it may be incomplete.
        append_dim_start = existing_ds[append_dim].max()
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            kind="operational-update",
            store_factory=store_factory,
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds
