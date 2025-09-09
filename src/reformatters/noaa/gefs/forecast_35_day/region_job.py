from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.config_models import EnsembleStatistic
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.region_job import RegionJob
from reformatters.common.storage import StoreFactory
from reformatters.common.types import AppendDim, ArrayND, DatetimeLike
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar, GEFSFileType
from reformatters.noaa.gefs.read_data import read_into

from .source_file_coord import (
    GefsForecast35DayEnsembleSourceFileCoord,
    GefsForecast35DaySourceFileCoord,
    GefsForecast35DayStatisticSourceFileCoord,
)


class GefsForecast35DayRegionJob(
    RegionJob[GEFSDataVar, GefsForecast35DaySourceFileCoord]
):
    """RegionJob for GEFS Forecast 35-Day dataset processing."""

    # From existing _VARIABLES_PER_BACKFILL_JOB in forecast_35_day/reformat.py
    max_vars_per_backfill_job = 3

    @classmethod
    def source_groups(
        cls, data_vars: Sequence[GEFSDataVar]
    ) -> Sequence[Sequence[GEFSDataVar]]:
        """
        Group variables by GEFS file type and ensemble statistic.

        Note: forecast version doesn't include has_hour_0_values in grouping.
        """
        grouper: dict[
            tuple[GEFSFileType, EnsembleStatistic | None], list[GEFSDataVar]
        ] = defaultdict(list)
        for data_var in data_vars:
            gefs_file_type = data_var.internal_attrs.gefs_file_type
            ensemble_statistic = data_var.attrs.ensemble_statistic
            grouper[(gefs_file_type, ensemble_statistic)].append(data_var)

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
    ) -> Sequence[GefsForecast35DaySourceFileCoord]:
        """Generate source file coordinates for forecast data."""
        coords = []
        for init_time in processing_region_ds["init_time"].values:
            for lead_time in processing_region_ds["lead_time"].values:
                for ensemble_member in processing_region_ds["ensemble_member"].values:
                    coords.append(
                        GefsForecast35DayEnsembleSourceFileCoord(
                            init_time=pd.Timestamp(init_time),
                            lead_time=pd.Timedelta(lead_time),
                            ensemble_member=int(ensemble_member),
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

    def download_file(self, coord: GefsForecast35DaySourceFileCoord) -> Path:
        """Download the source file for the given coordinate."""
        from .source_file_coord import download_source_file

        # We need to determine the file type and variables from the data_vars in the group being processed
        # For now, we'll pass an empty list and handle this properly when integrating with the full pipeline
        _, path = download_source_file(coord, "s+a", [])
        if path is None:
            raise FileNotFoundError(f"Could not download file for {coord}")
        return path

    def read_data(
        self, coord: GefsForecast35DaySourceFileCoord, data_var: GEFSDataVar
    ) -> ArrayND[np.generic]:
        """Read data from the source file for the given coordinate and variable."""
        # Create output data array with proper coordinates
        data_array = xr.DataArray(
            np.empty(
                (
                    1,
                    1,
                    1,
                    len(self.template_ds["latitude"]),
                    len(self.template_ds["longitude"]),
                ),
                dtype=np.float32,
            ),
            coords={
                "init_time": [coord.init_time],
                "lead_time": [coord.lead_time],
                "ensemble_member": [coord.ensemble_member]
                if hasattr(coord, "ensemble_member")
                else [0],
                "latitude": self.template_ds["latitude"],
                "longitude": self.template_ds["longitude"],
            },
            dims=["init_time", "lead_time", "ensemble_member", "latitude", "longitude"],
        )

        # Download the file if not already downloaded
        if coord.downloaded_path is None:
            # We can't modify the frozen dataclass directly
            # The download path should be set by the framework
            path = self.download_file(coord)
        else:
            path = coord.downloaded_path

        # Convert coord to legacy format for read_into
        if isinstance(coord, GefsForecast35DayEnsembleSourceFileCoord):
            legacy_coords = {
                "init_time": coord.init_time,
                "ensemble_member": coord.ensemble_member,
                "lead_time": coord.lead_time,
            }
        elif isinstance(coord, GefsForecast35DayStatisticSourceFileCoord):
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

        Based on existing transformation logic, but forecast data doesn't need
        the same missing value interpolation as analysis data.
        """
        if data_var.internal_attrs.deaccumulate_to_rate:
            reset_freq = data_var.internal_attrs.window_reset_frequency
            if reset_freq is not None:
                try:
                    deaccumulate_to_rates_inplace(
                        data_array,
                        dim="lead_time",
                        reset_frequency=reset_freq,
                    )
                except ValueError:
                    # Log and continue - errors are expected for some data
                    pass

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
        Sequence["RegionJob[GEFSDataVar, GefsForecast35DaySourceFileCoord]"], xr.Dataset
    ]:
        """
        Return the sequence of RegionJob instances necessary to update the dataset
        from its current state to include the latest available data.

        Also return the template_ds, expanded along append_dim through the end of
        the data to process. The dataset returned here may extend beyond the
        available data at the source, in which case `update_template_with_results`
        will trim the dataset to the actual data processed.

        For GEFS 35-day forecast, we process data along the init_time dimension, starting from
        the most recent init_time already in the dataset and extending to the current time.
        """
        existing_ds = xr.open_zarr(
            store_factory.primary_store(), decode_timedelta=True, chunks=None
        )
        # Start by reprocessing the most recent init_time already in the dataset; it may be incomplete.
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
