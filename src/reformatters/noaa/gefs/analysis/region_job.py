from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.interpolation import linear_interpolate_1d_inplace
from reformatters.common.iterating import item
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    Array1D,
    ArrayND,
    DatetimeLike,
    Dim,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_INIT_TIME_FREQUENCY,
    GEFS_REFORECAST_END,
    GEFS_REFORECAST_INIT_TIME_FREQUENCY,
    GEFS_REFORECAST_START,
    GEFSDataVar,
    GefsEnsembleSourceFileCoord,
    is_v12_index,
)
from reformatters.noaa.gefs.read_data import read_data
from reformatters.noaa.gefs.utils import gefs_download_file
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)


def is_available_time(times: pd.DatetimeIndex) -> Array1D[np.bool]:
    """Before v12, GEFS files had a 6 hour step."""
    # pre-v12 data is all we have for the 9 month period after the v12 reforecast ends
    # 2019-12-31 and before the v12 forecast archive starts 2020-10-01.
    return is_v12_index(times) | (times.hour % 6 == 0)


def filter_available_times(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return times[is_available_time(times)]


class GefsAnalysisSourceFileCoord(GefsEnsembleSourceFileCoord):
    ensemble_member: int = 0  # Control member for analysis

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {
            "time": self.init_time + self.lead_time,
        }


class GefsAnalysisRegionJob(RegionJob[GEFSDataVar, GefsAnalysisSourceFileCoord]):
    """RegionJob for GEFS Analysis dataset processing."""

    # 1 makes logic simpler when accessing GEFSv12 reforecast which has a file per variable
    max_vars_per_backfill_job = 1
    max_vars_per_download_group = 1

    def get_processing_region(self) -> slice:
        """
        Return processing region with 2-step buffer for interpolation and deaccumulation.
        """
        # Buffer by 2 steps to ensure accumulation starts at a reset step
        buffer_size = 2
        return slice(
            max(0, self.region.start - buffer_size), self.region.stop + buffer_size
        )

    @classmethod
    def source_groups(
        cls, data_vars: Sequence[GEFSDataVar]
    ) -> Sequence[Sequence[GEFSDataVar]]:
        # max_vars_per_download_group = 1 will cause all variables to be processed independently
        return [data_vars]

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[GEFSDataVar]
    ) -> Sequence[GefsAnalysisSourceFileCoord]:
        """Generate source file coordinates for analysis data from forecast files."""
        times = pd.to_datetime(processing_region_ds["time"].values)

        times = filter_available_times(times)

        var_has_hour_0_values = item({has_hour_0_values(var) for var in data_var_group})

        # If var doesn't have hour 0 values we have to go back one forecast
        # so the first step after the reforecast will still be drawn from the reforecast.
        if var_has_hour_0_values:
            is_reforecast = times < GEFS_REFORECAST_END
        else:
            is_reforecast = times <= GEFS_REFORECAST_END

        reforecast_init_times = times[is_reforecast].floor(
            f"{whole_hours(GEFS_REFORECAST_INIT_TIME_FREQUENCY)}h"
        )
        forecast_init_times = times[~is_reforecast].floor(
            f"{whole_hours(GEFS_INIT_TIME_FREQUENCY)}h"
        )
        init_times = pd.DatetimeIndex(
            list(reforecast_init_times) + list(forecast_init_times)
        )

        # If var doesn't have hour 0 values OR we are in the reforecast period which
        # does not have hour 0 values for any variable, we have to go back one forecast.
        # eg. Get the 6th hour rather than the 0th hour for 6-hourly init times.
        is_hour_0 = times == init_times
        do_shift = is_hour_0 & ((not var_has_hour_0_values) | is_reforecast)
        shifted_init_times = pd.DatetimeIndex(
            np.where(
                is_reforecast,
                init_times - GEFS_REFORECAST_INIT_TIME_FREQUENCY,
                init_times - GEFS_INIT_TIME_FREQUENCY,
            )
        )
        init_times = pd.DatetimeIndex(
            np.where(~do_shift, init_times.values, shifted_init_times.values)
        )

        lead_times = times - init_times

        ensemble_coords: Sequence[GefsAnalysisSourceFileCoord] = [
            GefsAnalysisSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                data_vars=data_var_group,
            )
            for init_time, lead_time in zip(init_times, lead_times, strict=True)
            # skip if we shifted init to before start of archive
            if init_time >= GEFS_REFORECAST_START
        ]
        return ensemble_coords

    def download_file(self, coord: GefsAnalysisSourceFileCoord) -> Path:
        """Download the source file for the given coordinate."""
        # Download grib index file
        return gefs_download_file(self.dataset_id, coord)

    def read_data(
        self, coord: GefsAnalysisSourceFileCoord, data_var: GEFSDataVar
    ) -> ArrayND[np.generic]:
        """Read data from the source file for the given coordinate and variable."""
        return read_data(self.template_ds, coord, data_var)

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
                    log.exception(f"Error deaccumulating {data_var.name}")

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
        primary_store: Store,
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
        existing_ds = xr.open_zarr(primary_store, decode_timedelta=True, chunks=None)
        # Start by reprocessing the most recent time already in the dataset; it may be incomplete.
        append_dim_start = existing_ds[append_dim].max()
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            kind="operational-update",
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[GefsAnalysisSourceFileCoord]]
    ) -> xr.Dataset:
        # Remove the last hour because most variables (except precip) lack hour 0 values.
        # Precipitation extends to hour 6 of the latest forecast, but other variables do not,
        # so we trim the final hour to keep all variables aligned.
        return (
            super()
            .update_template_with_results(process_results)
            .isel(time=slice(None, -1))
        )
