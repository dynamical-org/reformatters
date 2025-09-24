from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import (
    http_download_to_disk,
)
from reformatters.common.iterating import digest, item
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.types import AppendDim, ArrayND, DatetimeLike
from reformatters.noaa.gefs.gefs_config_models import (
    GEFSDataVar,
    GefsEnsembleSourceFileCoord,
    GEFSFileType,
)
from reformatters.noaa.gefs.read_data import read_data
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)


class GefsForecast35DaySourceFileCoord(GefsEnsembleSourceFileCoord):
    # We share the name attributes (init_time, lead_time, data_vars) with
    # GefsSourceFileCoord, and the default out_loc implementation of
    # {"init_time": self.init_time, "lead_time": self.lead_time}
    # is correct for this dataset.
    pass


class GefsForecast35DayRegionJob(
    RegionJob[GEFSDataVar, GefsForecast35DaySourceFileCoord]
):
    """RegionJob for GEFS Forecast 35-Day dataset processing."""

    max_vars_per_backfill_job = 3

    @classmethod
    def source_groups(
        cls, data_vars: Sequence[GEFSDataVar]
    ) -> Sequence[Sequence[GEFSDataVar]]:
        """
        Group variables by GEFS file type and ensemble statistic.

        Note: forecast version doesn't include has_hour_0_values in grouping.
        """
        grouper: dict[tuple[GEFSFileType, bool], list[GEFSDataVar]] = defaultdict(list)
        for data_var in data_vars:
            gefs_file_type = data_var.internal_attrs.gefs_file_type
            grouper[(gefs_file_type, has_hour_0_values(data_var))].append(data_var)

        groups = []
        for idx_data_vars in grouper.values():
            # Sort by index position for better coalescing of byte range requests to the grib
            groups.append(
                sorted(idx_data_vars, key=lambda dv: dv.internal_attrs.index_position)
            )

        # Sort groups for consistent ordering
        return sorted(groups, key=lambda g: str(g[0].internal_attrs.gefs_file_type))

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[GEFSDataVar]
    ) -> Sequence[GefsForecast35DaySourceFileCoord]:
        """Generate source file coordinates for forecast data."""
        # Filter out lead_time=0 for variables that don't have hour 0 values
        # (accumulated and last N hour avg values don't exist in the 0-hour forecast)
        var_has_hour_0_values = item({has_hour_0_values(v) for v in data_var_group})
        if not var_has_hour_0_values:
            processing_region_ds = processing_region_ds.sel(lead_time=slice("1h", None))

        coords = []
        for init_time in processing_region_ds["init_time"].values:
            for lead_time in processing_region_ds["lead_time"].values:
                for ensemble_member in processing_region_ds["ensemble_member"].values:
                    coords.append(
                        GefsForecast35DaySourceFileCoord(
                            init_time=pd.Timestamp(init_time),
                            lead_time=pd.Timedelta(lead_time),
                            data_vars=data_var_group,
                            ensemble_member=int(ensemble_member),
                        )
                    )
        return coords

    def download_file(self, coord: GefsForecast35DaySourceFileCoord) -> Path:
        """Download the source file for the given coordinate."""
        # Download grib index file
        idx_url = f"{coord.get_url()}.idx"
        idx_local_path = http_download_to_disk(idx_url, self.dataset_id)

        # Download the grib messages for the data vars in the coord using byte ranges
        starts, ends = grib_message_byte_ranges_from_index(
            idx_local_path, coord.data_vars, coord.init_time, coord.lead_time
        )
        vars_suffix = digest(f"{s}-{e}" for s, e in zip(starts, ends, strict=True))
        return http_download_to_disk(
            coord.get_url(),
            self.dataset_id,
            byte_ranges=(starts, ends),
            local_path_suffix=f"-{vars_suffix}",
        )

    def read_data(
        self, coord: GefsForecast35DaySourceFileCoord, data_var: GEFSDataVar
    ) -> ArrayND[np.generic]:
        """Read data from the source file for the given coordinate and variable."""
        return read_data(self.template_ds, coord, data_var)

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
            assert reset_freq is not None
            try:
                deaccumulate_to_rates_inplace(
                    data_array,
                    dim="lead_time",
                    reset_frequency=reset_freq,
                )
            except ValueError:
                log.exception(f"Error deaccumulating {data_var.name}")

        keep_mantissa_bits = data_var.internal_attrs.keep_mantissa_bits
        if isinstance(keep_mantissa_bits, int):
            round_float32_inplace(
                data_array.values,
                keep_mantissa_bits=keep_mantissa_bits,
            )

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: zarr.abc.store.Store,
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
        existing_ds = xr.open_zarr(primary_store, decode_timedelta=True, chunks=None)
        # Start by reprocessing the most recent init_time already in the dataset; it may be incomplete.
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
