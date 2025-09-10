from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.config_models import EnsembleStatistic
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import (
    http_download_to_disk,
)
from reformatters.common.interpolation import linear_interpolate_1d_inplace
from reformatters.common.iterating import digest
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
)
from reformatters.common.storage import StoreFactory
from reformatters.common.types import (
    AppendDim,
    ArrayND,
    DatetimeLike,
    Dim,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFSDataVar,
    GefsEnsembleSourceFileCoord,
    GEFSFileType,
)
from reformatters.noaa.gefs.read_data import (
    filter_available_times,
    is_available_time,
    parse_grib_index,
    read_data,
)
from reformatters.noaa.noaa_utils import has_hour_0_values


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
        times = filter_available_times(
            pd.to_datetime(processing_region_ds["time"].values)
        )
        coords = []
        for init_time in times:
            lead_time = pd.Timedelta(hours=init_time.hour % 24)
            init_time = init_time.replace(hour=0)

            # Analysis dataset uses control member only (ensemble_memeoner=0)
            coords.append(
                GefsAnalysisSourceFileCoord(
                    init_time=init_time,
                    lead_time=lead_time,
                    data_vars=data_var_group,
                )
            )
        return coords

    def download_file(self, coord: GefsAnalysisSourceFileCoord) -> Path:
        """Download the source file for the given coordinate."""
        # Download grib index file
        idx_url = f"{coord.get_url()}.idx"
        idx_local_path = http_download_to_disk(idx_url, self.dataset_id)
        index_contents = idx_local_path.read_text()

        # Download the grib messages for the data vars in the coord using byte ranges
        starts, ends = parse_grib_index(index_contents, coord)
        vars_suffix = digest(f"{s}-{e}" for s, e in zip(starts, ends, strict=True))
        return http_download_to_disk(
            coord.get_url(),
            self.dataset_id,
            byte_ranges=(starts, ends),
            local_path_suffix=f"-{vars_suffix}",
        )

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
