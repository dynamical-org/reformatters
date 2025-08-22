import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio  # type: ignore
import xarray as xr

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import (
    http_download_to_disk,
)
from reformatters.common.iterating import digest, item
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.storage import StoreFactory
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
)
from reformatters.noaa.gfs.gfs_common import (
    NoaaGfsSourceFileCoord,
    parse_grib_index,
)
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)


class NoaaGfsForecastSourceFileCoord(NoaaGfsSourceFileCoord):
    # We share the name attributes (init_time, lead_time, data_vars) with
    # NoaaGfsSourceFileCoord, and the default out_loc implementation of
    # {"init_time": self.init_time, "lead_time": self.lead_time}
    # is correct for this dataset.
    pass


class NoaaGfsForecastRegionJob(RegionJob[NoaaDataVar, NoaaGfsForecastSourceFileCoord]):
    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[NoaaDataVar],
    ) -> Sequence[Sequence[NoaaDataVar]]:
        """
        Return groups of variables, where all variables in a group can be retrieived from the same source file.

        By separating variables that have hour 0 values from those that don't here,
        generate_source_file coords can skip attempting to download hour 0 data,
        reducing error log noise.
        """
        grouped = defaultdict(list)
        for data_var in data_vars:
            grouped[has_hour_0_values(data_var)].append(data_var)
        return list(grouped.values())

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[NoaaDataVar]
    ) -> Sequence[NoaaGfsForecastSourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds."""
        var_has_hour_0_values = item({has_hour_0_values(v) for v in data_var_group})
        if not var_has_hour_0_values:
            processing_region_ds = processing_region_ds.sel(lead_time=slice("1h", None))

        return [
            NoaaGfsForecastSourceFileCoord(
                init_time=pd.Timestamp(init_time),
                lead_time=pd.Timedelta(lead_time),
                data_vars=data_var_group,
            )
            for init_time, lead_time in product(
                processing_region_ds["init_time"].values,
                processing_region_ds["lead_time"].values,
            )
        ]

    def download_file(self, coord: NoaaGfsForecastSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path."""
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
        self,
        coord: NoaaGfsSourceFileCoord,
        data_var: NoaaDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        grib_element = data_var.internal_attrs.grib_element
        # GFS accumulations add a zero-padded accumulation hours suffix to the grib element, e.g. `APCP04`
        # This is only present in the grib element metadata, the grib index looks like `:APCP:surface:0-4 hour acc fcst:`
        if data_var.internal_attrs.deaccumulate_to_rate:
            assert data_var.internal_attrs.window_reset_frequency is not None
            lead_hours = whole_hours(coord.lead_time)
            window_reset_hours = whole_hours(
                data_var.internal_attrs.window_reset_frequency
            )
            lead_hours_window = lead_hours % window_reset_hours
            if lead_hours_window == 0:
                lead_hours_window = window_reset_hours
            grib_element += str(lead_hours_window).zfill(2)

        grib_description = data_var.internal_attrs.grib_description

        with warnings.catch_warnings(), rasterio.open(coord.downloaded_path) as reader:
            matching_bands = [
                rasterio_band_i
                for band_i in range(reader.count)
                if reader.descriptions[band_i] == grib_description
                and reader.tags(rasterio_band_i := band_i + 1)["GRIB_ELEMENT"]
                == grib_element
            ]

            assert len(matching_bands) == 1, (
                f"Expected exactly 1 matching band, found {matching_bands}. "
                f"{grib_element=}, {grib_description=}, {coord.downloaded_path=}"
            )
            rasterio_band_index = matching_bands[0]

            result: ArrayFloat32 = reader.read(
                rasterio_band_index,
                out_dtype=np.float32,
            )
            return result

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: NoaaDataVar
    ) -> None:
        """Apply in-place data transformations to the output data array for a given data variable."""
        if data_var.internal_attrs.deaccumulate_to_rate:
            assert data_var.internal_attrs.window_reset_frequency is not None
            log.info(f"Converting {data_var.name} from accumulations to rates")
            try:
                deaccumulate_to_rates_inplace(
                    data_array,
                    dim="lead_time",
                    reset_frequency=data_var.internal_attrs.window_reset_frequency,
                )
            except ValueError:
                # Log exception so we are notified if deaccumulation errors are larger than expected.
                log.exception(f"Error deaccumulating {data_var.name}")

        keep_mantissa_bits = data_var.internal_attrs.keep_mantissa_bits
        if isinstance(keep_mantissa_bits, int):
            round_float32_inplace(data_array.values, keep_mantissa_bits)

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[NoaaGfsForecastSourceFileCoord]]
    ) -> xr.Dataset:
        """Update template dataset based on processing results."""
        # TODO: add and update ingested forecast length coord
        return super().update_template_with_results(process_results)

    @classmethod
    def operational_update_jobs(
        cls,
        store_factory: StoreFactory,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[NoaaDataVar, NoaaGfsForecastSourceFileCoord]"], xr.Dataset
    ]:
        """
        Return the sequence of RegionJob instances necessary to update the dataset
        from its current state to include the latest available data.
        """
        existing_ds = xr.open_zarr(
            store_factory.primary_store(), decode_timedelta=True, chunks=None
        )
        # Start by reprocessing the most recent forecast already in the dataset; it may be incomplete.
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
