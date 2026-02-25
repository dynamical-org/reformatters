import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from obstore.exceptions import GenericError
from zarr.abc.store import Store

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import (
    http_download_to_disk,
)
from reformatters.common.iterating import digest, group_by
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.retry import retry
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timedelta,
    Timestamp,
)
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)

type DownloadSource = Literal["s3", "nomads"]

# Limit concurrent NOMADS requests to avoid overloading their servers
_nomads_semaphore = threading.Semaphore(4)


class NoaaGfsSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process."""

    init_time: Timestamp
    lead_time: Timedelta
    data_vars: Sequence[NoaaDataVar]

    def get_url(self, source: DownloadSource = "s3") -> str:
        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")
        lead_hours = whole_hours(self.lead_time)
        path = f"gfs.{init_date_str}/{init_hour_str}/atmos/gfs.t{init_hour_str}z.pgrb2.0p25.f{lead_hours:03d}"
        match source:
            case "nomads":
                base = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
            case "s3":
                base = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
            case _ as unreachable:
                assert_never(unreachable)

        return f"{base}/{path}"

    def get_idx_url(self, source: DownloadSource = "s3") -> str:
        return f"{self.get_url(source=source)}.idx"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        raise NotImplementedError("Subclasses must implement out_loc()")


class NoaaGfsCommonRegionJob(RegionJob[NoaaDataVar, NoaaGfsSourceFileCoord]):
    """Common RegionJob for GFS datasets."""

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[NoaaDataVar],
    ) -> Sequence[Sequence[NoaaDataVar]]:
        """Return groups of variables that can be downloaded from the same source file."""
        return group_by(data_vars, has_hour_0_values)

    def _get_download_source(self, init_time: pd.Timestamp) -> DownloadSource:
        if init_time >= (pd.Timestamp.now() - pd.Timedelta(hours=18)):
            return "nomads"
        else:
            return "s3"

    def _download_from_source(
        self, coord: NoaaGfsSourceFileCoord, source: DownloadSource
    ) -> Path:
        idx_local_path = http_download_to_disk(
            coord.get_idx_url(source=source), self.dataset_id
        )
        starts, ends = grib_message_byte_ranges_from_index(
            idx_local_path, coord.data_vars, coord.init_time, coord.lead_time
        )
        vars_suffix = digest(f"{s}-{e}" for s, e in zip(starts, ends, strict=True))
        return http_download_to_disk(
            coord.get_url(source=source),
            self.dataset_id,
            byte_ranges=(starts, ends),
            local_path_suffix=f"-{vars_suffix}",
        )

    def download_file(self, coord: NoaaGfsSourceFileCoord) -> Path:
        source = self._get_download_source(coord.init_time)
        if source == "nomads":

            def attempt() -> Path:
                with _nomads_semaphore:
                    return self._download_from_source(coord, source="nomads")

            return retry(attempt, max_attempts=4, retryable_exceptions=(GenericError,))
        return self._download_from_source(coord, source=source)

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
            matching_bands: list[int] = []
            for band_i in range(reader.count):
                rasterio_band_i = band_i + 1
                if (
                    reader.descriptions[band_i] == grib_description
                    and reader.tags(rasterio_band_i)["GRIB_ELEMENT"] == grib_element
                ):
                    matching_bands.append(rasterio_band_i)

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
            deaccum_dim = "lead_time" if "lead_time" in data_array.dims else "time"
            log.info(
                f"Converting {data_var.name} from accumulations to rates along {deaccum_dim}"
            )
            try:
                deaccumulate_to_rates_inplace(
                    data_array,
                    dim=deaccum_dim,
                    reset_frequency=data_var.internal_attrs.window_reset_frequency,
                )
            except ValueError:
                # Log exception so we are notified if deaccumulation errors are larger than expected.
                log.exception(f"Error deaccumulating {data_var.name}")

        keep_mantissa_bits = data_var.internal_attrs.keep_mantissa_bits
        if isinstance(keep_mantissa_bits, int):
            round_float32_inplace(data_array.values, keep_mantissa_bits)

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaDataVar],
        reformat_job_name: str,
    ) -> tuple[Sequence["RegionJob[NoaaDataVar, NoaaGfsSourceFileCoord]"], xr.Dataset]:
        """
        Return the sequence of RegionJob instances necessary to update the dataset
        from its current state to include the latest available data.
        """
        existing_ds = xr.open_zarr(primary_store, decode_timedelta=True, chunks=None)
        # Start by reprocessing the most recent forecast already in the dataset; it may be incomplete.
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
