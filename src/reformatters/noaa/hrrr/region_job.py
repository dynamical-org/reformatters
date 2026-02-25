import threading
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
from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import digest, group_by, item
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
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
    NoaaHrrrDomain,
    NoaaHrrrFileType,
)
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)

type DownloadSource = Literal["s3", "nomads"]

# Limit concurrent NOMADS requests to avoid overloading their servers
_nomads_semaphore = threading.Semaphore(1)


class NoaaHrrrSourceFileCoord(SourceFileCoord):
    """Source file coordinate for HRRR forecast data."""

    init_time: Timestamp
    lead_time: Timedelta
    domain: NoaaHrrrDomain
    file_type: NoaaHrrrFileType
    data_vars: Sequence[NoaaHrrrDataVar]

    def get_url(self, source: DownloadSource = "s3") -> str:
        """Return the URL for this HRRR file."""
        lead_time_hours = whole_hours(self.lead_time)
        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")
        path = f"hrrr.{init_date_str}/{self.domain}/hrrr.t{init_hour_str}z.wrf{self.file_type}f{int(lead_time_hours):02d}.grib2"
        match source:
            case "nomads":
                base = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
            case "s3":
                base = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
            case _ as unreachable:
                assert_never(unreachable)

        return f"{base}/{path}"

    def get_idx_url(self, source: DownloadSource = "s3") -> str:
        """Return the URL for the GRIB index file."""
        return f"{self.get_url(source=source)}.idx"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        raise NotImplementedError  # depends on if the dataset is a forecast or analysis


class NoaaHrrrRegionJob(RegionJob[NoaaHrrrDataVar, NoaaHrrrSourceFileCoord]):
    """Base RegionJob for HRRR based datasets.  Subclassed by specific HRRR datasets."""

    max_vars_per_download_group = 5

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[NoaaHrrrDataVar],
    ) -> Sequence[Sequence[NoaaHrrrDataVar]]:
        return group_by(
            data_vars, lambda v: (v.internal_attrs.hrrr_file_type, has_hour_0_values(v))
        )

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaHrrrDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[NoaaHrrrDataVar, NoaaHrrrSourceFileCoord]],
        xr.Dataset,
    ]:
        """Generate operational update jobs for HRRR forecast data."""
        # For operational updates, we want to process recent forecast data
        # HRRR provides forecasts every hour, but 48-hour forecasts are only available
        # every 6 hours (00, 06, 12, 18 UTC)

        existing_ds = xr.open_zarr(primary_store, chunks=None, decode_timedelta=True)
        append_dim_start = cls._update_append_dim_start(existing_ds[append_dim])

        append_dim_end = cls._update_append_dim_end()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            kind="operational-update",
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
            filter_end=append_dim_end,
        )
        return jobs, template_ds

    def _get_download_source(self, init_time: pd.Timestamp) -> DownloadSource:
        if init_time >= (pd.Timestamp.now() - pd.Timedelta(hours=18)):
            return "nomads"
        else:
            return "s3"

    def _download_from_source(
        self, coord: NoaaHrrrSourceFileCoord, source: DownloadSource
    ) -> Path:
        idx_local_path = http_download_to_disk(
            coord.get_idx_url(source=source), self.dataset_id
        )
        byte_range_starts, byte_range_ends = grib_message_byte_ranges_from_index(
            idx_local_path, coord.data_vars, coord.init_time, coord.lead_time
        )
        vars_suffix = digest(
            f"{s}-{e}" for s, e in zip(byte_range_starts, byte_range_ends, strict=True)
        )
        return http_download_to_disk(
            coord.get_url(source=source),
            self.dataset_id,
            byte_ranges=(byte_range_starts, byte_range_ends),
            local_path_suffix=f"-{vars_suffix}",
        )

    def download_file(self, coord: NoaaHrrrSourceFileCoord) -> Path:
        """Download a subset of variables from a HRRR file and return the local path."""
        source = self._get_download_source(coord.init_time)
        if source == "nomads":

            def attempt() -> Path:
                with _nomads_semaphore:
                    return self._download_from_source(coord, source="nomads")

            return retry(attempt, max_attempts=4, retryable_exceptions=(GenericError,))
        return self._download_from_source(coord, source=source)

    def read_data(
        self,
        coord: NoaaHrrrSourceFileCoord,
        data_var: NoaaHrrrDataVar,
    ) -> ArrayFloat32:
        """Read data from a HRRR file for a specific variable."""
        assert coord.downloaded_path is not None  # for type check, system guarantees it
        grib_description = data_var.internal_attrs.grib_description

        grib_element = data_var.internal_attrs.grib_element
        # grib element has the accumulation window as a suffix in the grib file attributes, but not in the .idx file
        if (reset_freq := data_var.internal_attrs.window_reset_frequency) is not None:
            grib_element = f"{grib_element}{whole_hours(reset_freq):02d}"

        with rasterio.open(coord.downloaded_path) as reader:
            matching_bands: list[int] = []
            for band_i in range(reader.count):
                rasterio_band_i = band_i + 1
                if (
                    reader.descriptions[band_i] == grib_description
                    and reader.tags(rasterio_band_i)["GRIB_ELEMENT"] == grib_element
                ):
                    matching_bands.append(rasterio_band_i)

            assert len(matching_bands) == 1, (
                f"Expected exactly 1 matching band, found {len(matching_bands)}: {matching_bands}. "
                f"{grib_element=}, {grib_description=}, {coord.downloaded_path=}"
            )
            rasterio_band_index = item(matching_bands)

            result: ArrayFloat32 = reader.read(
                rasterio_band_index, out_dtype=np.float32
            )
            return result

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: NoaaHrrrDataVar
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
    def _update_append_dim_end(cls) -> pd.Timestamp:
        """Get the end time for operational updates."""
        return pd.Timestamp.now()

    @classmethod
    def _update_append_dim_start(cls, append_dim_coords: xr.DataArray) -> pd.Timestamp:
        """Get the start time for operational updates based on existing data."""
        ds_max_time = append_dim_coords.max().item()
        return pd.Timestamp(ds_max_time)
