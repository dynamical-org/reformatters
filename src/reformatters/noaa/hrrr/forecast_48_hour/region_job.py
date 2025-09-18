from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio  # type: ignore[import-untyped]
import xarray as xr
import zarr

from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import digest, group_by, item
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
    SourceFileCoord,
)
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
    HRRRDataVar,
    HRRRDomain,
    HRRRFileType,
)
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)


class HRRRSourceFileCoord(SourceFileCoord):
    """Source file coordinate for HRRR forecast data."""

    init_time: Timestamp
    lead_time: Timedelta
    domain: HRRRDomain
    file_type: HRRRFileType
    data_vars: Sequence[HRRRDataVar]

    def get_url(self) -> str:
        """Return the URL for this HRRR file."""
        lead_time_hours = whole_hours(self.lead_time)
        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")

        return f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{init_date_str}/{self.domain}/hrrr.t{init_hour_str}z.wrf{self.file_type}f{int(lead_time_hours):02d}.grib2"

    def get_idx_url(self) -> str:
        """Return the URL for the GRIB index file."""
        return f"{self.get_url()}.idx"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        """Return the output location for this file's data in the dataset."""
        # Map to the standard dimension names used in the template
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }


class NoaaHrrrForecast48HourRegionJob(RegionJob[HRRRDataVar, HRRRSourceFileCoord]):
    """Region job for HRRR 48-hour forecast data processing."""

    max_vars_per_download_group = 8  # currently a best guess

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[HRRRDataVar],
    ) -> Sequence[Sequence[HRRRDataVar]]:
        return group_by(
            data_vars, lambda v: (v.internal_attrs.hrrr_file_type, has_hour_0_values(v))
        )

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: zarr.abc.store.Store,
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

        existing_ds = xr.open_zarr(primary_store, chunks=None, decode_timedelta=True)
        append_dim_start = cls._update_append_dim_start(existing_ds)

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

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[HRRRDataVar],
    ) -> Sequence[HRRRSourceFileCoord]:
        """Generate source file coordinates for the processing region."""
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        group_has_hour_0 = item({has_hour_0_values(var) for var in data_var_group})
        if not group_has_hour_0:
            lead_times = lead_times[lead_times > pd.Timedelta(hours=0)]

        file_type = item({var.internal_attrs.hrrr_file_type for var in data_var_group})

        return [
            HRRRSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                domain="conus",
                file_type=file_type,
                data_vars=data_var_group,
            )
            for init_time in init_times
            for lead_time in lead_times
        ]

    def download_file(self, coord: HRRRSourceFileCoord) -> Path:
        """Download a subset of variables from a HRRR file and return the local path."""
        idx_url = coord.get_idx_url()
        idx_local_path = http_download_to_disk(idx_url, self.dataset_id)

        byte_range_starts, byte_range_ends = grib_message_byte_ranges_from_index(
            idx_local_path, coord.data_vars, coord.init_time, coord.lead_time
        )
        vars_suffix = digest(
            f"{s}-{e}" for s, e in zip(byte_range_starts, byte_range_ends, strict=True)
        )

        return http_download_to_disk(
            coord.get_url(),
            self.dataset_id,
            byte_ranges=(byte_range_starts, byte_range_ends),
            local_path_suffix=f"-{vars_suffix}",
        )

    def read_data(
        self,
        coord: HRRRSourceFileCoord,
        data_var: HRRRDataVar,
    ) -> ArrayFloat32:
        """Read data from a HRRR file for a specific variable."""
        assert coord.downloaded_path is not None  # for type check, system guarantees it
        grib_description = data_var.internal_attrs.grib_description

        grib_element = data_var.internal_attrs.grib_element
        # grib element has the accumulation window as a suffix in the grib file attributes, but not in the .idx file
        if (reset_freq := data_var.internal_attrs.window_reset_frequency) is not None:
            grib_element = f"{grib_element}{whole_hours(reset_freq):02d}"

        with rasterio.open(coord.downloaded_path) as reader:
            matching_bands = [
                rasterio_band_i
                for band_i in range(reader.count)
                if reader.descriptions[band_i] == grib_description
                and reader.tags(rasterio_band_i := band_i + 1)["GRIB_ELEMENT"]
                == grib_element
            ]

            assert len(matching_bands) == 1, (
                f"Expected exactly 1 matching band, found {len(matching_bands)}: {matching_bands}. "
                f"{grib_element=}, {grib_description=}, {coord.downloaded_path=}"
            )
            rasterio_band_index = item(matching_bands)

            result: ArrayFloat32 = reader.read(
                rasterio_band_index, out_dtype=np.float32
            )
            return result

    @classmethod
    def _update_append_dim_end(cls) -> pd.Timestamp:
        """Get the end time for operational updates."""
        return pd.Timestamp.now(tz="UTC")

    @classmethod
    def _update_append_dim_start(cls, existing_ds: xr.Dataset) -> pd.Timestamp:
        """Get the start time for operational updates based on existing data."""
        ds_max_time = existing_ds["init_time"].max().item()
        return pd.Timestamp(ds_max_time)
