import gzip
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import http_download_to_disk
from reformatters.common.logging import get_logger
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

from .template_config import MRMS_V12_START, NoaaMrmsDataVar

log = get_logger(__name__)

type DownloadSource = Literal["iowa", "s3", "ncep"]


class NoaaMrmsSourceFileCoord(SourceFileCoord):
    time: Timestamp
    product: str
    level: str = "00.00"

    def get_url(self, source: DownloadSource = "s3") -> str:
        date_str = self.time.strftime("%Y%m%d")
        time_str = self.time.strftime("%Y%m%d-%H%M%S")

        match source:
            case "s3":
                filename = f"MRMS_{self.product}_{self.level}_{time_str}.grib2.gz"
                return f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/{self.product}_{self.level}/{date_str}/{filename}"
            case "iowa":
                # Iowa Mesonet doesn't use the MRMS_ prefix in filenames
                filename = f"{self.product}_{self.level}_{time_str}.grib2.gz"
                year = self.time.strftime("%Y")
                month = self.time.strftime("%m")
                day = self.time.strftime("%d")
                return f"https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/mrms/ncep/{self.product}/{filename}"
            case "ncep":
                filename = f"MRMS_{self.product}_{self.level}_{time_str}.grib2.gz"
                return f"https://mrms.ncep.noaa.gov/2D/{self.product}/{filename}"
            case _ as unreachable:
                assert_never(unreachable)

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"time": self.time}


class NoaaMrmsRegionJob(RegionJob[NoaaMrmsDataVar, NoaaMrmsSourceFileCoord]):
    def get_processing_region(self) -> slice:
        """Buffer start by one step to allow deaccumulation without gaps in resulting output."""
        return slice(max(0, self.region.start - 1), self.region.stop)

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[NoaaMrmsDataVar],
    ) -> Sequence[Sequence[NoaaMrmsDataVar]]:
        # Each MRMS variable comes from a separate file, so each is its own group
        return [[v] for v in data_vars]

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaMrmsDataVar],
    ) -> Sequence[NoaaMrmsSourceFileCoord]:
        times = pd.to_datetime(processing_region_ds["time"].values)
        data_var = data_var_group[0]
        internal = data_var.internal_attrs

        coords: list[NoaaMrmsSourceFileCoord] = []
        for time in times:
            # Skip times before the product is available
            if internal.available_from is not None and time < internal.available_from:
                continue

            # Use pre-v12 product name for times before v12 launch
            if internal.mrms_product_pre_v12 is not None and time < MRMS_V12_START:
                product = internal.mrms_product_pre_v12
            else:
                product = internal.mrms_product

            coords.append(
                NoaaMrmsSourceFileCoord(
                    time=time,
                    product=product,
                    level=internal.mrms_level,
                )
            )
        return coords

    def _download_from_source(
        self, coord: NoaaMrmsSourceFileCoord, source: DownloadSource
    ) -> Path:
        gz_path = http_download_to_disk(coord.get_url(source=source), self.dataset_id)
        return _decompress_gzip(gz_path)

    def download_file(self, coord: NoaaMrmsSourceFileCoord) -> Path:
        if coord.time < MRMS_V12_START:
            return self._download_from_source(coord, source="iowa")
        try:
            return self._download_from_source(coord, source="s3")
        except FileNotFoundError:
            if coord.time > (pd.Timestamp.now() - pd.Timedelta(hours=12)):
                return self._download_from_source(coord, source="ncep")
            raise

    def read_data(
        self,
        coord: NoaaMrmsSourceFileCoord,
        data_var: NoaaMrmsDataVar,  # noqa: ARG002
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None
        with rasterio.open(coord.downloaded_path) as reader:
            assert reader.count == 1, (
                f"Expected exactly 1 band, found {reader.count} in {coord.downloaded_path}"
            )
            result: ArrayFloat32 = reader.read(1, out_dtype=np.float32)
            return result

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: NoaaMrmsDataVar
    ) -> None:
        if data_var.internal_attrs.deaccumulate_to_rate:
            assert data_var.internal_attrs.window_reset_frequency is not None
            log.info(
                f"Converting {data_var.name} from accumulations to rates along time"
            )
            try:
                deaccumulate_to_rates_inplace(
                    data_array,
                    dim="time",
                    reset_frequency=data_var.internal_attrs.window_reset_frequency,
                )
            except ValueError:
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
        all_data_vars: Sequence[NoaaMrmsDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[NoaaMrmsDataVar, NoaaMrmsSourceFileCoord]"],
        xr.Dataset,
    ]:
        existing_ds = xr.open_zarr(primary_store, chunks=None, decode_timedelta=True)
        ds_max_time = existing_ds[append_dim].max().item()
        append_dim_start = pd.Timestamp(ds_max_time)

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
            filter_end=append_dim_end,
        )
        return jobs, template_ds


def _decompress_gzip(gz_path: Path) -> Path:
    decompressed_path = gz_path.with_suffix("")
    with gzip.open(gz_path, "rb") as f_in, open(decompressed_path, "wb") as f_out:
        f_out.write(f_in.read())
    return decompressed_path
