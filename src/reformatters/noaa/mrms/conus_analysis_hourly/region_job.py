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
from reformatters.common.download import (
    http_download_to_disk,
    s3_list_first_key_with_prefix,
)
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import replace
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

_NOAA_MRMS_S3_BASE_URL = "https://noaa-mrms-pds.s3.amazonaws.com"


class NoaaMrmsSourceFileCoord(SourceFileCoord):
    time: Timestamp
    product: str
    level: str
    fallback_products: tuple[str, ...]

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
        assert len(data_var_group) == 1
        times = pd.to_datetime(processing_region_ds["time"].values)
        data_var = data_var_group[0]
        internal = data_var.internal_attrs

        coords: list[NoaaMrmsSourceFileCoord] = []
        for time in times:
            # Skip times before the product is available
            if internal.available_from is not None and time < internal.available_from:
                continue

            # Use pre-v12 product name for times before v12 launch
            if time < MRMS_V12_START:
                product = internal.mrms_product_pre_v12
                fallback_products = internal.mrms_fallback_products_pre_v12
            else:
                product = internal.mrms_product
                fallback_products = internal.mrms_fallback_products

            coords.append(
                NoaaMrmsSourceFileCoord(
                    time=time,
                    product=product,
                    level=internal.mrms_level,
                    fallback_products=fallback_products,
                )
            )
        return coords

    def _download_from_source(
        self, coord: NoaaMrmsSourceFileCoord, source: DownloadSource
    ) -> Path:
        try:
            gz_path = http_download_to_disk(
                coord.get_url(source=source), self.dataset_id
            )
        except FileNotFoundError:
            # RadarOnly QPE is published at 2-min intervals. When the system starts late
            # the exact HH:00:00 file is missing but a file at HH:MM:00 (MM > 0) exists.
            # List the hour's directory and try the first file found.
            if source != "s3" or not coord.product.startswith("RadarOnly_QPE_"):
                raise
            date_str = coord.time.strftime("%Y%m%d")
            hour_str = coord.time.strftime("%Y%m%d-%H")
            key_prefix = f"CONUS/{coord.product}_{coord.level}/{date_str}/MRMS_{coord.product}_{coord.level}_{hour_str}"
            first_key = s3_list_first_key_with_prefix(
                _NOAA_MRMS_S3_BASE_URL, key_prefix
            )
            if first_key is None:
                raise
            gz_path = http_download_to_disk(
                f"{_NOAA_MRMS_S3_BASE_URL}/{first_key}", self.dataset_id
            )
        return _decompress_gzip(gz_path)

    def download_file(self, coord: NoaaMrmsSourceFileCoord) -> Path:
        is_pre_v12 = coord.time < MRMS_V12_START
        is_recent = coord.time > (pd.Timestamp.now() - pd.Timedelta(hours=12))

        sources: tuple[DownloadSource, ...]
        if is_pre_v12:
            sources = ("iowa",)
        elif is_recent:
            sources = ("s3", "ncep")
        else:
            sources = ("s3",)

        products = (coord.product, *coord.fallback_products)

        last_exception: FileNotFoundError | None = None
        for product in products:
            for source in sources:
                try:
                    return self._download_from_source(
                        replace(coord, product=product), source=source
                    )
                except FileNotFoundError as exc:
                    last_exception = exc

        assert last_exception is not None
        raise last_exception

    def read_data(
        self,
        coord: NoaaMrmsSourceFileCoord,
        data_var: NoaaMrmsDataVar,  # noqa: ARG002
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None
        with rasterio.open(coord.downloaded_path) as reader:
            if reader.count == 2 and coord.time < MRMS_V12_START:
                rasterio_band = next(
                    (
                        band
                        for band in (1, 2)
                        if reader.tags(band)
                        .get("GRIB_DISCIPLINE", "")
                        .startswith("209")
                    ),
                    None,
                )
                assert rasterio_band is not None, (
                    f"Expected one band with GRIB_DISCIPLINE 209 in {coord.downloaded_path}"
                )
            else:
                assert reader.count == 1, (
                    f"Expected exactly 1 band, found {reader.count} in {coord.downloaded_path}"
                )
                rasterio_band = 1
            result: ArrayFloat32 = reader.read(rasterio_band, out_dtype=np.float32)
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
