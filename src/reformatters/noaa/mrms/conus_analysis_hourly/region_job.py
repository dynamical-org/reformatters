import gzip
import uuid
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


class NoaaMrmsSourceFileCoord(SourceFileCoord):
    time: Timestamp
    product: str
    level: str
    fallback_products: tuple[str, ...]
    data_var_name: str
    # Only set for precipitation_surface timestamps in _PRECIPITATION_SURFACE_RADAR_ONLY_OVERRIDES.
    # When set and product is RadarOnly_QPE_01H, get_url uses this timestamp instead of self.time.
    radar_only_time_override: Timestamp | None = None

    def get_url(self, source: DownloadSource = "s3") -> str:
        if (
            self.product == "RadarOnly_QPE_01H"
            and self.radar_only_time_override is not None
        ):
            time = self.radar_only_time_override
        else:
            time = self.time
        date_str = time.strftime("%Y%m%d")
        time_str = time.strftime("%Y%m%d-%H%M%S")

        match source:
            case "s3":
                filename = f"MRMS_{self.product}_{self.level}_{time_str}.grib2.gz"
                return f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/{self.product}_{self.level}/{date_str}/{filename}"
            case "iowa":
                # Iowa Mesonet doesn't use the MRMS_ prefix in filenames
                filename = f"{self.product}_{self.level}_{time_str}.grib2.gz"
                year = time.strftime("%Y")
                month = time.strftime("%m")
                day = time.strftime("%d")
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

            radar_only_time_override = None
            if data_var.name == "precipitation_surface":
                radar_only_time_override = (
                    _PRECIPITATION_SURFACE_RADAR_ONLY_OVERRIDES.get(time)
                )

            coords.append(
                NoaaMrmsSourceFileCoord(
                    time=time,
                    product=product,
                    level=internal.mrms_level,
                    fallback_products=fallback_products,
                    data_var_name=data_var.name,
                    radar_only_time_override=radar_only_time_override,
                )
            )
        return coords

    def _download_from_source(
        self, coord: NoaaMrmsSourceFileCoord, source: DownloadSource
    ) -> Path:
        local_path_suffix = f"_{coord.data_var_name}"
        gz_path = http_download_to_disk(
            coord.get_url(source=source),
            self.dataset_id,
            local_path_suffix=local_path_suffix,
        )
        return _decompress_gzip(gz_path, local_path_suffix)

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
        data_var: NoaaMrmsDataVar,
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None
        with rasterio.open(coord.downloaded_path) as reader:
            if reader.count == 2 and coord.time < MRMS_V12_START:
                # A handful of pre v12 files have two messages within the same product, we want the grib discipline 209 message
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

        nodata_sentinel = data_var.internal_attrs.nodata_sentinel
        if nodata_sentinel is not None:
            result[result == nodata_sentinel] = np.nan

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
                    expected_invalid_fraction=data_var.internal_attrs.expected_invalid_fraction,
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
        # Start 3 hours before the dataset's latest timestamp so precipitation_surface
        # (which falls back to radar-only when pass_2 is unavailable) gets reprocessed
        # and overwritten with pass_2 data once it becomes available (~60-min latency).
        append_dim_start = pd.Timestamp(ds_max_time) - pd.Timedelta(hours=3)

        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
            filter_end=append_dim_end,
        )
        return jobs, template_ds


# For precipitation_surface only: maps missing hourly timestamps to the RadarOnly file
# timestamp to use as fallback. Pre-v12 entries use GaugeCorr as primary; when that
# is absent, RadarOnly exists at the same timestamp on Iowa Mesonet. Post-v12 entries
# use the nearest RadarOnly_QPE_01H 2-minute file within ±15min of the missing hour.
_PRECIPITATION_SURFACE_RADAR_ONLY_OVERRIDES: dict[pd.Timestamp, pd.Timestamp] = {
    # Pre-v12 (Iowa Mesonet): GaugeCorr absent for this period, RadarOnly at exact timestamp
    pd.Timestamp("2014-11-03T18:00"): pd.Timestamp("2014-11-03T18:00"),
    pd.Timestamp("2014-11-06T11:00"): pd.Timestamp("2014-11-06T11:00"),
    pd.Timestamp("2014-11-11T06:00"): pd.Timestamp("2014-11-11T06:00"),
    pd.Timestamp("2014-11-11T07:00"): pd.Timestamp("2014-11-11T07:00"),
    pd.Timestamp("2014-11-11T08:00"): pd.Timestamp("2014-11-11T08:00"),
    pd.Timestamp("2014-11-11T19:00"): pd.Timestamp("2014-11-11T19:00"),
    pd.Timestamp("2014-11-15T12:00"): pd.Timestamp("2014-11-15T12:00"),
    pd.Timestamp("2014-11-18T19:00"): pd.Timestamp("2014-11-18T19:00"),
    pd.Timestamp("2014-11-18T20:00"): pd.Timestamp("2014-11-18T20:00"),
    pd.Timestamp("2014-11-18T21:00"): pd.Timestamp("2014-11-18T21:00"),
    pd.Timestamp("2014-11-20T06:00"): pd.Timestamp("2014-11-20T06:00"),
    pd.Timestamp("2014-11-20T14:00"): pd.Timestamp("2014-11-20T14:00"),
    pd.Timestamp("2014-11-20T19:00"): pd.Timestamp("2014-11-20T19:00"),
    pd.Timestamp("2014-11-21T15:00"): pd.Timestamp("2014-11-21T15:00"),
    pd.Timestamp("2014-11-30T04:00"): pd.Timestamp("2014-11-30T04:00"),
    pd.Timestamp("2014-11-30T08:00"): pd.Timestamp("2014-11-30T08:00"),
    pd.Timestamp("2014-11-30T10:00"): pd.Timestamp("2014-11-30T10:00"),
    pd.Timestamp("2014-11-30T11:00"): pd.Timestamp("2014-11-30T11:00"),
    pd.Timestamp("2014-11-30T12:00"): pd.Timestamp("2014-11-30T12:00"),
    pd.Timestamp("2014-11-30T13:00"): pd.Timestamp("2014-11-30T13:00"),
    pd.Timestamp("2014-12-02T08:00"): pd.Timestamp("2014-12-02T08:00"),
    pd.Timestamp("2014-12-07T11:00"): pd.Timestamp("2014-12-07T11:00"),
    pd.Timestamp("2014-12-07T21:00"): pd.Timestamp("2014-12-07T21:00"),
    pd.Timestamp("2014-12-07T23:00"): pd.Timestamp("2014-12-07T23:00"),
    pd.Timestamp("2014-12-08T04:00"): pd.Timestamp("2014-12-08T04:00"),
    pd.Timestamp("2014-12-15T02:00"): pd.Timestamp("2014-12-15T02:00"),
    # Post-v12 (S3): Pass2 and Pass1 absent; nearest RadarOnly_QPE_01H file within ±15min
    pd.Timestamp("2021-03-03T21:00"): pd.Timestamp("2021-03-03T20:46"),
    pd.Timestamp("2021-03-04T00:00"): pd.Timestamp("2021-03-04T00:10"),
    pd.Timestamp("2021-03-14T02:00"): pd.Timestamp("2021-03-14T01:58"),
    pd.Timestamp("2021-06-22T07:00"): pd.Timestamp("2021-06-22T06:58"),
    pd.Timestamp("2021-07-02T08:00"): pd.Timestamp("2021-07-02T07:58"),
    pd.Timestamp("2021-07-15T18:00"): pd.Timestamp("2021-07-15T17:48"),
    pd.Timestamp("2021-07-15T19:00"): pd.Timestamp("2021-07-15T18:58"),
    pd.Timestamp("2021-07-19T05:00"): pd.Timestamp("2021-07-19T04:58"),
    pd.Timestamp("2021-09-16T22:00"): pd.Timestamp("2021-09-16T22:08"),
    pd.Timestamp("2021-09-29T04:00"): pd.Timestamp("2021-09-29T03:54"),
    pd.Timestamp("2022-02-28T15:00"): pd.Timestamp("2022-02-28T15:04"),
    pd.Timestamp("2022-06-09T12:00"): pd.Timestamp("2022-06-09T11:58"),
    pd.Timestamp("2023-07-24T15:00"): pd.Timestamp("2023-07-24T14:56"),
    pd.Timestamp("2024-03-19T18:00"): pd.Timestamp("2024-03-19T17:50"),
    pd.Timestamp("2024-03-19T19:00"): pd.Timestamp("2024-03-19T19:02"),
}


def _decompress_gzip(gz_path: Path, local_path_suffix: str = "") -> Path:
    # gz_path.with_suffix("") strips the last extension (.gz or .gz_<suffix>),
    # then we append local_path_suffix to make the decompressed path unique per variable group.
    base = gz_path.with_suffix("")
    decompressed_path = base.with_name(base.name + local_path_suffix)
    temp_path = decompressed_path.with_name(
        f"{decompressed_path.name}.{uuid.uuid4().hex[:8]}"
    )
    with gzip.open(gz_path, "rb") as f_in, open(temp_path, "wb") as f_out:
        f_out.write(f_in.read())
    temp_path.rename(decompressed_path)
    gz_path.unlink(missing_ok=True)
    return decompressed_path
