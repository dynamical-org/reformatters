import itertools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import digest, group_by
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
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ecmwf_grib_index import get_message_byte_ranges_from_index
from reformatters.ecmwf.ecmwf_utils import all_variables_available

log = get_logger(__name__)

# Path changed from aifs/ to aifs-single/ on this date
AIFS_SINGLE_PATH_CHANGE_DATE = pd.Timestamp("2025-02-26T00:00")

# GRIB master table version changes caused metadata differences for precipitation.
# Early data (table v27): generic product template codes.
# Recent data (table v34+): specific parameter names with different step encoding.
# Maps grib_index_param -> (alt_grib_comment, alt_grib_description)
_PRECIP_ALT_GRIB_METADATA: dict[str, tuple[str, str]] = {
    "tp": (
        "Total precipitation rate [kg/(m^2*s)]",
        '0[-] SFC="Ground or water surface"',
    ),
    "cp": (
        "Convective precipitation rate [kg/(m^2*s)]",
        '0[-] SFC="Ground or water surface"',
    ),
}


class EcmwfAifsForecastSourceFileCoord(SourceFileCoord):
    init_time: Timestamp
    lead_time: Timedelta
    data_var_group: Sequence[EcmwfDataVar]

    s3_bucket_url: ClassVar[str] = "ecmwf-forecasts"

    def _get_base_url(self) -> str:
        base_url = f"https://{self.s3_bucket_url}.s3.eu-central-1.amazonaws.com"

        init_time_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")
        lead_time_hour_str = whole_hours(self.lead_time)

        if self.init_time >= AIFS_SINGLE_PATH_CHANGE_DATE:
            model_dir = "aifs-single"
        else:
            model_dir = "aifs"

        directory_path = f"{init_time_str}/{init_hour_str}z/{model_dir}/0p25/oper"
        filename = f"{init_time_str}{init_hour_str}0000-{lead_time_hour_str}h-oper-fc"
        return f"{base_url}/{directory_path}/{filename}"

    def get_url(self) -> str:
        return self._get_base_url() + ".grib2"

    def get_index_url(self) -> str:
        return self._get_base_url() + ".index"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }


class EcmwfAifsForecastRegionJob(
    RegionJob[EcmwfDataVar, EcmwfAifsForecastSourceFileCoord]
):
    max_vars_per_download_group: ClassVar[int] = 10

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[EcmwfDataVar],
    ) -> Sequence[Sequence[EcmwfDataVar]]:
        return group_by(data_vars, lambda v: v.internal_attrs.date_available)

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcmwfDataVar],
    ) -> Sequence[EcmwfAifsForecastSourceFileCoord]:
        coords = []
        for init_time, lead_time in itertools.product(
            processing_region_ds["init_time"].values,
            processing_region_ds["lead_time"].values,
        ):
            if not all_variables_available(data_var_group, init_time):
                continue

            coords.append(
                EcmwfAifsForecastSourceFileCoord(
                    init_time=init_time,
                    lead_time=lead_time,
                    data_var_group=data_var_group,
                )
            )
        return coords

    def download_file(self, coord: EcmwfAifsForecastSourceFileCoord) -> Path:
        idx_url = coord.get_index_url()
        idx_local_path = http_download_to_disk(idx_url, self.dataset_id)

        byte_range_starts, byte_range_ends = get_message_byte_ranges_from_index(
            idx_local_path,
            coord.data_var_group,
        )
        suffix = digest(
            f"{s}-{e}" for s, e in zip(byte_range_starts, byte_range_ends, strict=True)
        )
        return http_download_to_disk(
            coord.get_url(),
            self.dataset_id,
            byte_ranges=(byte_range_starts, byte_range_ends),
            local_path_suffix=f"-{suffix}",
        )

    def read_data(
        self,
        coord: EcmwfAifsForecastSourceFileCoord,
        data_var: EcmwfDataVar,
    ) -> ArrayFloat32:
        expected_comment = data_var.internal_attrs.grib_comment
        expected_description = data_var.internal_attrs.grib_description

        alt = _PRECIP_ALT_GRIB_METADATA.get(data_var.internal_attrs.grib_index_param)
        allowed_comments = {expected_comment}
        allowed_descriptions = {expected_description}
        if alt is not None:
            allowed_comments.add(alt[0])
            allowed_descriptions.add(alt[1])

        with rasterio.open(coord.downloaded_path) as reader:
            matching_bands: list[int] = []
            for band_i in range(reader.count):
                rasterio_band_i = band_i + 1
                if (
                    reader.tags(rasterio_band_i)["GRIB_COMMENT"] in allowed_comments
                    and reader.descriptions[band_i] in allowed_descriptions
                ):
                    matching_bands.append(rasterio_band_i)

            assert len(matching_bands) == 1, (
                f"Expected exactly 1 matching band, found {len(matching_bands)}. "
                f"{expected_comment=}, {expected_description=}, {coord.downloaded_path=}"
            )
            result: ArrayFloat32 = reader.read(matching_bands[0], out_dtype=np.float32)
            return result

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[EcmwfDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[EcmwfDataVar, EcmwfAifsForecastSourceFileCoord]"],
        xr.Dataset,
    ]:
        existing_ds = xr.open_zarr(primary_store, chunks=None)
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
