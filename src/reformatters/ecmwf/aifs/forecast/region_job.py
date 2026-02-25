import itertools
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import digest
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


def _get_deterministic_byte_ranges(
    index_local_path: PathLike[str],
    data_vars: Sequence[EcmwfDataVar],
) -> tuple[list[int], list[int]]:
    """Get byte ranges for deterministic (non-ensemble) AIFS data from a GRIB index file."""
    df = pd.read_json(index_local_path, lines=True)
    df = df.set_index(["param", "levtype", "levelist"]).sort_index()

    byte_range_starts: list[int] = []
    byte_range_ends: list[int] = []
    for data_var in data_vars:
        level_selector = (
            slice(None)
            if np.isnan(level_value := data_var.internal_attrs.grib_index_level_value)
            else level_value
        )
        row: pd.Series | pd.DataFrame = df.loc[
            (
                data_var.internal_attrs.grib_index_param,
                data_var.internal_attrs.grib_index_level_type,
                level_selector,
            ),
            ["_offset", "_length"],
        ]
        if isinstance(row, pd.DataFrame):
            if len(row) == 1:
                row = row.iloc[0]
            else:
                raise AssertionError(f"Expected exactly one match, but found: {row}")
        assert isinstance(row, pd.Series)
        start, length = row.values
        byte_range_starts.append(int(start))
        byte_range_ends.append(int(start + length))
    return byte_range_starts, byte_range_ends


class EcmwfAifsForecastSourceFileCoord(SourceFileCoord):
    init_time: Timestamp
    lead_time: Timedelta

    # Should contain one element (see max_vars_per_download_group below)
    data_var_group: Sequence[EcmwfDataVar]

    s3_bucket_url: ClassVar[str] = "ecmwf-forecasts"
    s3_region: ClassVar[str] = "eu-central-1"

    def _get_base_url(self) -> str:
        base_url = f"https://{self.s3_bucket_url}.s3.{self.s3_region}.amazonaws.com"

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
    # Download one variable at a time for parallelism via byte-range requests
    max_vars_per_download_group: ClassVar[int] = 1

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[EcmwfDataVar],
    ) -> Sequence[Sequence[EcmwfDataVar]]:
        """Group variables by date_available so variables added on the same date are processed together."""
        vars_by_date_available: dict[Timestamp | None, list[EcmwfDataVar]] = (
            defaultdict(list)
        )
        for data_var in data_vars:
            vars_by_date_available[data_var.internal_attrs.date_available].append(
                data_var
            )
        return list(vars_by_date_available.values())

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
        # Download the GRIB index file
        idx_url = coord.get_index_url()
        idx_local_path = http_download_to_disk(idx_url, self.dataset_id)

        # Use the index to get byte ranges for the requested variables
        byte_range_starts, byte_range_ends = _get_deterministic_byte_ranges(
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
        with rasterio.open(coord.downloaded_path) as reader:
            assert reader.count == 1, f"Expected 1 band, found {reader.count}"
            rasterio_band_index = 1

            grib_comment = reader.tags(rasterio_band_index)["GRIB_COMMENT"]
            grib_description = reader.descriptions[rasterio_band_index - 1]

            expected_comment = data_var.internal_attrs.grib_comment
            expected_description = data_var.internal_attrs.grib_description

            alt = _PRECIP_ALT_GRIB_METADATA.get(
                data_var.internal_attrs.grib_index_param
            )
            if alt is not None:
                alt_comment, alt_description = alt
                assert grib_comment in (expected_comment, alt_comment), (
                    f"{grib_comment=} not in ({expected_comment!r}, {alt_comment!r})"
                )
                assert grib_description in (expected_description, alt_description), (
                    f"{grib_description=} not in ({expected_description!r}, {alt_description!r})"
                )
            else:
                assert grib_comment == expected_comment, (
                    f"{grib_comment=} != {expected_comment=}"
                )
                assert grib_description == expected_description, (
                    f"{grib_description=} != {expected_description=}"
                )

            result: ArrayFloat32 = reader.read(
                rasterio_band_index, out_dtype=np.float32
            )
            expected_shape = (721, 1440)
            assert result.shape == expected_shape, (
                f"Expected {expected_shape} shape, found {result.shape}"
            )
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
