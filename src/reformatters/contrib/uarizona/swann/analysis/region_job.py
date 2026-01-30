from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field
from rasterio import rasterio
from zarr.abc.store import Store

from reformatters.common.download import http_download_to_disk
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.types import (
    AppendDim,
    Array2D,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timestamp,
)

from .template_config import UarizonaSwannDataVar


class UarizonaSwannAnalysisSourceFileCoord(SourceFileCoord):
    # SWANN data is revised over time and a single UarizonaSwannAnalysisSourceFileCoord represents the
    # most stable available version of the data for a single day. Mechanically, we optimistically
    # start with the most stable version first and rely on the `.get_url()` caller to call
    # `advance_data_status()` if a file is not found at the previously returned url.
    possible_data_statuses: ClassVar[tuple[str, ...]] = (
        "stable",
        "provisional",
        "early",
    )

    remaining_data_statuses: list[str] = Field(
        default_factory=lambda: list(
            UarizonaSwannAnalysisSourceFileCoord.possible_data_statuses
        )
    )

    time: Timestamp

    def get_url(self) -> str:
        water_year = self.get_water_year()
        year_month_day = self.time.strftime("%Y%m%d")

        try:
            data_status = self.get_data_status()
        except IndexError:
            # Allow get_url calls to return a URL for the last attempted status, even if it failed
            data_status = UarizonaSwannAnalysisSourceFileCoord.possible_data_statuses[
                -1
            ]

        return f"https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY{water_year}/UA_SWE_Depth_4km_v1_{year_month_day}_{data_status}.nc"

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"time": self.time}

    def get_water_year(self) -> int:
        """A water year starts October 1st."""
        october = 10
        return self.time.year if self.time.month < october else self.time.year + 1

    def get_data_status(self) -> str:
        return self.remaining_data_statuses[0]

    def advance_data_status(self) -> bool:
        self.remaining_data_statuses.pop(0)
        return len(self.remaining_data_statuses) > 0


class UarizonaSwannAnalysisRegionJob(
    RegionJob[UarizonaSwannDataVar, UarizonaSwannAnalysisSourceFileCoord]
):
    # Be gentle to UA HTTP servers
    download_parallelism: int = 2

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[UarizonaSwannDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[UarizonaSwannDataVar, UarizonaSwannAnalysisSourceFileCoord]],
        xr.Dataset,
    ]:
        existing_ds = xr.open_zarr(primary_store)
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
        )
        return jobs, template_ds

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        _data_var_group: Sequence[UarizonaSwannDataVar],
    ) -> Sequence[UarizonaSwannAnalysisSourceFileCoord]:
        times = processing_region_ds["time"].values
        return [UarizonaSwannAnalysisSourceFileCoord(time=t) for t in times]

    def download_file(self, coord: UarizonaSwannAnalysisSourceFileCoord) -> Path:
        while True:
            try:
                return http_download_to_disk(coord.get_url(), self.dataset_id)
            except FileNotFoundError:
                if not coord.advance_data_status():
                    raise

    def read_data(
        self,
        coord: UarizonaSwannAnalysisSourceFileCoord,
        data_var: UarizonaSwannDataVar,
    ) -> ArrayFloat32:
        var_name = data_var.internal_attrs.netcdf_var_name
        netcdf_path = f"netcdf:{coord.downloaded_path}:{var_name}"
        band = 1  # because rasterio netcdf requires selecting the band in the file path we always want band 1
        no_data_value = -999
        with rasterio.open(netcdf_path) as reader:
            result: Array2D[np.float32] = reader.read(band, out_dtype=np.float32)
            # We are using a different fill value here than the data var encoding fill value
            # This is because encoding fill value was previously NaN, and so when we matched
            # matched our no data value, we set values to NaN. We have now changed the
            # encoding fill value to 0. This is to accomdate the fact that due to an Xarray bug,
            # the encoding fill value was not round tripped (it was persisted as 0 despite the
            # definition in our encoding). We have updated the encoding fill value to 0 to match
            # what was written at the time of our backfill. That change ensures that empty chunks
            # continue to be interpreted as 0. But consequently, we need to ensure that when we
            # are setting the no data value when reading the netcdf data, we continue to use NaN.
            if data_var.internal_attrs.read_data_fill_value is not None:
                result[result == no_data_value] = (
                    data_var.internal_attrs.read_data_fill_value
                )
            assert result.shape == (621, 1405)
            return result

    @classmethod
    def _update_append_dim_end(cls) -> pd.Timestamp:
        return pd.Timestamp.now()

    @classmethod
    def _update_append_dim_start(cls, existing_ds: xr.Dataset) -> pd.Timestamp:
        ds_max_time = existing_ds["time"].max().item()
        # UArizona updates the data, renamining files to reflect their status (early, provisional, stable)
        # We go back a year to try to pull in the latest stable data.
        return pd.Timestamp(ds_max_time) - pd.Timedelta(days=365)
