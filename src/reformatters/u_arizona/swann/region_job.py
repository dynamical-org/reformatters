from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from pydantic import Field
from rasterio import rasterio  # type: ignore

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

from .template_config import SWANNDataVar


class SWANNSourceFileCoord(SourceFileCoord):
    # SWANN data is revised over time and a single SWANNSourceFileCoord represents the
    # most stable available version of the data for a single day. Mechanically, we optimistically
    # start with the most stable version first and rely on the `.get_url()` caller to call
    # `advance_data_status()` if a file is not found at the previously returned url.
    possible_data_statuses: ClassVar[tuple[str, ...]] = ("stable", "provisional")

    remaining_data_statuses: list[str] = Field(
        default_factory=lambda: list(SWANNSourceFileCoord.possible_data_statuses)
    )

    time: Timestamp

    def get_url(self, status: str = "stable") -> str:
        water_year = self.get_water_year()
        year_month_day = self.time.strftime("%Y%m%d")

        try:
            data_status = self.get_data_status()
        except IndexError:
            # Allow get_url calls to return a URL for the last attempted status, even if it failed
            data_status = SWANNSourceFileCoord.possible_data_statuses[-1]

        return f"https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY{water_year}/UA_SWE_Depth_4km_v1_{year_month_day}_{data_status}.nc"

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"time": self.time}

    def get_water_year(self) -> int:
        return self.time.year if self.time.month < 10 else self.time.year + 1

    def get_data_status(self) -> str:
        return self.remaining_data_statuses[0]

    def advance_data_status(self) -> bool:
        self.remaining_data_statuses.pop(0)
        return len(self.remaining_data_statuses) > 0


class SWANNRegionJob(RegionJob[SWANNDataVar, SWANNSourceFileCoord]):
    # Be gentle to UA HTTP servers
    download_parallelism: int = 2

    @classmethod
    def operational_update_jobs(
        cls,
        final_store: zarr.abc.store.Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[SWANNDataVar],
        reformat_job_name: str,
    ) -> tuple[Sequence[RegionJob[SWANNDataVar, SWANNSourceFileCoord]], xr.Dataset]:
        append_dim_end = cls._operational_append_dim_end()

        template_ds = get_template_fn(append_dim_end)
        existing_ds = xr.open_zarr(final_store)

        append_dim_start = cls._operational_append_dim_start(existing_ds)

        jobs = cls.get_jobs(
            kind="operational-update",
            final_store=final_store,
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
        data_var_group: Sequence[SWANNDataVar],
    ) -> Sequence[SWANNSourceFileCoord]:
        times = processing_region_ds["time"].values
        return [SWANNSourceFileCoord(time=t) for t in times]

    def download_file(self, coord: SWANNSourceFileCoord) -> Path:
        while True:
            try:
                return http_download_to_disk(coord.get_url(), self.dataset_id)
            except FileNotFoundError:
                if not coord.advance_data_status():
                    raise

    def read_data(
        self,
        coord: SWANNSourceFileCoord,
        data_var: SWANNDataVar,
    ) -> ArrayFloat32:
        var_name = data_var.internal_attrs.netcdf_var_name
        netcdf_path = f"netcdf:{coord.downloaded_path}:{var_name}"
        band = 1  # because rasterio netcdf requires selecting the band in the file path we always want band 1
        with rasterio.open(netcdf_path) as reader:
            result: Array2D[np.float32] = reader.read(band, out_dtype=np.float32)
            result[result == -999] = np.nan
            assert result.shape == (621, 1405)
            return result

    @classmethod
    def _operational_append_dim_end(cls) -> pd.Timestamp:
        return pd.Timestamp.now()

    @classmethod
    def _operational_append_dim_start(cls, existing_ds: xr.Dataset) -> pd.Timestamp:
        ds_max_time = existing_ds["time"].max().item()
        # UArizona updates the data, renamining files to reflect their status (early, provisional, stable)
        # We go back a year to try to pull in the latest stable data.
        return pd.Timestamp(ds_max_time) - pd.Timedelta(days=365)
