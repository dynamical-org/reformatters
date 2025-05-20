from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr
from rasterio import rasterio  # type: ignore

from reformatters.common.download import http_download
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.types import Array2D, ArrayFloat32, Timestamp

from .template_config import SWANNDataVar


class SWANNSourceFileCoord(SourceFileCoord):
    time: Timestamp

    def get_url(self) -> str:
        # TODO: fix this example
        # Example: 4km_SWE_Depth_WY1982_v01.nc
        # Example: UA_SWE_Depth_4km_v1_20241014_stable.nc
        # https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY2025/UA_SWE_Depth_4km_v1_20241014_stable.nc
        water_year = self.get_water_year()
        year_month_day = self.time.strftime("%Y%m%d")
        return f"https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY{water_year}/UA_SWE_Depth_4km_v1_{year_month_day}_stable.nc"

    def get_water_year(self) -> int:
        return self.time.year if self.time.month < 10 else self.time.year + 1


class SWANNRegionJob(RegionJob[SWANNDataVar, SWANNSourceFileCoord]):
    # Be gentle to UA HTTP servers
    download_parallelism: int = 4

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[SWANNDataVar],
    ) -> Sequence[SWANNSourceFileCoord]:
        times = processing_region_ds["time"].values
        return [SWANNSourceFileCoord(time=t) for t in times]

    def download_file(self, coord: SWANNSourceFileCoord) -> Path:
        url = coord.get_url()
        filename = Path(url).name
        return http_download(url, filename, "U_ARIZONA_SWANN", overwrite_existing=False)

    def read_data(
        self,
        coord: SWANNSourceFileCoord,
        data_var: SWANNDataVar,
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None, "downloaded_path must not be None"

        var_name = "SWE" if data_var.name == "snow_water_equivalent" else "DEPTH"
        netcdf_path = f"netcdf:{coord.downloaded_path}:{var_name}"
        band = 1
        return _read_netcdf(netcdf_path, band)


def _read_netcdf(netcdf_path: str, band: int) -> Array2D[np.float32]:
    """Helper function to read a netcdf file with rasterio.

    This is split out from read_data for easier testing and manual invocation.

    Args:
        netcdf_path: The path to the netcdf file.

    Returns:
        The data as a 2D array.
    """
    with rasterio.open(netcdf_path) as reader:
        result: Array2D[np.float32] = reader.read(band, out_dtype=np.float32)
        result[result == -999] = np.nan
        assert result.shape == (621, 1405)
        return result
