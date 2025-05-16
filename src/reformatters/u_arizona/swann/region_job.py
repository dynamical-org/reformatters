from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.types import ArrayFloat32, Dim, Timestamp

from .template_config import SWANNDataVar


class SWANNSourceFileCoord(SourceFileCoord):
    water_year: int
    time_start: Timestamp
    time_end: Timestamp

    def get_url(self) -> str:
        # Example: 4km_SWE_Depth_WY1982_v01.nc
        return f"https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/4km_SWE_Depth_WY{self.water_year}_v01.nc"

    def out_loc(self) -> Mapping[Dim, slice]:
        # This coord covers a range of times (one water year)
        return {"time": slice(self.time_start, self.time_end)}


class SWANNRegionJob(RegionJob[SWANNDataVar, SWANNSourceFileCoord]):
    max_vars_per_backfill_job: ClassVar[int] = 2

    @classmethod
    def group_data_vars(
        cls, data_vars: Sequence[SWANNDataVar]
    ) -> Sequence[Sequence[SWANNDataVar]]:
        # All variables are in the same file, so group all together
        return [data_vars]

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[SWANNDataVar],
    ) -> Sequence[SWANNSourceFileCoord]:
        # Each file covers a water year: Oct 1 (prev year) to Sep 30 (year)
        # Find all unique water years in the region
        times = pd.to_datetime(processing_region_ds["time"].values)
        years = np.unique([(t + pd.DateOffset(months=3)).year for t in times])
        coords = []
        for wy in years:
            # Water year N: Oct 1 (N-1) to Sep 30 (N)
            start = pd.Timestamp(year=wy - 1, month=10, day=1)
            end = pd.Timestamp(year=wy, month=9, day=30)
            # Only include times in this file
            region_times = times[(times >= start) & (times <= end)]
            if len(region_times) == 0:
                continue
            coords.append(
                SWANNSourceFileCoord(
                    water_year=wy,
                    time_start=region_times[0],
                    time_end=region_times[-1],
                )
            )
        return coords

    def download_file(self, coord: SWANNSourceFileCoord) -> Path:
        # For now, assume files are already available locally or use a stub
        # In production, implement actual download logic
        filename = f"4km_SWE_Depth_WY{coord.water_year}_v01.nc"
        # TODO: Should this location be set as a default somewhere?
        download_dir = Path("data/download/")
        local_path = download_dir / filename
        if not local_path.exists():
            # Download logic would go here
            raise FileNotFoundError(f"File not found: {local_path}")
        return local_path

    def read_data(
        self,
        coord: SWANNSourceFileCoord,
        data_var: SWANNDataVar,
    ) -> ArrayFloat32:
        # Open the NetCDF file and extract the relevant variable and time range
        assert coord.downloaded_path is not None, "downloaded_path must not be None"
        ds = xr.open_dataset(coord.downloaded_path)
        # The NetCDF time variable is days since 1900-01-01
        time_var = ds["time"]
        times = pd.to_datetime("1900-01-01") + pd.to_timedelta(
            time_var.values, unit="D"
        )
        # Find indices for the requested time range
        mask = (times >= coord.time_start) & (times <= coord.time_end)
        arr = ds[data_var.name].sel(time=mask).values.astype(np.float32)
        return arr
