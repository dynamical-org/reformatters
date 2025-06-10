from itertools import product

import numpy as np
import pandas as pd

from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.noaa.gfs.read_data import (
    GFS_ACCUMULATION_RESET_FREQUENCY,
    GFS_ACCUMULATION_RESET_HOURS,
    read_rasterio,
)
from reformatters.noaa.gfs.read_data import (
    SourceFileCoords as _GfsCoords,
)
from reformatters.noaa.gfs.read_data import (
    download_file as _gfs_download,
)
from reformatters.noaa.noaa_config_models import NOAADataVar


class GFSSourceFileCoord(SourceFileCoord):
    """Coordinates for a single GFS .idx/.grb2 request."""

    init_time: pd.Timestamp
    lead_time: pd.Timedelta

    def get_url(self) -> str:
        # reconstruct the NOAA OPeNDAP URL for debugging/logging
        hrs = int(self.lead_time.total_seconds() / 3600)
        d = self.init_time.strftime("%Y%m%d")
        h = self.init_time.strftime("%H")
        base = f"gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f{hrs:03d}"
        return f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/{base}"


class GFSRegionJob(RegionJob[NOAADataVar, GFSSourceFileCoord]):
    """RegionJob for NOAA GFS 0.25° forecasts."""

    def generate_source_file_coords(
        self,
        processing_region_ds,
        data_var_group,
    ) -> list[GFSSourceFileCoord]:
        # One file per (init_time, lead_time) pair
        init_times = processing_region_ds["init_time"].values
        lead_times = processing_region_ds["lead_time"].values
        return [
            GFSSourceFileCoord(init_time=pd.Timestamp(it), lead_time=pd.Timedelta(lt))
            for it, lt in product(init_times, lead_times)
        ]

    def download_file(self, coord: GFSSourceFileCoord) -> pd.PathLike:
        # reuse existing function: returns (coords_dict, Path|None)
        coords_dict: _GfsCoords = {
            "init_time": coord.init_time,
            "lead_time": coord.lead_time,
        }
        _, local_path = _gfs_download(coords_dict, self.data_vars)
        return local_path  # may be None, RegionJob will set status accordingly

    def read_data(self, coord: GFSSourceFileCoord, data_var: NOAADataVar) -> np.ndarray:
        path = coord.downloaded_path
        assert path is not None, "read_data called on missing file"

        # build the GRIB_ELEMENT (with lead‐time suffix if needed)
        elem = data_var.internal_attrs.grib_element
        if data_var.internal_attrs.include_lead_time_suffix:
            hours = coord.lead_time.total_seconds() / 3600
            acc = int(hours % GFS_ACCUMULATION_RESET_HOURS)
            if acc == 0:
                acc = GFS_ACCUMULATION_RESET_HOURS
            elem = f"{elem}{acc:02d}"

        desc = data_var.internal_attrs.grib_description

        # pull spatial metadata off the template
        da = self.template_ds[data_var.name]
        out_shape = da.rio.shape
        out_transform = da.rio.transform()
        out_crs = da.rio.crs

        return read_rasterio(
            path,
            elem,
            desc,
            out_shape,
            out_transform,
            out_crs,
        )

    def apply_data_transformations(self, data_array, data_var: NOAADataVar) -> None:
        # first deaccumulate if requested
        if data_var.internal_attrs.deaccumulate_to_rates:
            from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace

            deaccumulate_to_rates_inplace(
                data_array,
                dim="lead_time",
                reset_frequency=GFS_ACCUMULATION_RESET_FREQUENCY,
            )
        # then binary‐round as in the base class
        super().apply_data_transformations(data_array, data_var)
