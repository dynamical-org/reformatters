import bz2

import numpy as np
from rasterio.io import MemoryFile

from reformatters.common.time_utils import whole_hours
from reformatters.common.types import ArrayFloat32
from reformatters.dwd.icon_eu.icon_eu_region_job import (
    DwdIconEuRegionJob,
    DwdIconEuSourceFileCoord,
)
from reformatters.dwd.icon_eu.icon_eu_template_config import DwdIconEuDataVar


class DwdIconEuForecast5DaySourceFileCoord(DwdIconEuSourceFileCoord):
    """Coordinates of a single source file to process.

    ICON-EU is published as one Bzip2-compressed GRIB2 file per variable.
    """

    def get_url(self) -> str:
        """Return URL to .grib2.bz2 files on Dynamical.org's public archive of ICON-EU hosted on Source Co-Op."""
        init_time_str = self.init_time.strftime("%Y-%m-%dT%H")
        return (
            "https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
            "dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/"
            f"{init_time_str}/{self.variable_name_in_filename}/"
        ) + self._get_basename()

    def get_fallback_url(self) -> str:
        """Return URLs to .grib2.bz2 files on DWD's HTTP server.

        Note that DWD may change their directory structure for ICON-EU. See this comment for
        details: https://github.com/dynamical-org/reformatters/issues/183#issuecomment-3365068327
        """
        # Example DWD URL:
        # https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025090700_000_ALB_RAD.grib2.bz2

        init_hour_str = self.init_time.strftime("%H")
        return (
            "https://opendata.dwd.de/weather/nwp/icon-eu/grib/"
            f"{init_hour_str}/{self.variable_name_in_filename}/"
        ) + self._get_basename()

    def _get_basename(self) -> str:
        # Note that this only handles single-level variables.
        init_time_str: str = self.init_time.strftime("%Y%m%d%H")
        lead_time_hours: int = whole_hours(self.lead_time)
        return (
            f"icon-eu_europe_regular-lat-lon_single-level_{init_time_str}_{lead_time_hours:03d}_"
            f"{self.variable_name_in_filename.upper()}.grib2.bz2"
        )


class DwdIconEuForecast5DayRegionJob(
    DwdIconEuRegionJob[DwdIconEuForecast5DaySourceFileCoord]
):
    source_file_coord_class = DwdIconEuForecast5DaySourceFileCoord

    def read_data(
        self,
        coord: DwdIconEuForecast5DaySourceFileCoord,
        data_var: DwdIconEuDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        assert coord.downloaded_path is not None  # for type check, system guarantees it

        with bz2.open(coord.downloaded_path, "rb") as f:
            grib_data = f.read()

        with MemoryFile(grib_data) as memfile, memfile.open() as reader:
            assert reader.count == 1, (
                f"Expected exactly 1 element in each ICON-EU grib file, found {reader.count=}. "
                f"{data_var.internal_attrs.variable_name_in_filename=}, "
                f"{coord.downloaded_path=}"
            )
            result: ArrayFloat32 = reader.read(indexes=1, out_dtype=np.float32)
        return result
