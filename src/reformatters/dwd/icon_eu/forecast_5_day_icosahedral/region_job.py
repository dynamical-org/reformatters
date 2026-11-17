import eccodes
import numpy as np

from reformatters.common.time_utils import whole_hours
from reformatters.common.types import ArrayFloat32
from reformatters.dwd.icon_eu.icon_eu_region_job import (
    DwdIconEuRegionJob,
    DwdIconEuSourceFileCoord,
)
from reformatters.dwd.icon_eu.icon_eu_template_config import DwdIconEuDataVar

from . import icosahedral_grid


class DwdIconEuForecast5DayIcosahedralSourceFileCoord(DwdIconEuSourceFileCoord):
    def get_url(self) -> str:
        return (
            "https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
            "dynamical/dwd-icon-grib/icon-eu/icosahedral/"
            f"{self.init_time:%Y-%m-%dT%H}/{self.variable_name_in_filename.upper()}/"
            f"PT{whole_hours(self.lead_time):03d}H00M.grib2"
        )

    def get_fallback_url(self) -> str:
        return (
            "https://opendata.dwd.de/weather/nwp/v1/m/icon-eu/p/"
            f"{self.variable_name_in_filename.upper()}/r/{self.init_time:%Y-%m-%dT%H}:00/"
            f"s/PT{whole_hours(self.lead_time):03d}H00M.grib2"
        )


class DwdIconEuForecast5DayIcosahedralRegionJob(
    DwdIconEuRegionJob[DwdIconEuForecast5DayIcosahedralSourceFileCoord]
):
    source_file_coord_class = DwdIconEuForecast5DayIcosahedralSourceFileCoord

    def read_data(
        self,
        coord: DwdIconEuForecast5DayIcosahedralSourceFileCoord,
        data_var: DwdIconEuDataVar,
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None
        path = coord.downloaded_path
        with path.open("rb") as file:
            message_count = eccodes.codes_count_in_file(file)
            assert message_count == 1, (
                f"Expected exactly one GRIB message, got {message_count} in {path}"
            )
            handle = eccodes.codes_grib_new_from_file(file)
            assert handle is not None
            try:
                template = eccodes.codes_get(handle, "gridDefinitionTemplateNumber")
                assert template == 101, (
                    f"Expected an unstructured grid (template 3.101), got {template} in {path}"
                )
                grid_uuid = eccodes.codes_get(handle, "uuidOfHGrid")
                expected_grid_uuid = icosahedral_grid.GRID_UUID.replace("-", "")
                assert grid_uuid == expected_grid_uuid, (
                    f"Expected grid UUID {expected_grid_uuid}, got {grid_uuid} in {path}"
                )
                number_of_points = eccodes.codes_get(handle, "numberOfDataPoints")
                assert number_of_points == icosahedral_grid.CELL_COUNT, (
                    f"Expected {icosahedral_grid.CELL_COUNT} grid points, got {number_of_points} in {path}"
                )
                result = eccodes.codes_get_values(handle).astype(np.float32)
                if eccodes.codes_get(handle, "bitmapPresent"):
                    result[eccodes.codes_get_array(handle, "bitmap") == 0] = np.nan
            finally:
                eccodes.codes_release(handle)
        if data_var.attrs.units == "degree_Celsius":
            result -= np.float32(273.15)
        return result
