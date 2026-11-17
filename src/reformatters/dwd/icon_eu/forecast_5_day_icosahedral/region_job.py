from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.grib_eccodes import read_unstructured_grib
from reformatters.common.iterating import item
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
    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[DwdIconEuDataVar],
    ) -> Sequence[DwdIconEuForecast5DayIcosahedralSourceFileCoord]:
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        data_var = item(data_var_group)

        assert len(init_times) > 0
        assert len(lead_times) > 0

        return [
            DwdIconEuForecast5DayIcosahedralSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                data_var=data_var,
            )
            for init_time in init_times
            for lead_time in lead_times
        ]

    def read_data(
        self,
        coord: DwdIconEuForecast5DayIcosahedralSourceFileCoord,
        data_var: DwdIconEuDataVar,
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None
        result = read_unstructured_grib(
            coord.downloaded_path,
            expected_grid_uuid=icosahedral_grid.GRID_UUID.replace("-", ""),
            expected_number_of_points=icosahedral_grid.CELL_COUNT,
        )
        if data_var.attrs.units == "degree_Celsius":
            result -= np.float32(273.15)
        return result
