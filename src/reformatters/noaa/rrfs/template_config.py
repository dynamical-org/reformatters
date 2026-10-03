from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.types import Dims, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.projected_forecast_template_config import (
    NoaaProjectedForecastTemplateConfig,
)
from reformatters.noaa.rrfs.models import NoaaRrfsDataVar
from reformatters.noaa.rrfs.variables import data_vars

PRESSURE_LEVELS = (2, 5, 7, 10, 20, 30, 50, 70, 100, *range(125, 1001, 25))
HEIGHT_LEVELS = (305, 457, 610, 914, 1524, 1829, 2134, 2743, 3658, 4572)
DEPTH_LEVELS = (0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0)


class NoaaRrfsForecastTemplateConfig(
    NoaaProjectedForecastTemplateConfig[NoaaRrfsDataVar]
):
    dims: Dims = {
        ROOT: ("init_time", "lead_time", "y", "x"),
        "pressure_level": ("init_time", "lead_time", "y", "x", "pressure_level"),
        "height_above_mean_sea_level": (
            "init_time",
            "lead_time",
            "y",
            "x",
            "height_above_mean_sea_level",
        ),
        "depth_below_ground": (
            "init_time",
            "lead_time",
            "y",
            "x",
            "depth_below_ground",
        ),
    }
    dataset_id: ClassVar[str]
    dataset_name: ClassVar[str]
    append_dim_start: Timestamp = pd.Timestamp("2026-08-13T00:00")
    members: bool = False
    sub_hourly: bool = False

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        model = "REFS" if self.members else "RRFS"
        description = (
            "CONUS forecasts from the RRFS Ensemble Forecast System (REFS), including every unique RRFS-based member run: current deterministic RRFS as member 0 and five perturbed members as members 1-5. The full REFS products additionally use these runs from the previous cycle and current and previous-cycle HRRR."
            if self.members
            else "CONUS weather forecasts from the Rapid Refresh Forecast System (RRFS) operated by NOAA NWS NCEP."
        )
        return DatasetAttributes(
            dataset_id=self.dataset_id,
            dataset_version="0.1.0",
            name=self.dataset_name,
            description=description,
            attribution=f"NOAA NWS NCEP {model} data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            license="CC-BY-4.0",
            spatial_domain="Continental United States",
            spatial_resolution="3 km",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=(
                "Forecasts initialized hourly with 15-minute forecast steps"
                if self.sub_hourly
                else f"Forecasts initialized every {int(self.append_dim_frequency / pd.Timedelta('1h'))} hours with hourly forecast steps"
            ),
            forecast_domain=(
                "Forecast lead time 15 minutes to 18 hours ahead"
                if self.sub_hourly
                else f"Forecast lead time 0-{int(self.forecast_length / pd.Timedelta('1h'))} hours ahead"
            ),
            forecast_resolution="15 minute" if self.sub_hourly else "Hourly",
        )

    def _vertical_dimension_coordinates(self) -> dict[str, Any]:
        values = {
            "pressure_level": PRESSURE_LEVELS,
            "height_above_mean_sea_level": HEIGHT_LEVELS,
            "depth_below_ground": DEPTH_LEVELS,
        }
        return {
            name: np.asarray(values[name], dtype=np.float64)
            for name in values
            if name in self.dims
        }

    def _vertical_coords(self) -> list[Coordinate]:
        attrs: dict[str, tuple[str, str, str, Literal["up", "down"]]] = {
            "pressure_level": ("Pressure level", "air_pressure", "hPa", "down"),
            "height_above_mean_sea_level": (
                "Height above mean sea level",
                "height_above_mean_sea_level",
                "m",
                "up",
            ),
            "depth_below_ground": ("Depth below ground", "depth", "m", "down"),
        }
        return [
            Coordinate(
                name=name,
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    chunks=len(values),
                    shards=None,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=CoordinateAttrs(
                    long_name=attrs[name][0],
                    standard_name=attrs[name][1],
                    units=attrs[name][2],
                    axis="Z",
                    positive=attrs[name][3],
                    statistics_approximate=StatisticsApproximate(
                        min=float(values.min()), max=float(values.max())
                    ),
                ),
            )
            for name, values in self._vertical_dimension_coordinates().items()
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[NoaaRrfsDataVar]:
        return data_vars(self.dims, sub_hourly=self.sub_hourly, members=self.members)
