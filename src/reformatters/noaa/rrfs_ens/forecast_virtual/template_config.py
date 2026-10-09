from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.types import Dims, Timedelta, Timestamp
from reformatters.noaa.rrfs.template_config import (
    NoaaRrfsForecastTemplateConfig,
)


class NoaaRrfsEnsForecastVirtualTemplateConfig(NoaaRrfsForecastTemplateConfig):
    dataset_id: ClassVar[str] = "noaa-rrfs-ens-forecast-virtual"
    dataset_name: ClassVar[str] = "NOAA RRFS ENS forecast, virtual"
    forecast_length: Timedelta = pd.Timedelta("60h")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")
    dims: Dims = {
        ROOT: ("init_time", "ensemble_member", "lead_time", "y", "x"),
        "pressure_level": (
            "init_time",
            "ensemble_member",
            "lead_time",
            "y",
            "x",
            "pressure_level",
        ),
    }
    members: bool = True
    append_dim_start: Timestamp = pd.Timestamp("2026-09-10T00:00")

    def _vertical_dimension_coordinates(self) -> dict[str, Any]:
        return {
            "pressure_level": np.asarray(
                (250, 300, 400, 500, 600, 700, 750, 800, 850, 900, 925, 950, 975, 1000),
                dtype=np.float64,
            )
        }

    def dimension_coordinates(self) -> dict[str, Any]:
        return {**super().dimension_coordinates(), "ensemble_member": np.arange(6)}

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        return [
            *super().coords,
            Coordinate(
                name="ensemble_member",
                encoding=Encoding(dtype="int16", fill_value=-1, chunks=6, shards=None),
                attrs=CoordinateAttrs(
                    long_name="Ensemble member",
                    standard_name="realization",
                    units="1",
                    statistics_approximate=StatisticsApproximate(min=0, max=5),
                ),
            ),
        ]
