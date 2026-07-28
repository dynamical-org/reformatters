from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    DatasetAttributes,
    StatisticsApproximate,
)
from reformatters.common.types import Timedelta
from reformatters.noaa.hrrr.forecast_virtual_template_config import (
    NoaaHrrrForecastVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.template_config import (
    EXPECTED_FORECAST_LENGTH_V3,
    EXPECTED_FORECAST_LENGTH_V4,
    HRRR_V4_FIRST_INIT,
)


class NoaaHrrrForecast48HourVirtualTemplateConfig(
    NoaaHrrrForecastVirtualTemplateConfig
):
    """Virtual HRRR 48-hour forecast: the 00/06/12/18 UTC extended-length cycles.

    Mirrors the materialized noaa-hrrr-forecast-48-hour temporal structure.
    """

    forecast_length: Timedelta = EXPECTED_FORECAST_LENGTH_V4
    append_dim_frequency: Timedelta = pd.Timedelta("6h")  # only 00/06/12/18 reach f48

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return self._dataset_attributes(
            dataset_id="noaa-hrrr-forecast-48-hour-virtual",
            dataset_version="0.5.0",
            name="NOAA HRRR forecast, 48 hour, virtual",
        )

    def _expected_forecast_length_values(self, ds: xr.Dataset) -> np.ndarray[Any, Any]:
        return np.where(
            ds["init_time"].values < HRRR_V4_FIRST_INIT.to_datetime64(),
            EXPECTED_FORECAST_LENGTH_V3.to_timedelta64(),
            EXPECTED_FORECAST_LENGTH_V4.to_timedelta64(),
        )

    def _expected_forecast_length_statistics(self) -> StatisticsApproximate:
        return StatisticsApproximate(
            min=str(EXPECTED_FORECAST_LENGTH_V3), max=str(EXPECTED_FORECAST_LENGTH_V4)
        )
