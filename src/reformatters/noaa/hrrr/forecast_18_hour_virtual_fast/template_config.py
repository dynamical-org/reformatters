from collections.abc import Sequence
from typing import Final

import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    DatasetAttributes,
    StatisticsApproximate,
)
from reformatters.common.pydantic import replace
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import DatetimeLike, Timestamp
from reformatters.noaa.hrrr.forecast_18_hour_virtual.template_config import (
    NoaaHrrrForecast18HourVirtualTemplateConfig,
)

# How far back init_time reaches. Must stay under the NOMADS mirror bucket's object
# expiry by more than an update outage we intend to ride out.
RETENTION: Final = pd.Timedelta("72h")


class NoaaHrrrForecast18HourVirtualFastTemplateConfig(
    NoaaHrrrForecast18HourVirtualTemplateConfig
):
    """The 18-hour virtual product's variables and structure over a moving window of
    the most recent inits, fed from the NOMADS mirror. The NODD-fed sibling holds
    the history."""

    # The earliest the window can start; init_time otherwise starts RETENTION back.
    append_dim_start: Timestamp = pd.Timestamp("2026-09-01T00:00")

    def append_dim_coordinates(self, end: DatetimeLike) -> pd.DatetimeIndex:
        window_start = pd.Timestamp(end).floor("1h") - RETENTION
        return pd.date_range(
            max(self.append_dim_start, window_start),
            end,
            freq=self.append_dim_frequency,
            inclusive="left",
        )

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        attrs = self._dataset_attributes(
            dataset_id="noaa-hrrr-forecast-18-hour-virtual-fast",
            dataset_version="0.1.0",
            name="NOAA HRRR forecast, 18 hour, fast, virtual",
        )
        retention_days = whole_hours(RETENTION) // 24
        return replace(
            attrs,
            description=(
                f"{attrs.description} Files are taken from NOAA's NOMADS server as "
                "soon as they are published, minutes before NOAA's AWS archive lists "
                f"them. Only the most recent {retention_days} days of forecasts are "
                "kept: the first init_time advances every hour and older forecasts "
                "are removed. Use noaa-hrrr-forecast-18-hour-virtual for history."
            ),
            attribution="NOAA NWS NCEP HRRR data processed by dynamical.org from NOAA NOMADS.",
            time_domain=(
                f"Forecasts initialized in the most recent {retention_days} days, "
                "a moving window"
            ),
        )

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        return [
            _with_window_minimum(coord)
            if coord.name in ("init_time", "valid_time")
            else coord
            for coord in super().coords
        ]


def _with_window_minimum(coord: Coordinate) -> Coordinate:
    statistics = coord.attrs.statistics_approximate
    assert statistics is not None
    return replace(
        coord,
        attrs=replace(
            coord.attrs,
            statistics_approximate=StatisticsApproximate(
                min=f"Present - {whole_hours(RETENTION)} hours", max=statistics.max
            ),
        ),
    )
