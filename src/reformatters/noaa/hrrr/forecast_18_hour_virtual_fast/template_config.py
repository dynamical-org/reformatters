import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.pydantic import replace
from reformatters.common.types import Timestamp
from reformatters.noaa.hrrr.forecast_18_hour_virtual.template_config import (
    NoaaHrrrForecast18HourVirtualTemplateConfig,
)


class NoaaHrrrForecast18HourVirtualFastTemplateConfig(
    NoaaHrrrForecast18HourVirtualTemplateConfig
):
    """The 18-hour virtual product's variables and structure, fed from the NOMADS
    mirror ahead of NODD. Starts where the mirror does; the NODD-fed sibling holds
    the history."""

    append_dim_start: Timestamp = pd.Timestamp("2026-09-01T00:00")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        attrs = self._dataset_attributes(
            dataset_id="noaa-hrrr-forecast-18-hour-virtual-fast",
            dataset_version="0.1.0",
            name="NOAA HRRR forecast, 18 hour, fast, virtual",
        )
        return replace(
            attrs,
            description=(
                f"{attrs.description} Files are taken from NOAA's NOMADS server as "
                "soon as they are published, minutes before NOAA's AWS archive lists them."
            ),
        )
