from typing import Literal

import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.types import Timedelta, Timestamp
from reformatters.ecmwf.ifs_ens.virtual_template_config import (
    EcmwfIfsEnsForecastVirtualTemplateConfig,
)


class EcmwfIfsEnsForecast6Day01DegreeVirtualTemplateConfig(
    EcmwfIfsEnsForecastVirtualTemplateConfig
):
    resolution: Literal["0p25", "0p1"] = "0p1"
    forecast_days: Literal[6, 15] = 6
    append_dim_frequency: Timedelta = pd.Timedelta("6h")
    append_dim_start: Timestamp = pd.Timestamp("2026-10-21T00")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-ifs-ens-forecast-6-day-0-1-degree-virtual",
            dataset_version="0.1.0",
            name="ECMWF IFS ENS forecast, 6 day, 0.1 degree, virtual",
            description="Ensemble weather forecasts from the ECMWF Integrated Forecasting System (IFS).",
            attribution="ECMWF IFS ENS forecast data processed by dynamical.org from ECMWF Open Data.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.1 degrees (~10km)",
            time_domain="Forecasts initialized 2026-10-21 00:00:00 UTC to Present",
            time_resolution="Forecasts initialized every 6 hours",
            forecast_domain="Forecast lead time 0-144 hours (0-6 days) ahead",
            forecast_resolution="3 hourly",
        )
