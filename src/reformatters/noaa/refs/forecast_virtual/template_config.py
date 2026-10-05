from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
)
from reformatters.common.types import Dim, Dims, Timedelta, Timestamp
from reformatters.noaa.projected_forecast_template_config import (
    NoaaProjectedForecastTemplateConfig,
)

from .models import NoaaRefsDataVar
from .variables import data_vars


class NoaaRefsForecastVirtualTemplateConfig(
    NoaaProjectedForecastTemplateConfig[NoaaRefsDataVar]
):
    dataset_id: ClassVar[str] = "noaa-refs-forecast-virtual"
    dataset_name: ClassVar[str] = "NOAA REFS forecast, virtual"
    dims: Dims = {ROOT: ("init_time", "statistic", "lead_time", "y", "x")}
    append_dim_start: Timestamp = pd.Timestamp("2026-08-13T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")
    first_lead: Timedelta = pd.Timedelta("1h")
    forecast_length: Timedelta = pd.Timedelta("60h")

    def data_var_dims(self, var: NoaaRefsDataVar) -> tuple[Dim, ...]:
        return tuple(
            dim
            for dim in self.dims[var.group]
            if var.has_statistic or dim != "statistic"
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            **super().dimension_coordinates(),
            "statistic": np.asarray(("mean", "standard_deviation"), dtype=object),
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        return [
            *super().coords,
            Coordinate(
                name="statistic",
                encoding=Encoding(dtype="object", fill_value="", chunks=2, shards=None),
                attrs=CoordinateAttrs(
                    long_name="Ensemble statistic",
                    units="1",
                    statistics_approximate=None,
                    comment="The mean label denotes the ensemble mean; standard_deviation denotes NOAA's spread: the member-weighted population standard deviation about that mean, with weights normalized over contributing members. Exceptions are described on the affected variables.",
                ),
            ),
        ]

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id=self.dataset_id,
            dataset_version="0.1.0",
            name=self.dataset_name,
            description="CONUS derived forecasts from NOAA's RRFS Ensemble Forecast System (REFS): ensemble mean and standard deviation (NOAA spread), probability-matched and localized probability-matched means, mean/PMM averages, threshold and neighborhood probabilities, ensemble-agreement-scale probabilities, and flash-flood guidance and recurrence-interval exceedance probabilities. The products combine current and previous-cycle RRFS deterministic and perturbed runs with HRRR, using up to 14 members; the number of contributing members depends on field and lead time. Temperature, dewpoint and soil-temperature means use Celsius. Their standard deviations are provided as <variable>_standard_deviation, in kelvin differences; 1 K of difference equals 1 °C of difference. The standard_deviation slices of the Celsius arrays are not populated; read <variable>_standard_deviation.",
            attribution="NOAA NWS NCEP REFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            license="CC-BY-4.0",
            spatial_domain="Continental United States",
            spatial_resolution="3 km",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution="Forecasts initialized every 6 hours with hourly forecast steps",
            forecast_domain="Forecast lead time 1-60 hours ahead",
            forecast_resolution="Hourly",
        )

    @computed_field
    @property
    def data_vars(self) -> Sequence[NoaaRefsDataVar]:
        return data_vars(self.dims)
