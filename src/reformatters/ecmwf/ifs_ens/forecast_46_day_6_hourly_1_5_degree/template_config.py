from collections.abc import Sequence

import numpy as np
import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import ROOT, DatasetAttributes, Encoding
from reformatters.common.pydantic import replace
from reformatters.common.types import Dim, Dims, Timedelta
from reformatters.common.zarr import BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.ecmwf.ifs_ens.forecast_46_day_1_5_degree.template_config import (
    EcmwfIfsEnsForecast46Day15DegreeTemplateConfig,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_config_models import (
    EcmwfIfsEns46DayDataVar,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_template_config import (
    EcmwfIfsEns46DayCommonTemplateConfig,
)

DATA_VAR_NAMES = (
    "precipitation_surface",
    "wind_u_10m",
    "wind_v_10m",
    "maximum_temperature_2m",
    "minimum_temperature_2m",
)

SIX_HOURLY_COMMENTS = {
    "precipitation_surface": "Average precipitation rate over the previous 6 hours. Units equivalent to mm/s.",
    "maximum_temperature_2m": "Maximum temperature at 2 metres over the previous 6 hours.",
    "minimum_temperature_2m": "Minimum temperature at 2 metres over the previous 6 hours.",
}


class EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig(
    EcmwfIfsEns46DayCommonTemplateConfig
):
    dims: Dims = {
        ROOT: (
            "init_time",
            "lead_time",
            "ensemble_member",
            "latitude",
            "longitude",
        )
    }

    lead_time_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree",
            dataset_version="0.1.0",
            name="ECMWF IFS ENS forecast, 46 day, 6 hourly, 1.5 degree",
            description="Sub-seasonal-range ensemble weather forecasts from the ECMWF Integrated Forecasting System (IFS).",
            attribution="ECMWF IFS ENS sub-seasonal-range forecast data processed by dynamical.org from the ECMWF Data Store.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="1.5 degrees (~165km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-1104 hours (0-46 days) ahead",
            forecast_resolution="Forecast step 6 hourly",
        )

    @computed_field
    @property
    def data_vars(self) -> Sequence[EcmwfIfsEns46DayDataVar]:
        chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 37,
            "ensemble_member": 101,
            "latitude": 32,
            "longitude": 30,
        }
        shards: dict[Dim, int] = {
            "init_time": chunks["init_time"],
            "lead_time": chunks["lead_time"] * 5,
            "ensemble_member": chunks["ensemble_member"],
            "latitude": chunks["latitude"] * 4,
            "longitude": chunks["longitude"] * 4,
        }
        encoding = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(chunks[dim] for dim in self.dims[ROOT]),
            shards=tuple(shards[dim] for dim in self.dims[ROOT]),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )
        daily_data_vars = {
            data_var.name: data_var
            for data_var in EcmwfIfsEnsForecast46Day15DegreeTemplateConfig().data_vars
        }

        data_vars = []
        for name in DATA_VAR_NAMES:
            daily_data_var = daily_data_vars[name]
            attrs = daily_data_var.attrs
            internal_attrs = daily_data_var.internal_attrs
            if comment := SIX_HOURLY_COMMENTS.get(name):
                attrs = replace(attrs, comment=comment)
            if name in {"maximum_temperature_2m", "minimum_temperature_2m"}:
                internal_attrs = replace(internal_attrs, sub_step_reduction=None)
            data_vars.append(
                replace(
                    daily_data_var,
                    encoding=encoding,
                    attrs=attrs,
                    internal_attrs=internal_attrs,
                )
            )
        return data_vars
