from typing import ClassVar, Literal

import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.pydantic import replace
from reformatters.common.storage import StoreFactory
from reformatters.common.types import Timestamp
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)
from reformatters.google.weathernext2.storage import (
    Weathernext2IcechunkDatasetStorageConfig,
)
from reformatters.google.weathernext_terms import (
    COPYRIGHT,
    LICENSE,
    NON_ENDORSEMENT,
    REAL_TIME_ATTRIBUTION,
    USAGE_RESTRICTIONS,
    repo_notice_files,
)


class GoogleWeathernext2ForecastOperationalVirtualTemplateConfig(
    GoogleWeathernext2ForecastVirtualTemplateConfig
):
    source_layout: Literal["operational"] = "operational"
    product: ClassVar[str] = "Google WeatherNext 2"
    modifications: ClassVar[str] = (
        "The publisher, dynamical.org, restructured the source data into an "
        "Icechunk virtual Zarr dataset of references to the unmodified source "
        "chunks, renamed its variables and dimensions, and applies read-time "
        "filters that convert temperatures from kelvin to degrees Celsius, "
        "geopotential to geopotential height (dividing by 9.80665 m s-2), and "
        "6-hour total precipitation from metres of water equivalent to kg m-2. "
        "Values are otherwise the unchanged model output."
    )
    usage_restrictions: ClassVar[str] = USAGE_RESTRICTIONS
    non_endorsement: ClassVar[str] = NON_ENDORSEMENT
    append_dim_start: Timestamp = pd.Timestamp("2025-01-01T00:00")
    dataset_id_value: ClassVar[str] = "google-weathernext2-forecast-operational-virtual"
    dataset_name_value: ClassVar[str] = (
        "Google WeatherNext 2 operational forecast, virtual"
    )
    time_domain_end: ClassVar[str] = "Present"
    dataset_description: ClassVar[str] = (
        "Weather forecasts from the 64-member Google DeepMind WeatherNext 2 ensemble "
        "model."
    )

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        attributes = replace(
            super().dataset_attributes,
            license=LICENSE,
            attribution=(
                "Google WeatherNext 2 forecast data processed by dynamical.org "
                "from Google's WeatherNext 2 experimental data. "
                f"{COPYRIGHT}. {REAL_TIME_ATTRIBUTION}"
            ),
        )
        public_url = StoreFactory(
            primary_storage_config=Weathernext2IcechunkDatasetStorageConfig(),
            dataset_id=attributes.dataset_id,
            template_config_version=attributes.dataset_version,
        ).primary_url(public=True)
        return replace(
            attributes,
            notice_files=tuple(
                f"{public_url}/{path}"
                for path in repo_notice_files(
                    attributes,
                    product=self.product,
                    modifications=self.modifications,
                    usage_restrictions=self.usage_restrictions,
                    non_endorsement=self.non_endorsement,
                )
            ),
        )
