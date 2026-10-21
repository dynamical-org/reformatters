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
    LEGALLY_BINDING_NOTICE,
    REAL_TIME_ATTRIBUTION,
    repo_notice_files,
)


class GoogleWeathernext2ForecastOperationalVirtualTemplateConfig(
    GoogleWeathernext2ForecastVirtualTemplateConfig
):
    source_layout: Literal["operational"] = "operational"
    product: ClassVar[str] = "Google WeatherNext 2"
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
        historical = super().dataset_attributes
        attributes = replace(
            historical,
            attribution=(
                "Google requires this attribution for data relating to a time less "
                "than 1 hour ago and the future: Google WeatherNext 2, "
                f"{REAL_TIME_ATTRIBUTION} For data relating to a time 1 hour ago "
                "or more, Google requires this attribution: "
                + historical.attribution.removeprefix(
                    "Google requires this attribution: "
                )
            ),
        )
        public_url = StoreFactory(
            primary_storage_config=Weathernext2IcechunkDatasetStorageConfig(),
            dataset_id=attributes.dataset_id,
            template_config_version=attributes.dataset_version,
        ).primary_url(public=True)
        return replace(
            attributes,
            license="other",
            copyright=COPYRIGHT,
            terms_of_use=LEGALLY_BINDING_NOTICE,
            notice_files=tuple(
                f"{public_url}/{path}"
                for path in repo_notice_files(attributes, product=self.product)
            ),
        )
