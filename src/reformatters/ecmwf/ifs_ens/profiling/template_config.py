import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.pydantic import replace
from reformatters.common.types import Timestamp
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)

DATASET_ID = "ifsens-profile"
VERSION = "0.0.1-profiling"
START = pd.Timestamp("2024-03-30T00:00Z")
END = pd.Timestamp("2024-04-01T00:00Z")


class ProfilingTemplateConfig(EcmwfIfsEnsForecast15Day025DegreeTemplateConfig):
    # Zarr datetime coordinates store UTC without a timezone extension dtype.
    append_dim_start: Timestamp = START.tz_localize(None)

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return replace(
            super().dataset_attributes,
            dataset_id=DATASET_ID,
            dataset_version=VERSION,
        )
