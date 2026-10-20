from collections.abc import Sequence

import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.pydantic import replace
from reformatters.common.types import Timedelta, Timestamp
from reformatters.noaa.rtma.analysis_virtual.template_config import (
    NoaaRtmaAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.rtma.rtma_config_models import RtmaDataVar


class NoaaRtmaRuAnalysisVirtualTemplateConfig(NoaaRtmaAnalysisVirtualTemplateConfig):
    append_dim_start: Timestamp = pd.Timestamp("2017-12-13T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("15min")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return replace(
            super().dataset_attributes,
            dataset_id="noaa-rtma-ru-analysis-virtual",
            name="NOAA RTMA-RU analysis, virtual",
            description="Real-Time Mesoscale Analysis Rapid Update (RTMA-RU) operated every 15 minutes by NOAA NWS NCEP on the native CONUS grid.",
            time_resolution="15 minutes",
        )

    @computed_field
    @property
    def data_vars(self) -> Sequence[RtmaDataVar]:
        return [
            replace(
                var,
                internal_attrs=replace(
                    var.internal_attrs,
                    available_from=pd.Timestamp("2020-07-28T00:00"),
                ),
            )
            if var.name == "total_cloud_cover_atmosphere"
            else var
            for var in super().data_vars
        ]
