from collections.abc import Sequence

import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.pydantic import replace
from reformatters.noaa.rtma.analysis_virtual.template_config import (
    NoaaRtmaAnalysisVirtualTemplateConfig,
    _var,
)
from reformatters.noaa.rtma.rtma_config_models import RtmaDataVar


class NoaaUrmaAnalysisVirtualTemplateConfig(NoaaRtmaAnalysisVirtualTemplateConfig):
    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return replace(
            super().dataset_attributes,
            dataset_id="noaa-urma-analysis-virtual",
            name="NOAA URMA analysis, virtual",
            description="Hourly Unrestricted Mesoscale Analysis (URMA) operated by NOAA NWS NCEP on the native CONUS grid, incorporating observations received after the RTMA analysis.",
            attribution="NOAA NWS NCEP URMA data processed by dynamical.org from NOAA Open Data Dissemination archives.",
        )

    @computed_field
    @property
    def data_vars(self) -> Sequence[RtmaDataVar]:
        wave = _var(
            "significant_wave_height_surface",
            "HTSGW",
            (10, 0, 3, 1, 1.0),
            "swh",
            "Significant height of combined wind waves and swell",
            "m",
            "sea_surface_wave_significant_height",
            "NaN where no wave analysis is made, such as over land and ice-covered water.",
        )
        return [
            *super().data_vars,
            replace(
                wave,
                internal_attrs=replace(
                    wave.internal_attrs,
                    available_from=pd.Timestamp("2018-12-03T12:00"),
                    requires_bitmap=True,
                ),
            ),
        ]
