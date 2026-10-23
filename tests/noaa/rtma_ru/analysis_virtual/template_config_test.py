import numpy as np
import pandas as pd

from reformatters.noaa.rtma.analysis_virtual.template_config import (
    NoaaRtmaAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.rtma_ru.analysis_virtual.template_config import (
    NoaaRtmaRuAnalysisVirtualTemplateConfig,
)


def test_ru_reuses_rtma_grid_variables_and_codecs() -> None:
    hourly = NoaaRtmaAnalysisVirtualTemplateConfig()
    config = NoaaRtmaRuAnalysisVirtualTemplateConfig()
    assert config.dataset_id == "noaa-rtma-ru-analysis-virtual"
    assert config.dataset_attributes.name == "NOAA RTMA-RU analysis, virtual"
    assert config.dataset_attributes.time_resolution == "15 minutes"
    assert config.append_dim_start == pd.Timestamp("2017-12-13T00:00")
    assert config.append_dim_frequency == pd.Timedelta("15min")
    assert config.dims == hourly.dims
    assert len(config.data_vars) == 13
    for actual, expected in zip(config.data_vars, hourly.data_vars, strict=True):
        assert actual.name == expected.name
        assert actual.attrs == expected.attrs
        assert actual.encoding.model_dump_json() == expected.encoding.model_dump_json()
        assert (
            actual.internal_attrs.grib_identity == expected.internal_attrs.grib_identity
        )
        assert actual.internal_attrs.available_from == (
            pd.Timestamp("2020-07-28T00:00")
            if actual.name == "total_cloud_cover_atmosphere"
            else None
        )
    for name in ("y", "x"):
        np.testing.assert_array_equal(
            config.dimension_coordinates()[name], hourly.dimension_coordinates()[name]
        )
    ceiling = next(v for v in config.data_vars if v.name == "cloud_ceiling_height")
    assert ceiling.attrs.standard_name is None
    assert ceiling.attrs.comment is None
