import numpy as np
import pandas as pd

from reformatters.noaa.rtma.analysis_virtual.template_config import (
    NoaaRtmaAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.urma.analysis_virtual.template_config import (
    NoaaUrmaAnalysisVirtualTemplateConfig,
)


def test_urma_reuses_first_thirteen_fields_and_defines_wave_metadata() -> None:
    hourly = NoaaRtmaAnalysisVirtualTemplateConfig()
    config = NoaaUrmaAnalysisVirtualTemplateConfig()
    assert config.dataset_id == "noaa-urma-analysis-virtual"
    assert config.dataset_attributes.name == "NOAA URMA analysis, virtual"
    assert config.append_dim_start == pd.Timestamp("2017-05-02T00:00")
    assert config.append_dim_frequency == pd.Timedelta("1h")
    assert config.dims == hourly.dims
    assert len(config.data_vars) == 14
    for actual, expected in zip(config.data_vars[:13], hourly.data_vars, strict=True):
        assert actual.name == expected.name
        assert actual.attrs == expected.attrs
        assert actual.internal_attrs == expected.internal_attrs
        assert actual.encoding.model_dump_json() == expected.encoding.model_dump_json()
    wave = config.data_vars[-1]
    assert wave.name == "significant_wave_height_surface"
    assert wave.internal_attrs.grib_identity == (10, 0, 3, 1, 1.0)
    assert wave.internal_attrs.available_from == pd.Timestamp("2018-12-03T12:00")
    assert wave.internal_attrs.requires_bitmap
    assert wave.internal_attrs.keep_mantissa_bits == 7
    assert wave.attrs.short_name == "swh"
    assert wave.attrs.long_name == "Significant height of combined wind waves and swell"
    assert wave.attrs.standard_name == "sea_surface_wave_significant_height"
    assert wave.attrs.units == "m"
    assert (
        wave.attrs.comment
        == "NaN where no wave analysis is made, such as over land and ice-covered water."
    )
    assert wave.encoding.dtype == "float64"
    assert np.isnan(wave.encoding.fill_value)
    assert wave.encoding.chunks == (1, 1597, 2345)
    assert wave.encoding.shards is None
    assert wave.encoding.serializer is not None
    assert wave.encoding.serializer["configuration"]["var"] == "HTSGW"
