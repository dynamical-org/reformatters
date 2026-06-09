import pandas as pd

from reformatters.nasa.imerg.analysis_late_v7.template_config import (
    NasaImergAnalysisLateV7TemplateConfig,
)


def test_template_config_attrs() -> None:
    config = NasaImergAnalysisLateV7TemplateConfig()
    assert config.dataset_attributes.dataset_id == "nasa-imerg-analysis-late-v7"
    assert config.dims == ("time", "latitude", "longitude")
    assert config.append_dim == "time"
    assert config.append_dim_start == pd.Timestamp("1998-01-01T00:00")
    assert config.append_dim_frequency == pd.Timedelta("30min")


def test_data_vars_match_shared_definition() -> None:
    config = NasaImergAnalysisLateV7TemplateConfig()
    assert {v.name for v in config.data_vars} == {
        "precipitation_surface",
        "probability_of_liquid_precipitation_surface",
        "precipitation_quality_index_surface",
    }
    precip = next(v for v in config.data_vars if v.name == "precipitation_surface")
    assert precip.encoding.chunks == (1440, 45, 45)
    assert precip.encoding.shards == (1440, 450, 450)
