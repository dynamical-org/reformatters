import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.eccc.hrdps.analysis.template_config import (
    EcccHrdpsAnalysisTemplateConfig,
)
from reformatters.eccc.hrdps.forecast.template_config import (
    EcccHrdpsForecastTemplateConfig,
)


def test_template_config_attrs() -> None:
    config = EcccHrdpsAnalysisTemplateConfig()

    assert config.dims[ROOT] == ("time", "y", "x")
    assert config.append_dim == "time"
    assert config.append_dim_start == pd.Timestamp("2026-07-09T00:00")
    assert config.append_dim_frequency == pd.Timedelta("1h")

    attrs = config.dataset_attributes
    assert attrs.dataset_id == "eccc-hrdps-analysis"
    assert attrs.license == "ECCC Data Servers End-use Licence v2.1"
    assert "Environment and Climate Change Canada" in attrs.attribution


def test_data_vars_match_forecast_dataset() -> None:
    analysis_vars = {
        v.name: v.attrs for v in EcccHrdpsAnalysisTemplateConfig().data_vars
    }
    forecast_vars = {
        v.name: v.attrs for v in EcccHrdpsForecastTemplateConfig().data_vars
    }
    assert analysis_vars == forecast_vars


def test_dimension_coordinates() -> None:
    config = EcccHrdpsAnalysisTemplateConfig()
    dim_coords = config.dimension_coordinates()

    assert set(dim_coords) == {"time", "y", "x"}
    assert len(dim_coords["y"]) == 1290
    assert len(dim_coords["x"]) == 2540


def test_derive_coordinates() -> None:
    config = EcccHrdpsAnalysisTemplateConfig()
    template_ds = config.get_template(config.append_dim_start + pd.Timedelta("6h"))

    assert template_ds.coords["latitude"].dims == ("y", "x")
    assert template_ds.coords["longitude"].dims == ("y", "x")
    assert "spatial_ref" in template_ds.coords
