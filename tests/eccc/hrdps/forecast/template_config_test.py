import numpy as np
import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.eccc.hrdps.forecast.template_config import (
    EcccHrdpsForecastTemplateConfig,
)


def test_template_config_attrs() -> None:
    config = EcccHrdpsForecastTemplateConfig()

    assert config.dims[ROOT] == ("init_time", "lead_time", "y", "x")
    assert config.append_dim == "init_time"
    assert config.append_dim_start == pd.Timestamp("2026-07-09T00:00")
    assert config.append_dim_frequency == pd.Timedelta("6h")

    attrs = config.dataset_attributes
    assert attrs.dataset_id == "eccc-hrdps-forecast"
    assert attrs.license == "ECCC Data Servers End-use Licence v2.1"
    assert "Environment and Climate Change Canada" in attrs.attribution

    var_names = {v.name for v in config.data_vars}
    assert "temperature_2m" in var_names
    assert "wind_u_10m" in var_names
    assert "wind_v_10m" in var_names
    assert "precipitation_surface" in var_names
    assert "downward_short_wave_radiation_flux_surface" in var_names
    assert "pressure_surface" in var_names
    assert "snow_thickness_surface" in var_names


def test_dimension_coordinates() -> None:
    config = EcccHrdpsForecastTemplateConfig()
    dim_coords = config.dimension_coordinates()

    assert set(dim_coords) == {"init_time", "lead_time", "y", "x"}

    lead_times = dim_coords["lead_time"]
    assert len(lead_times) == 49
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("48h")

    assert len(dim_coords["y"]) == 1290
    assert len(dim_coords["x"]) == 2540


def test_derive_coordinates() -> None:
    config = EcccHrdpsForecastTemplateConfig()
    template_ds = config.get_template(config.append_dim_start + pd.Timedelta(days=3))

    assert (
        template_ds.coords["valid_time"]
        == (template_ds.coords["init_time"] + template_ds.coords["lead_time"])
    ).all()
    assert template_ds.coords["valid_time"].dims == ("init_time", "lead_time")
    assert (
        template_ds.coords["expected_forecast_length"].values == np.timedelta64(48, "h")
    ).all()
    assert template_ds.coords["latitude"].dims == ("y", "x")
    assert template_ds.coords["longitude"].dims == ("y", "x")
