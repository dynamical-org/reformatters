import pandas as pd
import pytest

from reformatters.common.pydantic import replace
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar


@pytest.fixture
def template_config() -> NoaaHrrrForecast48HourTemplateConfig:
    return NoaaHrrrForecast48HourTemplateConfig()


def _get_var(
    template_config: NoaaHrrrForecast48HourTemplateConfig, name: str
) -> NoaaHrrrDataVar:
    return next(v for v in template_config.data_vars if v.name == name)


def test_has_hour_0_values_instant_var(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    # instant step_type, no override → True
    assert _get_var(template_config, "temperature_2m").has_hour_0_values() is True


def test_has_hour_0_values_avg_var(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    # avg step_type, no override → False
    assert (
        _get_var(template_config, "precipitation_surface").has_hour_0_values() is False
    )


def test_has_hour_0_values_instant_var_with_override_false(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    # instant step_type but _has_hour_0_values=False override → False
    assert (
        _get_var(template_config, "categorical_rain_surface").has_hour_0_values()
        is False
    )


def test_has_hour_0_values_avg_var_with_override_true(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    # avg step_type but _has_hour_0_values=True override → True
    var = _get_var(template_config, "precipitation_surface")
    overridden = replace(
        var,
        internal_attrs=replace(var.internal_attrs, hour_0_values_override=True),
    )
    assert overridden.has_hour_0_values() is True


def test_analysis_lead_time_follows_hour_0_values(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    assert _get_var(template_config, "temperature_2m").analysis_lead_time() == (
        pd.Timedelta("0h")
    )
    assert _get_var(template_config, "precipitation_surface").analysis_lead_time() == (
        pd.Timedelta("1h")
    )


def test_analysis_lead_time_unusable_hour_0(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    var = _get_var(template_config, "dew_point_temperature_2m")
    # The forecast still serves hour 0; only the analysis avoids it.
    assert var.has_hour_0_values() is True
    assert var.analysis_lead_time() == pd.Timedelta("1h")


def test_analysis_lead_time_unusable_hour_0_without_hour_0_values(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    var = _get_var(template_config, "precipitation_surface")
    flagged = replace(
        var,
        internal_attrs=replace(var.internal_attrs, analysis_hour_0_unusable=True),
    )
    assert flagged.analysis_lead_time() == pd.Timedelta("1h")
