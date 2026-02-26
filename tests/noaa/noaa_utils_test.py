import pytest

from reformatters.common.pydantic import replace
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.noaa_utils import has_hour_0_values


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
    assert has_hour_0_values(_get_var(template_config, "temperature_2m")) is True


def test_has_hour_0_values_avg_var(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    # avg step_type, no override → False
    assert (
        has_hour_0_values(_get_var(template_config, "precipitation_surface")) is False
    )


def test_has_hour_0_values_instant_var_with_override_false(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    # instant step_type but _has_hour_0_values=False override → False
    assert (
        has_hour_0_values(_get_var(template_config, "categorical_rain_surface"))
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
    assert has_hour_0_values(overridden) is True
