import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.ecmwf.ifs_ens.forecast_46_day_6_hourly_1_5_degree.template_config import (
    EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig,
)


def test_template_has_the_native_six_hourly_axis_and_variables() -> None:
    config = EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig()

    assert config.dims == {
        ROOT: (
            "init_time",
            "lead_time",
            "ensemble_member",
            "latitude",
            "longitude",
        )
    }
    lead_time = config.dimension_coordinates()["lead_time"]
    assert len(lead_time) == 185
    assert lead_time.equals(pd.timedelta_range("0h", "1104h", freq="6h"))
    assert {data_var.name for data_var in config.data_vars} == {
        "precipitation_surface",
        "wind_u_10m",
        "wind_v_10m",
        "maximum_temperature_2m",
        "minimum_temperature_2m",
    }


def test_template_uses_the_layout_derived_for_185_lead_times() -> None:
    config = EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig()

    for data_var in config.data_vars:
        assert data_var.encoding.chunks == (1, 37, 101, 32, 30)
        assert data_var.encoding.shards == (1, 185, 101, 128, 120)


def test_precipitation_is_a_six_hour_rate_from_a_running_total() -> None:
    config = EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig()
    precipitation = next(
        data_var
        for data_var in config.data_vars
        if data_var.name == "precipitation_surface"
    )

    assert precipitation.attrs.comment == (
        "Average precipitation rate over the previous 6 hours. Units equivalent to mm/s."
    )
    assert precipitation.internal_attrs.window_reset_frequency == pd.Timedelta.max
    assert precipitation.internal_attrs.hour_0_values_override is True
    assert precipitation.has_hour_0_values()
    assert not precipitation.stores_hour_0_values()


def test_extremes_are_native_six_hour_windows() -> None:
    config = EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig()
    extremes = [
        data_var
        for data_var in config.data_vars
        if data_var.name in {"maximum_temperature_2m", "minimum_temperature_2m"}
    ]

    assert {data_var.attrs.comment for data_var in extremes} == {
        "Maximum temperature at 2 metres over the previous 6 hours.",
        "Minimum temperature at 2 metres over the previous 6 hours.",
    }
    assert all(
        data_var.internal_attrs.sub_step_reduction is None for data_var in extremes
    )
    assert all(not data_var.has_hour_0_values() for data_var in extremes)


def test_dataset_metadata_identifies_the_six_hourly_product() -> None:
    attrs = EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig().dataset_attributes

    assert attrs.dataset_id == ("ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree")
    assert attrs.name == "ECMWF IFS ENS forecast, 46 day, 6 hourly, 1.5 degree"
    assert attrs.forecast_resolution == "Forecast step 6 hourly"
