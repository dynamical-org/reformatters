import numpy as np
import pandas as pd

from reformatters.noaa.hrrr.forecast_18_hour_virtual.template_config import (
    NoaaHrrrForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.template_config import (
    NoaaHrrrForecast48HourVirtualTemplateConfig,
)

CONFIG = NoaaHrrrForecast18HourVirtualTemplateConfig()


def test_hourly_inits_and_18_hour_leads() -> None:
    assert CONFIG.append_dim_frequency == pd.Timedelta("1h")
    dim_coords = CONFIG.dimension_coordinates()
    lead_times = dim_coords["lead_time"]
    assert len(lead_times) == 19
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("18h")


def test_full_catalog_matches_48_hour_dataset() -> None:
    full_48h_config = NoaaHrrrForecast48HourVirtualTemplateConfig()
    assert list(CONFIG.data_vars) == list(full_48h_config.data_vars)


def test_expected_forecast_length_is_constant_18h() -> None:
    # Every hourly HRRR cycle reaches f18 across the whole v3+ archive, so there is
    # no per-init variation (unlike the 48-hour dataset's v3 36h -> v4 48h split).
    template = CONFIG.get_template(pd.Timestamp("2018-07-14T00:00"))
    expected = template.to_dataset()["expected_forecast_length"].values
    assert (expected == pd.Timedelta("18h").to_timedelta64()).all()

    coord = next(c for c in CONFIG.coords if c.name == "expected_forecast_length")
    assert coord.attrs.statistics_approximate is not None
    assert coord.attrs.statistics_approximate.min == str(pd.Timedelta("18h"))
    assert coord.attrs.statistics_approximate.max == str(pd.Timedelta("18h"))


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-hrrr-forecast-18-hour-virtual"
    assert attrs.name == "NOAA HRRR forecast, 18 hour, virtual"
    assert attrs.time_resolution == "Forecasts initialized every hour"
    assert attrs.forecast_domain == "Forecast lead time 0-18 hours ahead"


def test_template_starts_at_hrrr_v3_hourly() -> None:
    template = CONFIG.get_template(pd.Timestamp("2018-07-13T15:00"))
    init_times = template.to_dataset().get_index("init_time")
    assert list(init_times) == [
        pd.Timestamp("2018-07-13T12:00"),
        pd.Timestamp("2018-07-13T13:00"),
        pd.Timestamp("2018-07-13T14:00"),
    ]
    assert np.all(np.diff(init_times) == pd.Timedelta("1h").to_timedelta64())
