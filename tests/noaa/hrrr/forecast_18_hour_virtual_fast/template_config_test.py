import pandas as pd

from reformatters.noaa.hrrr.forecast_18_hour_virtual.template_config import (
    NoaaHrrrForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.template_config import (
    NoaaHrrrForecast18HourVirtualFastTemplateConfig,
)

CONFIG = NoaaHrrrForecast18HourVirtualFastTemplateConfig()


def test_dataset_attributes_describe_the_moving_window() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-hrrr-forecast-18-hour-virtual-fast"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.name == "NOAA HRRR forecast, 18 hour, fast, virtual"
    assert "NOMADS" in attrs.description
    assert "most recent 3 days" in attrs.description
    assert "moving window" in attrs.time_domain
    sibling = NoaaHrrrForecast18HourVirtualTemplateConfig().dataset_attributes
    assert attrs.description.startswith(sibling.description)


def test_variables_and_structure_match_the_18_hour_dataset() -> None:
    sibling = NoaaHrrrForecast18HourVirtualTemplateConfig()
    assert len(CONFIG.data_vars) == 176
    assert list(CONFIG.data_vars) == list(sibling.data_vars)
    assert CONFIG.dims == sibling.dims
    assert CONFIG.append_dim_frequency == sibling.append_dim_frequency


def test_window_with_partial_final_hour_has_73_labels() -> None:
    end = pd.Timestamp("2026-09-10T12:50")
    init_times = CONFIG.append_dim_coordinates(end)

    assert len(init_times) == 73
    assert init_times[0] == pd.Timestamp("2026-09-07T12:00")
    assert init_times[-1] == pd.Timestamp("2026-09-10T12:00")


def test_window_ending_on_hour_has_72_labels() -> None:
    end = pd.Timestamp("2026-09-10T12:00")
    init_times = CONFIG.append_dim_coordinates(end)

    assert len(init_times) == 72
    assert init_times[0] == pd.Timestamp("2026-09-07T12:00")
    assert init_times[-1] == pd.Timestamp("2026-09-10T11:00")


def test_window_never_starts_before_append_dim_start() -> None:
    init_times = CONFIG.append_dim_coordinates(pd.Timestamp("2026-09-01T02:00"))

    assert list(init_times) == [
        pd.Timestamp("2026-09-01T00:00"),
        pd.Timestamp("2026-09-01T01:00"),
    ]


def test_time_coordinate_statistics_describe_retention() -> None:
    coords = {coord.name: coord for coord in CONFIG.coords}

    for name in ("init_time", "valid_time"):
        statistics = coords[name].attrs.statistics_approximate
        assert statistics is not None
        assert statistics.min == "Present - 72 hours"
