import re

import pandas as pd
import pytest
from pydantic import ValidationError

from reformatters.noaa.gefs.gefs_config_models import GEFSSourceFileType
from reformatters.noaa.gefs.virtual_template_config import (
    NoaaGefsForecastVirtualTemplateConfig,
)
from reformatters.noaa.noaa_grib_index import grib_index_window_str

THREE_HOURLY_TO_240 = pd.timedelta_range("0h", "240h", freq="3h")


def tail(start: str, end: str, freq: str) -> pd.TimedeltaIndex:
    """THREE_HOURLY_TO_240 continued by lead times from `start` to `end` every `freq`."""
    return pd.TimedeltaIndex(
        THREE_HOURLY_TO_240.append(pd.timedelta_range(start, end, freq=freq))
    )


def config_with_lead_times(
    leads: pd.TimedeltaIndex,
) -> NoaaGefsForecastVirtualTemplateConfig:
    class Config(NoaaGefsForecastVirtualTemplateConfig):
        source_file_types: tuple[GEFSSourceFileType, ...] = ("s",)

        def lead_times(self) -> pd.TimedeltaIndex:
            return leads

    return Config(forecast_length=leads[-1])


def test_ten_day_lead_times_are_described() -> None:
    """3 hourly through the 240 hour lead where the 0.25 degree s file ends."""
    assert len(THREE_HOURLY_TO_240) == 81
    config_with_lead_times(THREE_HOURLY_TO_240)


def test_sixteen_day_lead_times_are_described() -> None:
    """3 hourly to 240 hours, then 6 hourly to the 384 hour lead."""
    leads = tail("246h", "384h", "6h")
    assert len(leads) == 105
    config_with_lead_times(leads)


def test_thirty_five_day_lead_times_are_described() -> None:
    """3 hourly to 240 hours, then 6 hourly to the 840 hour lead."""
    leads = tail("246h", "840h", "6h")
    assert len(leads) == 181
    config_with_lead_times(leads)


def test_lead_times_the_window_comments_cannot_describe_are_rejected() -> None:
    """A 4 hourly tail puts lead times off both sequences window_comments enumerates."""
    leads = tail("244h", "384h", "4h")
    assert len(leads) == 117
    with pytest.raises(
        ValidationError,
        match="window_comments does not describe lead time 244 hours, whose window is 4 hours",
    ):
        config_with_lead_times(leads)


def test_forecast_length_shorter_than_the_first_window_is_described() -> None:
    """A single 3 hour lead time, the shortest domain the wording has to cover."""
    config = NoaaGefsForecastVirtualTemplateConfig(
        source_file_types=("s",), forecast_length=pd.Timedelta("3h")
    )
    assert list(config.lead_times()) == [pd.Timedelta("0h"), pd.Timedelta("3h")]


def test_window_comments_name_the_lead_times_that_carry_each_window() -> None:
    """Every "N hour period (lead times a, b, c, ... hours)" clause the wording emits,
    read back and checked against the window string the source's own idx line carries
    at each named lead. A clause naming the wrong leads mislabels every windowed value
    a reader averages or differences."""
    config = config_with_lead_times(THREE_HOURLY_TO_240)
    windowed = next(var for var in config.data_vars if var.attrs.step_type == "accum")

    clauses = re.findall(
        r"(\d+) hour period \(lead times ([\d, ]+), \.\.\. hours\)",
        config.window_comments["accum"],
    )
    assert len(clauses) == 2, config.window_comments["accum"]

    for window_hours, named_leads in clauses:
        for lead in [int(hours) for hours in named_leads.split(", ")]:
            source_window = grib_index_window_str(windowed, lead)
            match = re.fullmatch(r"(\d+)-(\d+) hour acc fcst", source_window)
            assert match is not None, (lead, source_window)
            start, end = int(match.group(1)), int(match.group(2))
            assert end == lead, (lead, source_window)
            assert end - start == int(window_hours), (lead, source_window)
