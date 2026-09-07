import numpy as np
import pandas as pd

from reformatters.noaa.gefs.forecast_16_day_0_5_degree_virtual.template_config import (
    NoaaGefsForecast16Day05DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.forecast_35_day_0_5_degree_virtual.template_config import (
    NoaaGefsForecast35Day05DegreeVirtualTemplateConfig,
)

CONFIG = NoaaGefsForecast35Day05DegreeVirtualTemplateConfig()


def test_only_the_00z_cycle_reaches_this_forecast_length() -> None:
    """The a and b files run to 840 hours from 00z alone, so the init axis is daily and
    every one of its labels is a 00z cycle."""
    assert CONFIG.append_dim == "init_time"
    assert CONFIG.append_dim_frequency == pd.Timedelta("24h")
    assert CONFIG.append_dim_start == pd.Timestamp("2020-10-01T00:00")

    init_times = CONFIG.append_dim_coordinates(pd.Timestamp("2020-10-05T00:00"))
    assert set(init_times.hour) == {0}


def test_lead_times_reach_the_end_of_the_00z_extension() -> None:
    lead_times = CONFIG.dimension_coordinates()["lead_time"]
    assert len(lead_times) == 181
    assert lead_times[-1] == pd.Timedelta("840h")


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-gefs-forecast-35-day-0-5-degree-virtual"
    assert attrs.name == "NOAA GEFS forecast, 35 day, 0.5 degree, virtual"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.time_domain == "Forecasts initialized 2020-10-01 00:00:00 UTC to Present"  # fmt: skip
    assert attrs.time_resolution == "Forecasts initialized every 24 hours"
    assert attrs.forecast_domain == "Forecast lead time 0-840 hours ahead"


def test_the_init_axis_is_a_subset_of_the_16_day_dataset_s() -> None:
    """The two datasets deliberately overlap: this one re-serves every 00z init the 16
    day dataset holds, out to a lead time that one stops short of. A reader comparing
    them at a shared init must find the same labels on both sides."""
    config_16_day = NoaaGefsForecast16Day05DegreeVirtualTemplateConfig()
    end = pd.Timestamp("2020-10-08T00:00")
    assert set(CONFIG.append_dim_coordinates(end)) < set(
        config_16_day.append_dim_coordinates(end)
    )
    shared_leads = config_16_day.dimension_coordinates()["lead_time"]
    assert (
        list(shared_leads)
        == list(CONFIG.dimension_coordinates()["lead_time"])[: len(shared_leads)]
    )


def test_template_carries_the_forecast_coordinates_in_every_group() -> None:
    """A group is opened on its own, so it repeats the shared coordinates."""
    template = CONFIG.get_template(pd.Timestamp("2020-10-03T00:00"))
    for node in template.subtree:
        ds = node.to_dataset()
        assert list(ds.get_index("init_time")) == [
            pd.Timestamp("2020-10-01T00:00"),
            pd.Timestamp("2020-10-02T00:00"),
        ], node.path
        assert ds["valid_time"].dims == ("init_time", "lead_time"), node.path
        assert ds["valid_time"].isel(init_time=0, lead_time=-1) == pd.Timestamp(
            "2020-11-05T00:00"
        ), node.path
        assert set(np.unique(ds["expected_forecast_length"].values)) == {
            pd.Timedelta("840h").to_timedelta64()
        }, node.path
