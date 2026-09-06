import numpy as np
import pandas as pd

from reformatters.noaa.gefs.forecast_16_day_0_5_degree_virtual.template_config import (
    NoaaGefsForecast16Day05DegreeVirtualTemplateConfig,
)

CONFIG = NoaaGefsForecast16Day05DegreeVirtualTemplateConfig()


def test_forecast_time_structure() -> None:
    assert CONFIG.append_dim == "init_time"
    assert CONFIG.append_dim_frequency == pd.Timedelta("6h")
    assert CONFIG.append_dim_start == pd.Timestamp("2020-10-01T00:00")


def test_lead_times_end_where_every_cycle_does() -> None:
    """Every cycle publishes the a and b files through 384 hours; only 00z goes on."""
    lead_times = CONFIG.dimension_coordinates()["lead_time"]
    assert len(lead_times) == 105
    assert lead_times[-1] == pd.Timedelta("384h")


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-gefs-forecast-16-day-0-5-degree-virtual"
    assert attrs.name == "NOAA GEFS forecast, 16 day, 0.5 degree, virtual"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.time_domain == "Forecasts initialized 2020-10-01 00:00:00 UTC to Present"  # fmt: skip
    assert attrs.time_resolution == "Forecasts initialized every 6 hours"
    assert attrs.forecast_domain == "Forecast lead time 0-384 hours ahead"
    # The axis coarsens at 240 hours, so naming only the finer step would be false
    # about the 246-384 hour leads.
    assert attrs.forecast_resolution == (
        "Forecast step 0-240 hours: 3 hourly, 246-384 hours: 6 hourly"
    )


def test_template_carries_the_forecast_coordinates_in_every_group() -> None:
    """A group is opened on its own, so it repeats the shared coordinates."""
    template = CONFIG.get_template(pd.Timestamp("2020-10-01T12:00"))
    for node in template.subtree:
        ds = node.to_dataset()
        assert list(ds.get_index("init_time")) == [
            pd.Timestamp("2020-10-01T00:00"),
            pd.Timestamp("2020-10-01T06:00"),
        ], node.path
        assert ds["valid_time"].dims == ("init_time", "lead_time"), node.path
        assert ds["valid_time"].isel(init_time=0, lead_time=-1) == pd.Timestamp(
            "2020-10-17T00:00"
        ), node.path
        assert set(np.unique(ds["expected_forecast_length"].values)) == {
            pd.Timedelta("384h").to_timedelta64()
        }, node.path
