from typing import ClassVar

from reformatters.noaa.refs.dynamical_dataset import NoaaRefsDataset

from .region_job import NoaaRefsForecastHourlyVirtualRegionJob
from .template_config import NoaaRefsForecastHourlyVirtualTemplateConfig


class NoaaRefsForecastHourlyVirtualDataset(NoaaRefsDataset):
    # NOAA omits these messages starting with the 2026-10-15 12Z cycle.
    allow_missing_vars: ClassVar[tuple[str, ...]] = (
        "neighborhood_probability_of_freezing_rain_1_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_1_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_1_hour_surface_above_6p35",
    )
    template_config: NoaaRefsForecastHourlyVirtualTemplateConfig = (
        NoaaRefsForecastHourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsForecastHourlyVirtualRegionJob] = (
        NoaaRefsForecastHourlyVirtualRegionJob
    )
