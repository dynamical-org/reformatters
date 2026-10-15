from typing import ClassVar

from reformatters.noaa.refs.dynamical_dataset import NoaaRefsDataset

from .region_job import NoaaRefsForecast3HourlyVirtualRegionJob
from .template_config import NoaaRefsForecast3HourlyVirtualTemplateConfig


class NoaaRefsForecast3HourlyVirtualDataset(NoaaRefsDataset):
    # NOAA omits these messages starting with the 2026-10-15 12Z cycle.
    allow_missing_vars: ClassVar[tuple[str, ...]] = (
        "neighborhood_probability_of_freezing_rain_3_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_3_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_3_hour_surface_above_6p35",
        "neighborhood_probability_of_freezing_rain_6_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_6_hour_surface_above_12p7",
        "neighborhood_probability_of_freezing_rain_6_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_6_hour_surface_above_6p35",
        "neighborhood_probability_of_freezing_rain_12_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_12_hour_surface_above_12p7",
        "neighborhood_probability_of_freezing_rain_12_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_12_hour_surface_above_6p35",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_12p7",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_25p4",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_6p35",
    )
    template_config: NoaaRefsForecast3HourlyVirtualTemplateConfig = (
        NoaaRefsForecast3HourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsForecast3HourlyVirtualRegionJob] = (
        NoaaRefsForecast3HourlyVirtualRegionJob
    )
