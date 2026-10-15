from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsEnsForecastVirtualRegionJob
from .template_config import NoaaRrfsEnsForecastVirtualTemplateConfig


class NoaaRrfsEnsForecastVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +81-212 min; polling starts before the first lead.
    update_offset_minutes: NonNegativeInt = 75
    update_deadline: timedelta = timedelta(minutes=160)
    # NOAA omits these messages starting with the 2026-10-15 12Z cycle.
    allow_missing_vars: ClassVar[tuple[str, ...]] = (
        "freezing_rain_run_total_surface",
        "freezing_rain_surface",
    )
    template_config: NoaaRrfsEnsForecastVirtualTemplateConfig = (
        NoaaRrfsEnsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsEnsForecastVirtualRegionJob] = (
        NoaaRrfsEnsForecastVirtualRegionJob
    )
