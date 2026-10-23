from datetime import timedelta

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsEnsForecastVirtualRegionJob
from .template_config import NoaaRrfsEnsForecastVirtualTemplateConfig


class NoaaRrfsEnsForecastVirtualDataset(NoaaRrfsDataset):
    # Poll before early leads while preserving late-cycle coverage.
    update_offset_minutes: NonNegativeInt = 70
    # Poll until init + 6 h to cover late cycles, ending before the next fire.
    update_deadline: timedelta = timedelta(minutes=295)
    template_config: NoaaRrfsEnsForecastVirtualTemplateConfig = (
        NoaaRrfsEnsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsEnsForecastVirtualRegionJob] = (
        NoaaRrfsEnsForecastVirtualRegionJob
    )
