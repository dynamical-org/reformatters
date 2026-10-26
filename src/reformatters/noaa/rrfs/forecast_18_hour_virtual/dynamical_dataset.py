from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast18HourVirtualRegionJob
from .template_config import NoaaRrfsForecast18HourVirtualTemplateConfig


class NoaaRrfsForecast18HourVirtualDataset(NoaaRrfsDataset):
    # Poll before early leads while preserving late-cycle coverage.
    update_offset_minutes: NonNegativeInt = 65
    # Poll until init + 3 h 50 min to cover late cycles, ending before the next fire.
    update_deadline: timedelta = timedelta(minutes=170)
    # Snow density is all NaN when the air 300 m above ground is 5 °C or warmer everywhere.
    allow_all_nan_vars: ClassVar[tuple[str, ...]] = ("new_snow_density_surface",)
    template_config: NoaaRrfsForecast18HourVirtualTemplateConfig = (
        NoaaRrfsForecast18HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast18HourVirtualRegionJob] = (
        NoaaRrfsForecast18HourVirtualRegionJob
    )
