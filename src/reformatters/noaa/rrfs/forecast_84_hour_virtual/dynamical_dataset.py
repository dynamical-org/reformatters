from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast84HourVirtualRegionJob
from .template_config import NoaaRrfsForecast84HourVirtualTemplateConfig


class NoaaRrfsForecast84HourVirtualDataset(NoaaRrfsDataset):
    # Poll before early leads while preserving late-cycle coverage.
    update_offset_minutes: NonNegativeInt = 65
    update_deadline: timedelta = timedelta(minutes=170)
    # These variables are all NaN when temperatures are above freezing everywhere
    # (no snow/freezing precipitation occurs during warm weather conditions).
    allow_all_nan_vars: ClassVar[tuple[str, ...]] = (
        "freezing_rain_run_total_surface",
        "freezing_rain_surface",
        "frozen_precipitation_run_total_surface",
        "frozen_precipitation_surface",
        "new_snow_density_surface",
    )
    template_config: NoaaRrfsForecast84HourVirtualTemplateConfig = (
        NoaaRrfsForecast84HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast84HourVirtualRegionJob] = (
        NoaaRrfsForecast84HourVirtualRegionJob
    )
