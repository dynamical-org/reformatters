from typing import ClassVar

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast84HourVirtualRegionJob
from .template_config import NoaaRrfsForecast84HourVirtualTemplateConfig


class NoaaRrfsForecast84HourVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +108-212 min; polling starts before the first lead.
    _operational_timing: ClassVar[tuple[int, int]] = (100, 135)
    _allow_all_nan_vars: ClassVar[frozenset[str]] = frozenset(
        {
            "specific_humidity_surface",
            "potential_evaporation_rate_surface",
            "potential_evaporation_surface",
        }
    )
    template_config: NoaaRrfsForecast84HourVirtualTemplateConfig = (
        NoaaRrfsForecast84HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast84HourVirtualRegionJob] = (
        NoaaRrfsForecast84HourVirtualRegionJob
    )
