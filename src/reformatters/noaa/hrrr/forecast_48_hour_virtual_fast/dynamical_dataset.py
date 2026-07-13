from reformatters.noaa.hrrr.forecast_48_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualDataset,
)

from .template_config import NoaaHrrrForecast48HourVirtualFastTemplateConfig


class NoaaHrrrForecast48HourVirtualFastDataset(NoaaHrrrForecast48HourVirtualDataset):
    """NOAA HRRR 48-hour virtual forecast trimmed to the materialized dataset's
    variable set. All operational configuration is inherited from the full dataset,
    so the two products' ingest timing differs only by variable set."""

    template_config: NoaaHrrrForecast48HourVirtualFastTemplateConfig = (
        NoaaHrrrForecast48HourVirtualFastTemplateConfig()
    )
