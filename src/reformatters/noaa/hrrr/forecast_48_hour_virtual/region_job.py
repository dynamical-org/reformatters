from typing import ClassVar

import pandas as pd

from reformatters.common.materialized_update_trigger import MaterializedUpdateTrigger
from reformatters.common.types import Timedelta
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    SOURCE_AVAILABILITY,
)
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualRegionJob,
)


class NoaaHrrrForecast48HourVirtualRegionJob(NoaaHrrrForecastVirtualRegionJob):
    """RegionJob for the HRRR 48-hour virtual forecast dataset."""

    # 14h = two 6h cycles back + ~2h publication slack, so a couple of missed runs
    # still self-heal.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("14h")
    materialized_update_triggers: ClassVar[tuple[MaterializedUpdateTrigger, ...]] = (
        MaterializedUpdateTrigger(
            source_cronjob="noaa-hrrr-forecast-48-hour-virtual-update",
            target_cronjob="noaa-hrrr-forecast-48-hour-update",
            availability=SOURCE_AVAILABILITY,
        ),
    )
