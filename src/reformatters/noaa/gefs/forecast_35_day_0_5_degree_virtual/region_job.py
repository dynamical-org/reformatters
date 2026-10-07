from typing import ClassVar

import pandas as pd

from reformatters.common.materialized_update_trigger import MaterializedUpdateTrigger
from reformatters.common.types import Timedelta
from reformatters.noaa.gefs.forecast_35_day.dynamical_dataset import SOURCE_AVAILABILITY
from reformatters.noaa.gefs.gefs_config_models import GEFS_EXTENSION_REQUEST_MIN_AGE
from reformatters.noaa.gefs.virtual_region_job import NoaaGefsForecastVirtualRegionJob


class NoaaGefsForecast35Day05DegreeVirtualRegionJob(NoaaGefsForecastVirtualRegionJob):
    """RegionJob for the GEFS 35 day 0.5 degree virtual forecast dataset."""

    # Three update cron fires' span, so two consecutive missed runs still self-heal.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("72h")
    materialized_update_triggers: ClassVar[tuple[MaterializedUpdateTrigger, ...]] = (
        MaterializedUpdateTrigger(
            source_cronjob="noaa-gefs-forecast-35-day-0-5-virtual-update",
            target_cronjob="noaa-gefs-forecast-35-day-update",
            availability=SOURCE_AVAILABILITY.model_copy(update={"lead_hours": 840}),
            init_offset=pd.Timedelta("1D"),
            min_init_age=GEFS_EXTENSION_REQUEST_MIN_AGE,
        ),
        MaterializedUpdateTrigger(
            source_cronjob="noaa-gefs-forecast-35-day-0-5-virtual-update",
            target_cronjob="noaa-gefs-forecast-35-day-update",
            availability=SOURCE_AVAILABILITY,
        ),
    )
