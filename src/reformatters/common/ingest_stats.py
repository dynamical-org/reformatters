from collections.abc import Sequence
from typing import Protocol

import pandas as pd
import xarray as xr

from reformatters.common.logging import get_logger
from reformatters.common.types import Timedelta, Timestamp

log = get_logger(__name__)


# This Protocol tells the type checker: "Trust me, these objects have time info"
class HasTimeInfo(Protocol):
    init_time: Timestamp
    lead_time: Timedelta


def update_ingested_forecast_length(
    template_ds: xr.Dataset,
    results_coords: Sequence[HasTimeInfo],
) -> None:
    """
    Updates the 'ingested_forecast_length' coordinate in the template dataset.
    """
    if "ingested_forecast_length" not in template_ds.coords:
        log.warning(
            "ingested_forecast_length coordinate not found in template dataset."
        )
        return

    # 1. Group lead times by init_time
    max_lead_per_init: dict[Timestamp, Timedelta] = {}

    for coord in results_coords:
        # We check if we found a new 'longest' forecast for this specific start time
        if (
            coord.init_time not in max_lead_per_init
            or coord.lead_time > max_lead_per_init[coord.init_time]
        ):
            max_lead_per_init[coord.init_time] = coord.lead_time

    # 2. Update the dataset
    for init_time, max_lead in max_lead_per_init.items():
        if init_time in template_ds.coords["init_time"]:
            current_val = template_ds["ingested_forecast_length"].loc[
                {"init_time": init_time}
            ]

            # Use .values and pd.isnull to safely check for NaT (Not a Time)
            if pd.isnull(current_val.values) or max_lead > current_val:
                log.info(
                    f"Updating ingested_forecast_length for {init_time} to {max_lead}"
                )
                template_ds["ingested_forecast_length"].loc[
                    {"init_time": init_time}
                ] = max_lead
