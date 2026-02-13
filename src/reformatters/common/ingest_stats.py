from collections.abc import Sequence
from typing import Protocol

import xarray as xr

from reformatters.common.logging import get_logger
from reformatters.common.types import Timedelta, Timestamp

log = get_logger(__name__)


class DeterministicForecastSourceFileCoord(Protocol):
    init_time: Timestamp
    lead_time: Timedelta


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
    assert "ingested_forecast_length" in template_ds.coords

    max_lead_per_init: dict[Timestamp, Timedelta] = {}

    for coord in results_coords:
        if (
            coord.init_time not in max_lead_per_init
            or coord.lead_time > max_lead_per_init[coord.init_time]
        ):
            max_lead_per_init[coord.init_time] = coord.lead_time

    for init_time, max_lead in max_lead_per_init.items():
        if init_time in template_ds.coords["init_time"]:
            template_ds["ingested_forecast_length"].loc[{"init_time": init_time}] = (
                max_lead
            )
