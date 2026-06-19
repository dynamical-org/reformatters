from collections.abc import Mapping, Sequence
from typing import Protocol

import xarray as xr

from reformatters.common.logging import get_logger
from reformatters.common.types import Timedelta, Timestamp

log = get_logger(__name__)


class DeterministicForecastSourceFileCoord(Protocol):
    init_time: Timestamp
    lead_time: Timedelta


def update_ingested_forecast_length(
    template_ds: xr.Dataset,
    results_coords: Mapping[str, Sequence[DeterministicForecastSourceFileCoord]],
) -> xr.Dataset:
    """
    Updates the 'ingested_forecast_length' coordinate in the template dataset.

    The maximum processed lead time across all variables is set as the
    ingested_forecast_length. This can hide the nuance of a specific variable
    having fewer lead times processed than others.
    """
    assert "ingested_forecast_length" in template_ds.coords

    max_lead_per_init: dict[Timestamp, Timedelta] = {}

    for coords_seq in results_coords.values():
        for coord in coords_seq:
            if (
                coord.init_time not in max_lead_per_init
                or coord.lead_time > max_lead_per_init[coord.init_time]
            ):
                max_lead_per_init[coord.init_time] = coord.lead_time

    for init_time, max_lead in max_lead_per_init.items():
        template_ds["ingested_forecast_length"].loc[{"init_time": init_time}] = max_lead
    return template_ds
