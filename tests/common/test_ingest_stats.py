from collections.abc import Mapping
from typing import Any, cast

import pandas as pd
import xarray as xr

from reformatters.common.ingest_stats import update_ingested_forecast_length
from reformatters.common.region_job import CoordinateValueOrRange, SourceFileCoord
from reformatters.common.types import Dim, Timedelta, Timestamp


# --- Mock Class ---
class MockSourceFileCoord(SourceFileCoord):
    init_time: Timestamp
    lead_time: Timedelta

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {}


# ------------------


def test_update_ingested_forecast_length_simple() -> None:
    # 1. Setup a dummy dataset
    init_times = [
        pd.Timestamp("2025-01-01 12:00"),
        pd.Timestamp("2025-01-01 18:00"),
    ]

    # We use 'cast' to silence the strict type checker here
    empty_deltas = pd.to_timedelta([pd.NaT, pd.NaT]).values

    ds = xr.Dataset(
        coords={
            "init_time": init_times,
            "ingested_forecast_length": (("init_time",), empty_deltas),
        }
    )

    # 2. Setup the Results
    coord1 = MockSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01 12:00"),
        lead_time=pd.Timedelta(hours=6),
    )
    coord2 = MockSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01 18:00"),
        lead_time=pd.Timedelta(hours=48),
    )

    results = [coord1, coord2]

    # 3. Run the function
    update_ingested_forecast_length(ds, results)

    # 4. Check the answers
    assert ds["ingested_forecast_length"].sel(
        init_time="2025-01-01 12:00"
    ).values == pd.Timedelta(hours=6)
    assert ds["ingested_forecast_length"].sel(
        init_time="2025-01-01 18:00"
    ).values == pd.Timedelta(hours=48)


def test_update_ingested_forecast_length_update_existing() -> None:
    init_time = pd.Timestamp("2025-01-01 12:00")

    # Start with 6 hours already recorded
    ds = xr.Dataset(
        coords={
            "init_time": [init_time],
            "ingested_forecast_length": (("init_time",), [pd.Timedelta(hours=6)]),
        }
    )

    new_coord = MockSourceFileCoord(
        init_time=init_time,
        lead_time=pd.Timedelta(hours=12),
    )

    update_ingested_forecast_length(ds, [new_coord])

    assert ds["ingested_forecast_length"].sel(
        init_time=init_time
    ).values == pd.Timedelta(hours=12)
