from collections.abc import Mapping

import pandas as pd
import xarray as xr

from reformatters.common.ingest_stats import update_ingested_forecast_length
from reformatters.common.region_job import CoordinateValueOrRange, SourceFileCoord
from reformatters.common.types import Dim, Timedelta, Timestamp


class MockSourceFileCoord(SourceFileCoord):
    init_time: Timestamp
    lead_time: Timedelta

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {}


def test_update_ingested_forecast_length_simple() -> None:
    init_times = [
        pd.Timestamp("2025-01-01 12:00"),
        pd.Timestamp("2025-01-01 18:00"),
    ]

    empty_deltas = pd.to_timedelta([pd.NaT, pd.NaT]).to_numpy()  # type: ignore[call-overload]

    ds = xr.Dataset(
        coords={
            "init_time": init_times,
            "ingested_forecast_length": (("init_time",), empty_deltas),
        }
    )

    coord1 = MockSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01 12:00"),
        lead_time=pd.Timedelta(hours=6),
    )
    coord2 = MockSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01 18:00"),
        lead_time=pd.Timedelta(hours=48),
    )

    results = {"var1": [coord1, coord2]}
    ds = update_ingested_forecast_length(ds, results)

    assert ds["ingested_forecast_length"].sel(
        init_time="2025-01-01 12:00"
    ).values == pd.Timedelta(hours=6)
    assert ds["ingested_forecast_length"].sel(
        init_time="2025-01-01 18:00"
    ).values == pd.Timedelta(hours=48)


def test_update_ingested_forecast_length_update_existing() -> None:
    init_time = pd.Timestamp("2025-01-01 12:00")

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

    ds = update_ingested_forecast_length(ds, {"var1": [new_coord]})

    assert ds["ingested_forecast_length"].sel(
        init_time=init_time
    ).values == pd.Timedelta(hours=12)
