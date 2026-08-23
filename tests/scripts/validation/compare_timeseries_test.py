from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.validation.compare_timeseries import (
    _load_timeseries_for_var,
    select_time_period_for_comparison,
)
from scripts.validation.utils import RunContext
from tests.scripts.validation.grids import (
    ASCENDING_LAT,
    DESCENDING_LAT,
    LON_0_360,
    LON_180,
)

_TIMES = pd.date_range("2024-01-01", periods=3, freq="6h")


def _timeseries_ds(longitude: np.ndarray, latitude: np.ndarray) -> xr.Dataset:
    """Every cell's value is its longitude normalized to 0-360, so the same physical
    location has the same value in either longitude convention."""
    values = np.broadcast_to(
        (longitude % 360)[None, None, :], (_TIMES.size, latitude.size, longitude.size)
    )
    return xr.Dataset(
        {"t": (("time", "latitude", "longitude"), values.astype("float64"))},
        coords={"time": _TIMES, "latitude": latitude, "longitude": longitude},
    )


def _ctx(validation_ds: xr.Dataset, reference_ds: xr.Dataset) -> RunContext:
    lat = validation_ds.get_index("latitude")
    lon = validation_ds.get_index("longitude")
    point1_lat, point1_lon = float(lat[27]), float(lon[45])
    point2_lat, point2_lon = float(lat[9]), float(lon[63])
    return RunContext(
        output_dir=Path("unused"),
        validation_url="unused",
        reference_url="unused",
        validation_ds=validation_ds,
        reference_ds=reference_ds,
        started_at=pd.Timestamp.now(tz="UTC"),
        point1_sel={"latitude": 27, "longitude": 45},
        point2_sel={"latitude": 9, "longitude": 63},
        point1_lat=point1_lat,
        point1_lon=point1_lon,
        point2_lat=point2_lat,
        point2_lon=point2_lon,
        ensemble_member=None,
        variables=["t"],
    )


def _selected_reference_longitudes(
    validation_ds: xr.Dataset, reference_ds: xr.Dataset
) -> tuple[set[float], set[float]]:
    ctx = _ctx(validation_ds, reference_ds)
    _, ref_p1, _, ref_p2 = _load_timeseries_for_var(
        "t", ctx, validation_ds, reference_ds, {}
    )
    assert ref_p1 is not None
    assert ref_p2 is not None
    return set(ref_p1.values.tolist()), set(ref_p2.values.tolist())


def test_reference_point_selection_spans_a_0_360_dataset() -> None:
    # Without a convention match, a nearest-selection at 225 and 315 both clamp to the
    # reference's eastern edge (175), silently comparing the wrong half of the globe.
    validation_ds = _timeseries_ds(LON_0_360, ASCENDING_LAT)
    p1, p2 = _selected_reference_longitudes(
        validation_ds, _timeseries_ds(LON_180, DESCENDING_LAT)
    )
    assert p1 == {225.0}
    assert p2 == {315.0}


def test_reference_point_selection_unchanged_for_a_180_dataset() -> None:
    validation_ds = _timeseries_ds(LON_180, DESCENDING_LAT)
    p1, p2 = _selected_reference_longitudes(
        validation_ds, _timeseries_ds(LON_180, DESCENDING_LAT)
    )
    assert p1 == {float(LON_180[45] % 360)}
    assert p2 == {float(LON_180[63] % 360)}


def _reference_dataset() -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=200, freq="6h")
    return xr.Dataset(
        {"temperature_2m": (("time",), np.zeros(time.size))},
        coords={"time": time},
    )


def _forecast_dataset() -> xr.Dataset:
    init_time = pd.date_range("2020-01-01", periods=8, freq="D")
    lead_time = pd.to_timedelta([0, 6, 12], unit="h")
    valid_time = xr.DataArray(
        init_time.values[:, None] + lead_time.values[None, :],
        dims=("init_time", "lead_time"),
    )
    return xr.Dataset(
        {
            "temperature_2m": (
                ("init_time", "lead_time"),
                np.zeros((init_time.size, lead_time.size)),
            )
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "valid_time": valid_time,
        },
    )


def _analysis_dataset() -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=200, freq="6h")
    return xr.Dataset(
        {"temperature_2m": (("time",), np.zeros(time.size))},
        coords={"time": time},
    )


def test_init_time_pins_the_forecast() -> None:
    _, _, title_suffix, time_coord, _ = select_time_period_for_comparison(
        _forecast_dataset(), _reference_dataset(), init_time="2020-01-05T00:00"
    )
    assert title_suffix == "init=2020-01-05T00:00"
    assert time_coord == "valid_time"


def test_time_pins_the_analysis_window_start() -> None:
    _, _, title_suffix, time_coord, _ = select_time_period_for_comparison(
        _analysis_dataset(), _reference_dataset(), time="2020-01-10T00:00"
    )
    assert title_suffix.startswith("2020-01-10T00:00 - ")
    assert time_coord == "time"


def test_analysis_pin_past_the_last_full_window_clamps_to_it() -> None:
    """A pinned time inside the final 10 days still yields a full-width window."""
    ds = _analysis_dataset()
    last = pd.Timestamp(ds.time.max().item())
    _, _, title_suffix, _, _ = select_time_period_for_comparison(
        ds, _reference_dataset(), time=last.isoformat()
    )
    start = pd.Timestamp(title_suffix.split(" - ")[0])
    assert start + pd.Timedelta(days=10) <= last


def test_unpinned_selection_stays_within_the_archive() -> None:
    ds = _analysis_dataset()
    _, _, title_suffix, _, _ = select_time_period_for_comparison(
        ds, _reference_dataset()
    )
    start, end = (pd.Timestamp(part) for part in title_suffix.split(" - "))
    assert pd.Timestamp(ds.time.min().item()) <= start
    assert end <= pd.Timestamp(ds.time.max().item())
