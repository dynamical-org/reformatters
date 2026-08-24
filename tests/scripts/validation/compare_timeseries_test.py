import numpy as np
import pandas as pd
import xarray as xr

from scripts.validation.compare_timeseries import select_time_period_for_comparison


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
