import numpy as np
import pandas as pd
import xarray as xr

from scripts.validation.utils import (
    choose_level,
    var_slug,
    vertical_dims,
    virtual_message_count,
)


def _grouped_ds() -> xr.Dataset:
    init = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    lead = pd.to_timedelta([0, 1, 2, 3], unit="h")
    y = np.arange(5)
    x = np.arange(6)
    pressure_level = np.array([1000, 850, 700, 500, 300, 100])
    return xr.Dataset(
        {
            "temperature_2m": (
                ("init_time", "lead_time", "y", "x"),
                np.zeros((3, 4, 5, 6)),
            ),
            "pressure_level/temperature": (
                ("init_time", "lead_time", "y", "x", "pressure_level"),
                np.zeros((3, 4, 5, 6, 6)),
            ),
        },
        coords={
            "init_time": init,
            "lead_time": lead,
            "y": y,
            "x": x,
            "pressure_level": pressure_level,
        },
    )


def test_var_slug() -> None:
    assert var_slug("temperature_2m") == "temperature_2m"
    assert var_slug("pressure_level/temperature") == "pressure_level__temperature"


def test_vertical_dims() -> None:
    ds = _grouped_ds()
    assert vertical_dims(ds, "temperature_2m") == []
    assert vertical_dims(ds, "pressure_level/temperature") == ["pressure_level"]


def test_choose_level_single_level_var_returns_empty() -> None:
    ds = _grouped_ds()
    assert choose_level(ds, "temperature_2m", None) == {}


def test_choose_level_default_is_middle() -> None:
    ds = _grouped_ds()
    # 6 levels -> middle index 3 -> 500
    assert choose_level(ds, "pressure_level/temperature", None) == {
        "pressure_level": 500
    }


def test_choose_level_override_selects_nearest() -> None:
    ds = _grouped_ds()
    assert choose_level(ds, "pressure_level/temperature", 720) == {
        "pressure_level": 700
    }
    assert choose_level(ds, "pressure_level/temperature", 50) == {"pressure_level": 100}


def test_virtual_message_count() -> None:
    ds = _grouped_ds()
    # A point column (y, x scalar) over init x lead x level = 3 * 4 * 6 messages.
    point = ds["pressure_level/temperature"].isel(y=0, x=0)
    assert virtual_message_count(point) == 3 * 4 * 6
    # A single spatial field at one (init, lead, level) is one message.
    field = ds["pressure_level/temperature"].isel(
        init_time=0, lead_time=0, pressure_level=0
    )
    assert virtual_message_count(field) == 1
