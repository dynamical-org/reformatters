import numpy as np
import pandas as pd
import xarray as xr

from scripts.validation.utils import (
    align_longitude_convention,
    choose_level,
    get_random_spatial_indices,
    get_two_random_points,
    nearest_point_index,
    parse_point_options,
    var_slug,
    vertical_dims,
)
from tests.scripts.validation.grids import (
    ASCENDING_LAT,
    DESCENDING_LAT,
    LON_0_360,
    LON_180,
    global_lonlat_ds,
)


def _projected_ds() -> xr.Dataset:
    y = np.arange(20)
    x = np.arange(30)
    # 2D lat/lon like a projected (Lambert) grid: lat increases with y, lon with x.
    lat2d = np.broadcast_to((30.0 + y * 0.5)[:, None], (20, 30)).copy()
    lon2d = np.broadcast_to((-110.0 + x * 0.5)[None, :], (20, 30)).copy()
    return xr.Dataset(
        {"t": (("y", "x"), np.zeros((20, 30)))},
        coords={
            "y": y,
            "x": x,
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
        },
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


def test_align_longitude_convention_is_a_noop_within_one_convention() -> None:
    ds = global_lonlat_ds(LON_180, DESCENDING_LAT)
    reference = global_lonlat_ds(LON_180, DESCENDING_LAT)
    assert align_longitude_convention(ds, reference) is reference


def test_align_longitude_convention_relabels_reference_to_0_360() -> None:
    ds = global_lonlat_ds(LON_0_360, ASCENDING_LAT)
    aligned = align_longitude_convention(ds, global_lonlat_ds(LON_180, DESCENDING_LAT))
    np.testing.assert_array_equal(aligned.longitude.values, LON_0_360)
    # 225 in the validation dataset's convention is -135 in the reference's.
    assert aligned["t"].sel(longitude=225.0, method="nearest").values[0] == -135.0


def test_align_longitude_convention_relabels_reference_to_180() -> None:
    ds = global_lonlat_ds(LON_180, DESCENDING_LAT)
    aligned = align_longitude_convention(ds, global_lonlat_ds(LON_0_360, ASCENDING_LAT))
    np.testing.assert_array_equal(aligned.longitude.values, LON_180)
    assert aligned["t"].sel(longitude=-135.0, method="nearest").values[0] == 225.0


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


def test_parse_point_options() -> None:
    assert parse_point_options(None) == []
    assert parse_point_options([]) == []
    assert parse_point_options(["39.0,-98.5"]) == [(39.0, -98.5)]
    assert parse_point_options(["39,-98.5", "33.75,-84.4"]) == [
        (39.0, -98.5),
        (33.75, -84.4),
    ]


def test_parse_point_options_rejects_more_than_two() -> None:
    try:
        parse_point_options(["1,2", "3,4", "5,6"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for more than two points")


def test_random_spatial_indices_within_middle_50_percent() -> None:
    ds = _projected_ds()
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    for _ in range(500):
        p1, p2 = get_random_spatial_indices(ds, "y", "x")
        for p in (p1, p2):
            assert ny // 4 <= p["y"] < ny - ny // 4
            assert nx // 4 <= p["x"] < nx - nx // 4


def test_nearest_point_index_projected_grid() -> None:
    ds = _projected_ds()
    # lat = 30 + 0.5*y, lon = -110 + 0.5*x -> (35, -100) is y=10, x=20.
    assert nearest_point_index(ds, 35.0, -100.0) == {"y": 10, "x": 20}


def test_get_two_random_points_pins_provided_points() -> None:
    ds = _projected_ds()
    # lat = 30 + 0.5*y, lon = -110 + 0.5*x.
    p1_sel, p2_sel, (lat1, lon1), (lat2, lon2) = get_two_random_points(
        ds, [(35.0, -100.0), (39.0, -96.0)]
    )
    assert p1_sel == {"y": 10, "x": 20}
    assert p2_sel == {"y": 18, "x": 28}
    assert (lat1, lon1) == (35.0, -100.0)
    assert (lat2, lon2) == (39.0, -96.0)
