from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.validation.compare_timeseries import _load_timeseries_for_var
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
