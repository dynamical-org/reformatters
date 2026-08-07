"""Global lat/lon grids in both longitude conventions, for validation script tests."""

import numpy as np
import xarray as xr

ASCENDING_LAT = np.arange(-90.0, 91.0, 5.0)
DESCENDING_LAT = ASCENDING_LAT[::-1].copy()
LON_0_360 = np.arange(0.0, 360.0, 5.0)
LON_180 = np.arange(-180.0, 180.0, 5.0)


def global_lonlat_ds(longitude: np.ndarray, latitude: np.ndarray) -> xr.Dataset:
    """A grid whose values equal each cell's own longitude label, so a misaligned
    selection is detectable from the selected value alone."""
    values = np.broadcast_to(longitude[None, :], (latitude.size, longitude.size))
    return xr.Dataset(
        {"t": (("latitude", "longitude"), values.astype("float64"))},
        coords={"latitude": latitude, "longitude": longitude},
    )
