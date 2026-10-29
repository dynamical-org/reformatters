"""Grid coordinate derivation shared by datasets on a projected grid."""

import functools

import numpy as np
import pyproj

from reformatters.common.types import Array1D, Array2D


def y_x_coordinates(
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    resolution: tuple[float, float],
) -> tuple[Array1D[np.float64], Array1D[np.float64]]:
    """Pixel center y and x coordinates, in the projection's own units, of a grid
    described by (ny, nx), (left, bottom, right, top) and (dx, dy)."""
    dx, dy = resolution
    left, _bottom, _right, top = bounds
    ny, nx = shape
    # add 1/2 a pixel to corner of bounds to get pixel center
    y_coords = (top + (0.5 * dy)) + (np.arange(ny) * dy)
    x_coords = (left + (0.5 * dx)) + (np.arange(nx) * dx)
    # astype is no-op for type checker
    return y_coords.astype(np.float64), x_coords.astype(np.float64)


def latitude_longitude_grids(
    crs: str,
    x_coords: Array1D[np.float64],
    y_coords: Array1D[np.float64],
    *,
    degree_units: bool = False,
) -> tuple[Array2D[np.float32], Array2D[np.float32]]:
    """2D latitude and longitude of every cell center of the grid `y_coords` and
    `x_coords` span within `crs`.

    Set `degree_units` when the grid's coordinates are angles rather than metres
    (a rotated pole grid), which PROJ's inverse takes and returns in radians.

    Results are cached; callers must not mutate the returned arrays.
    """
    return _latitude_longitude_grids(
        crs, x_coords.tobytes(), y_coords.tobytes(), degree_units
    )


# The inverse transform costs 0.25-1s depending on grid size, datasets sharing a grid
# each request it, and a template build requests it once per zarr group.
@functools.cache
def _latitude_longitude_grids(
    crs: str, x_bytes: bytes, y_bytes: bytes, degree_units: bool
) -> tuple[Array2D[np.float32], Array2D[np.float32]]:
    x_coords = np.frombuffer(x_bytes, dtype=np.float64)
    y_coords = np.frombuffer(y_bytes, dtype=np.float64)
    xs, ys = np.meshgrid(x_coords, y_coords)
    if degree_units:
        xs, ys = np.radians(xs), np.radians(ys)
    lons, lats = pyproj.Proj(crs)(xs, ys, inverse=True)
    # Dropping to 32 bit precision still gets us < 1 meter precision and
    # halves the size of each array.
    lats = lats.astype(np.float32)
    lons = lons.astype(np.float32)
    return lats, lons
