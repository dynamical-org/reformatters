import numpy as np
import pyproj
import pytest

from reformatters.common.projection import latitude_longitude_grids, y_x_coordinates

# NOAA HRRR: a Lambert conformal grid whose coordinates are metres.
HRRR_SHAPE = (1059, 1799)
HRRR_BOUNDS = (
    -2699020.142521929,
    -1588806.152556665,
    2697979.857478071,
    1588193.847443335,
)
HRRR_RESOLUTION = (3000.0, -3000.0)
HRRR_CRS = "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs=True"

# ECCC HRDPS: a rotated pole grid whose coordinates are degrees.
HRDPS_SHAPE = (1290, 2540)
HRDPS_BOUNDS = (
    -14.832470000590822,
    -12.313751000775795,
    42.31753300059079,
    16.711251000775796,
)
HRDPS_RESOLUTION = (0.022500001181567565, -0.02250000155159038)
HRDPS_CRS = "+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p=36.08852 +lon_0=-114.694858 +R=6371229 +no_defs=True"


def coarse_grid(
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    resolution: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """A handful of points spanning the full grid, to keep the inverse transform cheap."""
    y_coords, x_coords = y_x_coordinates(shape, bounds, resolution)
    return (
        np.ascontiguousarray(y_coords[::300]),
        np.ascontiguousarray(x_coords[::400]),
    )


def transform_to_latitude_longitude(
    crs: str, x_coords: np.ndarray, y_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes via pyproj's Transformer, independent of the code under test."""
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4(crs), "EPSG:4326", always_xy=True
    )
    xs, ys = np.meshgrid(x_coords, y_coords)
    longitudes, latitudes = transformer.transform(xs, ys)
    return latitudes, longitudes


def test_coordinates_are_pixel_centers() -> None:
    y_coords, x_coords = y_x_coordinates((3, 4), (0.0, -30.0, 40.0, 0.0), (10.0, -10.0))

    # Half a pixel in from the top and left bounds, then stepping by the resolution.
    np.testing.assert_array_equal(y_coords, [-5.0, -15.0, -25.0])
    np.testing.assert_array_equal(x_coords, [5.0, 15.0, 25.0, 35.0])
    assert y_coords.dtype == np.float64
    assert x_coords.dtype == np.float64


def test_lengths_match_shape_and_spacing_matches_resolution() -> None:
    y_coords, x_coords = y_x_coordinates(HRRR_SHAPE, HRRR_BOUNDS, HRRR_RESOLUTION)

    assert (len(y_coords), len(x_coords)) == HRRR_SHAPE
    np.testing.assert_allclose(np.diff(y_coords), HRRR_RESOLUTION[1])
    np.testing.assert_allclose(np.diff(x_coords), HRRR_RESOLUTION[0])


def test_positive_y_resolution_gives_ascending_coordinates() -> None:
    y_coords, _ = y_x_coordinates((3, 1), (0.0, 0.0, 1.0, 0.0), (1.0, 10.0))

    np.testing.assert_array_equal(y_coords, [5.0, 15.0, 25.0])


def test_spans_the_grid_bounds() -> None:
    y_coords, x_coords = y_x_coordinates(HRRR_SHAPE, HRRR_BOUNDS, HRRR_RESOLUTION)
    left, bottom, right, top = HRRR_BOUNDS
    half_x, half_y = HRRR_RESOLUTION[0] / 2, HRRR_RESOLUTION[1] / 2

    # The first and last cell centers sit half a pixel inside each bound.
    assert y_coords[0] == pytest.approx(top + half_y)
    assert y_coords[-1] == pytest.approx(bottom - half_y)
    assert x_coords[0] == pytest.approx(left + half_x)
    assert x_coords[-1] == pytest.approx(right - half_x)


def test_latitude_longitude_grids_shape_and_dtype() -> None:
    y_coords, x_coords = coarse_grid(HRRR_SHAPE, HRRR_BOUNDS, HRRR_RESOLUTION)

    latitudes, longitudes = latitude_longitude_grids(HRRR_CRS, x_coords, y_coords)

    assert latitudes.shape == longitudes.shape == (len(y_coords), len(x_coords))
    assert latitudes.dtype == np.float32
    assert longitudes.dtype == np.float32


def test_metre_grid_matches_an_independent_transformer() -> None:
    y_coords, x_coords = coarse_grid(HRRR_SHAPE, HRRR_BOUNDS, HRRR_RESOLUTION)

    latitudes, longitudes = latitude_longitude_grids(HRRR_CRS, x_coords, y_coords)

    expected_lat, expected_lon = transform_to_latitude_longitude(
        HRRR_CRS, x_coords, y_coords
    )
    np.testing.assert_allclose(latitudes, expected_lat, atol=1e-4)
    np.testing.assert_allclose(longitudes, expected_lon, atol=1e-4)


def test_rotated_pole_grid_matches_an_independent_transformer() -> None:
    y_coords, x_coords = coarse_grid(HRDPS_SHAPE, HRDPS_BOUNDS, HRDPS_RESOLUTION)

    latitudes, longitudes = latitude_longitude_grids(
        HRDPS_CRS, x_coords, y_coords, degree_units=True
    )

    expected_lat, expected_lon = transform_to_latitude_longitude(
        HRDPS_CRS, x_coords, y_coords
    )
    np.testing.assert_allclose(latitudes, expected_lat, atol=1e-4)
    np.testing.assert_allclose(longitudes, expected_lon, atol=1e-4)


def test_rotated_pole_grid_without_degree_units_is_wrong_not_an_error() -> None:
    y_coords, x_coords = coarse_grid(HRDPS_SHAPE, HRDPS_BOUNDS, HRDPS_RESOLUTION)

    latitudes, _ = latitude_longitude_grids(
        HRDPS_CRS, x_coords, y_coords, degree_units=False
    )

    # PROJ reads the degrees as radians and returns finite coordinates from all over the
    # globe, rather than raising, so a caller that omits degree_units gets silent garbage.
    assert np.isfinite(latitudes).all()
    correct, _ = latitude_longitude_grids(
        HRDPS_CRS, x_coords, y_coords, degree_units=True
    )
    assert np.abs(latitudes - correct).max() > 10


def test_results_are_cached_and_keyed_on_degree_units() -> None:
    y_coords, x_coords = coarse_grid(HRDPS_SHAPE, HRDPS_BOUNDS, HRDPS_RESOLUTION)

    latitudes, _ = latitude_longitude_grids(
        HRDPS_CRS, x_coords, y_coords, degree_units=True
    )

    # Callers share one array, which is why they must not mutate it.
    assert (
        latitude_longitude_grids(HRDPS_CRS, x_coords, y_coords, degree_units=True)[0]
        is latitudes
    )
    assert (
        latitude_longitude_grids(HRDPS_CRS, x_coords, y_coords, degree_units=False)[0]
        is not latitudes
    )
