import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from reformatters.noaa.gefs.read_data import _reproject_bilinear_longitude_wrap

# Matches the source data's spherical-earth CRS (see common_gefs_template_config.py).
_CRS = rasterio.crs.CRS.from_proj4("+proj=longlat +a=6371229 +b=6371229 +no_defs")

# Destination 0.25° grid: longitude -180..179.75, latitude 90..-90.
_OUT_SHAPE = (721, 1440)
_OUT_TRANSFORM = Affine(0.25, 0, -180 - 0.125, 0, -0.25, 90 + 0.125)


@pytest.mark.parametrize("src_deg", [0.5, 1.0])
def test_reproject_bilinear_longitude_wrap_fills_eastern_edge(src_deg: float) -> None:
    n_lon = round(360 / src_deg)
    n_lat = round(180 / src_deg) + 1
    src_transform = Affine(
        src_deg, 0, -180 - src_deg / 2, 0, -src_deg, 90 + src_deg / 2
    )

    # Value encodes the source longitude so we can verify the antimeridian interpolation.
    lons = -180 + np.arange(n_lon) * src_deg
    raw = np.broadcast_to(lons.astype(np.float32), (n_lat, n_lon)).copy()

    result = _reproject_bilinear_longitude_wrap(
        raw, src_transform, _CRS, _OUT_SHAPE, _OUT_TRANSFORM, _CRS
    )

    assert np.all(np.isfinite(result)), "Reprojected data must not contain NaN"

    # Destination pixels whose centers align with source centers exactly retain the source value.
    step = round(src_deg / 0.25)
    assert np.array_equal(raw, result[::step, ::step])

    # The eastern edge column (longitude 179.75°) sits between the last source column
    # (180° - src_deg) and the wrapped longitude 180° == -180°, interpolating linearly.
    fraction = (179.75 - lons[-1]) / src_deg
    expected_east = (1 - fraction) * lons[-1] + fraction * (-180)
    np.testing.assert_allclose(result[:, -1], expected_east, rtol=0, atol=1e-4)
