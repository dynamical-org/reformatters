from pathlib import Path

import numpy as np
import pytest
import rasterio

from reformatters.noaa.refs.forecast_virtual.template_config import (
    NoaaRefsForecastVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.template_config import (
    NoaaRrfsForecast84HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.template_config import NoaaRrfsForecastTemplateConfig


@pytest.mark.parametrize(
    ("fixture_name", "config"),
    [
        ("spfh", NoaaRrfsForecast84HourVirtualTemplateConfig()),
        ("aotk", NoaaRefsForecastVirtualTemplateConfig()),
    ],
)
def test_conus_grid_matches_source_header(
    fixture_name: str, config: NoaaRrfsForecastTemplateConfig
) -> None:
    shape, bounds, resolution, crs = config._spatial_info()
    spatial_ref = next(c for c in config.coords if c.name == "spatial_ref")
    assert spatial_ref.attrs.GeoTransform is not None
    expected_transform = tuple(float(v) for v in spatial_ref.attrs.GeoTransform.split())
    path = Path(__file__).parent / "fixtures" / f"{fixture_name}-all-missing.grib2"
    with rasterio.open(path) as source:
        assert source.shape == shape
        np.testing.assert_allclose(source.bounds, bounds, rtol=0, atol=1e-8)
        assert source.res == (resolution[0], -resolution[1])
        assert source.crs == rasterio.crs.CRS.from_string(crs)
        np.testing.assert_allclose(
            source.transform.to_gdal(), expected_transform, rtol=0, atol=1e-8
        )
