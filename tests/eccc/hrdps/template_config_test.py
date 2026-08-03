import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS

from reformatters.common.download import http_download_to_disk
from reformatters.eccc.hrdps.forecast.template_config import (
    EcccHrdpsForecastTemplateConfig,
)
from reformatters.eccc.hrdps.region_job import EcccHrdpsSourceFileCoord


@pytest.fixture
def template_config() -> EcccHrdpsForecastTemplateConfig:
    return EcccHrdpsForecastTemplateConfig()


def test_y_x_coordinates(template_config: EcccHrdpsForecastTemplateConfig) -> None:
    y_coords, x_coords = template_config._y_x_coordinates()

    assert len(x_coords) == 2540
    assert np.allclose(np.diff(x_coords), 0.0225)
    assert np.isclose(x_coords[0], -14.82122)
    assert np.isclose(x_coords[-1], 42.306283)

    assert len(y_coords) == 1290
    assert np.allclose(np.diff(y_coords), -0.0225)  # descending north->south
    assert np.isclose(y_coords[0], 16.700001)
    assert np.isclose(y_coords[-1], -12.302501)


def test_latitude_longitude_coordinates(
    template_config: EcccHrdpsForecastTemplateConfig,
) -> None:
    y_coords, x_coords = template_config._y_x_coordinates()
    lats, lons = template_config._latitude_longitude_coordinates(x_coords, y_coords)

    assert lats.shape == (1290, 2540)
    assert lons.shape == (1290, 2540)

    # Grid corners match the MSC HRDPS documentation ("first grid point 39N 134W")
    assert np.isclose(lats[-1, 0], 39.626034)  # SW
    assert np.isclose(lons[-1, 0], -133.629520)
    assert np.isclose(lats[0, 0], 66.568541)  # NW
    assert np.isclose(lons[0, 0], -152.730666)
    assert np.isclose(lats[-1, -1], 27.284597)  # SE
    assert np.isclose(lons[-1, -1], -66.966422)
    assert np.isclose(lats[0, -1], 47.876457)  # NE
    assert np.isclose(lons[0, -1], -40.708561)

    assert np.isclose(lats.min(), 27.284597)
    assert np.isclose(lats.max(), 70.611480)
    assert np.isclose(lons.min(), -152.730666)
    assert np.isclose(lons.max(), -40.708561)


@pytest.mark.slow
def test_spatial_info_matches_file(
    template_config: EcccHrdpsForecastTemplateConfig,
) -> None:
    """Test that hard coded spatial information matches the real values derived from a source file."""
    shape, bounds, resolution, crs_wkt = template_config._spatial_info()

    coord = EcccHrdpsSourceFileCoord(
        init_time=pd.Timestamp("2026-07-09T00:00"),
        lead_time=pd.Timedelta("1h"),
        data_var=template_config.data_vars[0],
    )
    local_path = http_download_to_disk(coord.get_url(), template_config.dataset_id)

    ds = xr.open_dataset(local_path, engine="rasterio")

    assert shape == ds.rio.shape
    assert np.allclose(bounds, ds.rio.bounds())
    assert resolution == ds.rio.resolution()
    assert CRS.from_wkt(crs_wkt) == ds.rio.crs

    # Test that the attributes stored in the template match the file
    spatial_ref_coord = next(
        c for c in template_config.coords if c.name == "spatial_ref"
    )
    template_attrs = spatial_ref_coord.attrs.model_dump(exclude_none=True)
    template_attrs.pop("comment")

    file_attrs = ds.rio.write_crs(ds.rio.crs)["spatial_ref"].attrs
    assert file_attrs == template_attrs
