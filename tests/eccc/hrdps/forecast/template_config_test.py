import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.download import http_download_to_disk
from reformatters.eccc.hrdps.forecast.region_job import (
    EcccHrdpsForecastSourceFileCoord,
)
from reformatters.eccc.hrdps.forecast.template_config import (
    EcccHrdpsForecastTemplateConfig,
)


@pytest.fixture
def template_config() -> EcccHrdpsForecastTemplateConfig:
    return EcccHrdpsForecastTemplateConfig()


def test_y_x_coordinates(template_config: EcccHrdpsForecastTemplateConfig) -> None:
    y_coords, x_coords = template_config._y_x_coordinates()

    assert len(x_coords) == 2540
    assert np.allclose(np.diff(x_coords), 0.0225)
    assert np.isclose(x_coords.min(), -14.821220)
    assert np.isclose(x_coords.max(), 42.306283)

    # y descends north to south, matching the order values are stored in the source files
    assert len(y_coords) == 1290
    assert np.allclose(np.diff(y_coords), -0.0225)
    assert np.isclose(y_coords.max(), 16.700001)
    assert np.isclose(y_coords.min(), -12.302501)


def test_latitude_longitude_coordinates(
    template_config: EcccHrdpsForecastTemplateConfig,
) -> None:
    y_coords, x_coords = template_config._y_x_coordinates()
    latitudes, longitudes = template_config._latitude_longitude_coordinates(
        x_coords, y_coords
    )

    assert latitudes.shape == (1290, 2540)
    assert longitudes.shape == (1290, 2540)

    assert np.isclose(latitudes.min(), 27.284597)
    assert np.isclose(latitudes.max(), 70.61148)
    assert np.isclose(longitudes.min(), -152.73067)
    assert np.isclose(longitudes.max(), -40.70856)

    # The unrotated pole is north west of the domain, so latitude decreases
    # down every column and longitude increases across every row.
    assert (np.diff(latitudes, axis=0) < 0).all()
    assert (np.diff(longitudes, axis=1) > 0).all()


@pytest.mark.slow
def test_spatial_info_matches_file(
    template_config: EcccHrdpsForecastTemplateConfig,
) -> None:
    """Test that hard coded spatial information matches the real values derived from a source file."""
    shape, bounds, resolution, crs = template_config._spatial_info()

    coord = EcccHrdpsForecastSourceFileCoord(
        init_time=template_config.append_dim_start,
        lead_time=pd.Timedelta("0h"),
        # Any single variable will do
        data_var=template_config.data_vars[0],
    )
    local_path = http_download_to_disk(coord.get_url(), template_config.dataset_id)
    ds = xr.open_dataset(local_path, engine="rasterio")

    assert shape == ds.rio.shape
    assert np.allclose(bounds, ds.rio.bounds())
    assert resolution == ds.rio.resolution()
    assert crs == ds.rio.crs.to_proj4()

    # The file describes a subset of the CF grid mapping attributes we store.
    spatial_ref_coord = next(
        c for c in template_config.coords if c.name == "spatial_ref"
    )
    template_attrs = spatial_ref_coord.attrs.model_dump(exclude_none=True)
    assert ds.spatial_ref.attrs.items() <= template_attrs.items()
