import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.download import http_download_to_disk
from reformatters.noaa.hrrr.forecast_48_hour.region_job import (
    NoaaHrrrSourceFileCoord,
)
from reformatters.noaa.hrrr.template_config import (
    NoaaHrrrTemplateConfig,
)
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index


@pytest.fixture
def template_config() -> NoaaHrrrTemplateConfig:
    return NoaaHrrrTemplateConfig(
        dims=("time", "y", "x"),
        append_dim="time",
        append_dim_start=pd.Timestamp("2018-07-13T12:00"),
        append_dim_frequency=pd.Timedelta("1h"),
    )


def test_y_x_coordinates(template_config: NoaaHrrrTemplateConfig) -> None:
    """Test that _y_x_coordinates returns expected values."""
    y_coords, x_coords = template_config._y_x_coordinates()

    # Check x coordinates
    assert len(x_coords) == 1799
    assert np.allclose(np.diff(x_coords), 3000.0)
    assert np.isclose(x_coords.min() - (3000 / 2), -2699020.143)
    assert np.isclose(x_coords.max() + (3000 / 2), 2697979.857)

    # Check y coordinates
    assert len(y_coords) == 1059
    assert np.allclose(np.diff(y_coords), -3000.0)
    assert np.isclose(y_coords.min() - (3000 / 2), -1588806.153)
    assert np.isclose(y_coords.max() + (3000 / 2), 1588193.847)


def test_spatial_info_matches_file(template_config: NoaaHrrrTemplateConfig) -> None:
    """Test that hard coded spatial information matches the real values derived from a source file."""
    shape, bounds, resolution, crs = template_config._spatial_info()

    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2023-10-01T00:00"),
        lead_time=pd.Timedelta("0h"),
        domain="conus",
        file_type="sfc",
        data_vars=[template_config.data_vars[0]],  # Any one variable will do
    )
    idx_local_path = http_download_to_disk(
        coord.get_idx_url(), template_config.dataset_id
    )
    byte_range_starts, byte_range_ends = grib_message_byte_ranges_from_index(
        idx_local_path, coord.data_vars, coord.init_time, coord.lead_time
    )
    local_path = http_download_to_disk(
        coord.get_url(),
        template_config.dataset_id,
        byte_ranges=(byte_range_starts, byte_range_ends),
        local_path_suffix="spatial-info-test",
    )

    ds = xr.open_dataset(local_path, engine="rasterio")

    # Test that the spatial_info return values match the file
    assert shape == ds.rio.shape
    assert np.allclose(bounds, ds.rio.bounds())
    assert resolution == ds.rio.resolution()
    assert crs == ds.rio.crs.to_proj4()

    # Test that the attributes stored in the template match the file
    template_ds = template_config.get_template(pd.Timestamp("2025-01-01"))
    # The template has to round trip through JSON so tuples become lists
    assert ds.spatial_ref.attrs["standard_parallel"] == (38.5, 38.5)
    ds.spatial_ref.attrs["standard_parallel"] = list(
        ds.spatial_ref.attrs["standard_parallel"]
    )
    # Allow for a tiny floating point difference in the GeoTransform y offset last digit between arm64 and amd64
    if (
        ds.spatial_ref.attrs["GeoTransform"]
        == "-2699020.142521929 3000.0 0.0 1588193.8474433345 0.0 -3000.0"
    ):
        ds.spatial_ref.attrs["GeoTransform"] = (
            "-2699020.142521929 3000.0 0.0 1588193.847443335 0.0 -3000.0"
        )
    assert ds.spatial_ref.attrs == template_ds.spatial_ref.attrs
