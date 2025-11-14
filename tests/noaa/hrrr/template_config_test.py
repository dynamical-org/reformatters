import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.config_models import Encoding
from reformatters.common.download import http_download_to_disk
from reformatters.noaa.hrrr.forecast_48_hour.region_job import (
    NoaaHrrrSourceFileCoord,
)
from reformatters.noaa.hrrr.template_config import (
    NoaaHrrrTemplateConfig,
)
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index


@pytest.fixture
def template_config(monkeypatch: pytest.MonkeyPatch) -> NoaaHrrrTemplateConfig:
    from reformatters.common.config_models import DatasetAttributes

    mock_attrs = DatasetAttributes(
        dataset_id="noaa-hrrr-template",
        dataset_version="0.1.0",
        name="Test HRRR",
        description="Test dataset",
        attribution="Test",
        spatial_domain="CONUS",
        spatial_resolution="3km",
        time_domain="Test",
        time_resolution="1h",
    )

    monkeypatch.setattr(
        NoaaHrrrTemplateConfig, "dataset_attributes", property(lambda self: mock_attrs)
    )

    config = NoaaHrrrTemplateConfig(
        dims=("time", "y", "x"),
        append_dim="time",
        append_dim_start=pd.Timestamp("2018-07-13T12:00"),
        append_dim_frequency=pd.Timedelta("1h"),
    )
    return config


def test_y_x_coordinates(template_config: NoaaHrrrTemplateConfig) -> None:
    y_coords, x_coords = template_config._y_x_coordinates()

    assert len(x_coords) == 1799
    assert np.allclose(np.diff(x_coords), 3000.0)
    assert np.isclose(x_coords.min() - (3000 / 2), -2699020.143)
    assert np.isclose(x_coords.max() + (3000 / 2), 2697979.857)

    assert len(y_coords) == 1059
    assert np.allclose(np.diff(y_coords), -3000.0)
    assert np.isclose(y_coords.min() - (3000 / 2), -1588806.153)
    assert np.isclose(y_coords.max() + (3000 / 2), 1588193.847)


def test_latitude_longitude_coordinates(
    template_config: NoaaHrrrTemplateConfig,
) -> None:
    y_coords, x_coords = template_config._y_x_coordinates()
    lats, lons = template_config._latitude_longitude_coordinates(x_coords, y_coords)

    assert lats.shape == (1059, 1799)
    assert lons.shape == (1059, 1799)

    # Check latitude values
    assert lats.min() == 21.138123
    assert np.isclose(lats.mean(), 37.152527)
    # Note the maximum latitude is in the center north of CONUS, so this
    # max is larger than either of the upper corners latitudes.
    assert lats.max() == 52.615654

    # Check longitude values
    assert lons.min() == -134.09547
    assert np.isclose(lons.mean(), -97.50583)
    assert lons.max() == -60.917194

    # Check latitude differences
    # 1. decreasing
    # 2. the min and max diff should be similar because its roughly evenly spaced
    lat_diff_y = np.diff(lats, axis=0)
    assert np.isclose(lat_diff_y.min(), -0.02698135)
    assert np.isclose(lat_diff_y.max(), -0.0245285)

    # Check longitude differences
    # 1. increasing
    # 2. the min and max diff should be similar because its roughly evenly spaced
    lon_diff_x = np.diff(lons, axis=1)
    assert np.isclose(lon_diff_x.min(), 0.02666473)
    assert np.isclose(lon_diff_x.max(), 0.04299164)


def test_spatial_info_matches_file(template_config: NoaaHrrrTemplateConfig) -> None:
    """Test that hard coded spatial information matches the real values derived from a source file."""
    shape, bounds, resolution, crs = template_config._spatial_info()
    dummy_encoding = Encoding(
        dtype="float32",
        fill_value=0.0,
        chunks=(2000, 2000),
        shards=None,
        compressors=[],
    )

    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2023-10-01T00:00"),
        lead_time=pd.Timedelta("0h"),
        domain="conus",
        file_type="sfc",
        # Any single variable will do
        data_vars=template_config.get_data_vars(dummy_encoding)[:1],
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

    # Below, we can't call get_template because it will try to read the template from disk, which we don't have. Instead, call template_config.coords and check the attributes on the coord with name spatial_ref AI!

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
