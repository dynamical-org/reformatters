import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.download import http_download_to_disk
from reformatters.noaa.hrrr.analysis.template_config import (
    NoaaHrrrAnalysisTemplateConfig,
)
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index


def test_spatial_coordinates() -> None:
    """Ensure the template has the expected coordinate system."""
    template_config = NoaaHrrrAnalysisTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )

    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "x" in ds.coords
    assert "y" in ds.coords

    assert ds.latitude.dims == ("y", "x")
    assert ds.longitude.dims == ("y", "x")
    assert ds.x.dims == ("x",)
    assert ds.y.dims == ("y",)

    assert len(ds.x) == 1799
    assert (ds.x.diff(dim="x") == 3000.0).all()
    assert np.isclose(ds.x.min() - (3000 / 2), -2699020.143)
    assert np.isclose(ds.x.max() + (3000 / 2), 2697979.857)
    assert len(ds.y) == 1059
    assert (ds.y.diff(dim="y") == -3000.0).all()
    assert np.isclose(ds.y.min() - (3000 / 2), -1588806.153)
    assert np.isclose(ds.y.max() + (3000 / 2), 1588193.847)

    assert ds.latitude.min() == 21.138123
    assert ds.latitude.mean() == 37.152527
    assert ds.latitude.max() == 52.615654
    assert np.isclose(ds.latitude.diff(dim="y").min(), -0.02698135)
    assert np.isclose(ds.latitude.diff(dim="y").max(), -0.0245285)

    assert ds.longitude.min() == -134.09547
    assert ds.longitude.mean() == -97.50583
    assert ds.longitude.max() == -60.917194
    assert np.isclose(ds.longitude.diff(dim="x").min(), 0.02666473)
    assert np.isclose(ds.longitude.diff(dim="x").max(), 0.04299164)


def test_template_config_attrs() -> None:
    """Test basic template configuration attributes."""
    config = NoaaHrrrAnalysisTemplateConfig()

    assert config.dims == ("time", "y", "x")
    assert config.append_dim == "time"

    assert config.append_dim_start == pd.Timestamp("2018-07-14T00:00")
    assert config.append_dim_frequency == pd.Timedelta("1h")

    data_vars = config.data_vars
    assert len(data_vars) > 0

    refc_vars = [v for v in data_vars if v.internal_attrs.grib_element == "REFC"]
    assert len(refc_vars) == 1
    assert refc_vars[0].name == "composite_reflectivity"
    assert refc_vars[0].internal_attrs.hrrr_file_type == "sfc"


def test_dimension_coordinates() -> None:
    """Test dimension coordinates are properly configured."""
    config = NoaaHrrrAnalysisTemplateConfig()
    dim_coords = config.dimension_coordinates()

    assert "time" in dim_coords
    assert "x" in dim_coords
    assert "y" in dim_coords

    assert (
        dim_coords["time"]
        == pd.date_range("2018-07-14T00:00", "2018-07-14T00:00", freq="1h")
    ).all()

    assert len(dim_coords["x"]) == 1799
    assert len(dim_coords["y"]) == 1059


def test_template_variables_have_required_attrs() -> None:
    """Test that all data variables have required attributes."""
    config = NoaaHrrrAnalysisTemplateConfig()

    for var in config.data_vars:
        assert var.name
        assert var.encoding

        assert var.internal_attrs.grib_element
        assert var.internal_attrs.grib_index_level
        assert var.internal_attrs.hrrr_file_type in ["sfc", "prs", "nat", "subh"]

        assert var.attrs.short_name
        assert var.attrs.long_name
        assert var.attrs.units
        assert var.attrs.step_type in ["instant", "avg", "accum", "max", "min"]


def test_coordinate_configs() -> None:
    """Test coordinate configurations."""
    config = NoaaHrrrAnalysisTemplateConfig()
    coords = config.coords

    coord_names = [coord.name for coord in coords]

    required_coords = {
        "time",
        "x",
        "y",
        "latitude",
        "longitude",
        "spatial_ref",
    }

    assert set(coord_names) == required_coords, (
        f"Coordinate mismatch. Expected: {required_coords}, Got: {set(coord_names)}"
    )


def test_derive_coordinates_integration() -> None:
    """Integration test for derive_coordinates method."""
    config = NoaaHrrrAnalysisTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2018-07-14T12:00"))

    assert (
        template_ds.coords["time"]
        == pd.date_range("2018-07-14T00:00", "2018-07-14T11:00", freq="1h")
    ).all()


def test_spatial_info_matches_file() -> None:
    """Test that hard coded spatial information matches the real values derived from a source file."""
    config = NoaaHrrrAnalysisTemplateConfig()
    shape, bounds, resolution, crs = config._spatial_info()

    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2023-10-01T00:00"),
        lead_time=pd.Timedelta("0h"),
        domain="conus",
        file_type="sfc",
        data_vars=[config.data_vars[0]],
    )
    idx_local_path = http_download_to_disk(coord.get_idx_url(), config.dataset_id)
    byte_range_starts, byte_range_ends = grib_message_byte_ranges_from_index(
        idx_local_path, coord.data_vars, coord.init_time, coord.lead_time
    )
    local_path = http_download_to_disk(
        coord.get_url(),
        config.dataset_id,
        byte_ranges=(byte_range_starts, byte_range_ends),
        local_path_suffix="spatial-info-test",
    )

    ds = xr.open_dataset(local_path, engine="rasterio")

    assert shape == ds.rio.shape
    assert np.allclose(bounds, ds.rio.bounds())
    assert resolution == ds.rio.resolution()
    assert crs == ds.rio.crs.to_proj4()

    template_ds = config.get_template(pd.Timestamp("2025-01-01"))
    assert ds.spatial_ref.attrs["standard_parallel"] == (38.5, 38.5)
    ds.spatial_ref.attrs["standard_parallel"] = list(
        ds.spatial_ref.attrs["standard_parallel"]
    )
    if (
        ds.spatial_ref.attrs["GeoTransform"]
        == "-2699020.142521929 3000.0 0.0 1588193.8474433345 0.0 -3000.0"
    ):
        ds.spatial_ref.attrs["GeoTransform"] = (
            "-2699020.142521929 3000.0 0.0 1588193.847443335 0.0 -3000.0"
        )
    assert ds.spatial_ref.attrs == template_ds.spatial_ref.attrs
