import re
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.noaa.gfs.analysis.template_config import NoaaGfsAnalysisTemplateConfig


def test_get_template_spatial_ref() -> None:
    """Ensure the spatial reference system in the template matches our expectation."""
    template_config = NoaaGfsAnalysisTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )
    original_attrs = deepcopy(ds.spatial_ref.attrs)

    expected_crs = "+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs"
    calculated_spatial_ref_attrs = ds.rio.write_crs(expected_crs).spatial_ref.attrs
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs


def test_dataset_attributes() -> None:
    cfg = NoaaGfsAnalysisTemplateConfig()
    attrs = cfg.dataset_attributes
    assert attrs.dataset_id == "noaa-gfs-analysis"
    assert re.match(r"\d+\.\d+\.\d+", attrs.dataset_version) is not None
    assert str(cfg.append_dim_start) in attrs.time_domain
    assert "1 hour" in attrs.time_resolution


def test_dimension_coordinates_shapes_and_values() -> None:
    cfg = NoaaGfsAnalysisTemplateConfig()
    dc = cfg.dimension_coordinates()
    assert set(dc) == {"time", "latitude", "longitude"}

    # time: only the start timestamp
    time = dc["time"]
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 1
    assert time[0] == cfg.append_dim_start

    # latitude flips from +90 to -90 in 0.25° steps
    lat = dc["latitude"]
    assert isinstance(lat, np.ndarray)
    assert lat[0] == 90.0
    assert lat[-1] == -90.0
    assert len(lat) == 721
    assert np.allclose(np.diff(lat), -0.25)

    # longitude from -180 to +179.75
    lon = dc["longitude"]
    assert isinstance(lon, np.ndarray)
    assert lon[0] == -180.0
    assert lon[-1] == 179.75
    assert len(lon) == 1440
    assert np.allclose(np.diff(lon), 0.25)


def test_derive_coordinates_spatial_ref() -> None:
    cfg = NoaaGfsAnalysisTemplateConfig()
    dc = cfg.dimension_coordinates()
    ds = xr.Dataset(coords=dc)
    derived = cfg.derive_coordinates(ds)

    assert set(derived) == {"spatial_ref"}
    assert derived["spatial_ref"] == SPATIAL_REF_COORDS


def test_coords_property_order_and_names() -> None:
    cfg = NoaaGfsAnalysisTemplateConfig()
    names = {c.name for c in cfg.coords}
    assert names == {
        "latitude",
        "longitude",
        "spatial_ref",
        "time",
    }


def test_dims() -> None:
    cfg = NoaaGfsAnalysisTemplateConfig()
    assert cfg.dims == ("time", "latitude", "longitude")


def test_append_dim_configuration() -> None:
    cfg = NoaaGfsAnalysisTemplateConfig()
    assert cfg.append_dim == "time"
    assert cfg.append_dim_start == pd.Timestamp("2021-05-01T00:00")
    assert cfg.append_dim_frequency == pd.Timedelta("1h")


def test_data_vars_have_correct_encoding() -> None:
    cfg = NoaaGfsAnalysisTemplateConfig()
    data_vars = cfg.data_vars

    for var in data_vars:
        assert var.encoding.dtype == "float32"
        assert var.encoding.chunks == (1008, 64, 64)
        assert var.encoding.shards == (3024, 384, 384)


def test_data_vars_names() -> None:
    cfg = NoaaGfsAnalysisTemplateConfig()
    data_vars = cfg.data_vars

    expected_var_names = {
        "pressure_surface",
        "temperature_2m",
        "relative_humidity_2m",
        "maximum_temperature_2m",
        "minimum_temperature_2m",
        "wind_u_10m",
        "wind_v_10m",
        "wind_u_100m",
        "wind_v_100m",
        "percent_frozen_precipitation_surface",
        "precipitation_surface",
        "categorical_snow_surface",
        "categorical_ice_pellets_surface",
        "categorical_freezing_rain_surface",
        "categorical_rain_surface",
        "precipitable_water_atmosphere",
        "total_cloud_cover_atmosphere",
        "geopotential_height_cloud_ceiling",
        "downward_short_wave_radiation_flux_surface",
        "downward_long_wave_radiation_flux_surface",
        "pressure_reduced_to_mean_sea_level",
    }

    actual_var_names = {v.name for v in data_vars}
    assert expected_var_names <= actual_var_names
