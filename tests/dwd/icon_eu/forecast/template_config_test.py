from copy import deepcopy

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
from pyproj import CRS

from reformatters.dwd.icon_eu.forecast.template_config import (
    DwdIconEuForecastTemplateConfig,
)


def test_spatial_coordinates() -> None:
    template_config = DwdIconEuForecastTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )

    assert "latitude" in ds.coords
    assert "longitude" in ds.coords

    assert ds.latitude.dims == ("latitude",)
    assert ds.longitude.dims == ("longitude",)

    assert len(ds.latitude) == 657
    assert len(ds.longitude) == 1377

    assert np.isclose(ds.latitude.min(), 29.5)
    assert np.isclose(ds.latitude.max(), 70.5)
    assert np.isclose(ds.longitude.min(), -23.5)
    assert np.isclose(ds.longitude.max(), 62.5)

    # Resolution is 0.0625 degrees
    assert np.allclose(ds.latitude.diff(dim="latitude"), 0.0625)
    assert np.allclose(ds.longitude.diff(dim="longitude"), 0.0625)


def test_template_config_attrs() -> None:
    config = DwdIconEuForecastTemplateConfig()

    assert config.dims == ("init_time", "lead_time", "latitude", "longitude")
    assert config.append_dim == "init_time"
    assert config.append_dim_start == pd.Timestamp("2026-02-10T00:00")
    assert config.append_dim_frequency == pd.Timedelta("6h")

    data_vars = config.data_vars
    assert len(data_vars) > 0

    # Check key variables are present
    var_names = {v.name for v in data_vars}
    assert "temperature_2m" in var_names
    assert "wind_u_10m" in var_names
    assert "wind_v_10m" in var_names
    assert "precipitation_surface" in var_names
    assert "pressure_reduced_to_mean_sea_level" in var_names
    assert "dew_point_temperature_2m" in var_names
    assert "pressure_surface" in var_names
    assert "precipitable_water_atmosphere" in var_names


def test_dimension_coordinates() -> None:
    config = DwdIconEuForecastTemplateConfig()
    dim_coords = config.dimension_coordinates()

    assert "init_time" in dim_coords
    assert "lead_time" in dim_coords
    assert "latitude" in dim_coords
    assert "longitude" in dim_coords

    # Lead times: hourly 0-78h then 3-hourly 81-120h
    lead_times = dim_coords["lead_time"]
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("120h")
    assert len(lead_times) == 93

    # Hourly from 0 to 78
    hourly = lead_times[lead_times <= pd.Timedelta("78h")]
    assert len(hourly) == 79

    # 3-hourly from 81 to 120
    three_hourly = lead_times[lead_times > pd.Timedelta("78h")]
    assert len(three_hourly) == 14
    assert three_hourly[0] == pd.Timedelta("81h")


def test_template_variables_have_required_attrs() -> None:
    config = DwdIconEuForecastTemplateConfig()

    for var in config.data_vars:
        assert var.name
        assert var.encoding
        assert var.internal_attrs.variable_name_in_filename
        assert var.attrs.short_name
        assert var.attrs.long_name
        assert var.attrs.units
        assert var.attrs.step_type in ["instant", "avg", "accum", "max", "min"]


def test_coordinate_configs() -> None:
    config = DwdIconEuForecastTemplateConfig()
    coords = config.coords

    coord_names = [coord.name for coord in coords]

    required_coords = [
        "init_time",
        "lead_time",
        "latitude",
        "longitude",
        "valid_time",
        "ingested_forecast_length",
        "expected_forecast_length",
        "spatial_ref",
    ]

    for coord_name in required_coords:
        assert coord_name in coord_names, f"Missing coordinate: {coord_name}"


def test_derive_coordinates() -> None:
    config = DwdIconEuForecastTemplateConfig()
    template_ds = config.get_template(config.append_dim_start + pd.Timedelta(days=10))

    assert (
        template_ds.coords["valid_time"]
        == (template_ds.coords["init_time"] + template_ds.coords["lead_time"])
    ).all()
    assert template_ds.coords["valid_time"].dims == ("init_time", "lead_time")


def test_get_template_spatial_ref() -> None:
    template_config = DwdIconEuForecastTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )
    original_attrs = deepcopy(ds.spatial_ref.attrs)

    # This WKT string is extracted from the ICON-EU GRIB by gdalinfo:
    expected_crs = CRS.from_wkt("""GEOGCRS["Coordinate System imported from GRIB file",
    DATUM["unnamed",
        ELLIPSOID["Sphere",6371229,0,
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["degree",0.0174532925199433,
            ID["EPSG",9122]]],
    CS[ellipsoidal,2],
        AXIS["latitude",north,
            ORDER[1],
            ANGLEUNIT["degree",0.0174532925199433,
                ID["EPSG",9122]]],
        AXIS["longitude",east,
            ORDER[2],
            ANGLEUNIT["degree",0.0174532925199433,
                ID["EPSG",9122]]]]""")
    calculated_spatial_ref_attrs = ds.rio.write_crs(expected_crs).spatial_ref.attrs
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs


def test_dataset_attributes() -> None:
    config = DwdIconEuForecastTemplateConfig()
    attrs = config.dataset_attributes
    assert attrs.dataset_id == "dwd-icon-eu-forecast"
    assert attrs.dataset_version == "0.1.0"
    assert "Europe" in attrs.spatial_domain
    assert "0.0625" in attrs.spatial_resolution
