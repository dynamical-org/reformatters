import re

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  # Registers .rio accessor on xarray objects
import xarray as xr

from reformatters.common.iterating import item
from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.ecmwf.aifs_single.forecast.template_config import (
    EcmwfAifsSingleForecastTemplateConfig,
)


def test_template_config_attrs() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()

    assert config.dims == ("init_time", "lead_time", "latitude", "longitude")
    assert config.append_dim == "init_time"
    assert config.append_dim_start == pd.Timestamp("2024-04-01T00:00")
    assert config.append_dim_frequency == pd.Timedelta("6h")

    assert len(config.data_vars) > 0


def test_dataset_attributes() -> None:
    cfg = EcmwfAifsSingleForecastTemplateConfig()
    attrs = cfg.dataset_attributes
    assert attrs.dataset_id == "ecmwf-aifs-single-forecast"
    assert re.match(r"\d+\.\d+\.\d+", attrs.dataset_version) is not None
    assert str(cfg.append_dim_start) in attrs.time_domain
    assert "every 6 hours" in attrs.time_resolution


def test_dimension_coordinates() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    coords = config.dimension_coordinates()

    assert config.append_dim == "init_time"
    assert config.append_dim_frequency == pd.Timedelta("6h")

    lead_times = coords["lead_time"]
    assert len(lead_times) == 61  # 0h to 360h every 6h
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("360h")

    lat = coords["latitude"]
    assert len(lat) == 721
    assert float(lat.max()) == 90.0
    assert float(lat.min()) == -90.0

    lon = coords["longitude"]
    assert len(lon) == 1440
    assert float(lon.min()) == -180.0
    assert float(lon.max()) == 179.75
    assert float(item(np.unique(np.diff(lon)))) == 0.25
    assert float(item(np.unique(np.diff(lat)))) == -0.25


def test_dimension_coordinates_shapes_and_values() -> None:
    cfg = EcmwfAifsSingleForecastTemplateConfig()
    dc = cfg.dimension_coordinates()
    assert set(dc) == {"init_time", "lead_time", "latitude", "longitude"}

    init = dc["init_time"]
    assert isinstance(init, pd.DatetimeIndex)
    assert len(init) == 1
    assert init[0] == cfg.append_dim_start

    lt = dc["lead_time"]
    assert isinstance(lt, pd.TimedeltaIndex)
    assert lt[0] == pd.Timedelta("0h")
    assert lt[-1] == pd.Timedelta("360h")
    assert pd.Timedelta("6h") in lt
    assert pd.Timedelta("354h") in lt
    assert len(lt) == 61  # should match chunk sizes in data_vars encoding

    lat = dc["latitude"]
    assert isinstance(lat, np.ndarray)
    assert lat[0] == 90.0
    assert lat[-1] == -90.0
    assert len(lat) == 721

    lon = dc["longitude"]
    assert isinstance(lon, np.ndarray)
    assert lon[0] == -180.0
    assert lon[-1] == 179.75
    assert len(lon) == 1440


def test_data_vars_date_available() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    expanded_date = pd.Timestamp("2025-02-26T00:00")

    vars_with_date = [
        v for v in config.data_vars if v.internal_attrs.date_available is not None
    ]
    vars_without_date = [
        v for v in config.data_vars if v.internal_attrs.date_available is None
    ]

    assert len(vars_with_date) > 0
    assert len(vars_without_date) > 0

    for v in vars_with_date:
        assert v.internal_attrs.date_available == expanded_date


def test_template_variables_have_required_attrs() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()

    for var in config.data_vars:
        assert var.name
        assert var.encoding
        assert var.internal_attrs.grib_index_param
        assert var.attrs.short_name
        assert var.attrs.long_name
        assert var.attrs.units
        assert var.attrs.step_type in {"instant", "avg", "accum", "max", "min"}


def test_geopotential_height_vars_have_scale_factor() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    gh_vars = [v for v in config.data_vars if "geopotential_height" in v.name]
    assert len(gh_vars) == 3  # 500, 850, 925 hPa

    g = 9.80665
    for var in gh_vars:
        assert var.internal_attrs.grib_index_param == "z"
        assert var.internal_attrs.scale_factor == pytest.approx(1 / g)
        assert var.internal_attrs.grib_index_level_type == "pl"
        assert var.attrs.units == "m"
        assert var.attrs.short_name == "gh"


def test_derive_coordinates() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    ds = config.get_template(config.append_dim_start + config.append_dim_frequency)

    assert "valid_time" in ds.coords
    assert "ingested_forecast_length" in ds.coords
    assert "expected_forecast_length" in ds.coords
    assert "spatial_ref" in ds.coords

    # valid_time = init_time + lead_time
    expected_valid_time = ds["init_time"] + ds["lead_time"]
    np.testing.assert_array_equal(ds["valid_time"].values, expected_valid_time.values)


def test_derive_coordinates_and_spatial_ref() -> None:
    cfg = EcmwfAifsSingleForecastTemplateConfig()
    dc = cfg.dimension_coordinates()
    ds = xr.Dataset(coords=dc)
    derived = cfg.derive_coordinates(ds)

    assert set(derived) == {
        "valid_time",
        "ingested_forecast_length",
        "expected_forecast_length",
        "spatial_ref",
    }

    vt = derived["valid_time"]
    assert isinstance(vt, xr.DataArray)
    assert vt.dims == ("init_time", "lead_time")
    assert vt.shape == (len(dc["init_time"]), len(dc["lead_time"]))

    dims, arr = derived["ingested_forecast_length"]
    assert dims == (cfg.append_dim,)
    assert arr.dtype == "timedelta64[ns]"
    assert np.all(pd.isna(arr))

    dims2, arr2 = derived["expected_forecast_length"]
    assert dims2 == (cfg.append_dim,)
    assert np.all(arr2 == dc["lead_time"].max())

    assert derived["spatial_ref"] == SPATIAL_REF_COORDS


def test_coords_property_order_and_names() -> None:
    cfg = EcmwfAifsSingleForecastTemplateConfig()
    names = [c.name for c in cfg.coords]
    assert names == [
        "init_time",
        "lead_time",
        "latitude",
        "longitude",
        "valid_time",
        "ingested_forecast_length",
        "expected_forecast_length",
        "spatial_ref",
    ]


def test_get_template_spatial_ref() -> None:
    template_config = EcmwfAifsSingleForecastTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )

    expected_crs = "+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs"
    calculated_spatial_ref_attrs = ds.rio.write_crs(expected_crs).spatial_ref.attrs
    original_attrs = dict(ds.spatial_ref.attrs)
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs
