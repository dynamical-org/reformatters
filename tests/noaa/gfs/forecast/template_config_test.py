import re
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig


def test_get_template_spatial_ref() -> None:
    """Ensure the spatial reference system in the template matched our expectation."""
    template_config = NoaaGfsForecastTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )
    original_attrs = deepcopy(ds.spatial_ref.attrs)

    expected_crs = "+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs"
    calculated_spatial_ref_attrs = ds.rio.write_crs(expected_crs).spatial_ref.attrs
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs


def test_dataset_attributes() -> None:
    cfg = NoaaGfsForecastTemplateConfig()
    attrs = cfg.dataset_attributes
    assert attrs.dataset_id == "noaa-gfs-forecast"
    assert re.match(r"\d+\.\d+\.\d+", attrs.dataset_version) is not None
    # time_domain mentions the configured start
    assert str(cfg.append_dim_start) in attrs.time_domain
    # time_resolution mentions the 6-hour frequency
    assert "every 6 hours" in attrs.time_resolution


def test_dimension_coordinates_shapes_and_values() -> None:
    cfg = NoaaGfsForecastTemplateConfig()
    dc = cfg.dimension_coordinates()
    # must have exactly these four dims
    assert set(dc) == {"init_time", "lead_time", "latitude", "longitude"}
    # init_time: only the start timestamp
    init = dc["init_time"]
    assert isinstance(init, pd.DatetimeIndex)
    assert len(init) == 1
    assert init[0] == cfg.append_dim_start
    # lead_time: contains 0h, 120h, 123h, and ends at 384h
    lt = dc["lead_time"]
    assert isinstance(lt, pd.TimedeltaIndex)
    assert pd.Timedelta("0h") == lt[0]
    assert pd.Timedelta("120h") in lt
    assert pd.Timedelta("123h") in lt
    assert pd.Timedelta("384h") == lt[-1]
    # latitude flips from +90 to -90 in 0.25° steps
    lat = dc["latitude"]
    assert isinstance(lat, np.ndarray)
    assert lat[0] == 90.0
    assert lat[-1] == -90.0
    assert len(lat) == 721
    # longitude from -180 to +179.75
    lon = dc["longitude"]
    assert isinstance(lon, np.ndarray)
    assert lon[0] == -180.0
    assert lon[-1] == 179.75
    assert len(lon) == 1440


def test_derive_coordinates_and_spatial_ref() -> None:
    cfg = NoaaGfsForecastTemplateConfig()
    dc = cfg.dimension_coordinates()
    ds = xr.Dataset(coords=dc)
    derived = cfg.derive_coordinates(ds)

    # check keys
    assert set(derived) == {
        "valid_time",
        "ingested_forecast_length",
        "expected_forecast_length",
        "spatial_ref",
    }

    # valid_time: DataArray of shape (init_time, lead_time)
    vt = derived["valid_time"]
    assert isinstance(vt, xr.DataArray)
    assert vt.dims == ("init_time", "lead_time")
    assert vt.shape == (len(dc["init_time"]), len(dc["lead_time"]))
    # ingested_forecast_length: all NaT (timedeltas)
    dims, arr = derived["ingested_forecast_length"]
    assert dims == (cfg.append_dim,)
    # dtype is timedelta64[ns], not datetime64
    assert arr.dtype == "timedelta64[ns]"
    assert np.all(pd.isna(arr))
    # expected_forecast_length: filled with max lead_time
    dims2, arr2 = derived["expected_forecast_length"]
    assert dims2 == (cfg.append_dim,)
    assert np.all(arr2 == dc["lead_time"].max())
    # spatial_ref must match the constant
    assert derived["spatial_ref"] == SPATIAL_REF_COORDS


def test_coords_property_order_and_names() -> None:
    cfg = NoaaGfsForecastTemplateConfig()
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
