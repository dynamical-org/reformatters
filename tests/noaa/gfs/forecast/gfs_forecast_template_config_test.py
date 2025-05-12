import numpy as np
import pandas as pd
import xarray as xr

from reformatters.noaa.gfs.forecast.template_config import GFS_FORECAST_TEMPLATE_CONFIG
from reformatters.common.template_config import SPATIAL_REF_COORDS


def test_dataset_attributes():
    cfg = GFS_FORECAST_TEMPLATE_CONFIG
    attrs = cfg.dataset_attributes
    assert attrs.dataset_id == "noaa-gfs-forecast"
    assert attrs.dataset_version == "0.1.0"
    # time_domain mentions the configured start
    assert str(cfg.append_dim_start) in attrs.time_domain
    # time_resolution mentions the 6-hour frequency
    assert "every 6 hours" in attrs.time_resolution


def test_dimension_coordinates_shapes_and_values():
    cfg = GFS_FORECAST_TEMPLATE_CONFIG
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
    # latitude flips from +90 to –90 in 0.25° steps
    lat = dc["latitude"]
    assert isinstance(lat, np.ndarray)
    assert lat[0] == 90.0
    assert lat[-1] == -90.0
    assert len(lat) == 721
    # longitude from –180 to +179.75
    lon = dc["longitude"]
    assert isinstance(lon, np.ndarray)
    assert lon[0] == -180.0
    assert lon[-1] == 179.75
    assert len(lon) == 1440


def test_derive_coordinates_and_spatial_ref():
    cfg = GFS_FORECAST_TEMPLATE_CONFIG
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
    # ingested_forecast_length: all NaT
    dims, arr = derived["ingested_forecast_length"]
    assert dims == (cfg.append_dim,)
    assert arr.dtype == "datetime64[ns]"
    assert np.all(pd.isna(arr))
    # expected_forecast_length: filled with max lead_time
    dims2, arr2 = derived["expected_forecast_length"]
    assert dims2 == (cfg.append_dim,)
    assert np.all(arr2 == dc["lead_time"].max())
    # spatial_ref must match the constant
    assert derived["spatial_ref"] == SPATIAL_REF_COORDS


def test_coords_property_order_and_names():
    cfg = GFS_FORECAST_TEMPLATE_CONFIG
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
