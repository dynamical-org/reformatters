import warnings

import numpy as np
import pandas as pd
import xarray as xr
import zarr  # type: ignore

from common.config import Config
from common.types import DatetimeLike
from noaa.gefs.forecast.read_data import download_and_load_source_file

TEMPLATE_PATH = "noaa/gefs/forecast/templates/latest.zarr"

_INIT_TIME_START = pd.Timestamp("2024-01-01T00:00")
_INIT_TIME_FREQUENCY = pd.Timedelta("6h")
_CHUNKS = {"init_time": 2, "lead_time": 125, "latitude": 145, "longitude": 144}


def get_template(init_time_end: DatetimeLike) -> xr.Dataset:
    ds: xr.Dataset = xr.open_zarr(TEMPLATE_PATH)

    # Template is stored with a single init time, expand to the full range of init times.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore a "performace" warning
        ds = ds.reindex(init_time=_get_init_time_coordinates(init_time_end))

    # Init time chunks are 1 when stored, set them to desired.
    ds = ds.chunk(init_time=_CHUNKS["init_time"])

    if Config.is_dev():
        ds = ds.isel(init_time=slice(2), lead_time=slice(3))

    return ds


def _get_init_time_coordinates(
    init_time_end: DatetimeLike,
) -> pd.DatetimeIndex:
    return pd.date_range(
        _INIT_TIME_START, init_time_end, freq=_INIT_TIME_FREQUENCY, inclusive="left"
    )


def update_template() -> None:
    dims = ("init_time", "lead_time", "latitude", "longitude")
    assert dims == tuple(_CHUNKS.keys())

    coords = {
        "init_time": _get_init_time_coordinates(
            _INIT_TIME_START + _INIT_TIME_FREQUENCY
        ),
        "lead_time": pd.timedelta_range("3h", "240h", freq="3h"),
        # latitude descends when north is up
        "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
        "longitude": np.arange(-180, 180, 0.25),
    }
    assert dims == tuple(coords.keys())

    # Pull a single file to load variable names and metadata.
    # Use a lead time > 0 because not all variables are present at lead time == 0.
    ds = download_and_load_source_file(
        pd.Timestamp("2024-01-01T00:00"), pd.Timedelta("3h")
    )

    ds = (
        ds.chunk(-1)
        .reindex(lead_time=coords["lead_time"])
        .assign_coords(coords)
        .chunk(_CHUNKS)
    )

    for var in ds.data_vars:
        ds[var].encoding = {
            "dtype": np.float32,
            "chunks": [_CHUNKS[str(dim)] for dim in ds.dims],
            "compressor": zarr.Blosc(cname="zstd", clevel=4),
        }
    # TODO
    # Explicit coords encoding
    # Improve metadata
    ds.to_zarr(TEMPLATE_PATH, mode="w", compute=False)
