import warnings

import numpy as np
import pandas as pd
import xarray as xr
import zarr  # type: ignore

from common.config import Config
from common.download_directory import download_directory
from common.types import DatetimeLike
from noaa.gefs.forecast.read_data import download_file, read_file

TEMPLATE_PATH = "noaa/gefs/forecast/templates/latest.zarr"

_INIT_TIME_START = pd.Timestamp("2024-09-01T00:00")
_INIT_TIME_FREQUENCY = pd.Timedelta("6h")
_CHUNKS = {
    "init_time": 1,
    "ensemble_member": 31,  # all ensemble members in one chunk
    "lead_time": 125,  # all lead times in one chunk
    "latitude": 73,  # 10 chunks over 721 pixels
    "longitude": 72,  # 20 chunks over 1440 pixels
}


def get_template(init_time_end: DatetimeLike) -> xr.Dataset:
    ds: xr.Dataset = xr.open_zarr(TEMPLATE_PATH)

    # Template is stored with a single init time, expand to the full range of init times.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore a "performace" warning
        ds = ds.reindex(init_time=_get_init_time_coordinates(init_time_end))

    # Init time chunks are 1 when stored, set them to desired.
    ds = ds.chunk(init_time=_CHUNKS["init_time"])

    return ds


def _get_init_time_coordinates(
    init_time_end: DatetimeLike,
) -> pd.DatetimeIndex:
    return pd.date_range(
        _INIT_TIME_START, init_time_end, freq=_INIT_TIME_FREQUENCY, inclusive="left"
    )


def update_template() -> None:
    dims = ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")
    assert dims == tuple(_CHUNKS.keys())

    coords = {
        "init_time": _get_init_time_coordinates(
            _INIT_TIME_START + _INIT_TIME_FREQUENCY
        ),
        "ensemble_member": np.arange(31),
        "lead_time": pd.timedelta_range("0h", "240h", freq="3h"),
        # latitude descends when north is up
        "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
        "longitude": np.arange(-180, 180, 0.25),
    }

    # Pull a single file to load variable names and metadata.
    # Use a lead time > 0 because not all variables are present at lead time == 0.
    with download_directory() as directory:
        path = download_file(
            pd.Timestamp("2024-01-01T00:00"), pd.Timedelta("3h"), 0, directory
        )
        ds = read_file(path)

        ds = (
            ds.chunk(_CHUNKS)  # daskify for faster, lazy reindex
            .reindex(
                ensemble_member=coords["ensemble_member"], lead_time=coords["lead_time"]
            )
            .assign_coords(coords)
            .chunk(_CHUNKS)
        )

        for var_name, data_var in ds.data_vars.items():
            ds[var_name].encoding = {
                "dtype": np.float32,
                "chunks": [_CHUNKS[str(dim)] for dim in data_var.dims],
                "compressor": zarr.Blosc(cname="zstd", clevel=4),
            }
        # TODO
        # Explicit coords encoding
        # Improve metadata
        ds.to_zarr(TEMPLATE_PATH, mode="w", compute=False)
