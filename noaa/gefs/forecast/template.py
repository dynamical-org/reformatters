from collections.abc import Hashable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import BitRound, Blosc  # type: ignore

from common.config import Config  # noqa:F401
from common.download_directory import cd_into_download_directory
from common.types import DatetimeLike
from noaa.gefs.forecast.read_data import download_file, read_file

TEMPLATE_PATH = "noaa/gefs/forecast/templates/latest.zarr"

_DATASET_ID = "noaa-gefs-forecast"
_INIT_TIME_START = pd.Timestamp("2024-09-01T00:00")
_INIT_TIME_FREQUENCY = pd.Timedelta("6h")
_DIMS = ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")
_CHUNKS = {
    "init_time": 1,
    "ensemble_member": 31,  # all ensemble members in one chunk
    "lead_time": 125,  # all lead times in one chunk
    "latitude": 73,  # 10 chunks over 721 pixels
    "longitude": 72,  # 20 chunks over 1440 pixels
}
_CHUNKS_ORDERED = tuple(_CHUNKS[dim] for dim in _DIMS)

# Use this with an additional add_offset: <median> option
# to minimize loss from binary rounding
_FLOAT_DEFAULT = {
    "dtype": np.float32,
    "chunks": _CHUNKS_ORDERED,
    "filters": [BitRound(keepbits=7)],
    "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
}
_CATEGORICAL_WITH_MISSING_DEFAULT = {
    "dtype": np.float32,
    "chunks": _CHUNKS_ORDERED,
    "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
}
_ENCODING = {
    "cfrzr": _CATEGORICAL_WITH_MISSING_DEFAULT,
    "cicep": _CATEGORICAL_WITH_MISSING_DEFAULT,
    "cpofp": _FLOAT_DEFAULT,
    "crain": _CATEGORICAL_WITH_MISSING_DEFAULT,
    "csnow": _CATEGORICAL_WITH_MISSING_DEFAULT,
    "d2m": {**_FLOAT_DEFAULT, "add_offset": 273.15},
    "gh": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=8)]},
    "gust": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
    "hlcy": _FLOAT_DEFAULT,
    "mslet": {**_FLOAT_DEFAULT, "add_offset": 101_000.0},
    "mslhf": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
    "msshf": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
    "prmsl": {**_FLOAT_DEFAULT, "add_offset": 101_000.0},
    "pwat": _FLOAT_DEFAULT,
    "r2": {**_FLOAT_DEFAULT, "add_offset": 50.0, "filters": [BitRound(keepbits=6)]},
    "sde": _FLOAT_DEFAULT,
    "sdlwrf": {**_FLOAT_DEFAULT, "add_offset": 300.0},
    "sdswrf": _FLOAT_DEFAULT,
    "sdwe": _FLOAT_DEFAULT,
    "sithick": _FLOAT_DEFAULT,
    "soilw": _FLOAT_DEFAULT,
    "sp": {**_FLOAT_DEFAULT, "add_offset": 100_000.0},
    "st": {**_FLOAT_DEFAULT, "add_offset": 273.15},
    "suswrf": _FLOAT_DEFAULT,
    "t2m": {**_FLOAT_DEFAULT, "add_offset": 273.15},
    "tcc": {**_FLOAT_DEFAULT, "add_offset": 50.0},
    "tmax": {**_FLOAT_DEFAULT, "add_offset": 273.15},
    "tmin": {**_FLOAT_DEFAULT, "add_offset": 273.15},
    "tp": _FLOAT_DEFAULT,
    "u10": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
    "v10": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
    "vis": {**_FLOAT_DEFAULT, "add_offset": 15_000.0},
}


def get_template(init_time_end: DatetimeLike) -> xr.Dataset:
    ds: xr.Dataset = xr.open_zarr(TEMPLATE_PATH)

    # Expand init_time dimension with complete coordinates
    ds = ds.reindex(init_time=_get_init_time_coordinates(init_time_end))
    # Init time chunks are 1 when stored, set them to desired.
    ds = ds.chunk(init_time=_CHUNKS["init_time"])

    # Uncomment to make smaller zarr while developing
    # if Config.is_dev():
    #     ds = ds.isel(ensemble_member=slice(5), lead_time=slice(24))

    return ds


def _get_init_time_coordinates(
    init_time_end: DatetimeLike,
) -> pd.DatetimeIndex:
    return pd.date_range(
        _INIT_TIME_START, init_time_end, freq=_INIT_TIME_FREQUENCY, inclusive="left"
    )


def update_template() -> None:
    assert _DIMS == tuple(_CHUNKS.keys())

    coords = {
        "init_time": _get_init_time_coordinates(
            _INIT_TIME_START + _INIT_TIME_FREQUENCY
        ),
        "ensemble_member": np.arange(31),
        "lead_time": pd.timedelta_range("0h", "240h", freq="3h"),
        # TODO pull arange arguments from GRIB attributes
        # latitude descends when north is up
        "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
        "longitude": np.arange(-180, 180, 0.25),
    }

    # Resolve to absolue path before changing directories
    template_path = Path(TEMPLATE_PATH).absolute()

    # Pull a single file to load variable names and metadata.
    # Use a lead time > 0 because not all variables are present at lead time == 0.
    with cd_into_download_directory() as directory:
        path = download_file(
            pd.Timestamp("2024-01-01T00:00"), 0, pd.Timedelta("3h"), directory
        )
        ds = read_file(path.name)

        # Expand ensemble and lead time dimensions + set coordinates and chunking
        ds = (
            ds.sel(ensemble_member=coords["ensemble_member"], method="nearest")
            .sel(lead_time=coords["lead_time"], method="nearest")
            .assign_coords(coords)
            .chunk(_CHUNKS)
        )

        for var_name in ds.data_vars.keys():
            ds[var_name].encoding = _ENCODING[var_name]

        # TODO
        # Explicit coords encoding
        # Improve metadata
        ds.to_zarr(template_path, mode="w", compute=False)


def chunk_args(ds: xr.Dataset) -> dict[Hashable, int]:
    """Returns {dim: chunk_size} mapping suitable to pass to ds.chunk()"""
    return {dim: chunk_sizes[0] for dim, chunk_sizes in ds.chunksizes.items()}
