from collections.abc import Hashable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import BitRound, Blosc, Delta  # type: ignore

from common.config import Config  # noqa:F401
from common.download_directory import cd_into_download_directory
from common.types import DatetimeLike, StoreLike
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
    "init_time": {
        "dtype": np.int64,
        "filters": [Delta(np.int64)],
        "compressor": Blosc(cname="zstd"),
        "calendar": "proleptic_gregorian",
        "units": "seconds since 1970-01-01 00:00:00",
        "chunks": -1,
    },
    "ensemble_member": {
        "dtype": np.uint16,
        "chunks": -1,
    },
    "lead_time": {
        "dtype": np.int64,
        "compressor": Blosc(cname="zstd"),
        "units": "seconds",
        "chunks": -1,
    },
    "latitude": {
        "dtype": np.float64,
        "compressor": Blosc(cname="zstd"),
        "chunks": -1,
    },
    "longitude": {
        "dtype": np.float64,
        "compressor": Blosc(cname="zstd"),
        "chunks": -1,
    },
    "valid_time": {
        "dtype": np.int64,
        "filters": [Delta(np.int64)],
        "compressor": Blosc(cname="zstd"),
        "calendar": "proleptic_gregorian",
        "units": "seconds since 1970-01-01 00:00:00",
        "chunks": [-1, -1],
    },
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
    "u100": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
    "v10": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
    "v100": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
    "vis": {**_FLOAT_DEFAULT, "add_offset": 15_000.0},
}

# TODO: Some of the s+a should actually be s+b, check the
# index files to see which one.
_CUSTOM_ATTRIBUTES = {
    "cfrzr": {
        "noaa_file_type": "s+a",
        "noaa_variable": "CFRZR",
        "noaa_level": "surface",
    },
    "cicep": {"noaa_file_type": "s+a"},
    "cpofp": {"noaa_file_type": "s+a"},
    "crain": {"noaa_file_type": "s+a"},
    "csnow": {"noaa_file_type": "s+a"},
    "d2m": {"noaa_file_type": "s+a"},
    "gh": {"noaa_file_type": "s+a"},
    "gust": {"noaa_file_type": "s+a"},
    "hlcy": {"noaa_file_type": "s+a"},
    "mslet": {"noaa_file_type": "s+a"},
    "mslhf": {"noaa_file_type": "s+a"},
    "msshf": {"noaa_file_type": "s+a"},
    "prmsl": {
        "noaa_file_type": "s+a",
        "noaa_variable": "PRMSL",
        "noaa_level": "mean sea level",
    },
    "pwat": {"noaa_file_type": "s+a"},
    "r2": {"noaa_file_type": "s+a"},
    "sde": {"noaa_file_type": "s+a"},
    "sdlwrf": {"noaa_file_type": "s+a"},
    "sdswrf": {"noaa_file_type": "s+a"},
    "sdwe": {"noaa_file_type": "s+a"},
    "sithick": {"noaa_file_type": "s+a"},
    "soilw": {"noaa_file_type": "s+a"},
    "sp": {"noaa_file_type": "s+a"},
    "st": {"noaa_file_type": "s+a"},
    "suswrf": {"noaa_file_type": "s+a"},
    "t2m": {
        "noaa_file_type": "s+a",
        "noaa_variable": "TMP",
        "noaa_level": "2 m above ground",
    },
    "tcc": {"noaa_file_type": "s+a"},
    "tmax": {"noaa_file_type": "s+a"},
    "tmin": {"noaa_file_type": "s+a"},
    "tp": {"noaa_file_type": "s+a"},
    "u10": {
        "noaa_file_type": "s+a",
        "noaa_variable": "UGRD",
        "noaa_level": "10 m above ground",
    },
    "u100": {
        "noaa_file_type": "b",
        "noaa_variable": "UGRD",
        "noaa_level": "100 m above ground",
    },
    "v10": {
        "noaa_file_type": "s+a",
        "noaa_variable": "VGRD",
        "noaa_level": "10 m above ground",
    },
    "v100": {
        "noaa_file_type": "b",
        "noaa_variable": "VGRD",
        "noaa_level": "100 m above ground",
    },
    "vis": {"noaa_file_type": "s+b", "noaa_variable": "VIS", "noaa_level": "surface"},
}


def get_template(init_time_end: DatetimeLike) -> xr.Dataset:
    ds: xr.Dataset = xr.open_zarr(TEMPLATE_PATH)

    # Expand init_time dimension with complete coordinates
    ds = ds.reindex(init_time=_get_init_time_coordinates(init_time_end))
    # Init time chunks are 1 when stored, set them to desired.
    ds = ds.chunk(init_time=_CHUNKS["init_time"])
    # Recompute valid time after reindex
    ds.coords["valid_time"] = ds["init_time"] + ds["lead_time"]

    # Coordinates which are dask arrays are not written with
    # to_zarr(store, compute=False) so we ensure all coordinates are loaded.
    for coordinate in ds.coords.values():
        assert isinstance(coordinate.data, np.ndarray)

    # Uncomment to make smaller zarr while developing
    if Config.is_dev():
        ds = ds.isel(ensemble_member=slice(5), lead_time=slice(24))

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
            pd.Timestamp("2024-01-01T00:00"),
            0,
            "s+a",
            pd.Timedelta("3h"),
            [_CUSTOM_ATTRIBUTES[var] for var in ["u10"]],
            directory,
        )
        ds = read_file(path.name)

        # Expand ensemble and lead time dimensions + set coordinates and chunking
        ds = (
            ds.sel(ensemble_member=coords["ensemble_member"], method="nearest")
            .sel(lead_time=coords["lead_time"], method="nearest")
            .assign_coords(coords)
            .chunk(_CHUNKS)
        )

        # Remove left over coordinates encoding for coords we don't keep
        for data_var in ds.data_vars.values():
            del data_var.encoding["coordinates"]

        # This could be computed by users on the fly, but it compresses
        # really well so lets make things easy for users
        ds.coords["valid_time"] = ds["init_time"] + ds["lead_time"]

        # Add custom attributes
        for var_name, data_var in ds.data_vars.items():
            data_var.attrs.update(_CUSTOM_ATTRIBUTES[var_name])

        ds_keys = list(ds.keys()) + list(ds.coords.keys())
        if len(missing_encodings := [v for v in ds_keys if v not in _ENCODING]) != 0:
            raise ValueError(f"Missing encodings for {missing_encodings}")

        write_metadata(ds, template_path, mode="w")


def write_metadata(
    template_ds: xr.Dataset,
    store: StoreLike,
    mode: Literal["w", "w-", "a", "a-", "r+", "r"],
) -> None:
    template_ds.to_zarr(store, mode=mode, compute=False, encoding=_ENCODING)
    print(f"Wrote metadata to {store} with mode {mode}.")


def chunk_args(ds: xr.Dataset) -> dict[Hashable, int]:
    """Returns {dim: chunk_size} mapping suitable to pass to ds.chunk()"""
    return {dim: chunk_sizes[0] for dim, chunk_sizes in ds.chunksizes.items()}
