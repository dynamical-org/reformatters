import numpy as np
import pandas as pd
import xarray as xr
import zarr  # type: ignore

from noaa.gefs.forecast.reformat import download_and_load_source_file

TEMPLATE_PATH = "noaa/gefs/forecast/templates/v0.0.1.zarr"


def get_template() -> xr.Dataset:
    return xr.open_zarr(TEMPLATE_PATH)  # type: ignore


def update_template() -> None:
    dims = ("init_time", "lead_time", "latitude", "longitude")
    chunks = {"init_time": 1, "lead_time": 125, "latitude": 145, "longitude": 144}
    assert dims == tuple(chunks.keys())

    coords = {
        "init_time": pd.date_range("2024-01-01T00:00Z", "2024-01-01T00:00Z", freq="6h"),
        "lead_time": pd.timedelta_range("3h", "240h", freq="3h"),
        # latitude descends when north is up
        "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
        "longitude": np.arange(-180, 180, 0.25),
    }
    assert dims == tuple(coords.keys())

    ds = download_and_load_source_file(
        pd.Timestamp("2024-01-01T00:00Z"), pd.Timedelta(0)
    )

    ds = (
        ds.chunk(-1)
        .reindex(lead_time=coords["lead_time"])
        .assign_coords(coords)
        .chunk(chunks)
    )

    for var in ds.data_vars:
        ds[var].encoding = {
            "dtype": np.float32,
            "chunks": [chunks[str(dim)] for dim in ds.dims],
            "compressor": zarr.Blosc(cname="zstd", clevel=4),
        }
    # TODO
    # Explicit coords encoding
    # Improve metadata
    ds.to_zarr(TEMPLATE_PATH, mode="w", compute=False)
