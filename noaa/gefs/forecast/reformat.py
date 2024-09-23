from collections.abc import Iterable
from itertools import starmap
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from common.config import Config
from common.types import DatetimeLike, StoreLike
from noaa.gefs.forecast import template
from noaa.gefs.forecast.read_data import download_and_load_source_file


def local_reformat(init_time_end: DatetimeLike) -> None:
    template_ds = template.get_template(init_time_end)
    store = get_store()
    template_ds.to_zarr(store, mode=get_mode(store), compute=False)
    reformat_chunks(init_time_end, worker_index=0, workers_total=1)


def kubernetes_reformat(init_time_end: DatetimeLike) -> None:
    template_ds = template.get_template(init_time_end)
    store = get_store()
    template_ds.to_zarr(store, mode=get_mode(store), compute=False)
    # TODO
    # build and push docker image
    # create and launch kubernetes job


def reformat_chunks(
    init_time_end: DatetimeLike, *, worker_index: int, workers_total: int
) -> None:
    """Writes out array chunk data. Assumes the dataset metadata has already been written."""
    template_ds = template.get_template(init_time_end)
    store = get_store()

    worker_init_time_i_slices = get_worker_jobs(
        chunk_i_slices(template_ds, "init_time"), worker_index, workers_total
    )

    for init_time_i_slice in worker_init_time_i_slices:
        chunk_template_ds = template_ds.isel(init_time=init_time_i_slice)
        datasets = [
            download_and_load_source_file(init_time, lead_time)
            for lead_time in pd.to_timedelta(chunk_template_ds["lead_time"])  # type: ignore
            for init_time in pd.to_datetime(chunk_template_ds["init_time"])  # type: ignore
        ]
        ds = xr.merge(datasets)
        ds.to_zarr(store, region="auto")
        pass


def get_worker_jobs[T](
    jobs: Iterable[T], worker_index: int, workers_total: int
) -> Iterable[T]:
    """Returns the subset of `jobs` that worker_index should process if there are workers_total workers."""
    return list(jobs)[worker_index::workers_total]


def get_store() -> StoreLike:
    if Config.is_dev():
        return "output/noaa/gefs/forecast/dev.zarr"
    raise NotImplementedError(f"Store not defined in env {Config.env}")


def get_mode(store: StoreLike) -> Literal["w-", "w"]:
    if isinstance(store, str) and store.endswith("dev.zarr"):
        return "w"  # Allow overwritting dev store

    return "w-"  # Safe default - don't overwrite


def chunk_i_slices(ds: xr.Dataset, dim: str) -> Iterable[slice]:
    stop_idx = np.cumsum(ds.chunksizes[dim])
    start_idx = np.insert(stop_idx, 0, 0)
    return starmap(slice, zip(start_idx, stop_idx, strict=False))
