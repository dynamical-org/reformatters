import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from itertools import starmap
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from common.config import Config
from common.download_directory import download_directory
from common.types import DatetimeLike, StoreLike
from noaa.gefs.forecast import template
from noaa.gefs.forecast.read_data import download_file, read_file


def reformat_local(init_time_end: DatetimeLike) -> None:
    template_ds = template.get_template(init_time_end)
    store = get_store()
    print("writing meta")
    template_ds.to_zarr(store, mode=get_mode(store), compute=False)
    # Process all chunks
    print("starting reformat")
    reformat_chunks(init_time_end, worker_index=0, workers_total=1)


def reformat_kubernetes(init_time_end: DatetimeLike) -> None:
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

    thread_executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    # If we compile eccodes ourselves with thread safety enabled we could use threads for reading
    # https://confluence.ecmwf.int/display/ECC/ecCodes+installation ENABLE_ECCODES_THREADS
    proccess_executor = ProcessPoolExecutor(max_workers=os.cpu_count())

    for init_time_i_slice in worker_init_time_i_slices:
        chunk_template_ds = template_ds.isel(init_time=init_time_i_slice)

        chunk_init_times = pd.to_datetime(chunk_template_ds["init_time"].values)
        chunk_lead_times = pd.to_timedelta(chunk_template_ds["lead_time"].values)
        chunk_ensemble_members = chunk_template_ds["ensemble_member"]

        print("starting", chunk_init_times)

        with download_directory() as dir:
            init_time_datasets = []
            for init_time in chunk_init_times:
                ensemble_member_datasets = []
                for ensemble_member in chunk_ensemble_members:
                    download = partial(
                        download_file, init_time, ensemble_member, directory=dir
                    )
                    file_paths = tuple(thread_executor.map(download, chunk_lead_times))
                    datasets = tuple(proccess_executor.map(read_file, file_paths))
                    ensemble_member_ds = xr.concat(
                        datasets, dim="lead_time", join="exact"
                    )

                    # TODO decide on complete set of variables to include
                    if "orog" in ensemble_member_ds:
                        ensemble_member_ds = ensemble_member_ds.drop_vars("orog")

                    ensemble_member_datasets.append(ensemble_member_ds)

                init_time_ds = xr.concat(
                    ensemble_member_datasets, dim="ensemble_member", join="exact"
                )
                init_time_datasets.append(init_time_ds)

            chunk_ds = xr.concat(init_time_datasets, dim="init_time", join="exact")

            chunk_ds.chunk(-1).to_zarr(store, region="auto")

        print(chunk_init_times, "processed")


def get_worker_jobs[T](
    jobs: Iterable[T], worker_index: int, workers_total: int
) -> Iterable[T]:
    """Returns the subset of `jobs` that worker_index should process if there are workers_total workers."""
    return list(jobs)[worker_index::workers_total]


def get_store() -> StoreLike:
    return "data/output/noaa/gefs/forecast/dev.zarr"


def get_mode(store: StoreLike) -> Literal["w-", "w"]:
    if isinstance(store, str) and store.endswith("dev.zarr"):
        return "w"  # Allow overwritting dev store

    return "w-"  # Safe default - don't overwrite


def chunk_i_slices(ds: xr.Dataset, dim: str) -> Iterable[slice]:
    """Returns the integer offset slices which correspond to each chunk along `dim` of `ds`."""
    stop_idx = np.cumsum(ds.chunksizes[dim])  # 2, 4, 6
    start_idx = np.insert(stop_idx, 0, 0)  # 0, 2, 4, 6
    return starmap(slice, zip(start_idx, stop_idx, strict=False))  # (0,2), (2,4), (4,6)
