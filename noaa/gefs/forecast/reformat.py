import concurrent.futures
import os
import re
import subprocess
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from itertools import batched, islice, product, starmap
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import s3fs  # type: ignore
import xarray as xr

from common import string_template
from common.config import Config  # noqa:F401
from common.download_directory import cd_into_download_directory
from common.types import DatetimeLike, StoreLike
from noaa.gefs.forecast import template
from noaa.gefs.forecast.config_models import DataVar
from noaa.gefs.forecast.read_data import (
    NoaaFileType,
    SourceFileCoords,
    download_file,
    read_into,
)

_PROCESSING_CHUNK_DIMENSION = "init_time"


def reformat_local(init_time_end: DatetimeLike) -> None:
    template_ds = template.get_template(init_time_end)
    store = get_store()

    template.write_metadata(template_ds, store, get_mode(store))

    print("Starting reformat")
    # Process all chunks by setting worker_index=0 and worker_total=1
    reformat_chunks(init_time_end, worker_index=0, workers_total=1)
    print("Done writing to", store)


def reformat_kubernetes(
    init_time_end: DatetimeLike, jobs_per_pod: int, max_parallelism: int
) -> None:
    template_ds = template.get_template(init_time_end)

    job_timestamp = pd.Timestamp.now("UTC").strftime("%Y-%m-%dt%M-%S")

    docker_repo = os.environ["DOCKER_REPOSITORY"]
    assert re.fullmatch(r"[0-9a-zA-Z_\.\-\/]{1,1000}", docker_repo)
    image_tag = f"{docker_repo}:{job_timestamp}"

    subprocess.run(  # noqa: S603  allow passing variable to subprocess, it's realtively sanitized above
        [
            "/usr/bin/docker",
            "build",
            "--file",
            "deploy/Dockerfile",
            "--tag",
            image_tag,
            ".",
        ],
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["/usr/bin/docker", "push", image_tag], check=True
    )
    print("Pushed", image_tag)

    store = get_store()
    print("Writing zarr metadata")
    template.write_metadata(template_ds, store, get_mode(store))

    num_jobs = sum(1 for _ in chunk_i_slices(template_ds, _PROCESSING_CHUNK_DIMENSION))
    workers_total = int(np.ceil(num_jobs / jobs_per_pod))
    parallelism = min(workers_total, max_parallelism)

    dataset_id = template_ds.attrs["dataset_id"]
    job_name = f"{dataset_id}-{job_timestamp}"
    kubernetes_job = string_template.substitute(
        "deploy/kubernetes_ingest_job.yaml",
        {
            "NAME": job_name,
            "IMAGE": image_tag,
            "DATASET_ID": dataset_id,
            "INIT_TIME_END": pd.Timestamp(init_time_end).isoformat(),
            "WORKERS_TOTAL": workers_total,
            "PARALLELISM": parallelism,
            "CPU": 16,
            "MEMORY": "64G",
            "EPHEMERAL_STORAGE": "200G",
        },
    )

    subprocess.run(  # noqa: S603
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=kubernetes_job,
        text=True,
        check=True,
    )

    print("Submitted kubernetes job", job_name)


def reformat_chunks(
    init_time_end: DatetimeLike, *, worker_index: int, workers_total: int
) -> None:
    """Writes out array chunk data. Assumes the dataset metadata has already been written."""
    assert worker_index < workers_total
    template_ds = template.get_template(init_time_end)
    store = get_store()

    worker_init_time_i_slices = list(
        get_worker_jobs(
            chunk_i_slices(template_ds, _PROCESSING_CHUNK_DIMENSION),
            worker_index,
            workers_total,
        )
    )

    print(f"This is {worker_index = }, {workers_total = }, {worker_init_time_i_slices}")

    data_var_groups = group_data_vars_by_noaa_file_type(
        [d for d in template.DATA_VARIABLES if d.name in template_ds]
    )

    wait_executor = ThreadPoolExecutor(max_workers=256)
    io_executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)
    cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    # # If we compile eccodes ourselves with thread safety enabled we could use threads for reading
    # # https://confluence.ecmwf.int/display/ECC/ecCodes+installation ENABLE_ECCODES_THREADS
    # # but make sure to read thread safety comment in our `read_data` function.
    # proccess_executor = ProcessPoolExecutor(max_workers=os.cpu_count())

    for init_time_i_slice in worker_init_time_i_slices:
        chunk_template_ds = template_ds.isel(init_time=init_time_i_slice)

        chunk_init_times = pd.to_datetime(chunk_template_ds["init_time"].values)
        chunk_lead_times = pd.to_timedelta(chunk_template_ds["lead_time"].values)
        chunk_ensemble_members = chunk_template_ds["ensemble_member"].values
        chunk_coords: list[SourceFileCoords] = [
            {
                "init_time": init_time,
                "ensemble_member": ensemble_member,
                "lead_time": lead_time,
            }
            for init_time, ensemble_member, lead_time in product(
                chunk_init_times, chunk_ensemble_members, chunk_lead_times
            )
        ]

        chunk_init_times_str = ", ".join(chunk_init_times.strftime("%Y-%m-%dT%H:%M"))
        print("Starting chunk with init times", chunk_init_times_str)

        with cd_into_download_directory() as directory:
            download_var_group_futures: dict[
                Future[list[tuple[SourceFileCoords, Path]]], tuple[DataVar, ...]
            ] = {}
            for noaa_file_type, data_vars in data_var_groups:
                download_var_group_futures[
                    wait_executor.submit(
                        download_var_group_files,
                        data_vars,
                        chunk_coords,
                        noaa_file_type,
                        directory,
                        io_executor,
                    )
                ] = data_vars
                # TODO: this necessary? allow all of this group's downloads to be submitted to io_executor so a group is more likely to finish together and reading can begin
                time.sleep(0.1)

            for future in concurrent.futures.as_completed(download_var_group_futures):
                if (e := future.exception()) is not None:
                    raise e

                coords_and_paths = future.result()
                data_vars = download_var_group_futures[future]

                # Write variable by variable to avoid blowing up memory usage
                for data_var in data_vars:
                    # Skip reading the 0-hour for accumulated values
                    if data_var.attrs.step_type == "accum":
                        var_coords_and_paths = [
                            coords_and_path
                            for coords_and_path in coords_and_paths
                            if coords_and_path[0]["lead_time"] > pd.Timedelta(hours=0)
                        ]
                    else:
                        var_coords_and_paths = coords_and_paths
                    data_array = xr.full_like(chunk_template_ds[data_var.name], np.nan)
                    data_array.load()  # preallocate backing numpy arrays (for performance?)
                    # valid_time is a coordinate and already written with different chunks
                    data_array = data_array.drop_vars("valid_time")
                    print("reading...")
                    consume(
                        cpu_executor.map(
                            partial(
                                read_into,
                                data_array,
                                data_var=data_var,
                            ),
                            *zip(*var_coords_and_paths, strict=True),
                        )
                    )

                    assert np.isfinite(data_array).any()
                    print(f"Writing {data_var.name} {chunk_init_times_str}")
                    chunks = template.chunk_args(chunk_template_ds)
                    data_array.chunk(chunks).to_zarr(store, region="auto")


def download_var_group_files(
    idx_data_vars: Iterable[DataVar],
    chunk_coords: Iterable[SourceFileCoords],
    noaa_file_type: NoaaFileType,
    directory: Path,
    io_executor: ThreadPoolExecutor,
) -> list[tuple[SourceFileCoords, Path]]:
    local_paths = io_executor.map(
        lambda coord: download_file(
            **coord,
            noaa_file_type=noaa_file_type,
            directory=directory,
            noaa_idx_data_vars=idx_data_vars,
        ),
        chunk_coords,
    )
    return list(zip(chunk_coords, local_paths, strict=True))


def group_data_vars_by_noaa_file_type(
    data_vars: Iterable[DataVar], group_size: int = 4
) -> list[tuple[NoaaFileType, tuple[DataVar, ...]]]:
    grouper = defaultdict(list)
    for data_var in data_vars:
        noaa_file_type = data_var.internal_attrs.noaa_file_type
        grouper[noaa_file_type].append(data_var)
    chunks = []
    for file_type, idx_data_vars in grouper.items():
        idx_data_vars = sorted(
            idx_data_vars, key=lambda data_var: data_var.internal_attrs.index_position
        )
        chunks.extend(
            [
                (file_type, data_vars_chunk)
                for data_vars_chunk in batched(idx_data_vars, group_size)
            ]
        )

    return chunks


def get_worker_jobs[T](
    jobs: Iterable[T], worker_index: int, workers_total: int
) -> Iterable[T]:
    """Returns the subset of `jobs` that worker_index should process if there are workers_total workers."""
    return islice(jobs, worker_index, None, workers_total)


def get_store() -> StoreLike:
    if Config.is_dev():
        return Path("data/output/noaa/gefs/forecast/dev.zarr").absolute()

    s3 = s3fs.S3FileSystem()

    store: StoreLike = s3.get_mapper(
        "s3://us-west-2.opendata.source.coop/aldenks/noaa-gefs-dev/forecast/dev.zarr"
    )
    return store


def get_mode(store: StoreLike) -> Literal["w-", "w"]:
    store_root = store.name if isinstance(store, Path) else getattr(store, "root", "")
    if store_root.endswith("dev.zarr"):
        return "w"  # Allow overwritting dev store

    return "w-"  # Safe default - don't overwrite


def chunk_i_slices(ds: xr.Dataset, dim: str) -> Iterable[slice]:
    """Returns the integer offset slices which correspond to each chunk along `dim` of `ds`."""
    stop_idx = np.cumsum(ds.chunksizes[dim])  # 2, 4, 6
    start_idx = np.insert(stop_idx, 0, 0)  # 0, 2, 4, 6
    return starmap(slice, zip(start_idx, stop_idx, strict=False))  # (0,2), (2,4), (4,6)


def consume[T](iterator: Iterable[T], n: int | None = None) -> None:
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)
