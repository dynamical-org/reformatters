import os
import re
import subprocess
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from itertools import islice, starmap
from typing import Literal

import numpy as np
import pandas as pd
import s3fs  # type: ignore
import xarray as xr

from common import string_template
from common.config import Config
from common.download_directory import download_directory
from common.types import DatetimeLike, StoreLike
from noaa.gefs.forecast import template
from noaa.gefs.forecast.read_data import download_file, read_file

_PROCESSING_CHUNK_DIMENSION = "init_time"


def reformat_local(init_time_end: DatetimeLike) -> None:
    template_ds = template.get_template(init_time_end)
    store = get_store()

    print("Writing zarr metadata")
    template_ds.to_zarr(store, mode=get_mode(store), compute=False)

    print("Starting reformat")
    # Process all chunks
    reformat_chunks(init_time_end, worker_index=0, workers_total=1)


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
        ["/usr/bin/docker", "push", image_tag],
        check=True,
    )
    print("Pushed", image_tag)

    store = get_store()
    print("Writing zarr metadata")
    template_ds.to_zarr(store, mode=get_mode(store), compute=False)

    num_jobs = sum(1 for _ in chunk_i_slices(template_ds, _PROCESSING_CHUNK_DIMENSION))
    workers_total = int(np.ceil(num_jobs / jobs_per_pod))
    parallelism = min(workers_total, max_parallelism)

    # TODO read dataset id from template_ds.attrs
    dataset_id = template._DATASET_ID
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
            "MEMORY": "110Gi",
            "EPHEMERAL_STORAGE": "200Gi",
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_SESSION_TOKEN": os.environ["AWS_SESSION_TOKEN"],
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

    thread_executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    # If we compile eccodes ourselves with thread safety enabled we could use threads for reading
    # https://confluence.ecmwf.int/display/ECC/ecCodes+installation ENABLE_ECCODES_THREADS
    proccess_executor = ProcessPoolExecutor(max_workers=os.cpu_count())

    for init_time_i_slice in worker_init_time_i_slices:
        chunk_template_ds = template_ds.isel(init_time=init_time_i_slice)

        chunk_init_times = pd.to_datetime(chunk_template_ds["init_time"].values)
        chunk_lead_times = pd.to_timedelta(chunk_template_ds["lead_time"].values)
        chunk_ensemble_members = chunk_template_ds["ensemble_member"]

        print("Starting", chunk_init_times.to_list())

        with download_directory() as directory:
            init_time_datasets = []
            for init_time in chunk_init_times:
                ensemble_member_datasets = []
                for ensemble_member in chunk_ensemble_members:
                    download = partial(
                        download_file, init_time, ensemble_member, directory=directory
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

            # Write variable by variable to avoid blowing up memory usage
            for var_key in chunk_ds.data_vars.keys():
                init_times_str = ", ".join(
                    chunk_ds["init_time"].dt.strftime("%Y-%m-%dT%H:%M").values
                )
                print(f"Writing {var_key} {init_times_str}")
                chunks = template.chunk_args(template_ds)
                chunk_ds[var_key].chunk(chunks).to_zarr(store, region="auto")


def get_worker_jobs[T](
    jobs: Iterable[T], worker_index: int, workers_total: int
) -> Iterable[T]:
    """Returns the subset of `jobs` that worker_index should process if there are workers_total workers."""
    return islice(jobs, worker_index, None, workers_total)


def get_store() -> StoreLike:
    # return "data/output/noaa/gefs/forecast/dev.zarr"

    s3 = s3fs.S3FileSystem()

    store: StoreLike = s3.get_mapper(
        "s3://us-west-2.opendata.source.coop/aldenks/noaa-gefs-dev/forecast/dev.zarr"
    )
    return store


def get_mode(store: StoreLike) -> Literal["w-", "w"]:
    store_root = store if isinstance(store, str) else getattr(store, "root", "")
    if store_root.endswith("/dev.zarr"):
        return "w"  # Allow overwritting dev store

    return "w-"  # Safe default - don't overwrite


def chunk_i_slices(ds: xr.Dataset, dim: str) -> Iterable[slice]:
    """Returns the integer offset slices which correspond to each chunk along `dim` of `ds`."""
    stop_idx = np.cumsum(ds.chunksizes[dim])  # 2, 4, 6
    start_idx = np.insert(stop_idx, 0, 0)  # 0, 2, 4, 6
    return starmap(slice, zip(start_idx, stop_idx, strict=False))  # (0,2), (2,4), (4,6)
