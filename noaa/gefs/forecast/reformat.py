import concurrent.futures
import os
import re
import subprocess
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache, partial
from itertools import batched, groupby, islice, product, starmap
from pathlib import Path
from typing import Literal
from uuid import uuid4

import fsspec
import numpy as np
import pandas as pd
import s3fs  # type: ignore
import xarray as xr

from common import string_template
from common.config import Config  # noqa:F401
from common.download_directory import cd_into_download_directory
from common.types import Array1D, DatetimeLike, StoreLike
from noaa.gefs.forecast import template
from noaa.gefs.forecast.config_models import DataVar
from noaa.gefs.forecast.read_data import (
    NoaaFileType,
    SourceFileCoords,
    download_file,
    read_into,
)

_PROCESSING_CHUNK_DIMENSION = "init_time"


def reformat_operational_update() -> None:
    final_store = get_store()
    tmp_store = get_local_tmp_store()
    # Get the dataset, check what data is already present
    ds = xr.open_zarr(final_store)
    last_existing_init_time = ds.init_time.max()
    init_time_end = pd.Timestamp.utcnow().tz_localize(None)
    template_ds = template.get_template(init_time_end)
    template_ds.ingested_forecast_length.loc[{"init_time": ds.init_time.values}] = (
        ds.ingested_forecast_length
    )
    # Uncomment this line for local testing to scope down the number of init times
    # you will process.
    # template_ds = template_ds.isel(init_time=slice(0, len(ds.init_time) + 2))
    # We make some assumptions about what is safe to parallelize and how to
    # write the data based on the init_time dimension having a chunk size of one.
    # If this changes we will need to refactor.
    assert all(size == 1 for size in ds.chunksizes[_PROCESSING_CHUNK_DIMENSION])
    new_init_times = template_ds.init_time.loc[
        template_ds.init_time > last_existing_init_time
    ]
    new_init_time_indices = template_ds.get_index("init_time").get_indexer(
        new_init_times
    )  # type: ignore
    new_init_time_i_slices = list(
        starmap(
            slice,
            zip(
                new_init_time_indices,
                list(new_init_time_indices[1:]) + [None],
                strict=False,
            ),
        )
    )
    # In addition to new dates, reprocess old dates whose forecasts weren't
    # previously fully written.
    # On long forecasts, NOAA may fill in the first 10 days first, then publish
    # an additional 25 days after a delay.
    recent_incomplete_init_times = get_recent_init_times_for_reprocessing(ds)
    recent_incomplete_init_times_indices = template_ds.get_index(
        "init_time"
    ).get_indexer(recent_incomplete_init_times)  # type: ignore
    recent_incomplete_init_time_i_slices = list(
        starmap(
            slice,
            zip(
                recent_incomplete_init_times_indices,
                [x + 1 for x in recent_incomplete_init_times_indices],
                strict=False,
            ),
        )
    )
    upload_executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)
    template.write_metadata(template_ds, tmp_store, get_mode(final_store))
    futures = []
    for data_var, max_lead_times in reformat_init_time_i_slices(
        recent_incomplete_init_time_i_slices + new_init_time_i_slices,
        template_ds,
        tmp_store,
    ):
        for init_time, max_lead_time in max_lead_times.items():
            template_ds["ingested_forecast_length"].loc[{"init_time": init_time}] = (
                max_lead_time
            )
            # This only works because we know that chunks are size 1 in the
            # init_time dimension.
            chunk_index = template_ds.get_index("init_time").get_loc(init_time)
            futures.append(
                upload_executor.submit(
                    copy_data_var(data_var, chunk_index, tmp_store, final_store)
                )
            )

    concurrent.futures.wait(futures, return_when="ALL_COMPLETED")

    template.write_metadata(template_ds, final_store, get_mode(final_store))


def get_recent_init_times_for_reprocessing(ds: xr.Dataset) -> Array1D[np.datetime64]:
    max_init_time = ds.init_time.max().values
    # Get the last week of data
    recent_init_times = ds.init_time.where(
        ds.init_time > max_init_time - np.timedelta64(7, "D"), drop=True
    ).load()
    # Get the recent init_times where we have only partially completed
    # the ingest.
    return recent_init_times.where(  # type: ignore
        (recent_init_times.ingested_forecast_length.isnull())
        | (
            recent_init_times.ingested_forecast_length
            < recent_init_times.expected_forecast_length
        ),
        drop=True,
    ).init_time.values


def copy_data_var(
    data_var: DataVar,
    chunk_index: int,
    tmp_store: Path,
    final_store: fsspec.FSMap,
) -> Callable:
    files_to_copy = list(tmp_store.glob(f"{data_var.name}/{chunk_index}.*.*.*.*"))
    return lambda: final_store.fs.cp(
        files_to_copy, final_store.root + f"/{data_var.name}/"
    )


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
            "MEMORY": "80G",
            "EPHEMERAL_STORAGE": "60G",
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
    consume(reformat_init_time_i_slices(worker_init_time_i_slices, template_ds, store))


def reformat_init_time_i_slices(
    init_time_i_slices: list[slice], template_ds: xr.Dataset, store: StoreLike
) -> Generator[tuple[DataVar, dict[pd.Timestamp, pd.Timedelta]], None, None]:
    """
    Helper function to reformat the chunk data.
    Yields the data variable/init time combinations and their corresponding maximum
    ingested lead time as it processes.
    """
    data_var_groups = group_data_vars_by_noaa_file_type(
        [d for d in template.DATA_VARIABLES if d.name in template_ds]
    )

    wait_executor = ThreadPoolExecutor(max_workers=2)
    io_executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)
    cpu_executor = ThreadPoolExecutor(max_workers=int((os.cpu_count() or 1) * 3))

    # # If we compile eccodes ourselves with thread safety enabled we could use threads for reading
    # # https://confluence.ecmwf.int/display/ECC/ecCodes+installation ENABLE_ECCODES_THREADS
    # # but make sure to read thread safety comment in our `read_data` function.
    # proccess_executor = ProcessPoolExecutor(max_workers=os.cpu_count())

    for init_time_i_slice in init_time_i_slices:
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
                Future[list[tuple[SourceFileCoords, Path | None]]], tuple[DataVar, ...]
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

            for future in concurrent.futures.as_completed(download_var_group_futures):
                if (e := future.exception()) is not None:
                    raise e

                coords_and_paths = future.result()
                data_vars = download_var_group_futures[future]

                max_lead_times: dict[pd.Timestamp, pd.Timedelta] = {}
                for init_time, init_time_coords_and_paths in groupby(
                    coords_and_paths, key=lambda v: v[0]["init_time"]
                ):
                    max_ingested_lead_time = max(
                        filter(
                            lambda coord_and_path: coord_and_path[1] is not None,
                            init_time_coords_and_paths,
                        ),
                        key=lambda coord_and_path: coord_and_path[0]["lead_time"],
                        default=({"lead_time": pd.Timedelta("NaT")},),
                    )[0]["lead_time"]
                    max_lead_times[init_time] = max_ingested_lead_time

                # Write variable by variable to avoid blowing up memory usage
                for data_var in data_vars:
                    print("Reading", data_var.name)
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
                    # Drop all non-dimension coordinates.
                    # They are already written with different chunks.
                    data_array = data_array.drop_vars(
                        [
                            coord
                            for coord in data_array.coords
                            if coord not in data_array.dims
                        ]
                    )
                    data_array.load()  # preallocate backing numpy arrays (for performance?)
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

                    print(f"Writing {data_var.name} {chunk_init_times_str}")
                    chunks = template.chunk_args(chunk_template_ds)
                    data_array.chunk(chunks).to_zarr(store, region="auto")
                    yield (data_var, max_lead_times)


def download_var_group_files(
    idx_data_vars: Iterable[DataVar],
    chunk_coords: Iterable[SourceFileCoords],
    noaa_file_type: NoaaFileType,
    directory: Path,
    io_executor: ThreadPoolExecutor,
) -> list[tuple[SourceFileCoords, Path | None]]:
    done, not_done = concurrent.futures.wait(
        [
            io_executor.submit(
                download_file,
                coord,
                noaa_file_type=noaa_file_type,
                directory=directory,
                noaa_idx_data_vars=idx_data_vars,
            )
            for coord in chunk_coords
        ]
    )
    assert len(not_done) == 0

    for future in done:
        if (e := future.exception()) is not None:
            raise e

    print("Completed download for", [d.name for d in idx_data_vars])
    return [f.result() for f in done]


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


def get_store() -> fsspec.FSMap:
    if Config.is_dev():
        local_store: StoreLike = fsspec.get_mapper(
            "data/output/noaa/gefs/forecast/dev.zarr"
        )
        return local_store

    s3 = s3fs.S3FileSystem()

    store: StoreLike = s3.get_mapper(
        "s3://us-west-2.opendata.source.coop/aldenks/noaa-gefs-dev/forecast/dev.zarr"
    )
    return store


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/tmp-{uuid4()}.zarr").absolute()


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
