import concurrent.futures
import gc
import json
import logging
import os
import re
import subprocess
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache, partial
from itertools import batched, groupby, islice, pairwise, starmap
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import fsspec  # type: ignore
import numpy as np
import pandas as pd
import s3fs  # type: ignore
import sentry_sdk
import xarray as xr

from common.config import Config  # noqa:F401
from common.download_directory import cd_into_download_directory
from common.kubernetes import ReformatCronJob, ReformatJob
from common.types import Array1D, DatetimeLike, StoreLike
from noaa.gefs.forecast import template, template_config
from noaa.gefs.forecast.config_models import DataVar, EnsembleStatistic
from noaa.gefs.forecast.read_data import (
    NoaaFileType,
    SourceFileCoords,
    download_file,
    generate_chunk_coordinates,
    read_into,
)

_PROCESSING_CHUNK_DIMENSION = "init_time"
_CRON_SCHEDULE = "0 7 * * *"  # At 7:00 UTC every day.

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@sentry_sdk.monitor(
    monitor_slug=f"{template_config.DATASET_ID}-reformat-operational-update",
    monitor_config={
        "schedule": {"type": "crontab", "value": _CRON_SCHEDULE},
        "timezone": "UTC",
        # If an expected check-in doesn't come in `checkin_margin`
        # minutes, it'll be considered missed
        "checkin_margin": 10,
        # The check-in is allowed to run for `max_runtime` minutes
        # before it's considered failed
        "max_runtime": 120,  # minutes
        # It'll take `failure_issue_threshold` consecutive failed
        # check-ins to create an issue
        "failure_issue_threshold": 1,
        # It'll take `recovery_threshold` OK check-ins to resolve
        # an issue
        "recovery_threshold": 1,
    },
)
def reformat_operational_update() -> None:
    final_store = get_store()
    tmp_store = get_local_tmp_store()
    # Get the dataset, check what data is already present
    ds = xr.open_zarr(final_store)
    for coord in ds.coords.values():
        coord.load()
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
    all(
        all(1 == val for val in da.chunksizes[_PROCESSING_CHUNK_DIMENSION])
        for da in ds.data_vars.values()
    )
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

    for init_time_i_slice in (
        recent_incomplete_init_time_i_slices + new_init_time_i_slices
    ):
        truncated_template_ds = template_ds.isel(
            init_time=slice(0, init_time_i_slice.stop)
        )
        # Write through this i_slice
        template.write_metadata(
            truncated_template_ds,
            tmp_store,
            get_mode(tmp_store),
        )
        futures = []
        for data_var, max_lead_times in reformat_init_time_i_slices(
            [init_time_i_slice],
            template_ds,
            tmp_store,
        ):
            # This only works because we know that chunks are size 1 in the
            # init_time dimension.
            futures.append(
                upload_executor.submit(
                    copy_data_var(
                        data_var, init_time_i_slice.start, tmp_store, final_store
                    )
                )
            )
            for (init_time, ensemble_member), max_lead_time in max_lead_times.items():
                if np.issubdtype(type(ensemble_member), np.integer):
                    truncated_template_ds["ingested_forecast_length"].loc[
                        {"init_time": init_time, "ensemble_member": ensemble_member}
                    ] = max_lead_time

        concurrent.futures.wait(futures, return_when="ALL_COMPLETED")

        template.write_metadata(
            truncated_template_ds.isel(init_time=slice(0, init_time_i_slice.stop)),
            tmp_store,
            get_mode(tmp_store),
        )
        copy_zarr_metadata(truncated_template_ds, tmp_store, final_store)


def get_recent_init_times_for_reprocessing(ds: xr.Dataset) -> Array1D[np.datetime64]:
    # Get the last few days of data
    recent_init_times = ds.init_time.where(
        ds.init_time > pd.Timestamp.utcnow().tz_localize(None) - np.timedelta64(4, "D"),
        drop=True,
    ).load()
    # Ingested forecast length is along init time and ensemble member.
    # We care if any of the ensemble members for this init time
    # are not fully ingested.
    recent_init_times["reduced_ingested_forecast_length"] = (
        ds.ingested_forecast_length.min(dim="ensemble_member")
    )
    # Get the recent init_times where we have only partially completed
    # the ingest.
    return recent_init_times.where(  # type: ignore
        (recent_init_times.reduced_ingested_forecast_length.isnull())
        | (
            recent_init_times.reduced_ingested_forecast_length
            < recent_init_times.expected_forecast_length
        ),
        drop=True,
    ).init_time.values


def copy_data_var(
    data_var: DataVar,
    chunk_index: int,
    tmp_store: Path,
    final_store: fsspec.FSMap,
) -> Callable[[], None]:
    files_to_copy = list(
        tmp_store.glob(f"{data_var.name}/{chunk_index}.*.*.*")
    )  # matches any chunk with 4 or more dimensions

    def mv_files() -> None:
        logger.info(
            f"Copying data var chunks to final store ({final_store.root}) for {data_var.name}."
        )
        try:
            fs = final_store.fs
            fs.auto_mkdir = True
            fs.put(
                files_to_copy, final_store.root + f"/{data_var.name}/", auto_mkdir=True
            )
        except Exception as e:
            logger.warning(f"Failed to upload chunk: {e}")
        try:
            # Delete data to conserve space.
            for file in files_to_copy:
                file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete chunk after upload: {e}")

    return mv_files


def copy_zarr_metadata(
    template_ds: xr.Dataset, tmp_store: Path, final_store: fsspec.FSMap
) -> None:
    logger.info(
        f"Copying metadata to final store ({final_store.root}) from {tmp_store}"
    )
    metadata_files = []
    # Coordinates
    for coord in template_ds.coords:
        metadata_files.extend(list(tmp_store.glob(f"{coord}/*")))
    # zattrs, zarray, zgroup and zmetadata
    metadata_files.extend(list(tmp_store.glob("**/.z*")))
    # deduplicate
    metadata_files = list(dict.fromkeys(metadata_files))
    for file in metadata_files:
        relative = file.relative_to(tmp_store)
        final_store.fs.put_file(file, f"{final_store.root}/{relative}")


def reformat_local(init_time_end: DatetimeLike) -> None:
    template_ds = template.get_template(init_time_end)
    store = get_store()

    logger.info("Writing metadata")
    template.write_metadata(template_ds, store, get_mode(store))

    logger.info("Starting reformat")
    # Process all chunks by setting worker_index=0 and worker_total=1
    reformat_chunks(init_time_end, worker_index=0, workers_total=1)
    logger.info("Done writing to", store)


def reformat_kubernetes(
    init_time_end: DatetimeLike, jobs_per_pod: int, max_parallelism: int
) -> None:
    template_ds = template.get_template(init_time_end)

    job_timestamp = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H-%M-%SZ")

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
    logger.info(f"Pushed {image_tag}")

    store = get_store()
    logger.info(f"Writing zarr metadata to {store.root}")
    template.write_metadata(template_ds, store, get_mode(store))

    num_jobs = sum(1 for _ in chunk_i_slices(template_ds, _PROCESSING_CHUNK_DIMENSION))
    workers_total = int(np.ceil(num_jobs / jobs_per_pod))
    parallelism = min(workers_total, max_parallelism)

    dataset_id = template_ds.attrs["dataset_id"]
    job_name = f"{dataset_id}-{job_timestamp}"

    kubernetes_job = ReformatJob(
        image=image_tag,
        dataset_id=dataset_id,
        workers_total=workers_total,
        parallelism=parallelism,
        cpu="6",  # fit on 8 vCPU node
        memory="60G",  # fit on 64GB node
        ephemeral_storage="60G",
        command=[
            "reformat-chunks",
            pd.Timestamp(init_time_end).isoformat(),
        ],
    )
    subprocess.run(  # noqa: S603
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=json.dumps(kubernetes_job.as_kubernetes_object()),
        text=True,
        check=True,
    )

    logger.info(f"Submitted kubernetes job {job_name}")


def deploy_operational_updates() -> None:
    job_timestamp = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H-%M-%SZ")

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
    logger.info(f"Pushed {image_tag}")

    dataset_id = template_config.DATASET_ID

    cron_job = ReformatCronJob(
        name=f"{dataset_id}-operational-update",
        schedule=_CRON_SCHEDULE,
        image=image_tag,
        dataset_id=dataset_id,
        workers_total=1,
        parallelism=1,
        cpu="6",  # fit on 8 vCPU node
        memory="60G",  # fit on 64GB node
        ephemeral_storage="150G",
        command=[
            "reformat-operational-update",
        ],
    )
    subprocess.run(  # noqa: S603
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=json.dumps(cron_job.as_kubernetes_object()),
        text=True,
        check=True,
    )


def reformat_chunks(
    init_time_end: DatetimeLike, *, worker_index: int, workers_total: int
) -> None:
    """Writes out array chunk data. Assumes the dataset metadata has already been written."""
    assert worker_index < workers_total
    template_ds = template.get_template(init_time_end)
    store = get_store()

    logger.info("Getting i slices")
    worker_init_time_i_slices = list(
        get_worker_jobs(
            chunk_i_slices(template_ds, _PROCESSING_CHUNK_DIMENSION),
            worker_index,
            workers_total,
        )
    )

    logger.info(
        f"This is {worker_index = }, {workers_total = }, {worker_init_time_i_slices}"
    )
    consume(reformat_init_time_i_slices(worker_init_time_i_slices, template_ds, store))


# Integer ensemble member or an ensemble statistic
type EnsOrStat = int | np.integer[Any] | str


def reformat_init_time_i_slices(
    init_time_i_slices: list[slice], template_ds: xr.Dataset, store: StoreLike
) -> Generator[
    tuple[DataVar, dict[tuple[pd.Timestamp, EnsOrStat], pd.Timedelta]], None, None
]:
    """
    Helper function to reformat the chunk data.
    Yields the data variable/init time combinations and their corresponding maximum
    ingested lead time as it processes.
    """
    data_var_groups = group_data_vars_by_noaa_file_type(
        [d for d in template.DATA_VARIABLES if d.name in template_ds]
    )
    ensemble_statistics: set[EnsembleStatistic] = {
        statistic
        for var in template_ds.data_vars.values()
        if (statistic := var.attrs.get("ensemble_statistic")) is not None
    }

    wait_executor = ThreadPoolExecutor(max_workers=2)
    io_executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)
    cpu_executor = ThreadPoolExecutor(max_workers=int((os.cpu_count() or 1) * 1.5))

    for init_time_i_slice in init_time_i_slices:
        chunk_template_ds = template_ds.isel(init_time=init_time_i_slice)

        chunk_init_times = pd.to_datetime(chunk_template_ds["init_time"].values)
        chunk_lead_times = pd.to_timedelta(chunk_template_ds["lead_time"].values)
        chunk_ensemble_members = chunk_template_ds["ensemble_member"].values

        chunk_coords_by_type = generate_chunk_coordinates(
            chunk_init_times,
            chunk_ensemble_members,
            chunk_lead_times,
            ensemble_statistics,
        )

        chunk_init_times_str = ", ".join(chunk_init_times.strftime("%Y-%m-%dT%H:%M"))
        logger.info(f"Starting chunk with init times {chunk_init_times_str}")

        with cd_into_download_directory() as directory:
            download_var_group_futures: dict[
                Future[list[tuple[SourceFileCoords, Path | None]]],
                tuple[DataVar, ...],
            ] = {}
            for noaa_file_type, ensemble_statistic, data_vars in data_var_groups:
                chunk_coords: Iterable[SourceFileCoords]
                if ensemble_statistic is None:
                    chunk_coords = chunk_coords_by_type["ensemble"]
                else:
                    chunk_coords = chunk_coords_by_type["statistic"]

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

                def groupbykey(
                    v: tuple[SourceFileCoords, Path | None],
                ) -> tuple[pd.Timestamp, EnsOrStat]:
                    coords, _ = v

                    ensemble_portion: EnsOrStat
                    if isinstance(
                        ensemble_member := coords.get("ensemble_member"),
                        int | np.integer,
                    ):
                        ensemble_portion = ensemble_member
                    elif isinstance(statistic := coords.get("statistic"), str):
                        ensemble_portion = statistic
                    return (coords["init_time"], ensemble_portion)

                max_lead_times: dict[tuple[pd.Timestamp, EnsOrStat], pd.Timedelta] = {}
                for (init_time, ensemble_member), init_time_coords_and_paths in groupby(
                    sorted(coords_and_paths, key=groupbykey), key=groupbykey
                ):
                    ingested_lead_times = [
                        coord["lead_time"]
                        for coord, path in init_time_coords_and_paths
                        if path is not None
                    ]
                    max_lead_times[(init_time, ensemble_member)] = max(
                        ingested_lead_times, default=pd.Timedelta("NaT")
                    )

                # Write variable by variable to avoid blowing up memory usage
                for data_var in data_vars:
                    logger.info(f"Reading {data_var.name}")
                    # Skip reading the 0-hour for accumulated or last N hours avg values
                    if data_var.attrs.step_type in ("accum", "avg"):
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

                    logger.info(f"Writing {data_var.name} {chunk_init_times_str}")
                    (
                        data_array.chunk(
                            template_ds[data_var.name].encoding["preferred_chunks"]
                        ).to_zarr(store, region="auto")
                    )
                    yield (data_var, max_lead_times)

                    del data_array
                    gc.collect()

                # Reclaim space once done.
                for _, filepath in coords_and_paths:
                    if filepath is not None:
                        filepath.unlink()


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

    logger.info(f"Completed download for {[d.name for d in idx_data_vars]}")
    return [f.result() for f in done]


def group_data_vars_by_noaa_file_type(
    data_vars: Iterable[DataVar], group_size: int = 4
) -> list[tuple[NoaaFileType, EnsembleStatistic | None, tuple[DataVar, ...]]]:
    grouper = defaultdict(list)
    for data_var in data_vars:
        noaa_file_type = data_var.internal_attrs.noaa_file_type
        grouper[(noaa_file_type, data_var.attrs.ensemble_statistic)].append(data_var)
    chunks = []
    for (file_type, ensemble_statistic), idx_data_vars in grouper.items():
        idx_data_vars = sorted(
            idx_data_vars, key=lambda data_var: data_var.internal_attrs.index_position
        )
        chunks.extend(
            [
                (file_type, ensemble_statistic, data_vars_chunk)
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
    if Config.is_dev:
        local_store: StoreLike = fsspec.get_mapper(
            "data/output/noaa/gefs/forecast/dev.zarr"
        )
        return local_store

    s3 = s3fs.S3FileSystem(
        key=Config.source_coop.aws_access_key_id,
        secret=Config.source_coop.aws_secret_access_key,
    )
    store: StoreLike = s3.get_mapper(
        "s3://us-west-2.opendata.source.coop/dynamical/noaa-gefs-forecast/v0.1.0.zarr"
    )
    return store


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/{uuid4()}-tmp.zarr").absolute()


def get_mode(store: StoreLike) -> Literal["w-", "w"]:
    store_root = store.name if isinstance(store, Path) else getattr(store, "root", "")
    if store_root.endswith("dev.zarr") or store_root.endswith("-tmp.zarr"):
        return "w"  # Allow overwritting dev store

    return "w-"  # Safe default - don't overwrite


def chunk_i_slices(ds: xr.Dataset, dim: str) -> Iterable[slice]:
    """Returns the integer offset slices which correspond to each chunk along `dim` of `ds`."""
    vars_dim_chunk_sizes = {var.chunksizes[dim] for var in ds.data_vars.values()}
    assert (
        len(vars_dim_chunk_sizes) == 1
    ), f"Inconsistent chunk sizes among data variables along update dimension ({dim}): {vars_dim_chunk_sizes}"
    dim_chunk_sizes = next(iter(vars_dim_chunk_sizes))  # eg. 2, 2, 2
    stop_idx = np.cumsum(dim_chunk_sizes)  # eg.    2, 4, 6
    start_idx = np.insert(stop_idx, 0, 0)  # eg. 0, 2, 4, 6
    return starmap(slice, pairwise(start_idx))  # eg. slice(0,2), slice(2,4), slice(4,6)


def consume[T](iterator: Iterable[T], n: int | None = None) -> None:
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)
