import concurrent.futures
import gc
import json
import logging
import os
import subprocess
import warnings
from collections import defaultdict
from collections.abc import Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from itertools import batched, groupby, starmap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sentry_sdk
import xarray as xr
import zarr

from reformatters.common import docker, validation
from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.config import Config  # noqa:F401
from reformatters.common.config_models import EnsembleStatistic
from reformatters.common.download_directory import cd_into_download_directory
from reformatters.common.iterating import chunk_i_slices, consume, get_worker_jobs
from reformatters.common.kubernetes import Job, ReformatCronJob, ValidationCronJob
from reformatters.common.types import Array1D, DatetimeLike
from reformatters.common.zarr import (
    copy_data_var,
    copy_zarr_metadata,
    get_local_tmp_store,
    get_mode,
    get_zarr_store,
)
from reformatters.noaa.gefs.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.noaa.gefs.forecast_35_day import template
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar, GEFSFileType
from reformatters.noaa.gefs.read_data import (
    SourceFileCoords,
    download_file,
    generate_chunk_coordinates,
    read_into,
)

_PROCESSING_CHUNK_DIMENSION = "init_time"
_OPERATIONAL_CRON_SCHEDULE = "0 7 * * *"  # At 7:00 UTC every day.
_VALIDATION_CRON_SCHEDULE = "0 9 * * *"  # At 9:00 UTC every day.
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@sentry_sdk.monitor(
    monitor_slug=f"{template.DATASET_ID}-validation",
    monitor_config={
        "schedule": {"type": "crontab", "value": _VALIDATION_CRON_SCHEDULE},
        "timezone": "UTC",
    },
)
def validate_zarr() -> None:
    validation.validate_zarr(
        get_store(),
        validators=(
            validation.check_forecast_current_data,
            validation.check_forecast_recent_nans,
        ),
    )


@sentry_sdk.monitor(
    monitor_slug=f"{template.DATASET_ID}-reformat-operational-update",
    monitor_config={
        "schedule": {"type": "crontab", "value": _OPERATIONAL_CRON_SCHEDULE},
        "timezone": "UTC",
    },
)
def reformat_operational_update() -> None:
    final_store = get_store()
    tmp_store = get_local_tmp_store()
    # Get the dataset, check what data is already present
    ds = xr.open_zarr(final_store, decode_timedelta=True)
    for coord in ds.coords.values():
        coord.load()
    last_existing_init_time = ds.init_time.max()
    init_time_end = pd.Timestamp.utcnow().tz_localize(None)
    template_ds = template.get_template(init_time_end)
    template_ds.ingested_forecast_length.loc[{"init_time": ds.init_time.values}] = (
        ds.ingested_forecast_length
    )

    # Uncomment this line for local testing to scope down the number of init times
    # template_ds = template_ds.isel(init_time=slice(0, len(ds.init_time) + 2))

    # We make some assumptions about what is safe to parallelize and how to
    # write the data based on the init_time dimension having a chunk size of one.
    # If this changes we will need to refactor.
    assert all(
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
        # Write through this i_slice. Metadata is required for to_zarr's region="auto" option.
        template.write_metadata(
            truncated_template_ds,
            tmp_store,
            get_mode(tmp_store),
        )
        data_var_upload_futures = []
        for data_var, max_lead_times in reformat_init_time_i_slices(
            [init_time_i_slice],
            template_ds,
            tmp_store,
        ):
            # Writing a chunk without merging in existing data works because
            # the init_time dimension chunk size is 1.
            data_var_upload_futures.append(
                upload_executor.submit(
                    copy_data_var,
                    data_var.name,
                    init_time_i_slice.start,
                    tmp_store,
                    final_store,
                )
            )
            for (init_time, ensemble_member), max_lead_time in max_lead_times.items():
                if np.issubdtype(type(ensemble_member), np.integer):
                    truncated_template_ds["ingested_forecast_length"].loc[
                        {"init_time": init_time, "ensemble_member": ensemble_member}
                    ] = max_lead_time

        concurrent.futures.wait(data_var_upload_futures, return_when="FIRST_EXCEPTION")
        for future in data_var_upload_futures:
            if (e := future.exception()) is not None:
                raise e

        # Write metadata again to update the ingested_forecast_length
        template.write_metadata(
            truncated_template_ds,
            tmp_store,
            get_mode(tmp_store),
        )
        # Write the metadata last, the data must be written first
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


def reformat_local(init_time_end: DatetimeLike) -> None:
    template_ds = template.get_template(init_time_end)
    store = get_store()

    logger.info("Writing metadata")
    template.write_metadata(template_ds, store, get_mode(store))

    logger.info("Starting reformat")
    # Process all chunks by setting worker_index=0 and worker_total=1
    reformat_chunks(init_time_end, worker_index=0, workers_total=1)
    logger.info(f"Done writing to {store}")


def reformat_kubernetes(
    init_time_end: DatetimeLike, jobs_per_pod: int, max_parallelism: int
) -> None:
    image_tag = docker.build_and_push_image()

    template_ds = template.get_template(init_time_end)
    store = get_store()
    logger.info(f"Writing zarr metadata to {store.path}")
    template.write_metadata(template_ds, store, get_mode(store))

    num_jobs = sum(1 for _ in chunk_i_slices(template_ds, _PROCESSING_CHUNK_DIMENSION))
    workers_total = int(np.ceil(num_jobs / jobs_per_pod))
    parallelism = min(workers_total, max_parallelism)

    dataset_id = template_ds.attrs["dataset_id"]

    kubernetes_job = Job(
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

    logger.info(f"Submitted kubernetes job {kubernetes_job.job_name}")


def operational_kubernetes_resources(image_tag: str) -> Iterable[Job]:
    operational_update_cron_job = ReformatCronJob(
        name=f"{template.DATASET_ID}-operational-update",
        schedule=_OPERATIONAL_CRON_SCHEDULE,
        image=image_tag,
        dataset_id=template.DATASET_ID,
        cpu="6",  # fit on 8 vCPU node
        memory="60G",  # fit on 64GB node
        ephemeral_storage="150G",
    )
    validation_cron_job = ValidationCronJob(
        name=f"{template.DATASET_ID}-validation",
        schedule=_VALIDATION_CRON_SCHEDULE,
        image=image_tag,
        dataset_id=template.DATASET_ID,
        cpu="6",  # fit on 8 vCPU node
        memory="60G",  # fit on 64GB node
    )

    return [operational_update_cron_job, validation_cron_job]


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

    logger.info(
        f"This is {worker_index = }, {workers_total = }, {worker_init_time_i_slices}"
    )
    consume(reformat_init_time_i_slices(worker_init_time_i_slices, template_ds, store))


# Integer ensemble member or an ensemble statistic
type EnsOrStat = int | np.integer[Any] | str


def reformat_init_time_i_slices(
    init_time_i_slices: list[slice],
    template_ds: xr.Dataset,
    store: zarr.storage.FsspecStore | Path,
) -> Generator[
    tuple[GEFSDataVar, dict[tuple[pd.Timestamp, EnsOrStat], pd.Timedelta]], None, None
]:
    """
    Do the chunk data reformatting work - download files, read into memory, write to zarr.
    Yields the data variable/init time combinations and their corresponding maximum
    ingested lead time as it processes.
    """
    data_var_groups = group_data_vars_by_gefs_file_type(
        [d for d in template.DATA_VARIABLES if d.name in template_ds]
    )
    ensemble_statistics: set[EnsembleStatistic] = {
        statistic
        for var in template_ds.data_vars.values()
        if (statistic := var.attrs.get("ensemble_statistic")) is not None
    }

    with (
        ThreadPoolExecutor(max_workers=2) as wait_executor,
        ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2) as io_executor,
        ThreadPoolExecutor(
            max_workers=int((os.cpu_count() or 1) * 1.5)
        ) as cpu_executor,
    ):
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

            chunk_init_times_str = ", ".join(
                chunk_init_times.strftime("%Y-%m-%dT%H:%M")
            )
            logger.info(f"Starting chunk with init times {chunk_init_times_str}")

            with cd_into_download_directory() as directory:
                download_var_group_futures: dict[
                    Future[list[tuple[SourceFileCoords, Path | None]]],
                    tuple[GEFSDataVar, ...],
                ] = {}
                for gefs_file_type, ensemble_statistic, data_vars in data_var_groups:
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
                            gefs_file_type,
                            directory,
                            io_executor,
                        )
                    ] = data_vars

                for future in concurrent.futures.as_completed(
                    download_var_group_futures
                ):
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

                    max_lead_times: dict[
                        tuple[pd.Timestamp, EnsOrStat], pd.Timedelta
                    ] = {}
                    for (
                        init_time,
                        ensemble_member,
                    ), init_time_coords_and_paths in groupby(
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
                                if coords_and_path[0]["lead_time"]
                                > pd.Timedelta(hours=0)
                            ]
                        else:
                            var_coords_and_paths = coords_and_paths
                        data_array = xr.full_like(
                            chunk_template_ds[data_var.name], np.nan
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

                        if data_var.attrs.step_type == "accum":
                            logger.info(
                                f"Converting {data_var.name} from accumulations to rates"
                            )
                            try:
                                deaccumulate_to_rates_inplace(
                                    data_array, dim="lead_time"
                                )
                            except ValueError:
                                # Log exception so we are notified if deaccumulation errors are larger than expected.
                                logger.exception(
                                    f"Error deaccumulating {data_var.name}"
                                )

                        keep_mantissa_bits = data_var.internal_attrs.keep_mantissa_bits
                        if isinstance(keep_mantissa_bits, int):
                            logger.info(f"Starting bit rounding {data_var.name}")
                            round_float32_inplace(
                                data_array.values,
                                keep_mantissa_bits=keep_mantissa_bits,
                            )

                        logger.info(f"Writing {data_var.name} {chunk_init_times_str}")

                        # Dask chunks when writing must match shards
                        data_array = data_array.chunk(
                            template_ds[data_var.name].encoding["shards"]
                        )

                        # Drop all non-dimension coordinates.
                        # They are already written by write_metadata.
                        data_array = data_array.drop_vars(
                            [
                                coord
                                for coord in data_array.coords
                                if coord not in data_array.dims
                            ]
                        )

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="In a future version of xarray decode_timedelta will default to False rather than None.",
                                category=FutureWarning,
                            )
                            ## Write to zarr ##
                            data_array.to_zarr(store, region="auto")  # type: ignore[call-overload]

                        yield (data_var, max_lead_times)

                        del data_array
                        gc.collect()

                    # Reclaim space once done.
                    for _, filepath in coords_and_paths:
                        if filepath is not None:
                            filepath.unlink()


def download_var_group_files(
    idx_data_vars: Iterable[GEFSDataVar],
    chunk_coords: Iterable[SourceFileCoords],
    gefs_file_type: GEFSFileType,
    directory: Path,
    io_executor: ThreadPoolExecutor,
) -> list[tuple[SourceFileCoords, Path | None]]:
    done, not_done = concurrent.futures.wait(
        [
            io_executor.submit(
                download_file,
                coord,
                gefs_file_type=gefs_file_type,
                directory=directory,
                gefs_idx_data_vars=idx_data_vars,
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


def group_data_vars_by_gefs_file_type(
    data_vars: Iterable[GEFSDataVar], group_size: int = 4
) -> list[tuple[GEFSFileType, EnsembleStatistic | None, tuple[GEFSDataVar, ...]]]:
    """
    Group data variables by the things which determine which source file they come from:
    1. their GEFS file type (a, b, or s) and
    2. their ensemble statistic if present or None if they are an ensemble trace.

    Then, within each group, chunk them into groups of size `group_size`. We download
    all variables in a group together which can reduce the number of tiny requests if
    they are nearby in the source grib file. By breaking groups into `group_size`, we
    allow reading/writing to begin before we've downloaded _all_ variables from a file.
    """
    grouper = defaultdict(list)
    for data_var in data_vars:
        gefs_file_type = data_var.internal_attrs.gefs_file_type
        grouper[(gefs_file_type, data_var.attrs.ensemble_statistic)].append(data_var)
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


def get_store() -> zarr.storage.FsspecStore:
    return get_zarr_store(template.DATASET_ID, template.DATASET_VERSION)
