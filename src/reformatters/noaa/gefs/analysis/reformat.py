import concurrent.futures
import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sentry_sdk
import xarray as xr
import zarr

from reformatters.common import docker, template_utils, validation
from reformatters.common.iterating import (
    consume,
    dimension_slices,
    get_worker_jobs,
)
from reformatters.common.kubernetes import (
    Job,
    ReformatCronJob,
    ValidationCronJob,
)
from reformatters.common.logging import get_logger
from reformatters.common.reformat_utils import ChunkFilters
from reformatters.common.types import DatetimeLike
from reformatters.common.update_progress_tracker import UpdateProgressTracker
from reformatters.common.zarr import (
    copy_data_var,
    copy_zarr_metadata,
    get_local_tmp_store,
    get_mode,
    get_zarr_store,
)
from reformatters.noaa.gefs.analysis import template
from reformatters.noaa.gefs.analysis.reformat_internals import (
    group_data_vars_by_gefs_file_type,
    reformat_time_i_slices,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

# 1 makes logic simpler when accessing GEFSv12 reforecast which has a file per variable
_VARIABLES_PER_BACKFILL_JOB = 1
_OPERATIONAL_CRON_SCHEDULE = "0 0,6,12,18 * * *"  # UTC
_VALIDATION_CRON_SCHEDULE = "30 7,10,13,19 * * *"  # UTC 1.5 hours after update

logger = get_logger(__name__)


def reformat_local(time_end: DatetimeLike, chunk_filters: ChunkFilters) -> None:
    template_ds = template.get_template(time_end)
    store = get_store()

    logger.info("Writing metadata")
    template_utils.write_metadata(template_ds, store, get_mode(store))

    logger.info("Starting reformat")
    # Process all chunks by setting worker_index=0 and worker_total=1
    reformat_chunks(
        time_end, worker_index=0, workers_total=1, chunk_filters=chunk_filters
    )
    logger.info(f"Done writing to {store}")


def reformat_kubernetes(
    time_end: DatetimeLike,
    jobs_per_pod: int,
    max_parallelism: int,
    chunk_filters: ChunkFilters,
    docker_image: str | None = None,
) -> None:
    image_tag = docker_image or docker.build_and_push_image()

    template_ds = template.get_template(time_end)
    store = get_store()
    logger.info(f"Writing zarr metadata to {store.path}")
    template.write_metadata(template_ds, store, get_mode(store))

    num_jobs = len(
        all_jobs_ordered(
            template_ds,
            chunk_filters,
            template.DATA_VARIABLES,
            _VARIABLES_PER_BACKFILL_JOB,
        )
    )
    workers_total = int(np.ceil(num_jobs / jobs_per_pod))
    parallelism = min(workers_total, max_parallelism)

    dataset_id = template_ds.attrs["dataset_id"]

    command = ["reformat-chunks", pd.Timestamp(time_end).isoformat()]
    if chunk_filters.time_start is not None:
        command.append(f"--filter-time-start={chunk_filters.time_start}")
    if chunk_filters.time_end is not None:
        command.append(f"--filter-time-end={chunk_filters.time_end}")
    if chunk_filters.variable_names is not None:
        for variable_name in chunk_filters.variable_names:
            command.append(f"--filter-variable-names={variable_name}")

    kubernetes_job = Job(
        image=image_tag,
        dataset_id=dataset_id,
        workers_total=workers_total,
        parallelism=parallelism,
        cpu="14",  # fit on 16 vCPU node
        memory="30G",  # fit on 32GB node
        shared_memory="12G",
        ephemeral_storage="30G",
        command=command,
        pod_active_deadline=timedelta(minutes=30),
    )
    subprocess.run(  # noqa: S603
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=json.dumps(kubernetes_job.as_kubernetes_object()),
        text=True,
        check=True,
    )

    logger.info(f"Submitted kubernetes job {kubernetes_job.job_name}")


def reformat_chunks(
    time_end: DatetimeLike,
    *,
    worker_index: int,
    workers_total: int,
    chunk_filters: ChunkFilters,
) -> None:
    """Writes out array chunk data. Assumes the dataset metadata has already been written."""
    assert worker_index < workers_total
    template_ds = template.get_template(time_end)
    store = get_store()

    worker_jobs = get_worker_jobs(
        all_jobs_ordered(
            template_ds,
            chunk_filters,
            template.DATA_VARIABLES,
            _VARIABLES_PER_BACKFILL_JOB,
        ),
        worker_index,
        workers_total,
    )

    logger.info(f"This is {worker_index = }, {workers_total = }, {worker_jobs}")
    consume(
        reformat_time_i_slices(
            worker_jobs,
            template_ds,
            store,
            _VARIABLES_PER_BACKFILL_JOB,
        )
    )


@sentry_sdk.monitor(
    monitor_slug=f"{template.DATASET_ID}-reformat-operational-update",
    monitor_config={
        "schedule": {"type": "crontab", "value": _OPERATIONAL_CRON_SCHEDULE},
        "timezone": "UTC",
    },
)
def reformat_operational_update(job_name: str) -> None:
    # get stores (final and tmp)
    # figure out end time to process through, AND time slices to process to fill from existing through that
    # for update slice in update slices:
    #   prepare local tmp store to be written into
    #   track and skip variables that are already processed (if this pod restarted)
    #   process data chunks to local tmp store
    #   track max coordinates of processed data
    #   upload from tmp store to final store
    #   modify template ds based on what was processed
    #   write template ds and copy to final store
    #
    append_dim = template.APPEND_DIMENSION

    final_store = get_store()
    tmp_store = get_local_tmp_store()
    # Get the dataset, check what data is already present
    ds = xr.open_zarr(final_store, decode_timedelta=True)
    for coord in ds.coords.values():
        coord.load()

    last_existing_time = ds.time.max()
    time_end = _get_operational_update_time_end()
    template_ds = template.get_template(time_end)

    start_time_str = (
        template_ds.sel(time=template_ds.time > last_existing_time)
        .time.min()
        .dt.strftime("%Y-%m-%dT%H:%M")
        .item()
    )
    chunk_filters = ChunkFilters(time_dim=append_dim, time_start=start_time_str)
    update_jobs = get_worker_jobs(
        all_jobs_ordered(
            template_ds,
            chunk_filters,
            template.DATA_VARIABLES,
            len(template.DATA_VARIABLES),
        ),
        worker_index=0,
        workers_total=1,
    )
    # Consolidate jobs for different vars with the same shard time slice,
    # We want to process all variables for a shard before we write metadata.
    update_time_i_slices = sorted({j[0] for j in update_jobs}, key=lambda s: s.start)

    upload_executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)

    for time_i_slice in update_time_i_slices:
        # Write through this i_slice. Metadata is required for to_zarr's region="auto" option.
        truncated_template_ds = template_ds.isel(time=slice(0, time_i_slice.stop))
        template.write_metadata(
            truncated_template_ds,
            tmp_store,
            get_mode(tmp_store),
        )
        max_processed_time = pd.Timestamp.min

        progress_tracker = UpdateProgressTracker(
            final_store, job_name, time_i_slice.start
        )
        vars_to_process = progress_tracker.get_unprocessed_str(
            list(template_ds.data_vars.keys())
        )

        data_var_upload_futures = []
        for data_var, _ in reformat_time_i_slices(
            [(time_i_slice, vars_to_process)],
            template_ds,
            tmp_store,
            _VARIABLES_PER_BACKFILL_JOB,
        ):
            max_processed_time = max(
                max_processed_time,
                get_latest_processed_time(tmp_store, time_i_slice, data_var.name),
            )
            data_var_upload_futures.append(
                upload_executor.submit(
                    copy_data_var,
                    data_var.name,
                    time_i_slice,
                    template_ds,
                    append_dim,
                    tmp_store,
                    final_store,
                    partial(progress_tracker.record_completion, data_var.name),
                )
            )

        concurrent.futures.wait(data_var_upload_futures, return_when="FIRST_EXCEPTION")
        for future in data_var_upload_futures:
            if (e := future.exception()) is not None:
                raise e

        if max_processed_time == pd.Timestamp.min:
            logger.info(
                f"No data processed in time_i_slice={time_i_slice}, not updating metadata."
            )
            continue

        progress_tracker.close()

        # Trim off any steps that are not yet available and rewrite metadata locally.
        # We trim one less than the max_processed_time because the last step only has
        # values for variables that do not have hour 0 data for which we have to use a
        # longer lead time (eg 3 and 6 hour not 0 and 3 hour).
        truncated_template_ds = template_ds.sel(
            time=slice(None, max_processed_time)
        ).isel(time=slice(None, -1))
        logger.info(f"Writing updated metadata for dataset ending {max_processed_time}")
        template.write_metadata(
            truncated_template_ds,
            tmp_store,
            get_mode(tmp_store),
        )

        # Write the metadata to the final store last, the data must be written first
        copy_zarr_metadata(truncated_template_ds, tmp_store, final_store)


def _get_operational_update_time_end() -> pd.Timestamp:
    """Because we use early forecast hours, we may be able to process data a bit in the "future"."""
    return pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=6)


def get_latest_processed_time(
    tmp_store: Path, chunk_i_slice: slice, var_name: str
) -> pd.Timestamp:
    ds = xr.open_zarr(tmp_store, chunks=None, decode_timedelta=True)
    da = ds[var_name].isel(time=chunk_i_slice, latitude=0, longitude=0)
    da_finite = da[da.notnull()]
    if len(da_finite) == 0:
        return pd.Timestamp.min
    return pd.Timestamp(da_finite.time.max().item())


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
            validation.check_analysis_current_data,
            validation.check_analysis_recent_nans,
        ),
    )


def operational_kubernetes_resources(image_tag: str) -> Iterable[Job]:
    operational_update_cron_job = ReformatCronJob(
        name=f"{template.DATASET_ID}-operational-update",
        schedule=_OPERATIONAL_CRON_SCHEDULE,
        pod_active_deadline=timedelta(hours=1),
        image=image_tag,
        dataset_id=template.DATASET_ID,
        cpu="14",  # fit on 16 vCPU node
        memory="30G",  # fit on 32GB node
        shared_memory="12G",
        ephemeral_storage="30G",
    )
    validation_cron_job = ValidationCronJob(
        name=f"{template.DATASET_ID}-validation",
        schedule=_VALIDATION_CRON_SCHEDULE,
        pod_active_deadline=timedelta(minutes=10),
        image=image_tag,
        dataset_id=template.DATASET_ID,
        cpu="1.3",  # fit on 2 vCPU node
        memory="7G",  # fit on 8GB node
    )

    return [operational_update_cron_job, validation_cron_job]


def all_jobs_ordered(
    template_ds: xr.Dataset,
    chunk_filters: ChunkFilters,
    possible_data_vars: Sequence[GEFSDataVar],
    group_size: int,
) -> list[tuple[slice, list[str]]]:
    if chunk_filters.variable_names is not None:
        template_ds = template_ds[chunk_filters.variable_names]

    time_i_slices = dimension_slices(template_ds, chunk_filters.time_dim)

    if chunk_filters.time_start is not None:
        time_start = pd.Timestamp(chunk_filters.time_start)
        time_i_slices = tuple(
            time_i_slice
            for time_i_slice in time_i_slices
            if template_ds.isel({chunk_filters.time_dim: time_i_slice})[
                chunk_filters.time_dim
            ].max()
            >= time_start  # type: ignore[operator]
        )

    if chunk_filters.time_end is not None:
        time_end = pd.Timestamp(chunk_filters.time_end)
        time_i_slices = tuple(
            time_i_slice
            for time_i_slice in time_i_slices
            if template_ds.isel({chunk_filters.time_dim: time_i_slice})[
                chunk_filters.time_dim
            ].min()
            < time_end  # type: ignore[operator]
        )

    data_var_groups = group_data_vars_by_gefs_file_type(
        [d for d in possible_data_vars if d.name in template_ds], group_size=group_size
    )
    data_var_name_groups = [
        [data_var.name for data_var in data_vars]
        for _, _, _, data_vars in data_var_groups
    ]

    return list(product(time_i_slices, data_var_name_groups))


def get_store() -> zarr.storage.FsspecStore:
    return get_zarr_store(template.DATASET_ID, template.DATASET_VERSION)
