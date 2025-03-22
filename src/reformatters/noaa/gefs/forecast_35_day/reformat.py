import concurrent.futures
import json
import os
import subprocess
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import product, starmap

import numpy as np
import pandas as pd
import sentry_sdk
import xarray as xr
import zarr
from pydantic import BaseModel

from reformatters.common import docker, validation
from reformatters.common.config import Config  # noqa:F401
from reformatters.common.iterating import (
    consume,
    dimension_slices,
    get_worker_jobs,
)
from reformatters.common.kubernetes import Job, ReformatCronJob, ValidationCronJob
from reformatters.common.logging import get_logger
from reformatters.common.types import Array1D, DatetimeLike
from reformatters.common.zarr import (
    copy_data_var,
    copy_zarr_metadata,
    get_local_tmp_store,
    get_mode,
    get_zarr_store,
)
from reformatters.noaa.gefs.forecast_35_day import template
from reformatters.noaa.gefs.forecast_35_day.reformat_internals import (
    group_data_vars_by_gefs_file_type,
    reformat_init_time_i_slices,
)

_VARIABLES_PER_BACKFILL_JOB = 3
_OPERATIONAL_CRON_SCHEDULE = "0 7 * * *"  # At 7:00 UTC every day.
_VALIDATION_CRON_SCHEDULE = "0 10 * * *"  # At 10:00 UTC every day.

logger = get_logger(__name__)


class ChunkFilters(BaseModel):
    """
    Filters for controlling which chunks of data to process.
    Only necessary if you don't want to process all data.
    """

    time_dim: str
    time_start: str | None = None
    time_end: str | None = None
    variable_names: list[str] | None = None


def reformat_local(init_time_end: DatetimeLike, chunk_filters: ChunkFilters) -> None:
    template_ds = template.get_template(init_time_end)
    store = get_store()

    logger.info("Writing metadata")
    template.write_metadata(template_ds, store, get_mode(store))

    logger.info("Starting reformat")
    # Process all chunks by setting worker_index=0 and worker_total=1
    reformat_chunks(
        init_time_end, worker_index=0, workers_total=1, chunk_filters=chunk_filters
    )
    logger.info(f"Done writing to {store}")


def reformat_kubernetes(
    init_time_end: DatetimeLike,
    jobs_per_pod: int,
    max_parallelism: int,
    chunk_filters: ChunkFilters,
    docker_image: str | None = None,
    skip_write_template: bool = False,
) -> None:
    image_tag = docker_image or docker.build_and_push_image()

    template_ds = template.get_template(init_time_end)
    store = get_store()
    if not skip_write_template:
        logger.info(f"Writing zarr metadata to {store.path}")
        template.write_metadata(template_ds, store, get_mode(store))

    num_jobs = sum(1 for _ in all_jobs_ordered(template_ds, chunk_filters))
    workers_total = int(np.ceil(num_jobs / jobs_per_pod))
    parallelism = min(workers_total, max_parallelism)

    dataset_id = template_ds.attrs["dataset_id"]

    command = [
        "reformat-chunks",
        pd.Timestamp(init_time_end).isoformat(),
    ]
    if chunk_filters.time_start is not None:
        command.append(f"--filter-init-time-start={chunk_filters.time_start}")
    if chunk_filters.time_end is not None:
        command.append(f"--filter-init-time-end={chunk_filters.time_end}")
    if chunk_filters.variable_names is not None:
        for variable_name in chunk_filters.variable_names:
            command.append(f"--filter-variable-names={variable_name}")

    kubernetes_job = Job(
        image=image_tag,
        dataset_id=dataset_id,
        workers_total=workers_total,
        parallelism=parallelism,
        cpu="6",  # fit on 8 vCPU node
        memory="60G",  # fit on 64GB node
        shared_memory="24G",
        ephemeral_storage="60G",
        command=command,
    )
    subprocess.run(  # noqa: S603
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=json.dumps(kubernetes_job.as_kubernetes_object()),
        text=True,
        check=True,
    )

    logger.info(f"Submitted kubernetes job {kubernetes_job.job_name}")


def reformat_chunks(
    init_time_end: DatetimeLike,
    *,
    worker_index: int,
    workers_total: int,
    chunk_filters: ChunkFilters,
) -> None:
    """Writes out array chunk data. Assumes the dataset metadata has already been written."""
    assert worker_index < workers_total
    template_ds = template.get_template(init_time_end)
    store = get_store()

    worker_jobs = get_worker_jobs(
        all_jobs_ordered(template_ds, chunk_filters),
        worker_index,
        workers_total,
    )

    logger.info(f"This is {worker_index = }, {workers_total = }, {worker_jobs}")
    consume(
        reformat_init_time_i_slices(
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
def reformat_operational_update() -> None:
    final_store = get_store()
    tmp_store = get_local_tmp_store()
    # Get the dataset, check what data is already present
    ds = xr.open_zarr(final_store, decode_timedelta=True)
    for coord in ds.coords.values():
        coord.load()
    last_existing_init_time = ds.init_time.max()
    init_time_end = _get_operational_update_init_time_end()
    template_ds = template.get_template(init_time_end)
    template_ds.ingested_forecast_length.loc[{"init_time": ds.init_time.values}] = (
        ds.ingested_forecast_length
    )

    # Uncomment this line for local testing to scope down the number of init times
    # template_ds = template_ds.isel(init_time=slice(0, len(ds.init_time) + 2))

    # We make some assumptions about what is safe to parallelize and how to
    # write the data based on the init_time dimension having a shard size of one.
    # If this changes we will need to refactor.
    assert all(
        1 == da.encoding["shards"][da.dims.index(template.APPEND_DIMENSION)]
        for da in ds.data_vars.values()
    )
    new_init_times = template_ds.init_time.loc[
        template_ds.init_time > last_existing_init_time
    ]
    new_init_time_indices = template_ds.get_index(
        template.APPEND_DIMENSION
    ).get_indexer(new_init_times)  # type: ignore
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
    recent_incomplete_init_times = _get_recent_init_times_for_reprocessing(ds)
    recent_incomplete_init_times_indices = template_ds.get_index(
        template.APPEND_DIMENSION
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
            [(init_time_i_slice, list(template_ds.data_vars.keys()))],
            template_ds,
            tmp_store,
            _VARIABLES_PER_BACKFILL_JOB,
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


def _get_operational_update_init_time_end() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None)


def _get_recent_init_times_for_reprocessing(ds: xr.Dataset) -> Array1D[np.datetime64]:
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


def operational_kubernetes_resources(image_tag: str) -> Iterable[Job]:
    operational_update_cron_job = ReformatCronJob(
        name=f"{template.DATASET_ID}-operational-update",
        schedule=_OPERATIONAL_CRON_SCHEDULE,
        image=image_tag,
        dataset_id=template.DATASET_ID,
        cpu="6",  # fit on 8 vCPU node
        memory="60G",  # fit on 64GB node
        shared_memory="24G",
        ephemeral_storage="150G",
    )
    validation_cron_job = ValidationCronJob(
        name=f"{template.DATASET_ID}-validation",
        schedule=_VALIDATION_CRON_SCHEDULE,
        image=image_tag,
        dataset_id=template.DATASET_ID,
        cpu="3",  # fit on 4 vCPU node
        memory="30G",  # fit on 32GB node
    )

    return [operational_update_cron_job, validation_cron_job]


def all_jobs_ordered(
    template_ds: xr.Dataset,
    chunk_filters: ChunkFilters,
) -> list[tuple[slice, list[str]]]:
    if chunk_filters.time_start is not None:
        template_ds = template_ds.sel(
            {chunk_filters.time_dim: slice(chunk_filters.time_start, None)}
        )
    if chunk_filters.time_end is not None:
        # end point is exclusive
        template_ds = template_ds.sel(
            {
                chunk_filters.time_dim: template_ds[chunk_filters.time_dim]
                < pd.Timestamp(chunk_filters.time_end)  # type: ignore[operator]
            }
        )
    if chunk_filters.variable_names is not None:
        template_ds = template_ds[chunk_filters.variable_names]

    time_i_slices = dimension_slices(template_ds, chunk_filters.time_dim)

    data_var_groups = group_data_vars_by_gefs_file_type(
        [d for d in template.DATA_VARIABLES if d.name in template_ds],
        group_size=_VARIABLES_PER_BACKFILL_JOB,
    )
    data_var_name_groups = [
        [data_var.name for data_var in data_vars] for _, _, data_vars in data_var_groups
    ]

    return list(product(time_i_slices, data_var_name_groups))


def get_store() -> zarr.storage.FsspecStore:
    return get_zarr_store(template.DATASET_ID, template.DATASET_VERSION)
