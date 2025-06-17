import json
import subprocess
from itertools import batched, product

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from reformatters.common import docker, template_utils
from reformatters.common.iterating import (
    consume,
    dimension_slices,
    get_worker_jobs,
)
from reformatters.common.kubernetes import Job
from reformatters.common.logging import get_logger
from reformatters.common.types import DatetimeLike
from reformatters.common.zarr import (
    get_mode,
    get_zarr_store,
)
from reformatters.noaa.gfs.forecast.reformat_internals import (
    reformat_time_i_slices,
)
from reformatters.noaa.gfs.forecast.template_config import GFS_FORECAST_TEMPLATE_CONFIG

# More variables than we currently have but have a buffer in case we add more
_VARIABLES_PER_BACKFILL_JOB = 30

logger = get_logger(__name__)


def reformat_local(time_end: DatetimeLike) -> None:
    template_ds = GFS_FORECAST_TEMPLATE_CONFIG.get_template(time_end)
    store = get_store()

    logger.info("Writing metadata")
    template_utils.write_metadata(template_ds, store, get_mode(store))

    logger.info("Starting reformat")
    # Process all chunks by setting worker_index=0 and worker_total=1
    reformat_chunks(time_end, worker_index=0, workers_total=1)
    logger.info(f"Done writing to {store}")


def reformat_kubernetes(
    time_end: DatetimeLike,
    jobs_per_pod: int,
    max_parallelism: int,
    docker_image: str | None = None,
) -> None:
    image_tag = docker_image or docker.build_and_push_image()

    template_ds = GFS_FORECAST_TEMPLATE_CONFIG.get_template(time_end)
    store = get_store()
    logger.info(f"Writing zarr metadata to {store.path}")
    template_utils.write_metadata(template_ds, store, get_mode(store))

    num_jobs = sum(1 for _ in all_jobs_ordered(template_ds))
    workers_total = int(np.ceil(num_jobs / jobs_per_pod))
    parallelism = min(workers_total, max_parallelism)

    dataset_id = template_ds.attrs["dataset_id"]

    kubernetes_job = Job(
        image=image_tag,
        dataset_id=dataset_id,
        workers_total=workers_total,
        parallelism=parallelism,
        cpu="3.5",  # fit on 4 vCPU node
        memory="7G",  # fit on 8GB node
        shared_memory="1.5G",
        ephemeral_storage="20G",
        command=[
            "reformat-chunks",
            pd.Timestamp(time_end).isoformat(),
        ],
    )
    subprocess.run(  # noqa: S603
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=json.dumps(kubernetes_job.as_kubernetes_object()),
        text=True,
        check=True,
    )

    logger.info(f"Submitted kubernetes job {kubernetes_job.job_name}")


def reformat_chunks(
    time_end: DatetimeLike, *, worker_index: int, workers_total: int
) -> None:
    """Writes out array chunk data. Assumes the dataset metadata has already been written."""
    assert worker_index < workers_total
    template_ds = GFS_FORECAST_TEMPLATE_CONFIG.get_template(time_end)
    store = get_store()

    worker_jobs = get_worker_jobs(
        all_jobs_ordered(template_ds),
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


def all_jobs_ordered(
    template_ds: xr.Dataset,
) -> list[tuple[slice, list[str]]]:
    append_dim_slices = dimension_slices(
        template_ds, GFS_FORECAST_TEMPLATE_CONFIG.append_dim
    )
    data_var_groups = batched(
        [d for d in GFS_FORECAST_TEMPLATE_CONFIG.data_vars if d.name in template_ds],
        _VARIABLES_PER_BACKFILL_JOB,
    )
    data_var_name_groups = [
        [data_var.name for data_var in data_vars] for data_vars in data_var_groups
    ]
    return list(product(append_dim_slices, data_var_name_groups))


def get_store() -> zarr.storage.FsspecStore:
    return get_zarr_store(
        "s3://us-west-2.opendata.source.coop/dynamical",
        GFS_FORECAST_TEMPLATE_CONFIG.dataset_id,
        GFS_FORECAST_TEMPLATE_CONFIG.dataset_attributes.dataset_version,
    )
