from datetime import datetime
from typing import Any
from uuid import uuid4

import pandas as pd

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.storage import DatasetFormat
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.profiling.config import END, START, dataset
from reformatters.ecmwf.ifs_ens.profiling.worker import assert_image, jobs_for


def _utc(value: datetime | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def backfill_profile(
    source: DynamicalDataset[Any, Any],
    *,
    append_dim_end: datetime | None,
    jobs_per_pod: int,
    max_parallelism: int,
    filter_start: datetime | None,
    filter_end: datetime | None,
    filter_contains: list[datetime] | None,
    filter_variable_names: list[str] | None,
    docker_image: str | None,
    overwrite_chunks: bool,
    overwrite_metadata: bool,
    keep_branch: str | None,
) -> None:
    assert isinstance(source, EcmwfIfsEnsForecast15Day025DegreeDataset)
    assert source.dataset_id == "ecmwf-ifs-ens-forecast-15-day-0-25-degree", (
        "Profiling supports only the IFS ENS 15-day Icechunk dataset"
    )
    assert source.primary_storage_config.format == DatasetFormat.ICECHUNK
    assert not source.replica_storage_configs
    assert not any((overwrite_chunks, overwrite_metadata, keep_branch is not None)), (
        "Profiling requires create-new-store without --keep-branch"
    )
    assert _utc(append_dim_end) == END
    assert (_utc(filter_start), _utc(filter_end)) == (START, END), (
        "Profiling requires [2024-03-30, 2024-04-01)"
    )
    assert not any((filter_contains, filter_variable_names)), (
        "Profiling processes all 19 variables and both initializations"
    )
    assert jobs_per_pod in {1, 10}
    assert max_parallelism == 2
    assert docker_image is not None
    assert_image(docker_image)

    scratch = dataset(uuid4().hex)
    jobs_for(scratch)
    scratch.backfill_kubernetes(
        append_dim_end=END.tz_localize(None).to_pydatetime(),
        jobs_per_pod=jobs_per_pod,
        max_parallelism=2,
        filter_start=START.tz_localize(None).to_pydatetime(),
        filter_end=END.tz_localize(None).to_pydatetime(),
        docker_image=docker_image,
    )
