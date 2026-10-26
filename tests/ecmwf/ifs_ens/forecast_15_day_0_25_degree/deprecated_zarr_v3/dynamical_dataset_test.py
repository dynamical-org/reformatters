from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.iterating import item
from reformatters.common.storage import DatasetFormat
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.deprecated_zarr_v3 import (
    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.deprecated_zarr_v3.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)

PUBLISHED_DATASET_ID = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"


def _registered[T: DynamicalDataset[Any, Any]](cls: type[T]) -> T:
    dataset = item(d for d in DYNAMICAL_DATASETS if type(d) is cls)
    assert isinstance(dataset, cls)
    return dataset


def test_registered_zarr_dataset_keeps_published_store_and_metadata() -> None:
    dataset = _registered(EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset)

    assert dataset.dataset_id == "ecmwf-ifs-ens-15-day-deprecated-zarr-v3"
    assert dataset.template_config.dataset_id == PUBLISHED_DATASET_ID
    assert dataset.primary_storage_config.format == DatasetFormat.ZARR3
    assert dataset.store_factory.replica_urls() == []
    assert (
        dataset.store_factory.primary_url()
        == f"s3://us-west-2.opendata.source.coop/dynamical/{PUBLISHED_DATASET_ID}/v0.1.0.zarr"
    )


def test_registered_icechunk_dataset_has_no_replica() -> None:
    dataset = _registered(EcmwfIfsEnsForecast15Day025DegreeDataset)

    assert dataset.dataset_id == PUBLISHED_DATASET_ID
    assert dataset.primary_storage_config.format == DatasetFormat.ICECHUNK
    assert dataset.store_factory.replica_urls() == []


def test_template_keeps_april_2024_start() -> None:
    template_config = _registered(
        EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset
    ).template_config

    assert template_config.append_dim_start == pd.Timestamp("2024-04-01T00:00")
    template = template_config.get_template(pd.Timestamp("2024-04-03"))
    assert template["init_time"].to_numpy()[0] == pd.Timestamp("2024-04-01T00:00")
    assert template.attrs["dataset_id"] == PUBLISHED_DATASET_ID


def test_operational_kubernetes_resources() -> None:
    dataset = _registered(EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset)
    icechunk_dataset = _registered(EcmwfIfsEnsForecast15Day025DegreeDataset)

    (update_cron_job,) = dataset.operational_kubernetes_resources("test-image-tag")
    (icechunk_cron_job,) = icechunk_dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == "ecmwf-ifs-ens-15-day-deprecated-zarr-v3-update"
    assert update_cron_job.dataset_id == dataset.dataset_id
    assert update_cron_job.schedule == "45 8 * * *"
    assert update_cron_job.suspend is True
    assert update_cron_job.workers_total == 2 * dataset.num_variable_groups()
    assert update_cron_job.parallelism == min(update_cron_job.workers_total, 10)
    assert update_cron_job.secret_names == [
        dataset.primary_storage_config.k8s_secret_name
    ]
    assert dataset._validation_monitor_name() == (
        "ecmwf-ifs-ens-15-day-deprecated-zarr-v3-validate"
    )
    assert update_cron_job.name != icechunk_cron_job.name
    assert (
        update_cron_job.as_kubernetes_object()["metadata"]["name"]
        != icechunk_cron_job.as_kubernetes_object()["metadata"]["name"]
    )


def test_validators() -> None:
    validators = _registered(
        EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset
    ).validators()

    assert validators == (
        validation.CheckCurrentData(max_delay=timedelta(hours=9, minutes=20)),
        validation.CheckRecentNans(append_dim_window=3),
    )


def _shifted_store(end: str) -> xr.DataTree:
    template_config = EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig()
    return template_config.model_copy(
        update={"append_dim_start": pd.Timestamp("2024-04-02")}
    ).get_template(pd.Timestamp(end))


def test_backfill_rejects_store_with_shifted_labels() -> None:
    dataset = _registered(EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset)
    existing_ds = _shifted_store("2024-04-04")

    with pytest.raises(AssertionError, match="do not match the template"):
        dataset._resolve_backfill_template(
            existing_ds,
            datetime(2024, 4, 3),
            overwrite_chunks=False,
            overwrite_metadata=True,
        )


def test_backfill_accepts_store_with_template_labels() -> None:
    dataset = _registered(EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset)
    existing_ds = dataset.template_config.get_template(pd.Timestamp("2024-04-03"))

    template_ds, _ = dataset._resolve_backfill_template(
        existing_ds,
        datetime(2024, 4, 3),
        overwrite_chunks=False,
        overwrite_metadata=True,
    )

    assert template_ds["init_time"].to_numpy()[0] == pd.Timestamp("2024-04-01")


def test_writers_reject_store_with_shifted_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _registered(EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset)
    monkeypatch.setattr(
        type(dataset), "_open_existing_store", lambda self: _shifted_store("2024-04-04")
    )

    with pytest.raises(AssertionError, match="do not match the template"):
        dataset._process_region_jobs(
            all_jobs=[],
            worker_index=0,
            workers_total=1,
            reformat_job_name="test",
            template_ds=dataset.template_config.get_template(
                pd.Timestamp("2024-04-03")
            ),
            tmp_store=tmp_path,
            update_template_with_results=True,
        )
