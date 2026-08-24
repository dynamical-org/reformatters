import re
from datetime import timedelta
from pathlib import Path

import icechunk
import pytest

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.google.weathernext2.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastVirtualDataset,
)


@pytest.fixture
def dataset(tmp_path: Path) -> GoogleWeathernext2ForecastVirtualDataset:
    return GoogleWeathernext2ForecastVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


def test_operational_kubernetes_resources(
    dataset: GoogleWeathernext2ForecastVirtualDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))
    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    # Single-writer virtual update: no fan-out.
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    # A fire must not outlive the next one.
    assert update_cron_job.pod_active_deadline < timedelta(hours=6)
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert len(update_cron_job.secret_names) > 0
    # Nothing is backfilled yet.
    assert update_cron_job.suspend is True
    assert validation_cron_job.suspend is True


def test_validators(dataset: GoogleWeathernext2ForecastVirtualDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    (completeness,) = [
        v
        for v in validators
        if isinstance(v, validation.CheckVirtualManifestCompleteness)
    ]
    assert completeness.min_present_fraction == (1.0,)
    assert any(isinstance(v, validation.CheckVirtualDecodeHealth) for v in validators)


def test_current_data_validator_allows_publication_lag(
    dataset: GoogleWeathernext2ForecastVirtualDataset,
) -> None:
    (current_data,) = [
        v for v in dataset.validators() if isinstance(v, validation.CheckCurrentData)
    ]
    assert current_data.max_delay == timedelta(hours=60)


def _resolved_split_size(
    split: icechunk.ManifestSplittingConfig, array_path: str
) -> int:
    # Mirror icechunk's first-to-last rule matching: a path_matches condition
    # matches by regex search; the AnyArray catch-all matches every array.
    for condition, dim_splits in split.split_sizes:
        regex = getattr(condition, "regex", None)
        if regex is None or re.search(regex, array_path):
            [(_dim_condition, size)] = dim_splits
            return size
    raise AssertionError(f"no split rule matched {array_path}")


def test_manifest_split_size_resolves_per_group(
    dataset: GoogleWeathernext2ForecastVirtualDataset,
) -> None:
    split = dataset.icechunk_virtual_config.manifest_split
    assert _resolved_split_size(split, "/pressure_level/temperature") == 4
    assert _resolved_split_size(split, "/temperature_2m") == 32


def test_virtual_container_matches_ref_prefix(
    dataset: GoogleWeathernext2ForecastVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "https://wn.dynamical.org/chunks/"
    assert isinstance(container.store, icechunk.ObjectStoreConfig.Http)
