from pathlib import Path

import pytest

from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.forecast_48_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual_fast.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualFastDataset,
)


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaHrrrForecast48HourVirtualFastDataset:
    return NoaaHrrrForecast48HourVirtualFastDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


def test_manifest_split_is_a_single_root_rule(
    dataset: NoaaHrrrForecast48HourVirtualFastDataset,
) -> None:
    """Root-only dataset, so one catch-all rule and no vertical-group rules."""
    split = dataset.icechunk_virtual_config.manifest_split
    [(_condition, dim_splits)] = split.split_sizes
    [(_dim, size)] = dim_splits
    assert size == 600


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaHrrrForecast48HourVirtualFastDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-hrrr-bdp-pds/"


def test_operational_timing_matches_the_full_virtual_dataset(
    dataset: NoaaHrrrForecast48HourVirtualFastDataset,
) -> None:
    """Ingest latency between the two products must differ only by variable set, so
    their schedules and deadlines have to stay identical."""
    full = NoaaHrrrForecast48HourVirtualDataset(
        primary_storage_config=dataset.primary_storage_config
    )
    fast_jobs = {j.name: j for j in dataset.operational_kubernetes_resources("tag")}
    full_jobs = {j.name: j for j in full.operational_kubernetes_resources("tag")}
    for suffix in ("-update", "-validate"):
        fast_job = next(j for n, j in fast_jobs.items() if n.endswith(suffix))
        full_job = next(j for n, j in full_jobs.items() if n.endswith(suffix))
        assert fast_job.schedule == full_job.schedule, suffix
        assert fast_job.pod_active_deadline == full_job.pod_active_deadline, suffix


def test_has_a_mirror_job_the_full_dataset_lacks(
    dataset: NoaaHrrrForecast48HourVirtualFastDataset,
) -> None:
    names = [j.name for j in dataset.operational_kubernetes_resources("tag")]
    assert f"{dataset.dataset_id}-mirror" in names
    mirror = next(
        j
        for j in dataset.operational_kubernetes_resources("tag")
        if j.name.endswith("-mirror")
    )
    # The mirror must be warm before the update starts polling the cache.
    update = next(
        j
        for j in dataset.operational_kubernetes_resources("tag")
        if j.name.endswith("-update")
    )
    assert mirror.schedule.split()[0] < update.schedule.split()[0]


def test_ingest_is_suspended_until_backfilled(
    dataset: NoaaHrrrForecast48HourVirtualFastDataset,
) -> None:
    """The mirror can run alone; ingest must not write an unbackfilled store."""
    by_suffix = {
        j.name.rsplit("-", 1)[-1]: j
        for j in dataset.operational_kubernetes_resources("tag")
    }
    assert by_suffix["mirror"].suspend is False
    assert by_suffix["update"].suspend is True
    assert by_suffix["validate"].suspend is True


def test_cron_job_names_are_dns_safe(
    dataset: NoaaHrrrForecast48HourVirtualFastDataset,
) -> None:
    for job in dataset.operational_kubernetes_resources("tag"):
        assert len(job.name) <= 52, job.name


def test_validators_match_the_full_virtual_dataset(
    dataset: NoaaHrrrForecast48HourVirtualFastDataset,
) -> None:
    full = NoaaHrrrForecast48HourVirtualDataset(
        primary_storage_config=dataset.primary_storage_config
    )
    assert len(dataset.validators()) == len(full.validators())
