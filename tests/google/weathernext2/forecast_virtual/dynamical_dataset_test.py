import re
from datetime import timedelta
from pathlib import Path

import icechunk

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.google.weathernext2.forecast_historical_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastHistoricalVirtualDataset,
)
from reformatters.google.weathernext2.forecast_operational_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastOperationalVirtualDataset,
)


def _storage(tmp_path: Path) -> StorageConfig:
    return StorageConfig(base_path=str(tmp_path), format=DatasetFormat.ICECHUNK)


def _resolved_split_size(
    split: icechunk.ManifestSplittingConfig, array_path: str
) -> int:
    for condition, dim_splits in split.split_sizes:
        regex = getattr(condition, "regex", None)
        if regex is None or re.search(regex, array_path):
            [(_dim_condition, size)] = dim_splits
            return size
    raise AssertionError(f"no split rule matched {array_path}")


def test_historical_product_is_fixed_and_has_virtual_health_checks(
    tmp_path: Path,
) -> None:
    dataset = GoogleWeathernext2ForecastHistoricalVirtualDataset(
        primary_storage_config=_storage(tmp_path)
    )

    update, validate = dataset.operational_kubernetes_resources("test")
    assert update.name == f"{dataset.dataset_id}-update"
    assert update.suspend is True
    assert validate.name == f"{dataset.dataset_id}-validate"
    assert validate.suspend is True
    assert not any(
        isinstance(item, validation.CheckCurrentData) for item in dataset.validators()
    )
    assert any(
        isinstance(item, validation.CheckVirtualManifestCompleteness)
        for item in dataset.validators()
    )
    assert any(
        isinstance(item, validation.CheckVirtualDecodeHealth)
        for item in dataset.validators()
    )
    assert (
        _resolved_split_size(
            dataset.icechunk_virtual_config.manifest_split,
            "/pressure_level/temperature",
        )
        == 32
    )


def test_operational_product_has_lag_aware_crons_and_splits(tmp_path: Path) -> None:
    dataset = GoogleWeathernext2ForecastOperationalVirtualDataset(
        primary_storage_config=_storage(tmp_path)
    )

    update, validate = dataset.operational_kubernetes_resources("test")
    assert update.name == f"{dataset.dataset_id}-update"
    assert update.workers_total == 1
    assert update.parallelism == 1
    assert update.pod_active_deadline < timedelta(hours=6)
    assert update.suspend is True
    assert validate.suspend is True
    (current,) = [
        item
        for item in dataset.validators()
        if isinstance(item, validation.CheckCurrentData)
    ]
    assert current.max_delay == timedelta(hours=60)
    split = dataset.icechunk_virtual_config.manifest_split
    assert _resolved_split_size(split, "/pressure_level/temperature") == 4
    assert _resolved_split_size(split, "/temperature_2m") == 32


def test_both_products_use_the_raw_http_chunk_container(tmp_path: Path) -> None:
    datasets = (
        GoogleWeathernext2ForecastHistoricalVirtualDataset(
            primary_storage_config=_storage(tmp_path / "historical")
        ),
        GoogleWeathernext2ForecastOperationalVirtualDataset(
            primary_storage_config=_storage(tmp_path / "operational")
        ),
    )
    for dataset in datasets:
        (container,) = dataset.icechunk_virtual_config.containers
        assert container.url_prefix == "https://wn.dynamical.org/chunks/"
        assert isinstance(container.store, icechunk.ObjectStoreConfig.Http)
