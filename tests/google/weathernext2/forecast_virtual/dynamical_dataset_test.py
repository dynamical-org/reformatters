import re
from datetime import timedelta
from pathlib import Path

import icechunk
import pandas as pd
import xarray as xr
from zarr.storage import MemoryStore

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.types import DatetimeLike
from reformatters.google.weathernext2.forecast_historical_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastHistoricalVirtualDataset,
)
from reformatters.google.weathernext2.forecast_operational_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastOperationalVirtualDataset,
)
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    OPERATIONAL_SOURCE_LANDING_DELAY,
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

    (update,) = dataset.operational_kubernetes_resources("test")
    assert update.name == "google-wn2-forecast-historical-virtual-update"
    assert dataset.region_job_class.tick_interval == pd.Timedelta("1s")
    assert update.suspend is True
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
    assert "partial" not in dataset.template_config.dataset_attributes.description
    # The historical layout packs 4 members and all 13 levels per chunk, so both array
    # groups hold the same refs per init and take one coarse split.
    split = dataset.icechunk_virtual_config.manifest_split
    assert _resolved_split_size(split, "/pressure_level/temperature") == 128
    assert _resolved_split_size(split, "/temperature_2m") == 128


def test_operational_product_has_virtual_health_checks_and_splits(
    tmp_path: Path,
) -> None:
    dataset = GoogleWeathernext2ForecastOperationalVirtualDataset(
        primary_storage_config=_storage(tmp_path)
    )

    (update,) = dataset.operational_kubernetes_resources("test")
    assert dataset.region_job_class.tick_interval == pd.Timedelta("15s")
    assert update.name == "google-wn2-forecast-operational-virtual-update"
    assert update.schedule == "3 0,6,12,18 * * *"
    assert update.previous_fire_time(pd.Timestamp("2026-09-04T01:00")) == pd.Timestamp(
        "2026-09-04T00:03"
    )
    assert update.previous_fire_time(pd.Timestamp("2026-09-04T00:00")) == pd.Timestamp(
        "2026-09-03T18:03"
    )
    assert update.workers_total == 1
    assert update.parallelism == 1
    assert (
        update.pod_active_deadline
        == update.active_deadline
        == timedelta(hours=6, minutes=4, seconds=23)
    )
    assert update.concurrency_policy == "Allow"
    assert update.suspend is False
    fire = pd.Timestamp("2026-10-20T06:03")
    assert dataset._virtual_poll_deadline(
        fire + pd.Timedelta("1min")
    ) == fire + pd.Timedelta("6h1m16s")
    assert dataset.virtual_poll_deadline_grace == timedelta(minutes=3, seconds=7)
    (current,) = [
        item
        for item in dataset.validators()
        if isinstance(item, validation.CheckCurrentData)
    ]
    assert current.max_delay == timedelta(hours=9)
    (completeness,) = [
        item
        for item in dataset.validators()
        if isinstance(item, validation.CheckVirtualManifestCompleteness)
    ]
    assert completeness.min_present_fraction == (0.05, 1.0)
    (decode,) = [
        item
        for item in dataset.validators()
        if isinstance(item, validation.CheckVirtualDecodeHealth)
    ]
    assert decode.positions == "all"
    assert decode.max_positions == 2
    assert dataset.region_job_class.operational_update_window == timedelta(hours=24)
    description = dataset.template_config.dataset_attributes.description
    assert description == (
        "Weather forecasts from the 64-member Google DeepMind WeatherNext 2 ensemble model."
    )
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


def test_operational_fire_selects_arriving_init_and_keeps_successor_overlap(
    tmp_path: Path,
) -> None:
    dataset = GoogleWeathernext2ForecastOperationalVirtualDataset(
        primary_storage_config=_storage(tmp_path)
    )
    config = dataset.template_config
    (cron,) = dataset.operational_kubernetes_resources("test")

    def get_template(end: DatetimeLike) -> xr.DataTree:
        return xr.DataTree(
            xr.Dataset(coords={"init_time": config.append_dim_coordinates(end)})
        )

    for hour in (0, 6, 12, 18):
        anchor = pd.Timestamp("2026-10-28") + pd.Timedelta(hours=hour)
        fire = cron.previous_fire_time(anchor + pd.Timedelta("6h18m21s"))
        successor = cron.next_fire_time(fire)
        for selected_fire, newest in (
            (fire, anchor),
            (successor, anchor + pd.Timedelta("6h")),
        ):
            (job,), template = dataset.region_job_class.operational_update_jobs(
                MemoryStore(),
                tmp_path,
                get_template,
                "init_time",
                config.data_vars[:1],
                "window-test",
                selected_fire,
            )
            inits = template.to_dataset().get_index("init_time")[job.region]
            assert inits[-1] == newest
            assert anchor in inits
            assert newest - config.append_dim_frequency in inits
        assert fire - anchor >= OPERATIONAL_SOURCE_LANDING_DELAY
