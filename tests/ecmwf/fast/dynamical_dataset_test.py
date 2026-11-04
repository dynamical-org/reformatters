from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import validation
from reformatters.common.storage import DatasetFormat
from reformatters.ecmwf.fast.dynamical_dataset import EcmwfFastDataset
from reformatters.ecmwf.fast.source import OUTPUT_SECRET, SOURCE_PREFIX, SOURCE_SECRET

DATASETS = [d for d in DYNAMICAL_DATASETS if isinstance(d, EcmwfFastDataset)]


def test_registered_ids_and_separate_source_output_access() -> None:
    assert {d.dataset_id for d in DATASETS} == {
        "ecmwf-ifs-ens-forecast-0-25-degree-fast",
        "ecmwf-aifs-ens-forecast-fast",
    }
    for dataset in DATASETS:
        assert dataset.primary_storage_config.format == DatasetFormat.ICECHUNK
        assert dataset.primary_storage_config.k8s_secret_name == OUTPUT_SECRET
        assert dataset.icechunk_virtual_config.source_secret_names == {
            SOURCE_PREFIX: SOURCE_SECRET
        }
        assert dataset.store_factory.k8s_secret_names() == [
            OUTPUT_SECRET,
            SOURCE_SECRET,
        ]
        containers = dataset.icechunk_virtual_config.containers
        assert len(containers) == 1
        assert containers[0].url_prefix == SOURCE_PREFIX


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda d: d.dataset_id)
def test_schedule_deadline_currency_and_repair_window(
    dataset: EcmwfFastDataset, tmp_path: Path
) -> None:
    cron = dataset.operational_kubernetes_resources("test-image")[0]
    minute = 8 if dataset.template_config.model == "ifs" else 34
    assert cron.schedule == f"{minute} 5,11,17,23 * * *"
    assert cron.suspend
    assert cron.workers_total == cron.parallelism == 1
    assert cron.concurrency_policy == "Allow"
    assert cron.ephemeral_storage == "20G"
    assert (
        cron.pod_active_deadline
        == cron.active_deadline
        == timedelta(hours=6, minutes=3, seconds=30)
    )
    assert cron.pod_active_deadline - dataset.virtual_poll_deadline_grace == timedelta(
        hours=6, minutes=1
    )
    checks = dataset.validators()
    currency = checks[0]
    assert isinstance(currency, validation.CheckCurrentData)
    completeness = checks[1]
    assert isinstance(completeness, validation.CheckVirtualManifestCompleteness)
    assert completeness.min_present_fraction == (1.0,)
    decode = checks[2]
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    assert decode.max_workers <= 4
    assert decode.sampled_leads == 3
    assert dataset.region_job_class.operational_update_window == pd.Timedelta("24h")
    for hour in (0, 6, 12, 18):
        init = pd.Timestamp("2026-11-02") + pd.Timedelta(hours=hour)
        intended_fire = init + timedelta(hours=5, minutes=minute)
        assert cron.previous_fire_time(intended_fire) == intended_fire
        assert currency.max_delay >= intended_fire - init
