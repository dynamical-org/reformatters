from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import validation
from reformatters.common.storage import DatasetFormat
from reformatters.ecmwf.fast.dynamical_dataset import EcmwfFastDataset
from reformatters.ecmwf.fast.source import SOURCE_PREFIX, SOURCE_SECRET
from tests.common.validation_decode_test import _context, _spy_on_load

OUTPUT_SECRET = "dynamical-ecmwf-output-r2-storage-options-key"  # noqa: S105

DATASETS = [d for d in DYNAMICAL_DATASETS if isinstance(d, EcmwfFastDataset)]


def test_registered_ids_and_separate_source_output_access() -> None:
    assert {d.dataset_id for d in DATASETS} == {
        "ecmwf-ifs-ens-forecast-0-25-degree-fast",
        "ecmwf-aifs-ens-forecast-fast",
    }
    for dataset in DATASETS:
        assert dataset.rolling_positions == 12
        assert (
            dataset.template_config.dataset_attributes.time_domain
            == "Most recent 12 forecast initializations"
        )
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
    assert not cron.suspend
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


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda d: d.dataset_id)
@pytest.mark.parametrize("missing_index", [None, *range(9)])
def test_decode_health_checks_every_variable(
    dataset: EcmwfFastDataset,
    missing_index: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _spy_on_load(monkeypatch)
    init = pd.Timestamp("2026-11-10")
    variables = dataset.template_config.data_vars
    ds = xr.Dataset(
        {
            var.name: (
                ("init_time", "latitude", "longitude"),
                np.full((1, 1, 1), np.nan if i == missing_index else 1.0),
            )
            for i, var in enumerate(variables)
        },
        coords={"init_time": [init], "latitude": [0], "longitude": [0]},
    )
    context = _context(ds, [var.path for var in variables], [{"init_time": init}])
    check = next(
        c
        for c in dataset.validators()
        if isinstance(c, validation.CheckVirtualDecodeHealth)
    )
    result = check.check(context)
    assert result.passed == (missing_index is None), result.message
    assert result.checked_count == 9
    assert {call.name for call in loads} == {v.name for v in variables}
    if missing_index is not None:
        assert (
            f"{variables[missing_index].name}: every sampled chunk decoded entirely NaN"
            in result.message
        )


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda d: d.dataset_id)
def test_decode_health_checks_newest_init_with_missing_historical_variables(
    dataset: EcmwfFastDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads = _spy_on_load(monkeypatch)
    inits = pd.date_range("2026-11-10", periods=2, freq="6h")
    variables = dataset.template_config.data_vars
    ds = xr.Dataset(
        {
            var.name: (("init_time", "latitude", "longitude"), np.ones((2, 1, 1)))
            for var in variables
        },
        coords={"init_time": inits, "latitude": [0], "longitude": [0]},
    )
    for var in variables[7:]:
        ds[var.name][0] = np.nan
    check = next(
        c
        for c in dataset.validators()
        if isinstance(c, validation.CheckVirtualDecodeHealth)
    )
    assert check.positions == 1
    assert check.reference_presence is None
    context = _context(
        ds, [v.path for v in variables], [{"init_time": init} for init in inits]
    )
    result = check.check(context)
    assert result.passed, result.message
    assert result.checked_count == 9
    assert len(loads) == 9
    assert all(
        call.coordinates["init_time"] == ds.init_time.values[-1].tolist()
        for call in loads
    )
