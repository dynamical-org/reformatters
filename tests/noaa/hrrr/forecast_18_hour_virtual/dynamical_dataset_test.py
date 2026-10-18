import re
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pandas as pd
import pytest

from reformatters.common import template_utils, validation, virtual_region_job
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.noaa.hrrr.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast18HourVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual.region_job import (
    NoaaHrrrForecast18HourVirtualRegionJob,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from tests.common.dynamical_dataset_test import assert_configured_validators
from tests.common.virtual_region_job_test import (
    VirtualTestRegionJob,
    VirtualTestSourceFileCoord,
    _create_template_ds,
    _make_dataset,
    _make_region_job,
    _primary_repo,
)

_Y, _X = 635, 1062
_INIT = "2024-06-01T01:00"
_FILTER_VARS = [
    "temperature_2m",
    "wind_u_10m",
    "total_precipitation_surface",
    "temperature",
]


def make_dataset(tmp_path: Path) -> NoaaHrrrForecast18HourVirtualDataset:
    return NoaaHrrrForecast18HourVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaHrrrForecast18HourVirtualDataset:
    return make_dataset(tmp_path)


def test_polling_leaves_validation_time_inside_single_deadline(
    dataset: NoaaHrrrForecast18HourVirtualDataset,
) -> None:
    cron = dataset.operational_kubernetes_resources("test")[0]
    fire = pd.Timestamp("2026-10-02T00:50")
    deadline = pd.Timestamp("2026-10-02T01:55")

    assert cron.pod_active_deadline == timedelta(minutes=75)
    assert cron.active_deadline == cron.pod_active_deadline
    assert dataset._virtual_poll_deadline(fire) == deadline
    assert dataset._virtual_poll_deadline(fire + timedelta(minutes=10)) == deadline
    assert fire + cron.pod_active_deadline - deadline == timedelta(minutes=10)
    assert deadline - cron.next_fire_time(fire) == timedelta(minutes=5)
    rendered = cron.as_kubernetes_object()["spec"]
    assert rendered["concurrencyPolicy"] == "Allow"
    job_spec = rendered["jobTemplate"]["spec"]
    assert job_spec["activeDeadlineSeconds"] == 4500
    assert job_spec["template"]["spec"]["activeDeadlineSeconds"] == 4500


@pytest.mark.parametrize("ready_at", ["00:47:01", "00:49:59", "00:50:01"])
def test_late_source_pair_commits_during_hourly_handoff(
    dataset: NoaaHrrrForecast18HourVirtualDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ready_at: str,
) -> None:
    synthetic = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), synthetic.store_factory)
    repo = _primary_repo(synthetic.store_factory)
    now = [pd.Timestamp("2026-10-18T00:46:59")]
    paired_ready = pd.Timestamp(f"2026-10-18T{ready_at}")
    original_refs = VirtualTestRegionJob.process_virtual_refs

    def discover(
        self: VirtualTestRegionJob, pending: list[VirtualTestSourceFileCoord]
    ) -> list[tuple[VirtualTestSourceFileCoord, int]]:
        return [(coord, 48) for coord in pending] if now[0] >= paired_ready else []

    def refs(
        self: VirtualTestRegionJob, coord: VirtualTestSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        return list(next(original_refs(self, [coord]))[0][1])

    def advance(seconds: float) -> None:
        now[0] += pd.Timedelta(seconds=1)

    monkeypatch.setattr(pd.Timestamp, "now", lambda tz=None: now[0])
    monkeypatch.setattr(virtual_region_job.time, "sleep", advance)
    monkeypatch.setattr(VirtualTestRegionJob, "discover_available", discover)
    monkeypatch.setattr(VirtualTestRegionJob, "file_refs", refs)
    monkeypatch.setattr(
        VirtualTestRegionJob,
        "process_virtual_refs",
        VirtualRegionJob.process_virtual_refs,
    )
    job = _make_region_job(
        _create_template_ds(1), region=slice(0, 1), processing_mode="update"
    ).model_copy(
        update={
            "poll_deadline": dataset._virtual_poll_deadline(
                pd.Timestamp("2026-10-17T23:50")
            )
        }
    )

    job.process_virtual(repo, [], "main", job.source_file_coords())

    assert now[0] == paired_ready
    assert (
        job.filter_already_present(
            job.source_file_coords(), repo.readonly_session("main").store
        )
        == []
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)
    original_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: original_get_template(end_time).isel(lead_time=[0, 6]),
    )

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-06-01T02:00"),
        filter_start=pd.Timestamp(_INIT),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    cell = ds.isel(y=_Y, x=_X).sel(init_time=_INIT)
    f6 = cell.sel(lead_time=pd.Timedelta("6h"))
    np.testing.assert_allclose(f6["temperature_2m"].values, 20.892510986328148)
    np.testing.assert_allclose(f6["wind_u_10m"].values, -1.763387680053711)
    np.testing.assert_allclose(f6["total_precipitation_surface"].values, 0.004)
    np.testing.assert_allclose(
        f6["pressure_level/temperature"].sel(pressure_level=500).values,
        -11.32408752441404,
    )
    np.testing.assert_allclose(
        f6["model_level/temperature"].sel(model_level=1).values,
        20.603631591796898,
    )

    f0 = cell.sel(lead_time=pd.Timedelta("0h"))
    assert np.isnan(f0["total_precipitation_surface"].values)
    assert not np.isnan(f0["temperature_2m"].values)
    assert not np.isnan(f0["pressure_level/temperature"].sel(pressure_level=500).values)

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2024-06-01T05:00")),
    )
    original_update_jobs = (
        NoaaHrrrForecast18HourVirtualRegionJob.operational_update_jobs.__func__
    )

    def filtered_update_jobs(
        cls: type[NoaaHrrrForecast18HourVirtualRegionJob],
        *,
        all_data_vars: Sequence[NoaaHrrrDataVar],
        **kwargs: Any,  # noqa: ANN401 - passthrough to the wrapped classmethod
    ) -> object:
        return original_update_jobs(
            cls,
            all_data_vars=[v for v in all_data_vars if v.name in _FILTER_VARS],
            **kwargs,
        )

    monkeypatch.setattr(
        NoaaHrrrForecast18HourVirtualRegionJob,
        "operational_update_jobs",
        classmethod(filtered_update_jobs),
    )

    dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    update_f6 = updated.isel(y=_Y, x=_X).sel(
        init_time="2024-06-01T04:00", lead_time=pd.Timedelta("6h")
    )
    actual = [
        update_f6["temperature_2m"].item(),
        update_f6["wind_u_10m"].item(),
        update_f6["total_precipitation_surface"].item(),
        update_f6["pressure_level/temperature"].sel(pressure_level=500).item(),
        update_f6["model_level/temperature"].sel(model_level=1).item(),
    ]
    np.testing.assert_allclose(
        actual,
        [
            20.145928955078148,
            -1.2683000564575195,
            0.0,
            -11.303259277343727,
            19.885400390625023,
        ],
    )

    assert_configured_validators(dataset)


@pytest.mark.slow
def test_missing_source_file_remains_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)
    original_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: original_get_template(end_time).isel(lead_time=[18]),
    )

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2018-07-13T15:00"),
        filter_start=pd.Timestamp("2018-07-13T14:00"),
        filter_variable_names=["temperature"],
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    model_temperature = (
        ds["model_level/temperature"]
        .isel(y=_Y, x=_X)
        .sel(
            init_time="2018-07-13T14:00",
            lead_time=pd.Timedelta("18h"),
            model_level=1,
        )
    )
    assert np.isnan(model_temperature.item())


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrForecast18HourVirtualDataset,
) -> None:
    (update_cron_job,) = dataset.operational_kubernetes_resources("test-image-tag")

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert update_cron_job.schedule == "50 * * * *"
    assert update_cron_job.pod_active_deadline == timedelta(minutes=75)
    assert update_cron_job.cpu == "4"
    assert not update_cron_job.suspend
    assert len(update_cron_job.secret_names) > 0


def test_validators(dataset: NoaaHrrrForecast18HourVirtualDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    (current_data,) = [
        validator
        for validator in validators
        if isinstance(validator, validation.CheckCurrentData)
    ]
    assert current_data.max_delay == timedelta(minutes=52)
    completeness = next(
        validator
        for validator in validators
        if isinstance(validator, validation.CheckVirtualManifestCompleteness)
    )
    assert completeness.min_present_fraction == (0.05, 1.0)
    assert any(
        isinstance(validator, validation.CheckVirtualDecodeHealth)
        for validator in validators
    )


def _resolved_split_size(
    split: icechunk.ManifestSplittingConfig, array_path: str
) -> int:
    for condition, dim_splits in split.split_sizes:
        regex = getattr(condition, "regex", None)
        if regex is None or re.search(regex, array_path):
            [(_dim_condition, size)] = dim_splits
            return size
    raise AssertionError(f"no split rule matched {array_path}")


def test_manifest_split_size_resolves_per_group(
    dataset: NoaaHrrrForecast18HourVirtualDataset,
) -> None:
    split = dataset.icechunk_virtual_config.manifest_split
    assert _resolved_split_size(split, "/pressure_level/temperature") == 225
    assert _resolved_split_size(split, "/model_level/temperature") == 200
    assert _resolved_split_size(split, "/temperature_2m") == 1500


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaHrrrForecast18HourVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-hrrr-bdp-pds/"
