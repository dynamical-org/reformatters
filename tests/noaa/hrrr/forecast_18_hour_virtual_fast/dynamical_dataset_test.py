import re
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import icechunk
import numpy as np
import obstore.store
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils, validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.dynamical_dataset import (
    CheckMirrorRefsRepointed,
    NoaaHrrrForecast18HourVirtualFastDataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.region_job import (
    PENDING_REPOINT_JOB_NAME,
    NoaaHrrrForecast18HourVirtualFastRegionJob,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.nomads_mirror import MIRROR_SECRET_NAME
from tests.common.dynamical_dataset_test import assert_configured_validators

_Y, _X = 635, 1062
_INIT = "2026-09-01T01:00"
_FILTER_VARS = [
    "temperature_2m",
    "wind_u_10m",
    "total_precipitation_surface",
    "temperature",
]


def make_dataset(tmp_path: Path) -> NoaaHrrrForecast18HourVirtualFastDataset:
    return NoaaHrrrForecast18HourVirtualFastDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaHrrrForecast18HourVirtualFastDataset:
    return make_dataset(tmp_path)


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
        append_dim_end=pd.Timestamp("2026-09-01T02:00"),
        filter_start=pd.Timestamp(_INIT),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    cell = ds.isel(y=_Y, x=_X).sel(init_time=_INIT)
    f6 = cell.sel(lead_time=pd.Timedelta("6h"))
    np.testing.assert_allclose(f6["temperature_2m"].values, 28.350335693359398)
    np.testing.assert_allclose(f6["wind_u_10m"].values, -0.7518806457519531)
    np.testing.assert_allclose(f6["total_precipitation_surface"].values, 0.0)
    np.testing.assert_allclose(
        f6["pressure_level/temperature"].sel(pressure_level=500).values,
        -4.524816894531227,
    )
    np.testing.assert_allclose(
        f6["model_level/temperature"].sel(model_level=1).values,
        28.178521728515648,
    )

    f0 = cell.sel(lead_time=pd.Timedelta("0h"))
    assert np.isnan(f0["total_precipitation_surface"].values)
    assert not np.isnan(f0["temperature_2m"].values)
    assert not np.isnan(f0["pressure_level/temperature"].sel(pressure_level=500).values)

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2026-09-01T05:00")),
    )
    original_update_jobs = (
        NoaaHrrrForecast18HourVirtualFastRegionJob.operational_update_jobs.__func__
    )

    def filtered_update_jobs(
        cls: type[NoaaHrrrForecast18HourVirtualFastRegionJob],
        *,
        all_data_vars: Sequence[NoaaHrrrDataVar],
        **kwargs: Any,  # noqa: ANN401 - passthrough to the wrapped classmethod
    ) -> object:
        jobs, template_ds = original_update_jobs(
            cls,
            all_data_vars=[v for v in all_data_vars if v.name in _FILTER_VARS],
            **kwargs,
        )
        return jobs, template_ds

    monkeypatch.setattr(
        NoaaHrrrForecast18HourVirtualFastRegionJob,
        "operational_update_jobs",
        classmethod(filtered_update_jobs),
    )

    monkeypatch.setattr(
        NoaaHrrrForecast18HourVirtualFastRegionJob,
        "mirror_store",
        lambda self: obstore.store.LocalStore(tmp_path / "empty-mirror", mkdir=True),
    )
    dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    update_f6 = updated.isel(y=_Y, x=_X).sel(
        init_time="2026-09-01T04:00", lead_time=pd.Timedelta("6h")
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
            27.065789794921898,
            0.4588966369628906,
            0.0,
            -4.87865295410154,
            26.869805908203148,
        ],
    )

    assert (
        dataset.store_factory.list_coordination_files(PENDING_REPOINT_JOB_NAME, "")
        == []
    )
    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert update_cron_job.schedule == "50 * * * *"
    assert update_cron_job.pod_active_deadline == timedelta(minutes=59)
    assert update_cron_job.cpu == "4"
    assert not update_cron_job.suspend
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert validation_cron_job.schedule == "49 * * * *"
    assert not validation_cron_job.suspend
    assert update_cron_job.name == "noaa-hrrr-forecast-18-hour-virtual-fast-update"
    assert (
        validation_cron_job.name == "noaa-hrrr-forecast-18-hour-virtual-fast-validate"
    )
    store_secrets = dataset.store_factory.k8s_secret_names()
    assert update_cron_job.secret_names == [*store_secrets, MIRROR_SECRET_NAME]
    assert validation_cron_job.secret_names == store_secrets
    assert MIRROR_SECRET_NAME not in validation_cron_job.secret_names


def test_validators(dataset: NoaaHrrrForecast18HourVirtualFastDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 4
    assert any(isinstance(v, CheckMirrorRefsRepointed) for v in validators)
    (current_data,) = [
        validator
        for validator in validators
        if isinstance(validator, validation.CheckCurrentData)
    ]
    assert current_data.max_delay == timedelta(hours=1, minutes=49)
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
    dataset: NoaaHrrrForecast18HourVirtualFastDataset,
) -> None:
    split = dataset.icechunk_virtual_config.manifest_split
    assert _resolved_split_size(split, "/pressure_level/temperature") == 225
    assert _resolved_split_size(split, "/model_level/temperature") == 200
    assert _resolved_split_size(split, "/temperature_2m") == 1500


def test_virtual_containers_match_the_ref_prefixes_of_both_sources(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset,
) -> None:
    prefixes = [c.url_prefix for c in dataset.icechunk_virtual_config.containers]
    assert prefixes == [
        "s3://noaa-hrrr-bdp-pds/",
        "https://noaa-hrrr-nomads-mirror.r2.dynamical.org/",
    ]


def test_validation_job_probes_the_manifest_without_the_mirror_override(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2026-09-02T01:00")),
    )
    template_ds = dataset.template_config.get_template(pd.Timestamp("2026-09-01T02:00"))
    template_utils.write_metadata(
        template_ds.isel(lead_time=[0]), dataset.store_factory
    )
    job = dataset._virtual_validation_region_job(dataset.validators(), "test")
    assert isinstance(job, NoaaHrrrForecast18HourVirtualFastRegionJob)
    assert not job.repoint_mirrored
    assert job.pending_repoints() == []


def _check_with_records(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset,
    monkeypatch: pytest.MonkeyPatch,
    keys: Sequence[str],
    now: pd.Timestamp,
) -> validation.ValidationResult:
    for key in keys:
        dataset.store_factory.write_coordination_file(
            PENDING_REPOINT_JOB_NAME, key.replace("/", "__"), b""
        )
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda cls, tz=None: now.tz_localize(tz) if tz else now),
    )
    template_ds = dataset.template_config.get_template(now)
    template_utils.write_metadata(
        template_ds.isel(lead_time=[0]), dataset.store_factory
    )
    job = dataset._virtual_validation_region_job(dataset.validators(), "test")
    assert job is not None
    context = validation.ValidationContext(
        store=Mock(), ds=xr.Dataset(), append_dim="init_time", region_job=job
    )
    return CheckMirrorRefsRepointed().check(context)


_KEY_19Z = "hrrr.20260907/conus/hrrr.t19z.wrfsfcf01.grib2"
_KEY_20Z = "hrrr.20260907/conus/hrrr.t20z.wrfprsf01.grib2"


def test_mirror_refs_check_passes_with_no_pending_records(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_with_records(
        dataset, monkeypatch, [], pd.Timestamp("2026-09-08T00:00")
    )
    assert result.passed
    assert result.checked_count == 0


def test_mirror_refs_check_passes_with_a_young_pending_record(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_with_records(
        dataset, monkeypatch, [_KEY_19Z], pd.Timestamp("2026-09-08T06:00")
    )
    assert result.passed
    assert result.checked_count == 1


def test_mirror_refs_check_fails_with_a_record_older_than_36_hours(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 19Z is 36 h 30 m old, 20Z is 35 h 30 m old.
    result = _check_with_records(
        dataset,
        monkeypatch,
        [_KEY_19Z, _KEY_20Z],
        pd.Timestamp("2026-09-09T07:30"),
    )
    assert not result.passed
    assert result.checked_count == 2
    assert _KEY_19Z in result.message
    assert _KEY_20Z not in result.message
