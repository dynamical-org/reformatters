from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from typer.testing import CliRunner

from reformatters.common import template_utils, validation
from reformatters.common.kubernetes import CronJob
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast import (
    dynamical_dataset as fast_dynamical_dataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.dynamical_dataset import (
    CheckMirrorWindow,
    NoaaHrrrForecast18HourVirtualFastDataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.region_job import (
    NoaaHrrrForecast18HourVirtualFastRegionJob,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.template_config import (
    NoaaHrrrForecast18HourVirtualFastTemplateConfig,
)
from reformatters.noaa.hrrr.nomads_mirror import (
    MIRROR_LOCATION_PREFIX,
    MIRROR_SECRET_NAME,
)

TEMPLATE_CONFIG = NoaaHrrrForecast18HourVirtualFastTemplateConfig()


def make_dataset(tmp_path: Path) -> NoaaHrrrForecast18HourVirtualFastDataset:
    return NoaaHrrrForecast18HourVirtualFastDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaHrrrForecast18HourVirtualFastDataset:
    return make_dataset(tmp_path)


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset,
) -> None:
    mirror_cron_job, update_cron_job, validation_cron_job = (
        dataset.operational_kubernetes_resources("test-image-tag")
    )

    assert mirror_cron_job.name == "noaa-hrrr-nomads-mirror-gribs"
    assert mirror_cron_job.command == ["mirror-gribs"]
    assert mirror_cron_job.dataset_id == dataset.dataset_id
    assert mirror_cron_job.schedule == "45 * * * *"
    assert mirror_cron_job.pod_active_deadline == timedelta(minutes=59)
    assert mirror_cron_job.secret_names == [MIRROR_SECRET_NAME]
    assert not mirror_cron_job.suspend

    assert update_cron_job.name == "noaa-hrrr-forecast-18-hour-virtual-fast-update"
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert update_cron_job.schedule == "50 * * * *"
    assert update_cron_job.pod_active_deadline == timedelta(minutes=59)
    assert update_cron_job.cpu == "4"
    assert not update_cron_job.suspend
    assert validation_cron_job.name == (
        "noaa-hrrr-forecast-18-hour-virtual-fast-validate"
    )
    assert validation_cron_job.schedule == "49 * * * *"
    assert validation_cron_job.pod_active_deadline == timedelta(minutes=30)
    assert not validation_cron_job.suspend
    store_secrets = dataset.store_factory.k8s_secret_names()
    assert update_cron_job.secret_names == [*store_secrets, MIRROR_SECRET_NAME]
    assert validation_cron_job.secret_names == store_secrets
    assert MIRROR_SECRET_NAME not in validation_cron_job.secret_names


def test_mirror_gribs_command_runs_the_mirror_cron(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[CronJob, int]] = []
    monkeypatch.setattr(
        fast_dynamical_dataset,
        "mirror_gribs",
        lambda cron_job, poll_start_minutes: calls.append(
            (cron_job, poll_start_minutes)
        ),
    )

    result = CliRunner().invoke(
        dataset.get_cli(), ["mirror-gribs", "job-name", "--poll-start-minutes", "50"]
    )

    assert result.exit_code == 0, result.output
    ((cron_job, poll_start_minutes),) = calls
    assert cron_job.name == "noaa-hrrr-nomads-mirror-gribs"
    assert poll_start_minutes == 50


def test_virtual_config_has_only_the_mirror_container_and_a_manifest_split(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset,
) -> None:
    config = dataset.icechunk_virtual_config

    assert [container.url_prefix for container in config.containers] == [
        MIRROR_LOCATION_PREFIX
    ]
    assert config.manifest_split is not None


def test_validators(dataset: NoaaHrrrForecast18HourVirtualFastDataset) -> None:
    validators = tuple(dataset.validators())

    assert [type(validator) for validator in validators] == [
        validation.CheckCurrentData,
        validation.CheckVirtualManifestCompleteness,
        validation.CheckVirtualDecodeHealth,
        CheckMirrorWindow,
    ]
    current_data = cast("validation.CheckCurrentData", validators[0])
    completeness = cast("validation.CheckVirtualManifestCompleteness", validators[1])
    assert current_data.max_delay == timedelta(hours=1, minutes=49)
    assert completeness.min_present_fraction == (0.05, 1.0)


def test_virtual_validation_region_is_the_last_poll_window(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-09-10T12:00")

    def fixed_now(cls: type[pd.Timestamp], tz: str | None = None) -> pd.Timestamp:
        return now.tz_localize(tz) if tz else now

    monkeypatch.setattr(pd.Timestamp, "now", classmethod(fixed_now))
    fire_time = dataset.operational_kubernetes_resources("test-image")[
        0
    ].previous_fire_time(now)
    template = dataset.template_config.get_template(fire_time).isel(lead_time=[0])
    template_utils.write_metadata(template, dataset.store_factory)

    job = dataset._virtual_validation_region_job(dataset.validators(), "test")

    assert isinstance(job, NoaaHrrrForecast18HourVirtualFastRegionJob)
    assert job.region.stop - job.region.start == 6
    init_times = job.template_ds.to_dataset().get_index("init_time")
    assert init_times[job.region].equals(init_times[-6:])


VALIDATED_PATHS = (
    "composite_reflectivity",
    "pressure_level/temperature",
    "model_level/temperature",
)


def mirror_validation_context(
    first_init_with_data: dict[str, int],
) -> validation.ValidationContext:
    """Three inits; each path is NaN (no ref) before its first init with data."""
    init_times = pd.date_range("2026-09-10T10:00", periods=3, freq="1h")
    lead_times = pd.timedelta_range("0h", periods=2, freq="1h")
    data = {}
    for path in VALIDATED_PATHS:
        values = np.ones((len(init_times), len(lead_times), 1, 1))
        values[: first_init_with_data.get(path, len(init_times))] = np.nan
        data[path] = (("init_time", "lead_time", "y", "x"), values)
    ds = xr.Dataset(
        data,
        coords={"init_time": init_times, "lead_time": lead_times, "y": [0], "x": [0]},
    )
    configured = {var.path: var for var in TEMPLATE_CONFIG.data_vars}
    return validation.ValidationContext(
        store=cast("Any", None),
        ds=ds,
        append_dim="init_time",
        data_vars=tuple(configured[path] for path in VALIDATED_PATHS),
    )


def test_mirror_window_passes_when_each_file_types_oldest_data_decodes() -> None:
    context = mirror_validation_context(
        {
            "composite_reflectivity": 0,
            "pressure_level/temperature": 1,
            "model_level/temperature": 2,
        }
    )

    result = CheckMirrorWindow().check(context)

    assert result.passed, result.message
    assert "composite_reflectivity at 2026-09-10 10:00:00" in result.message
    assert "pressure_level/temperature at 2026-09-10 11:00:00" in result.message
    assert "model_level/temperature at 2026-09-10 12:00:00" in result.message


def test_mirror_window_fails_when_a_file_type_has_no_data() -> None:
    context = mirror_validation_context(
        {"composite_reflectivity": 0, "pressure_level/temperature": 0}
    )

    result = CheckMirrorWindow().check(context)

    assert not result.passed
    assert "of 3 file types" in result.message


def test_mirror_window_fails_when_the_oldest_chunk_does_not_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = mirror_validation_context(dict.fromkeys(VALIDATED_PATHS, 0))
    loaded: list[pd.Timestamp] = []

    def expired(self: xr.DataArray, **kwargs: object) -> xr.DataArray:
        loaded.append(pd.Timestamp(self["init_time"].item()))
        raise OSError("404 Not Found")

    monkeypatch.setattr(xr.DataArray, "load", expired)

    result = CheckMirrorWindow().check(context)

    assert not result.passed
    assert loaded == [pd.Timestamp("2026-09-10T10:00")]
    assert "does not decode" in result.message
    assert "404 Not Found" in result.message
