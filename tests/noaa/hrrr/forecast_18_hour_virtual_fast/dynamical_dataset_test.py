from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils, validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast import (
    dynamical_dataset as dynamical_dataset_module,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.dynamical_dataset import (
    CheckMirrorWindow,
    NoaaHrrrForecast18HourVirtualFastDataset,
    _mirror_failure,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.region_job import (
    NoaaHrrrForecast18HourVirtualFastRegionJob,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.template_config import (
    RETENTION,
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


def mirror_location(
    init_time: pd.Timestamp,
    *,
    file_type: str = "sfc",
    lead: int = 0,
) -> str:
    return (
        f"{MIRROR_LOCATION_PREFIX}hrrr.{init_time:%Y%m%d}/conus/"
        f"hrrr.t{init_time:%H}z.wrf{file_type}f{lead:02d}.grib2"
    )


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrForecast18HourVirtualFastDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

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


class FakeSession:
    def __init__(self, locations: list[str]) -> None:
        self._locations = locations

    def all_virtual_chunk_locations(self) -> list[str]:
        return self._locations


class FakeIcechunkStore:
    def __init__(self, locations: list[str]) -> None:
        self.session = FakeSession(locations)


def mirror_validation_context(locations: list[str]) -> validation.ValidationContext:
    init_times = pd.date_range("2026-09-10T10:00", periods=3, freq="1h")
    lead_times = pd.timedelta_range("0h", periods=5, freq="1h")
    paths = (
        "composite_reflectivity",
        "pressure_level/temperature",
        "model_level/temperature",
    )
    shape = (len(init_times), len(lead_times), 1, 1)
    ds = xr.Dataset(
        {
            path: (("init_time", "lead_time", "y", "x"), np.ones(shape))
            for path in paths
        },
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "y": [0],
            "x": [0],
        },
    )
    configured = {var.path: var for var in TEMPLATE_CONFIG.data_vars}
    return validation.ValidationContext(
        store=cast(Any, FakeIcechunkStore(locations)),
        ds=ds,
        append_dim="init_time",
        data_vars=tuple(configured[path] for path in paths),
    )


def test_mirror_window_checks_each_file_types_smallest_lead_at_oldest_ref_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oldest_ref_init = pd.Timestamp("2026-09-10T11:00")
    locations = [
        mirror_location(pd.Timestamp("2026-09-10T12:00"), lead=0),
        mirror_location(oldest_ref_init, file_type="sfc", lead=3),
        mirror_location(oldest_ref_init, file_type="sfc", lead=1),
        mirror_location(oldest_ref_init, file_type="prs", lead=4),
        mirror_location(oldest_ref_init, file_type="prs", lead=2),
        mirror_location(oldest_ref_init, file_type="nat", lead=3),
    ]
    context = mirror_validation_context(locations)
    loaded: list[tuple[str | None, pd.Timestamp, pd.Timedelta]] = []

    def load(chunk: xr.DataArray, **kwargs: object) -> xr.DataArray:
        loaded.append(
            (
                str(chunk.name) if chunk.name is not None else None,
                pd.Timestamp(chunk["init_time"].item()),
                pd.Timedelta(chunk["lead_time"].item()),
            )
        )
        return chunk

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda cls, *args: pd.Timestamp("2026-09-10T12:00")),
    )
    monkeypatch.setattr(dynamical_dataset_module, "IcechunkStore", FakeIcechunkStore)
    monkeypatch.setattr(xr.DataArray, "load", load)

    result = CheckMirrorWindow().check(context)

    assert result.passed
    assert result.checked_count == len(locations)
    assert set(loaded) == {
        ("composite_reflectivity", oldest_ref_init, pd.Timedelta("1h")),
        ("pressure_level/temperature", oldest_ref_init, pd.Timedelta("2h")),
        ("model_level/temperature", oldest_ref_init, pd.Timedelta("3h")),
    }


def test_window_failure_rejects_empty_store() -> None:
    assert CheckMirrorWindow()._window_failure(pd.DatetimeIndex([])) == (
        "Dataset has no init_time positions"
    )
    assert _mirror_failure(mirror_validation_context([]), []) == (
        "The store references no files"
    )


def test_window_failure_rejects_old_first_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-09-10T12:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, *args: now))
    init_times = pd.date_range(
        now - RETENTION - timedelta(hours=4), periods=72, freq="1h"
    )

    failure = CheckMirrorWindow()._window_failure(init_times)

    assert failure is not None
    assert "before" in failure


def test_window_failure_rejects_too_many_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-09-10T12:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, *args: now))
    init_times = pd.date_range(end=now, periods=74, freq="1h")

    failure = CheckMirrorWindow()._window_failure(init_times)

    assert failure is not None
    assert "74 init_time positions exceed" in failure


def test_mirror_failure_rejects_location_outside_mirror() -> None:
    foreign = "s3://noaa-hrrr-bdp-pds/hrrr.20260910/conus/file.grib2"

    locations = [mirror_location(pd.Timestamp("2026-09-10")), foreign]
    failure = _mirror_failure(mirror_validation_context(locations), locations)

    assert failure is not None
    assert foreign in failure


def test_mirror_failure_reports_read_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    location = mirror_location(pd.Timestamp("2026-09-10T11:00"), lead=2)
    context = mirror_validation_context([location])

    def fail_load(chunk: xr.DataArray, **kwargs: object) -> xr.DataArray:
        raise RuntimeError("expired mirror object")

    monkeypatch.setattr(xr.DataArray, "load", fail_load)
    failure = _mirror_failure(context, [location])

    assert failure is not None
    assert "composite_reflectivity" in failure
    assert "lead 0 days 02:00:00" in failure
    assert "expired mirror object" in failure
