from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast18HourVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual.region_job import (
    NoaaHrrrForecast18HourVirtualRegionJob,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from tests.common.dynamical_dataset_test import assert_configured_validators

# A non-synoptic (t01z) init - the hourly cycles between 00/06/12/18 are what this
# dataset adds over the 48-hour virtual dataset. Same heavy-rain cell as the 48-hour
# snapshot test (north-first y, row 0 = largest latitude).
_Y, _X = 635, 1062
_INIT = "2024-06-01T01:00"

# Variables spanning root + pressure_level + model_level. "temperature" matches both
# the pressure_level and model_level group vars (un-suffixed group var names).
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


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)

    # Trim to leads 0h and 6h to limit work (virtual backfill downloads only .idx
    # sidecars; decode happens when the snapshot cells are read).
    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).isel(lead_time=[0, 6]),
    )

    # 1. Backfill the single non-synoptic 2024-06-01T01 init.
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-06-01T02:00"),
        filter_start=pd.Timestamp(_INIT),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert ds.init_time.values[-1] == np.datetime64(_INIT)

    cell = ds.isel(y=_Y, x=_X).sel(init_time=_INIT)
    f6 = cell.sel(lead_time=pd.Timedelta("6h"))
    # Snapshot values (decoded raw GRIB; temperature converted K->C by the codec).
    np.testing.assert_allclose(f6["temperature_2m"].values, 20.892510986328148)
    np.testing.assert_allclose(f6["wind_u_10m"].values, -1.763387680053711)
    np.testing.assert_allclose(f6["total_precipitation_surface"].values, 0.004)
    np.testing.assert_allclose(
        f6["pressure_level/temperature"].sel(pressure_level=500).values,
        -11.32408752441404,
    )
    np.testing.assert_allclose(
        f6["model_level/temperature"].sel(model_level=1).values, 20.603631591796898
    )

    # Hour-0 handling: accumulated precip is absent (excluded in coord generation),
    # instant fields are present.
    f0 = cell.sel(lead_time=pd.Timedelta("0h"))
    assert np.isnan(f0["total_precipitation_surface"].values)
    assert not np.isnan(f0["temperature_2m"].values)
    assert not np.isnan(f0["pressure_level/temperature"].sel(pressure_level=500).values)

    # 2. Operational update: "now" a few hourly cycles later.
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2024-06-01T05:00")),
    )
    orig_update_jobs = (
        NoaaHrrrForecast18HourVirtualRegionJob.operational_update_jobs.__func__  # type: ignore[attr-defined]
    )

    def filtered_update_jobs(
        cls: type[NoaaHrrrForecast18HourVirtualRegionJob],
        *,
        all_data_vars: Sequence[NoaaHrrrDataVar],
        **kwargs: Any,  # noqa: ANN401 - passthrough to the wrapped classmethod
    ) -> object:
        return orig_update_jobs(
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
    # The 6h update window before 05:00 ingests every hourly init through 04z.
    assert updated.init_time.values[-1] == np.datetime64("2024-06-01T04:00")
    new_cell = updated.isel(y=_Y, x=_X).sel(
        init_time="2024-06-01T04:00", lead_time=pd.Timedelta("6h")
    )
    t6 = float(new_cell["temperature_2m"].values)
    assert -60.0 < t6 < 60.0  # plausible Celsius
    assert not np.isnan(new_cell["model_level/temperature"].sel(model_level=1).values)

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrForecast18HourVirtualDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))
    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    # Single-writer virtual update: no fan-out.
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    # Hourly fires must never overlap.
    assert update_cron_job.schedule == "50 * * * *"
    assert update_cron_job.pod_active_deadline < timedelta(hours=1)
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert len(update_cron_job.secret_names) > 0


def test_validators(dataset: NoaaHrrrForecast18HourVirtualDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    completeness = next(
        v
        for v in validators
        if isinstance(v, validation.CheckVirtualManifestCompleteness)
    )
    # The current init publishes over ~2 hourly cycles, so only positions ingested
    # >= 2 update fires ago are required to be complete.
    assert completeness.min_present_fraction == (0.0, 0.0, 1.0)
    assert any(isinstance(v, validation.CheckVirtualDecodeHealth) for v in validators)


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaHrrrForecast18HourVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-hrrr-bdp-pds/"
