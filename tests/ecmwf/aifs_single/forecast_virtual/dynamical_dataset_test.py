import re
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pandas as pd
import pytest

from reformatters.common import validation
from reformatters.common.pydantic import replace
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
)
from reformatters.ecmwf.aifs_single.forecast_virtual.dynamical_dataset import (
    EcmwfAifsSingleForecastVirtualDataset,
)
from reformatters.ecmwf.aifs_single.forecast_virtual.region_job import (
    EcmwfAifsSingleForecastVirtualRegionJob,
)
from reformatters.ecmwf.aifs_single.forecast_virtual.template_config import (
    EcmwfAifsSingleVirtualDataVar,
)
from tests.common.dynamical_dataset_test import assert_configured_validators

# A rainy Amazon cell in the 2025-03-01T00 init (after the 2025-02-26 format change).
_LAT, _LON = -5.0, -60.0
_INIT = "2025-03-01T00:00"

# Every variable here costs source range requests -- one per cell asserted below, plus
# one per variable in the decode-health validator -- and each request downloads and
# decodes a whole global field. So this covers only what real bytes in a real store can
# prove, and nothing that a unit test can:
#   temperature_2m               a root var's refs decode into the right cell, with the
#                                K->C filter applied on read
#   temperature                  the group-var layout: a pressure_level ref lands at the
#                                right level ("temperature" is the un-suffixed group var)
#   geopotential_height_surface  a lead_0_only var is real data at lead 0 and reads as
#                                fill (not stale bytes) at lead 6, and exercises the
#                                second read-time filter, geopotential -> height
# Message routing for every other variable -- soil levels, accumulations, absent levels,
# stale byte ranges -- is checked against a recorded real .index, for all 35 variables
# and with no network, by region_job_test.py.
_FILTER_VARS = [
    "temperature_2m",
    "geopotential_height_surface",
    "temperature",
]


def make_dataset(tmp_path: Path) -> EcmwfAifsSingleForecastVirtualDataset:
    return EcmwfAifsSingleForecastVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> EcmwfAifsSingleForecastVirtualDataset:
    return make_dataset(tmp_path)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)

    # Trim to leads 0h and 6h to limit work (virtual backfill downloads only .index
    # sidecars; decode happens when the snapshot cells are read).
    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).isel(lead_time=[0, 1]),
    )

    # 1. Backfill the single 2025-03-01T00 init.
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2025-03-01T01:00"),
        filter_start=pd.Timestamp(_INIT),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert ds.init_time.values[-1] == np.datetime64("2025-03-01T00:00")

    cell = ds.sel(latitude=_LAT, longitude=_LON, init_time=_INIT)
    f6 = cell.sel(lead_time=pd.Timedelta("6h"))
    # Snapshot values (decoded raw GRIB; the K->C and geopotential->height
    # ScaleOffset filters apply on read).
    np.testing.assert_allclose(f6["temperature_2m"].values, 24.408471679687523)
    np.testing.assert_allclose(
        f6["pressure_level/temperature"].sel(pressure_level=850).values,
        18.03638610839846,
    )

    f0 = cell.sel(lead_time=pd.Timedelta("0h"))
    np.testing.assert_allclose(f0["temperature_2m"].values, 26.940789794921898)
    # Statics are published at lead 0 only, so lead 6 has no reference and reads as fill.
    np.testing.assert_allclose(
        f0["geopotential_height_surface"].values, 41.25282353842801
    )
    assert np.isnan(f6["geopotential_height_surface"].values)

    # 2. Operational update: "now" at the update cron fire that covers the 06z init.
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2025-03-01T11:20")),
    )
    orig_update_jobs = (
        EcmwfAifsSingleForecastVirtualRegionJob.operational_update_jobs.__func__  # type: ignore[attr-defined]
    )

    def filtered_update_jobs(
        cls: type[EcmwfAifsSingleForecastVirtualRegionJob],
        *,
        all_data_vars: Sequence[EcmwfAifsSingleVirtualDataVar],
        **kwargs: Any,  # noqa: ANN401 - passthrough to the wrapped classmethod
    ) -> object:
        return orig_update_jobs(
            cls,
            all_data_vars=[v for v in all_data_vars if v.name in _FILTER_VARS],
            **kwargs,
        )

    monkeypatch.setattr(
        EcmwfAifsSingleForecastVirtualRegionJob,
        "operational_update_jobs",
        classmethod(filtered_update_jobs),
    )
    # Narrow the window to the 06z init alone. The production 20h window also re-walks
    # the two prior cycles, costing four more .index downloads and ~120 more refs for
    # nothing this test doesn't get from one init; the window arithmetic itself is
    # covered by test_operational_update_jobs_single_polling_job in region_job_test.py.
    monkeypatch.setattr(
        EcmwfAifsSingleForecastVirtualRegionJob,
        "operational_update_window",
        pd.Timedelta("6h"),
    )

    dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    # The update window (6h before the 11:20 fire) ingests the 06z init.
    assert updated.init_time.values[-1] == np.datetime64("2025-03-01T06:00")
    new_cell = updated.sel(
        latitude=_LAT,
        longitude=_LON,
        init_time="2025-03-01T06:00",
        lead_time=pd.Timedelta("6h"),
    )
    t6 = float(new_cell["temperature_2m"].values)
    assert -60.0 < t6 < 60.0  # plausible Celsius

    # Sample the decode-health validator down to one lead time and one level: every
    # sampled chunk is a range request to s3://ecmwf-forecasts, which answers reads with
    # 503 SlowDown often enough to flake CI. All three variables are still decoded, and
    # the production sampling config is asserted in test_validators below.
    orig_validators = type(dataset).validators
    monkeypatch.setattr(
        type(dataset),
        "validators",
        lambda self: tuple(
            replace(validator, sampled_leads=1, sampled_levels=1)
            if isinstance(validator, validation.CheckVirtualDecodeHealth)
            else validator
            for validator in orig_validators(self)
        ),
    )
    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(
    dataset: EcmwfAifsSingleForecastVirtualDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))
    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    # Single-writer virtual update: no fan-out.
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert update_cron_job.pod_active_deadline < timedelta(hours=6)
    assert update_cron_job.suspend is False
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert validation_cron_job.suspend is False
    assert len(update_cron_job.secret_names) > 0


def test_validators(dataset: EcmwfAifsSingleForecastVirtualDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    (completeness,) = [
        v
        for v in validators
        if isinstance(v, validation.CheckVirtualManifestCompleteness)
    ]
    # The source publishes all 61 leads in a ~2 minute burst, so an ingested init is
    # never half-published and the default whole-position threshold holds.
    assert completeness.min_present_fraction == (1.0,)
    assert any(isinstance(v, validation.CheckVirtualDecodeHealth) for v in validators)


def test_current_data_validator_allows_7_hours(
    dataset: EcmwfAifsSingleForecastVirtualDataset,
) -> None:
    (current_data,) = [
        v for v in dataset.validators() if isinstance(v, validation.CheckCurrentData)
    ]
    # Validation fires at init+6h20m, so the threshold must exceed that.
    assert current_data.max_age == timedelta(hours=7)


def _resolved_split_size(
    split: icechunk.ManifestSplittingConfig, array_path: str
) -> int:
    # Mirror icechunk's first-to-last rule matching: a path_matches condition
    # matches by regex search; the AnyArray catch-all matches every array.
    for condition, dim_splits in split.split_sizes:
        regex = getattr(condition, "regex", None)
        if regex is None or re.search(regex, array_path):
            [(_dim_condition, size)] = dim_splits
            return size
    raise AssertionError(f"no split rule matched {array_path}")


def test_manifest_split_size_resolves_per_group(
    dataset: EcmwfAifsSingleForecastVirtualDataset,
) -> None:
    split = dataset.icechunk_virtual_config.manifest_split
    assert _resolved_split_size(split, "/pressure_level/temperature") == 200
    assert _resolved_split_size(split, "/temperature_2m") == 600


def test_virtual_container_matches_ref_prefix(
    dataset: EcmwfAifsSingleForecastVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://ecmwf-forecasts/"
