import re
from collections.abc import Sequence
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pandas as pd
import pytest

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.analysis_virtual.dynamical_dataset import (
    NoaaHrrrAnalysisVirtualDataset,
)
from reformatters.noaa.hrrr.analysis_virtual.region_job import (
    NoaaHrrrAnalysisVirtualRegionJob,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from tests.common.dynamical_dataset_test import assert_configured_validators

_Y, _X = 635, 1062
_TIME = "2024-06-01T07:00"
_FILTER_VARS = [
    "temperature_2m",
    "wind_u_10m",
    "total_precipitation_surface",
    "temperature",
]


def make_dataset(tmp_path: Path) -> NoaaHrrrAnalysisVirtualDataset:
    return NoaaHrrrAnalysisVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaHrrrAnalysisVirtualDataset:
    return make_dataset(tmp_path)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-06-01T08:00"),
        filter_start=pd.Timestamp("2024-06-01T06:00"),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    cell = ds.isel(y=_Y, x=_X).sel(time=_TIME)
    # Instant vars come from f00 of the same hour's cycle; the hourly precipitation
    # accumulation comes from f01 of the prior hour's cycle.
    np.testing.assert_allclose(cell["temperature_2m"].values, 19.862695312500023)
    np.testing.assert_allclose(cell["wind_u_10m"].values, -1.5750274658203125)
    np.testing.assert_allclose(cell["total_precipitation_surface"].values, 0.0)
    assert (ds["total_precipitation_surface"].sel(time=_TIME).values > 0).any()
    np.testing.assert_allclose(
        cell["pressure_level/temperature"].sel(pressure_level=500).values,
        -11.205221557617165,
    )
    np.testing.assert_allclose(
        cell["model_level/temperature"].sel(model_level=1).values,
        19.547998046875023,
    )
    # Each level must land in its own chunk rather than repeating one level's
    # message: temperature falls with height through the troposphere.
    profile = cell["pressure_level/temperature"].sel(pressure_level=[1000, 700, 500])
    assert (profile.diff("pressure_level") < 0).all(), profile.values

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2024-06-01T10:00")),
    )
    original_update_jobs = (
        NoaaHrrrAnalysisVirtualRegionJob.operational_update_jobs.__func__
    )

    def filtered_update_jobs(
        cls: type[NoaaHrrrAnalysisVirtualRegionJob],
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
        NoaaHrrrAnalysisVirtualRegionJob,
        "operational_update_jobs",
        classmethod(filtered_update_jobs),
    )

    dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    update_cell = updated.isel(y=_Y, x=_X).sel(time="2024-06-01T09:00")
    actual = [
        update_cell["temperature_2m"].item(),
        update_cell["wind_u_10m"].item(),
        update_cell["total_precipitation_surface"].item(),
        update_cell["pressure_level/temperature"].sel(pressure_level=500).item(),
        update_cell["model_level/temperature"].sel(model_level=1).item(),
    ]
    np.testing.assert_allclose(
        actual,
        [
            20.107934570312523,
            -1.6848134994506836,
            0.0,
            -11.095617675781227,
            19.839074707031273,
        ],
    )

    assert_configured_validators(dataset)


@pytest.mark.slow
def test_backfill_earliest_hrrr_v1_era(tmp_path: Path) -> None:
    """The archive's first hour decodes, including the v1-only PRMSL spelling.

    HRRR v1 (through the 2016-08-23 cycles) writes mean sea level pressure as PRMSL
    rather than MSLMA, and its GRIB messages must decode through GribberishCodec just
    like modern ones.
    """
    dataset = make_dataset(tmp_path)
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2014-10-01T02:00"),
        filter_start=pd.Timestamp("2014-10-01T00:00"),
        filter_variable_names=[
            "temperature_2m",
            "pressure_reduced_to_mean_sea_level",
            "temperature",
        ],
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    cell = ds.isel(y=_Y, x=_X).sel(time="2014-10-01T00:00")
    np.testing.assert_allclose(cell["temperature_2m"].values, 24.221795654296898)
    np.testing.assert_allclose(
        cell["pressure_reduced_to_mean_sea_level"].values, 101059.0
    )
    np.testing.assert_allclose(
        cell["pressure_level/temperature"].sel(pressure_level=500).values,
        -6.440039062499977,
    )
    np.testing.assert_allclose(
        cell["model_level/temperature"].sel(model_level=1).values,
        25.420648193359398,
    )


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrAnalysisVirtualDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert update_cron_job.schedule == "50 * * * *"
    assert update_cron_job.pod_active_deadline == timedelta(minutes=30)
    assert update_cron_job.cpu == "4"
    assert update_cron_job.suspend
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert validation_cron_job.schedule == "20 * * * *"
    assert validation_cron_job.suspend
    assert len(update_cron_job.secret_names) > 0


def test_validators(dataset: NoaaHrrrAnalysisVirtualDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 4
    (current_data,) = [
        validator
        for validator in validators
        if isinstance(validator, partial)
        and validator.func is validation.check_analysis_current_data
    ]
    assert current_data.keywords == {"max_expected_delay": timedelta(hours=2)}

    # One completeness instance per source-file publication schedule, each held to a
    # whole 1.0 from the position where its variables are expected to be present.
    completeness = [
        validator
        for validator in validators
        if isinstance(validator, validation.CheckVirtualManifestCompleteness)
    ]
    f01_check = next(c for c in completeness if c.include_vars == "all")
    f00_check = next(c for c in completeness if c.include_vars != "all")
    hour_0_paths = {
        var.path for var in dataset.template_config.data_vars if var.has_hour_0_values()
    }
    other_paths = {
        var.path
        for var in dataset.template_config.data_vars
        if not var.has_hour_0_values()
    }
    assert hour_0_paths, "the f00-sourced partition must be non-empty"
    assert other_paths, "the f01-sourced partition must be non-empty"

    assert set(f01_check.exclude_vars) == hour_0_paths
    assert f01_check.include_vars == "all"
    assert f01_check.min_present_fraction == (1.0,)

    assert set(f00_check.include_vars) == hour_0_paths
    assert f00_check.min_present_fraction == (0.0, 1.0)

    decode_health = next(
        validator
        for validator in validators
        if isinstance(validator, validation.CheckVirtualDecodeHealth)
    )
    assert (decode_health.positions, decode_health.max_positions) == ("latest", None)


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
    dataset: NoaaHrrrAnalysisVirtualDataset,
) -> None:
    split = dataset.icechunk_virtual_config.manifest_split
    assert _resolved_split_size(split, "/pressure_level/temperature") == 4500
    assert _resolved_split_size(split, "/model_level/temperature") == 4000
    assert _resolved_split_size(split, "/temperature_2m") == 30000


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaHrrrAnalysisVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-hrrr-bdp-pds/"
