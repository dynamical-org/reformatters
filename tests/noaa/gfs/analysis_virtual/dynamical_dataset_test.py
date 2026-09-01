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
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.gfs.analysis_virtual.dynamical_dataset import (
    NoaaGfsAnalysisVirtualDataset,
)
from reformatters.noaa.gfs.analysis_virtual.region_job import (
    NoaaGfsAnalysisVirtualRegionJob,
)
from reformatters.noaa.models import NoaaDataVar
from tests.common.dynamical_dataset_test import assert_configured_validators

# A convective cell off the coast of Colombia: precipitating, at sea level so the
# 305 m above MSL fields are present, and inside the cloud fields.
_LATITUDE, _LONGITUDE = 343, 397
_FILTER_VARS = [
    "temperature_2m",  # pgrb2 root, instantaneous, K -> C filter
    "total_precipitation_surface",  # pgrb2 root, 6 hour accumulation bucket
    # A bare name selects that variable in every group, so this covers both the
    # pressure_level and height_above_mean_sea_level copies of temperature.
    "temperature",
]


def make_dataset(tmp_path: Path) -> NoaaGfsAnalysisVirtualDataset:
    return NoaaGfsAnalysisVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaGfsAnalysisVirtualDataset:
    return make_dataset(tmp_path)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2026-08-28T19:00"),
        filter_start=pd.Timestamp("2026-08-28T17:00"),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    cell = ds.isel(latitude=_LATITUDE, longitude=_LONGITUDE)
    assert (float(ds["latitude"][_LATITUDE]), float(ds["longitude"][_LONGITUDE])) == (
        4.25,
        -80.75,
    )
    # 18:00 is a synoptic hour, so the instantaneous fields come from f000 of the 18Z
    # cycle while the accumulation comes from f006 of the 12Z cycle.
    at_18 = cell.sel(time="2026-08-28T18:00")
    np.testing.assert_allclose(
        [
            at_18["temperature_2m"].item(),
            at_18["total_precipitation_surface"].item(),
            at_18["height_above_mean_sea_level/temperature"]
            .sel(height_above_mean_sea_level=305)
            .item(),
            at_18["pressure_level/temperature"].sel(pressure_level=500).item(),
            # 875 hPa is one of the 16 levels only pgrb2b carries.
            at_18["pressure_level/temperature"].sel(pressure_level=875).item(),
        ],
        [
            25.84406738281251,
            17.75,
            23.20923828125001,
            -3.4753906249999886,
            18.88984375000001,
        ],
    )
    np.testing.assert_allclose(
        cell.sel(time="2026-08-28T17:00")["temperature_2m"].item(), 26.188647460937545
    )
    # Four widely separated longitudes, each within the published materialized
    # noaa-gfs-analysis's rounding quantum of its value for the same message. This is
    # what pins the grid: GribberishCodec rolls the source's 0..360 longitudes onto our
    # -180..180 grid, and a wrong roll would move every value half a globe.
    np.testing.assert_allclose(
        [
            ds["temperature_2m"]
            .sel(time="2026-08-28T18:00", latitude=latitude, longitude=longitude)
            .item()
            for latitude, longitude in (
                (4.25, -80.75),
                (60.0, 100.0),
                (-33.0, 18.5),
                (35.0, 139.0),
            )
        ],
        [
            25.84406738281251,
            11.844067382812511,
            16.144067382812523,
            22.544067382812557,
        ],
    )

    # Each level must land in its own chunk rather than repeating one level's message:
    # temperature falls with height through the troposphere.
    profile = at_18["pressure_level/temperature"].sel(
        pressure_level=[1000, 875, 700, 500]
    )
    assert (profile.diff("pressure_level") < 0).all(), profile.values

    original_update_jobs = (
        NoaaGfsAnalysisVirtualRegionJob.operational_update_jobs.__func__
    )

    def filtered_update_jobs(
        cls: type[NoaaGfsAnalysisVirtualRegionJob],
        *,
        all_data_vars: Sequence[NoaaDataVar],
        **kwargs: Any,  # noqa: ANN401 - passthrough to the wrapped classmethod
    ) -> object:
        return original_update_jobs(
            cls,
            all_data_vars=[v for v in all_data_vars if v.name in _FILTER_VARS],
            **kwargs,
        )

    with monkeypatch.context() as update_monkeypatch:
        update_monkeypatch.setattr(
            pd.Timestamp,
            "now",
            classmethod(lambda *args, **kwargs: pd.Timestamp("2026-08-28T20:00")),
        )
        update_monkeypatch.setattr(
            NoaaGfsAnalysisVirtualRegionJob,
            "operational_update_jobs",
            classmethod(filtered_update_jobs),
        )
        dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    update_cell = updated.isel(latitude=_LATITUDE, longitude=_LONGITUDE).sel(
        time="2026-08-28T19:00"
    )
    np.testing.assert_allclose(
        [
            update_cell["temperature_2m"].item(),
            update_cell["total_precipitation_surface"].item(),
            update_cell["height_above_mean_sea_level/temperature"]
            .sel(height_above_mean_sea_level=305)
            .item(),
            update_cell["pressure_level/temperature"].sel(pressure_level=500).item(),
            update_cell["pressure_level/temperature"].sel(pressure_level=875).item(),
        ],
        [
            25.55000000000001,
            9.5625,
            23.10523437500001,
            -4.145390624999948,
            19.662753906250032,
        ],
    )
    # The update's window reaches back before the backfilled range and fills it in.
    np.testing.assert_allclose(
        updated.isel(latitude=_LATITUDE, longitude=_LONGITUDE)
        .sel(time="2026-08-28T13:00")["temperature_2m"]
        .item(),
        26.733032226562557,
    )

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(
    dataset: NoaaGfsAnalysisVirtualDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert update_cron_job.schedule == "50 3,9,15,21 * * *"
    assert update_cron_job.pod_active_deadline == timedelta(minutes=45)
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert validation_cron_job.schedule == "35 4,10,16,22 * * *"
    assert len(update_cron_job.secret_names) > 0
    assert update_cron_job.suspend
    assert validation_cron_job.suspend


def test_validators(dataset: NoaaGfsAnalysisVirtualDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    current_data = next(
        v for v in validators if isinstance(v, validation.CheckCurrentData)
    )
    assert current_data.max_delay == timedelta(hours=11)

    # The gate holds the frontier to a complete hour but can still release an earlier
    # incomplete one, so a whole 1.0 over every variable is what catches that.
    completeness = next(
        v
        for v in validators
        if isinstance(v, validation.CheckVirtualManifestCompleteness)
    )
    assert completeness.include_vars == "all"
    assert completeness.exclude_vars == ()
    assert completeness.min_present_fraction == (1.0,)

    decode_health = next(
        v for v in validators if isinstance(v, validation.CheckVirtualDecodeHealth)
    )
    assert (decode_health.positions, decode_health.max_positions) == (1, None)


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
    dataset: NoaaGfsAnalysisVirtualDataset,
) -> None:
    split = dataset.icechunk_virtual_config.manifest_split
    assert _resolved_split_size(split, "/pressure_level/temperature") == 512
    assert (
        _resolved_split_size(split, "/height_above_mean_sea_level/temperature") == 512
    )
    assert _resolved_split_size(split, "/temperature_2m") == 4096


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaGfsAnalysisVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-gfs-bdp-pds/"
