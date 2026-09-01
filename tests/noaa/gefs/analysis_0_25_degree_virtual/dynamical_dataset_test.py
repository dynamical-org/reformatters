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
from reformatters.noaa.gefs.analysis_0_25_degree_virtual.dynamical_dataset import (
    NoaaGefsAnalysis025DegreeVirtualDataset,
)
from reformatters.noaa.gefs.analysis_0_25_degree_virtual.region_job import (
    NoaaGefsAnalysis025DegreeVirtualRegionJob,
)
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsVirtualDataVar
from tests.common.dynamical_dataset_test import assert_configured_validators

# 40N 100W, a land cell so the soil and snow bitmaps carry values there.
_LATITUDE, _LONGITUDE = 200, 320
# 0N 160W, open Pacific, where those same bitmaps are masked.
_OCEAN_LATITUDE, _OCEAN_LONGITUDE = 360, 80

_FILTER_VARS = [
    "temperature_2m",  # Kelvin source, Celsius filter
    "soil_temperature_0_10cm",  # Kelvin source GDAL mislabels as Celsius
    "snow_water_equivalent_surface",  # kg m-2 source scaled to metres, bitmapped
    "total_precipitation_surface",  # accumulation, absent at lead 0
    "pressure_reduced_to_mean_sea_level",  # unscaled
]


def make_dataset(tmp_path: Path) -> NoaaGefsAnalysis025DegreeVirtualDataset:
    return NoaaGefsAnalysis025DegreeVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaGefsAnalysis025DegreeVirtualDataset:
    return make_dataset(tmp_path)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-06-01T06:00"),
        filter_start=pd.Timestamp("2024-06-01T00:00"),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert float(ds["latitude"][_LATITUDE]) == 40.0
    assert float(ds["longitude"][_LONGITUDE]) == -100.0

    # On a cycle boundary the instant variables come from that cycle's lead 0 file and
    # the accumulation from the previous cycle's lead 6 file, so its window is 6 hours.
    cell = ds.isel(latitude=_LATITUDE, longitude=_LONGITUDE).sel(
        time="2024-06-01T00:00"
    )
    np.testing.assert_allclose(cell["temperature_2m"].values, 22.70626953125003)
    np.testing.assert_allclose(
        cell["soil_temperature_0_10cm"].values, 21.750000000000057
    )
    np.testing.assert_allclose(cell["snow_water_equivalent_surface"].values, 0.0)
    np.testing.assert_allclose(cell["total_precipitation_surface"].values, 1.83)
    np.testing.assert_allclose(
        cell["pressure_reduced_to_mean_sea_level"].values, 101227.65000000001
    )

    # Between cycles every variable comes from one lead 3 file, so the accumulation
    # window is 3 hours and its total is correspondingly smaller.
    between = ds.isel(latitude=_LATITUDE, longitude=_LONGITUDE).sel(
        time="2024-06-01T03:00"
    )
    np.testing.assert_allclose(between["temperature_2m"].values, 17.959492187500018)
    np.testing.assert_allclose(
        between["soil_temperature_0_10cm"].values, 21.01998046875002
    )
    np.testing.assert_allclose(between["snow_water_equivalent_surface"].values, 0.0)
    np.testing.assert_allclose(between["total_precipitation_surface"].values, 0.13)
    np.testing.assert_allclose(
        between["pressure_reduced_to_mean_sea_level"].values, 101426.13750000001
    )

    # The source bitmaps soil and snow over open water; gribberish decodes those cells
    # to NaN, which is what the declared fill value means to a CF-aware reader.
    ocean = ds.isel(latitude=_OCEAN_LATITUDE, longitude=_OCEAN_LONGITUDE).sel(
        time="2024-06-01T00:00"
    )
    assert np.isnan(ocean["soil_temperature_0_10cm"].values)
    assert np.isnan(ocean["snow_water_equivalent_surface"].values)
    assert not np.isnan(ocean["temperature_2m"].values)

    original_update_jobs = (
        NoaaGefsAnalysis025DegreeVirtualRegionJob.operational_update_jobs.__func__
    )

    def filtered_update_jobs(
        cls: type[NoaaGefsAnalysis025DegreeVirtualRegionJob],
        *,
        all_data_vars: Sequence[NoaaGefsVirtualDataVar],
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
            classmethod(lambda *args, **kwargs: pd.Timestamp("2024-06-01T12:00")),
        )
        update_monkeypatch.setattr(
            NoaaGefsAnalysis025DegreeVirtualRegionJob,
            "operational_update_jobs",
            classmethod(filtered_update_jobs),
        )
        dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert updated.get_index("time").max() == pd.Timestamp("2024-06-01T09:00")

    for time, expected in (
        (
            "2024-06-01T06:00",
            [16.51618652343751, 19.750000000000057, 0.0, 0.2, 101524.01875],
        ),
        (
            "2024-06-01T09:00",
            [14.74871093750005, 18.750000000000057, 0.0, 0.02, 101404.38125],
        ),
    ):
        cell = updated.isel(latitude=_LATITUDE, longitude=_LONGITUDE).sel(time=time)
        np.testing.assert_allclose(
            [cell[name].item() for name in _FILTER_VARS], expected, err_msg=time
        )

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(
    dataset: NoaaGefsAnalysis025DegreeVirtualDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.schedule == "51 3,9,15,21 * * *"
    assert update_cron_job.pod_active_deadline == timedelta(minutes=30)
    # Virtual updates are single writer.
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert len(update_cron_job.secret_names) > 0

    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    # The update's fire plus its pod_active_deadline.
    assert validation_cron_job.schedule == "21 4,10,16,22 * * *"

    # Both stay suspended until the archive is backfilled.
    assert update_cron_job.suspend
    assert validation_cron_job.suspend


def test_validators(dataset: NoaaGefsAnalysis025DegreeVirtualDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3

    current_data = next(
        v for v in validators if isinstance(v, validation.CheckCurrentData)
    )
    assert current_data.max_delay == timedelta(hours=13)

    # discover_available extends time only to a step holding every file it needs, so
    # one instance covering every variable at a whole 1.0 is the right check: no
    # ingested position is ever partially published.
    completeness = next(
        v
        for v in validators
        if isinstance(v, validation.CheckVirtualManifestCompleteness)
    )
    assert completeness.include_vars == "all"
    assert completeness.exclude_vars == ()
    assert completeness.min_present_fraction == (1.0,)

    # Every variable carries real values somewhere on the globe at every step.
    decode_health = next(
        v for v in validators if isinstance(v, validation.CheckVirtualDecodeHealth)
    )
    assert decode_health.positions == 1
    assert decode_health.allow_all_nan_vars == ()


def _resolved_split_size(
    split: icechunk.ManifestSplittingConfig, array_path: str
) -> int:
    for condition, dim_splits in split.split_sizes:
        regex = getattr(condition, "regex", None)
        if regex is None or re.search(regex, array_path):
            [(_dim_condition, size)] = dim_splits
            return size
    raise AssertionError(f"no split rule matched {array_path}")


def test_manifest_split_holds_four_years_of_three_hourly_refs(
    dataset: NoaaGefsAnalysis025DegreeVirtualDataset,
) -> None:
    """Each array holds one ref per step, so a split is also its manifest's ref count."""
    split = dataset.icechunk_virtual_config.manifest_split
    split_size = _resolved_split_size(split, "/temperature_2m")
    assert split_size == 4 * 365 * 8

    # Above the 1000 refs icechunk needs before it compresses ref locations, and well
    # inside the 3 MiB a reader downloads to resolve any one chunk.
    assert split_size > 1000
    assert split_size * 16.4 < 3 * 1024 * 1024


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaGefsAnalysis025DegreeVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-gefs-pds/"
