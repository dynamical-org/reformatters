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
from reformatters.noaa.gfs.forecast_virtual.dynamical_dataset import (
    NoaaGfsForecastVirtualDataset,
)
from reformatters.noaa.gfs.forecast_virtual.region_job import (
    NoaaGfsForecastVirtualRegionJob,
)
from reformatters.noaa.gfs.forecast_virtual.template_config import (
    NoaaGfsForecastVirtualTemplateConfig,
)
from reformatters.noaa.models import NoaaDataVar
from tests.common.dynamical_dataset_test import assert_configured_validators

# A convective cell off the coast of Colombia: precipitating, at sea level so the
# 305 m above MSL fields are present, and inside the cloud fields.
_LATITUDE, _LONGITUDE = 343, 397
_INIT = "2026-08-28T12:00"
# f000 (no precipitation message), f003 (bucket and running total are one message),
# f009 (they separate) and f123 (the running total's non-day window form).
_LEAD_INDEX = [0, 3, 9, 121]
_FILTER_VARS = [
    "temperature_2m",  # pgrb2 root, instantaneous, K -> C filter
    "total_precipitation_surface",  # pgrb2 root, 6 hour accumulation bucket
    "total_precipitation_run_total_surface",  # the same element, since initialization
    "snow_water_equivalent_surface",  # pgrb2 root, kg m-2 -> m lwe filter
    # A bare name selects the variable in every group, covering the pressure_level and
    # height_above_mean_sea_level copies of temperature.
    "temperature",
]


def make_dataset(tmp_path: Path) -> NoaaGfsForecastVirtualDataset:
    return NoaaGfsForecastVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaGfsForecastVirtualDataset:
    return make_dataset(tmp_path)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)

    # Trim to four lead times to limit work. Chunk and shard geometry is untouched: a
    # virtual chunk is one whole GRIB message and cannot be reshaped.
    original_get_template = NoaaGfsForecastVirtualTemplateConfig.get_template
    monkeypatch.setattr(
        NoaaGfsForecastVirtualTemplateConfig,
        "get_template",
        lambda self, end_time: original_get_template(self, end_time).isel(
            lead_time=_LEAD_INDEX
        ),
    )

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2026-08-28T13:00"),
        filter_start=pd.Timestamp(_INIT),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert ds["init_time"].values[-1] == np.datetime64(_INIT)
    assert (float(ds["latitude"][_LATITUDE]), float(ds["longitude"][_LONGITUDE])) == (
        4.25,
        -80.75,
    )
    cell = ds.isel(latitude=_LATITUDE, longitude=_LONGITUDE).sel(init_time=_INIT)

    def at(lead: str) -> list[float]:
        step = cell.sel(lead_time=pd.Timedelta(lead))
        return [
            step["temperature_2m"].item(),
            step["total_precipitation_surface"].item(),
            step["total_precipitation_run_total_surface"].item(),
            step["height_above_mean_sea_level/temperature"]
            .sel(height_above_mean_sea_level=305)
            .item(),
            step["pressure_level/temperature"].sel(pressure_level=500).item(),
            # 875 hPa is one of the 16 levels only pgrb2b carries.
            step["pressure_level/temperature"].sel(pressure_level=875).item(),
        ]

    # GFS publishes no accumulation message at f000, so both precipitation arrays are
    # empty there while the instantaneous fields are filled.
    f000 = at("0h")
    assert np.isnan(f000[1])
    assert np.isnan(f000[2])
    np.testing.assert_allclose(
        [f000[0], *f000[3:]],
        [26.808813476562534, 24.23625000000004, -4.179667968749982, 19.52091796875004],
    )
    # At f003 the bucket and the running total are one index line, so both arrays hold
    # the same value; by f009 the running total has outgrown the 6 hour bucket.
    np.testing.assert_allclose(
        at("3h"),
        [
            26.410205078125045,
            8.5625,
            8.5625,
            23.74503906250004,
            -4.046406249999961,
            19.43576171875003,
        ],
    )
    np.testing.assert_allclose(
        at("9h"),
        [
            26.264453125000045,
            5.5625,
            23.25,
            23.293027343750055,
            -3.717031249999991,
            19.77873046875004,
        ],
    )
    np.testing.assert_allclose(
        at("123h"),
        [
            25.650012207031295,
            7.5625,
            176.1875,
            23.363789062500018,
            -4.905859374999977,
            18.511367187500014,
        ],
    )

    # The kg m-2 -> m filter, on the ice sheet where a missing divide-by-1000 would
    # read as hundreds of metres of snow rather than tenths.
    greenland = ds["snow_water_equivalent_surface"].sel(
        init_time=_INIT,
        lead_time=pd.Timedelta("9h"),
        latitude=72.0,
        longitude=-40.0,
    )
    np.testing.assert_allclose(greenland.item(), 0.143832)

    # Four widely separated longitudes. This is what pins the grid: GribberishCodec
    # rolls the source's 0..360 longitudes onto our -180..180 grid, and a wrong roll
    # would move every value half a globe.
    np.testing.assert_allclose(
        [
            ds["temperature_2m"]
            .sel(
                init_time=_INIT,
                lead_time=pd.Timedelta(0),
                latitude=latitude,
                longitude=longitude,
            )
            .item()
            for latitude, longitude in (
                (4.25, -80.75),
                (60.0, 100.0),
                (-33.0, 18.5),
                (35.0, 139.0),
            )
        ],
        [
            26.808813476562534,
            15.508813476562523,
            32.00881347656252,
            24.70881347656251,
        ],
    )

    # Each level must land in its own chunk rather than repeating one level's message:
    # temperature falls with height through the troposphere.
    profile = cell.sel(lead_time=pd.Timedelta(0))["pressure_level/temperature"].sel(
        pressure_level=[1000, 875, 700, 500]
    )
    assert (profile.diff("pressure_level") < 0).all(), profile.values

    original_update_jobs = (
        NoaaGfsForecastVirtualRegionJob.operational_update_jobs.__func__
    )

    def filtered_update_jobs(
        cls: type[NoaaGfsForecastVirtualRegionJob],
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
            NoaaGfsForecastVirtualRegionJob,
            "operational_update_jobs",
            classmethod(filtered_update_jobs),
        )
        dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert updated["init_time"].values[-1] == np.datetime64("2026-08-28T18:00")
    update_cell = updated.isel(latitude=_LATITUDE, longitude=_LONGITUDE).sel(
        init_time="2026-08-28T18:00", lead_time=pd.Timedelta("9h")
    )
    np.testing.assert_allclose(
        [
            update_cell["temperature_2m"].item(),
            update_cell["total_precipitation_surface"].item(),
            update_cell["total_precipitation_run_total_surface"].item(),
            update_cell["height_above_mean_sea_level/temperature"]
            .sel(height_above_mean_sea_level=305)
            .item(),
            update_cell["pressure_level/temperature"].sel(pressure_level=500).item(),
        ],
        [
            27.049316406250057,
            0.75,
            14.8125,
            24.322050781250027,
            -4.458593749999977,
        ],
    )
    # The 18 hour window reaches back past the backfilled init and fills the one before.
    np.testing.assert_allclose(
        updated.isel(latitude=_LATITUDE, longitude=_LONGITUDE)
        .sel(init_time="2026-08-28T06:00", lead_time=pd.Timedelta("9h"))[
            "total_precipitation_run_total_surface"
        ]
        .item(),
        15.9375,
    )

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(
    dataset: NoaaGfsForecastVirtualDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert update_cron_job.schedule == "29 3,9,15,21 * * *"
    assert update_cron_job.pod_active_deadline == timedelta(hours=2, minutes=30)
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert validation_cron_job.schedule == "59 5,11,17,23 * * *"
    assert len(update_cron_job.secret_names) > 0
    assert update_cron_job.suspend
    assert validation_cron_job.suspend


def test_validators(dataset: NoaaGfsForecastVirtualDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    current_data = next(
        v for v in validators if isinstance(v, validation.CheckCurrentData)
    )
    assert current_data.max_delay == timedelta(hours=6, minutes=30)

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


def test_manifest_split_sizes_stay_inside_the_reader_budget(
    dataset: NoaaGfsForecastVirtualDataset,
) -> None:
    """An init contributes 209 refs to a root array and 209 x 57 to a pressure-level
    one, so the split has to stay inside the budget a reader downloads to resolve one
    chunk."""
    split = dataset.icechunk_virtual_config.manifest_split
    root_split = _resolved_split_size(split, "/temperature_2m")
    group_split = _resolved_split_size(split, "/pressure_level/temperature")
    assert (root_split, group_split) == (128, 16)

    leads = len(dataset.template_config.dimension_coordinates()["lead_time"])
    levels = len(dataset.template_config.dimension_coordinates()["pressure_level"])
    # Measured GFS manifest cost, bytes per chunk reference.
    mebibyte = 2**20
    assert root_split * leads * 13.8 / mebibyte < 3.0
    assert 0.0 < group_split * leads * levels * 11.0 / mebibyte < 8.0
    # Both stay above the ~1000 refs a manifest needs to compress well.
    assert root_split * leads > 1000
    assert group_split * leads * levels > 1000


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaGfsForecastVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-gfs-bdp-pds/"
