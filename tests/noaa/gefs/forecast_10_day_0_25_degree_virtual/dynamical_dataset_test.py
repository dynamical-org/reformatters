import itertools
import re
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.iterating import item
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.time_utils import whole_hours
from reformatters.noaa.gefs.forecast_10_day_0_25_degree_virtual.dynamical_dataset import (
    NoaaGefsForecast10Day025DegreeVirtualDataset,
)
from reformatters.noaa.gefs.forecast_10_day_0_25_degree_virtual.region_job import (
    NoaaGefsForecast10Day025DegreeVirtualRegionJob,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_INIT_TIME_FREQUENCY,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.gefs.virtual_region_job import (
    NoaaGefsForecastVirtualSourceFileCoord,
)
from tests.common.dynamical_dataset_test import (
    assert_configured_validators,
    stalled_cycles_before_alerting,
)

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
# One init is 81 lead times x 31 members of source files. Keep the ends of both axes
# (lead 0's structural gap, the 6 hour window at lead 6, the 240 hour end of the s file,
# the control and the last perturbed member) and drop the interior.
_TEST_LEAD_TIMES = (
    pd.Timedelta("0h"),
    pd.Timedelta("3h"),
    pd.Timedelta("6h"),
    pd.Timedelta("240h"),
)
_TEST_ENSEMBLE_MEMBERS = (0, 30)


def _fire_minutes(schedule: str) -> list[int]:
    """Minutes past midnight a `<minute> <hours> * * *` cron schedule fires at."""
    minute, hours, *_ = schedule.split()
    return [int(hour) * 60 + int(minute) for hour in hours.split(",")]


def make_dataset(tmp_path: Path) -> NoaaGefsForecast10Day025DegreeVirtualDataset:
    return NoaaGefsForecast10Day025DegreeVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaGefsForecast10Day025DegreeVirtualDataset:
    return make_dataset(tmp_path)


def _narrow_source_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real coord generation but read only the corner lead times and members."""
    original = (
        NoaaGefsForecast10Day025DegreeVirtualRegionJob.generate_source_file_coords
    )

    def narrowed(
        self: NoaaGefsForecast10Day025DegreeVirtualRegionJob,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaGefsVirtualDataVar],
    ) -> Sequence[NoaaGefsForecastVirtualSourceFileCoord]:
        return [
            coord
            for coord in original(self, processing_region_ds, data_var_group)
            if coord.lead_time in _TEST_LEAD_TIMES
            and coord.ensemble_member in _TEST_ENSEMBLE_MEMBERS
        ]

    monkeypatch.setattr(
        NoaaGefsForecast10Day025DegreeVirtualRegionJob,
        "generate_source_file_coords",
        narrowed,
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)
    _narrow_source_files(monkeypatch)

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

    cell = ds.isel(latitude=_LATITUDE, longitude=_LONGITUDE).sel(
        init_time="2024-06-01T00:00"
    )

    # Lead 0 is the analysis step: the accumulation has no window there and stays NaN,
    # while the instant variables carry the same values noaa-gefs-analysis-0-25-degree-
    # virtual reads from this very file.
    lead_0 = cell.sel(lead_time=pd.Timedelta("0h"), ensemble_member=0)
    np.testing.assert_allclose(
        [lead_0[name].item() for name in _FILTER_VARS],
        [22.70626953125003, 21.750000000000057, 0.0, np.nan, 101227.65000000001],
    )

    # Lead 3's accumulation window is 0-3 hours, lead 6's is 0-6, so the same forecast
    # accumulates more by lead 6. Both differ from the lead 3 and 6 values of any other
    # init, which is why the window comment is phrased in lead time.
    for lead, expected in (
        ("3h", [17.959492187500018, 21.01998046875002, 0.0, 0.13, 101426.13750000001]),
        ("6h", [16.147812500000043, 19.950000000000045, 0.0, 0.2, 101535.95625]),
        ("240h", [24.30976562500001, 22.789282226562534, 0.0, 0.0, 101352.375]),
    ):
        values = cell.sel(lead_time=pd.Timedelta(lead), ensemble_member=0)
        np.testing.assert_allclose(
            [values[name].item() for name in _FILTER_VARS], expected, err_msg=lead
        )

    # A perturbed member is a different forecast, read from a different source file.
    member_30 = cell.sel(lead_time=pd.Timedelta("3h"), ensemble_member=30)
    np.testing.assert_allclose(
        [member_30[name].item() for name in _FILTER_VARS],
        [
            18.278652343750025,
            21.000000000000057,
            0.0,
            0.30000000000000004,
            101446.49375000001,
        ],
    )

    # The source bitmaps soil and snow over open water; gribberish decodes those cells
    # to NaN, which is what the declared fill value means to a CF-aware reader.
    ocean = ds.isel(latitude=_OCEAN_LATITUDE, longitude=_OCEAN_LONGITUDE).sel(
        init_time="2024-06-01T00:00", lead_time=pd.Timedelta("3h"), ensemble_member=0
    )
    assert np.isnan(ocean["soil_temperature_0_10cm"].values)
    assert np.isnan(ocean["snow_water_equivalent_surface"].values)
    assert not np.isnan(ocean["temperature_2m"].values)

    original_update_jobs = (
        NoaaGefsForecast10Day025DegreeVirtualRegionJob.operational_update_jobs.__func__
    )

    def filtered_update_jobs(
        cls: type[NoaaGefsForecast10Day025DegreeVirtualRegionJob],
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
            NoaaGefsForecast10Day025DegreeVirtualRegionJob,
            "operational_update_jobs",
            classmethod(filtered_update_jobs),
        )
        dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert updated.get_index("init_time").max() == pd.Timestamp("2024-06-01T06:00")

    appended = updated.isel(latitude=_LATITUDE, longitude=_LONGITUDE).sel(
        init_time="2024-06-01T06:00", lead_time=pd.Timedelta("3h"), ensemble_member=0
    )
    np.testing.assert_allclose(
        [appended[name].item() for name in _FILTER_VARS],
        [14.74871093750005, 18.750000000000057, 0.0, 0.02, 101404.38125],
    )

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(
    dataset: NoaaGefsForecast10Day025DegreeVirtualDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    # f000 publishes ~init+3h47m and the last member's f240 ~init+5h37m, so the fire
    # leads the burst and the deadline covers its end with over half an hour to spare.
    assert update_cron_job.schedule == "45 3,9,15,21 * * *"
    assert update_cron_job.pod_active_deadline == timedelta(hours=2, minutes=30)
    # Virtual updates are single writer.
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert len(update_cron_job.secret_names) > 0

    # The update's fire plus its pod_active_deadline, plus 10 minutes of margin so the
    # validator never reads the store while the update is still committing.
    assert validation_cron_job.schedule == "25 6,12,18,0 * * *"
    margin = timedelta(minutes=10)
    day_minutes = 24 * 60
    assert sorted(_fire_minutes(validation_cron_job.schedule)) == sorted(
        (
            fire
            + int((update_cron_job.pod_active_deadline + margin).total_seconds() // 60)
        )
        % day_minutes
        for fire in _fire_minutes(update_cron_job.schedule)
    )
    # Without this the 6 hour default returns and a stuck validation overlaps its next fire.
    assert validation_cron_job.pod_active_deadline == timedelta(minutes=30)

    # Both stay suspended until the archive is backfilled.
    assert update_cron_job.suspend
    assert validation_cron_job.suspend


def test_operational_update_window_spans_three_update_fires(
    dataset: NoaaGefsForecast10Day025DegreeVirtualDataset,
) -> None:
    """Two consecutive failed or lost updates still self-heal: the span the next fire
    re-sweeps reaches back past both. Derived from the schedule rather than pinned as
    hours, so changing the cron cadence alone cannot silently shrink the recovery."""
    update_cron_job, _ = dataset.operational_kubernetes_resources("test-image-tag")
    fires = _fire_minutes(update_cron_job.schedule)
    intervals = {b - a for a, b in itertools.pairwise(fires)}
    assert len(intervals) == 1, fires
    assert (
        NoaaGefsForecast10Day025DegreeVirtualRegionJob.operational_update_window
        == 3 * pd.Timedelta(minutes=item(intervals))
    )


def test_cron_job_names_fit_the_kubernetes_limit(
    dataset: NoaaGefsForecast10Day025DegreeVirtualDataset,
) -> None:
    """The dataset id plus "-validate" is two characters over, so the names drop
    "-degree". Constructing the CronJobs at all is the check -- the field validator
    rejects a longer name -- but pin the abbreviation so it stays recognizable."""
    for cron_job in dataset.operational_kubernetes_resources("test-image-tag"):
        assert cron_job.name.startswith("noaa-gefs-forecast-10-day-0-25-virtual-")
        assert len(cron_job.name) <= 52
        assert cron_job.dataset_id == dataset.dataset_id


def test_validators(
    dataset: NoaaGefsForecast10Day025DegreeVirtualDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3

    _, validation_cron_job = dataset.operational_kubernetes_resources("test-image-tag")
    validation_fire = pd.Timestamp("2026-09-01") + pd.Timedelta(
        minutes=min(_fire_minutes(validation_cron_job.schedule))
    )
    # The cycle the update that this fire follows ingested.
    newest_init = (
        validation_fire.floor(f"{whole_hours(GEFS_INIT_TIME_FREQUENCY)}h")
        - GEFS_INIT_TIME_FREQUENCY
    )

    # The newest init is 6h25m old when validation fires and the update that ingested
    # it stopped writing 10 minutes earlier, so the first stalled cycle alerts.
    assert validation_fire - newest_init == pd.Timedelta("6h25m")
    current_data = next(
        v for v in validators if isinstance(v, validation.CheckCurrentData)
    )
    assert current_data.max_delay == timedelta(hours=6, minutes=20)
    assert (
        stalled_cycles_before_alerting(
            current_data.max_delay,
            validation_fire,
            newest_init,
            GEFS_INIT_TIME_FREQUENCY,
            monkeypatch,
        )
        == 1
    )

    completeness = next(
        v
        for v in validators
        if isinstance(v, validation.CheckVirtualManifestCompleteness)
    )
    assert completeness.include_vars == "all"
    assert completeness.exclude_vars == ()
    # Every init the store reached must be whole: the source finishes publishing before
    # validation fires, and an init the source has not started is skipped by the
    # append-dim extent clamp rather than checked here.
    assert completeness.min_present_fraction == (1.0,)

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


def test_manifest_split_holds_four_days_of_inits(
    dataset: NoaaGefsForecast10Day025DegreeVirtualDataset,
) -> None:
    """Every array holds one ref per (lead time, ensemble member) of every init in the
    split, so the split size fixes both the manifest's ref count and its bytes."""
    split = dataset.icechunk_virtual_config.manifest_split
    split_size = _resolved_split_size(split, "/temperature_2m")
    assert split_size == 16

    refs_per_manifest = split_size * 81 * 31
    assert refs_per_manifest == 40176
    # Above the 1000 refs icechunk needs before it compresses ref locations, and well
    # inside the 3 MiB a reader downloads to resolve any one chunk. 17.8 bytes/ref is
    # measured on this dataset's own manifests, not carried over from another.
    assert refs_per_manifest > 1000
    assert refs_per_manifest * 17.8 < 3 * 1024 * 1024


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaGefsForecast10Day025DegreeVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-gefs-pds/"
