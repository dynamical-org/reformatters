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
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.time_utils import whole_hours
from reformatters.noaa.gefs.forecast_35_day_0_5_degree_virtual.dynamical_dataset import (
    NoaaGefsForecast35Day05DegreeVirtualDataset,
)
from reformatters.noaa.gefs.forecast_35_day_0_5_degree_virtual.region_job import (
    NoaaGefsForecast35Day05DegreeVirtualRegionJob,
)
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsVirtualDataVar
from reformatters.noaa.gefs.virtual_region_job import (
    NoaaGefsForecastVirtualSourceFileCoord,
)
from tests.common.dynamical_dataset_test import (
    assert_configured_validators,
    stalled_cycles_before_alerting,
)

_INIT_TIME_FREQUENCY = pd.Timedelta("24h")

# 40N 100W, a land cell so the soil and snow bitmaps carry values there.
_LATITUDE, _LONGITUDE = 100, 160
# 0N 160W, open Pacific, where those same bitmaps are masked.
_OCEAN_LATITUDE, _OCEAN_LONGITUDE = 180, 40
# 72N 40W, the Greenland ice sheet, wholly snow covered in June.
_SNOW_LATITUDE, _SNOW_LONGITUDE = 36, 280

# One variable of every kind this dataset builds: each source product, each vertical
# group, each decoding filter, and the windowed and lead-0-absent shapes.
_FILTER_VARS = [
    "temperature_2m",  # a file, Kelvin source, Celsius filter
    "total_precipitation_surface",  # a file, accumulation, absent at lead 0
    "visibility_surface",  # b file, unscaled
    "average_snow_area_fraction_surface",  # b file, percent scaled to a fraction
    "pressure_level/temperature",  # levels split between the a and b files
    "pressure_level/specific_humidity",  # b file only
    "model_level/temperature",  # b file only
    "height_above_mean_sea_level/wind_u",  # b file only
]
# Two levels per group: one each side of the a/b split for pressure, and the ends of
# the model and height axes.
_TEST_LEVELS = {
    "pressure_level": (850.0, 500.0),
    "model_level": (1, 4),
    "height_above_mean_sea_level": (305.0, 4572.0),
}
# One init is 181 lead times x 31 members x 2 products of source files. Keep the ends of
# both axes and every structural boundary along the lead axis (lead 0's gap, the 6 hour
# window at lead 6, the 240 hour step up in lead spacing, the 384 hour lead where every
# other cycle stops, the 390 hour lead where only 00z continues, and the 840 hour end)
# and drop the interior.
_TEST_LEAD_TIMES = (
    pd.Timedelta("0h"),
    pd.Timedelta("3h"),
    pd.Timedelta("6h"),
    pd.Timedelta("240h"),
    pd.Timedelta("246h"),
    pd.Timedelta("384h"),
    pd.Timedelta("390h"),
    pd.Timedelta("840h"),
)
_TEST_ENSEMBLE_MEMBERS = (0, 30)

# Every filtered variable, in _cell_values order, at 40N 100W.
_LEAD_0_EXPECTED = [22.69966796875002, np.nan, 24100.0, np.nan, 14.898339843750023, -12.369287109374966, 0.010082061462402345, 0.001004, 22.437011718750057, 20.682226562500034, np.nan, 14.1606884765625]  # fmt: skip
# fmt: off
_EXPECTED_BY_LEAD = {
    "3h": [17.95281250000005, 0.15, 24100.0, 0.0, 14.443408203125045, -12.860595703124943, 0.00995024296283722, 0.0008001989476680755, 18.794277343750025, 19.318652343750045, np.nan, 12.2342431640625],
    "6h": [16.147812500000043, 0.2, 24100.0, 0.0, 14.380444335937511, -13.761791992187455, 0.009562717080116273, 0.001320283229827881, 16.361660156250025, 16.20261718750004, np.nan, 7.46295166015625],
    "240h": [24.30976562500001, 0.0, 24100.0, 0.0, 16.095922851562534, -7.9500244140624545, 0.007299999995231629, 0.001937, 23.692871093750057, 21.937167968750032, np.nan, 3.2834423828125],
    "246h": [16.75788574218751, 0.0, 24100.0, 0.0, 14.933789062500011, -8.549999999999955, 0.007365050048828125, 0.0019446008300781248, 17.650000000000034, 18.750000000000057, np.nan, 3.7874462890625002],
    "384h": [35.78531250000003, 0.0, 24100.0, 0.0, 29.450000000000045, -7.958032226562466, 0.006689999995231629, 0.0025114374389648437, 35.80275390625002, 34.39171875000005, np.nan, -0.85630126953125],
    "390h": [23.908666992187534, 0.0, 24100.0, 0.0, 24.279150390625034, -8.837207031249989, 0.006899999990463258, 0.002508, 25.039453125000023, 25.73560546875001, np.nan, -2.6495874023437502],
    "840h": [36.406103515625034, 0.2, 24100.0, 0.0, 29.650000000000034, -3.3704345703124545, 0.010690000000000002, 0.003983999999761581, 35.25000000000006, 33.269946289062545, np.nan, 4.6202294921875],
}
# fmt: on
_MEMBER_30_EXPECTED = [
    18.26951171875004,
    0.30000000000000004,
    24100.0,
    0.0,
    13.743383789062534,
    -12.594702148437477,
    0.009816732581853867,
    0.0007689999999999999,
    18.721035156250025,
    18.80857421875004,
    np.nan,
    9.45343017578125,
]  # fmt: skip
_APPENDED_EXPECTED = [18.274511718750034, 1.82, 19300.0, 0.0, 16.356567382812557, -12.649999999999977, 0.010330000977516175, 0.0022130485686063764, 18.671992187500052, 18.90617187500004, np.nan, 7.5866113281250005]  # fmt: skip


def _fire_minutes(schedule: str) -> list[int]:
    """Minutes past midnight a `<minute> <hours> * * *` cron schedule fires at."""
    minute, hours, *_ = schedule.split()
    return [int(hour) * 60 + int(minute) for hour in hours.split(",")]


def make_dataset(tmp_path: Path) -> NoaaGefsForecast35Day05DegreeVirtualDataset:
    return NoaaGefsForecast35Day05DegreeVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaGefsForecast35Day05DegreeVirtualDataset:
    return make_dataset(tmp_path)


def _narrow_source_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real coord generation but read only the corner lead times and members."""
    original = NoaaGefsForecast35Day05DegreeVirtualRegionJob.generate_source_file_coords

    def narrowed(
        self: NoaaGefsForecast35Day05DegreeVirtualRegionJob,
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
        NoaaGefsForecast35Day05DegreeVirtualRegionJob,
        "generate_source_file_coords",
        narrowed,
    )


def _cell_values(cell: xr.Dataset) -> list[float]:
    """Every filtered variable at one (init, lead, member, latitude, longitude),
    a vertical-group variable once per test level."""
    values = []
    for path in _FILTER_VARS:
        group, _, _name = path.rpartition("/")
        if not group:
            values.append(cell[path].item())
            continue
        values.extend(
            cell[path].sel({group: level}).item() for level in _TEST_LEVELS[group]
        )
    return values


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)
    _narrow_source_files(monkeypatch)

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-06-02T00:00"),
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
    # while the instant variables carry the values the source's own analysis file holds.
    lead_0 = cell.sel(lead_time=pd.Timedelta("0h"), ensemble_member=0)
    np.testing.assert_allclose(_cell_values(lead_0), _LEAD_0_EXPECTED)

    # Lead 3's accumulation window is 0-3 hours, lead 6's is 0-6, so the same forecast
    # accumulates more by lead 6. The 246 hour lead is the first on the coarse 6 hourly
    # part of the axis; 390 and 840 hours are in the extension only 00z publishes.
    for lead, expected in _EXPECTED_BY_LEAD.items():
        values = cell.sel(lead_time=pd.Timedelta(lead), ensemble_member=0)
        np.testing.assert_allclose(_cell_values(values), expected, err_msg=lead)

    # A perturbed member is a different forecast, read from a different source file.
    member_30 = cell.sel(lead_time=pd.Timedelta("3h"), ensemble_member=30)
    np.testing.assert_allclose(_cell_values(member_30), _MEMBER_30_EXPECTED)

    # A lead 0 NaN is structural for the two variables the source omits there, not a
    # gap: the accumulation has no window and snow cover a zero length averaging one.
    assert np.isnan(lead_0["total_precipitation_surface"].values)
    assert np.isnan(lead_0["average_snow_area_fraction_surface"].values)
    # 305 m above mean sea level is below the ground at this cell, where the terrain is
    # near 1000 m, so the lowest height level is NaN while the highest carries wind, at
    # the far end of the extension as much as at lead 0.
    for lead in (pd.Timedelta("0h"), pd.Timedelta("840h")):
        at_lead = cell.sel(lead_time=lead, ensemble_member=0)
        assert np.isnan(at_lead["height_above_mean_sea_level/wind_u"].sel(height_above_mean_sea_level=305.0).values)  # fmt: skip

    # The source bitmaps snow cover over open water; gribberish decodes those cells to
    # NaN, which is what the declared fill value means to a CF-aware reader.
    at_lead_3 = ds.sel(
        init_time="2024-06-01T00:00", lead_time=pd.Timedelta("3h"), ensemble_member=0
    )
    ocean = at_lead_3.isel(latitude=_OCEAN_LATITUDE, longitude=_OCEAN_LONGITUDE)
    assert np.isnan(ocean["average_snow_area_fraction_surface"].values)
    assert not np.isnan(ocean["temperature_2m"].values)

    # The source publishes snow cover as a percentage; the declared filter divides it to
    # the fraction surface_snow_area_fraction means. A whole ice sheet cell reads 1, not
    # 100, and no cell anywhere exceeds 1.
    snow = at_lead_3["average_snow_area_fraction_surface"]
    assert float(snow.isel(latitude=_SNOW_LATITUDE, longitude=_SNOW_LONGITUDE)) == 1.0
    assert float(snow.max()) == 1.0

    # Visibility is clipped at the ceiling the field encodes, which its comment names.
    visibility = at_lead_3["visibility_surface"]
    assert float(visibility.max()) == 24100.0
    assert float((visibility == 24100.0).mean()) > 0.5

    original_update_jobs = (
        NoaaGefsForecast35Day05DegreeVirtualRegionJob.operational_update_jobs.__func__
    )

    def filtered_update_jobs(
        cls: type[NoaaGefsForecast35Day05DegreeVirtualRegionJob],
        *,
        all_data_vars: Sequence[NoaaGefsVirtualDataVar],
        **kwargs: Any,  # noqa: ANN401 - passthrough to the wrapped classmethod
    ) -> object:
        return original_update_jobs(
            cls,
            all_data_vars=[v for v in all_data_vars if v.path in _FILTER_VARS],
            **kwargs,
        )

    with monkeypatch.context() as update_monkeypatch:
        update_monkeypatch.setattr(
            pd.Timestamp,
            "now",
            classmethod(lambda *args, **kwargs: pd.Timestamp("2024-06-02T12:00")),
        )
        update_monkeypatch.setattr(
            NoaaGefsForecast35Day05DegreeVirtualRegionJob,
            "operational_update_jobs",
            classmethod(filtered_update_jobs),
        )
        dataset.update("test-update")

    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert updated.get_index("init_time").max() == pd.Timestamp("2024-06-02T00:00")

    appended = updated.isel(latitude=_LATITUDE, longitude=_LONGITUDE).sel(
        init_time="2024-06-02T00:00", lead_time=pd.Timedelta("3h"), ensemble_member=0
    )
    np.testing.assert_allclose(_cell_values(appended), _APPENDED_EXPECTED)

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(
    dataset: NoaaGefsForecast35Day05DegreeVirtualDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    # The first file lands ~init+3h46m and the lead times through 384 hours finish
    # ~init+6h43m, so the fire leads that stage and the deadline covers its end with
    # three hours to spare. It also reaches past ~init+28h05m of the cycle a day older,
    # where that cycle's 840 hour extension finishes.
    assert update_cron_job.schedule == "45 3 * * *"
    assert update_cron_job.pod_active_deadline == timedelta(hours=6)
    fire_to_next_fire = timedelta(hours=24)
    assert update_cron_job.pod_active_deadline < fire_to_next_fire
    (fire,) = _fire_minutes(update_cron_job.schedule)
    previous_cycle_complete = timedelta(hours=28, minutes=5) - timedelta(days=1)
    assert (
        timedelta(minutes=fire) + update_cron_job.pod_active_deadline
        > previous_cycle_complete
    )
    # Virtual updates are single writer.
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert len(update_cron_job.secret_names) > 0

    # The update's fire plus its pod_active_deadline, plus 10 minutes of margin so the
    # validator never reads the store while the update is still committing.
    assert validation_cron_job.schedule == "55 9 * * *"
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
    dataset: NoaaGefsForecast35Day05DegreeVirtualDataset,
) -> None:
    """Two consecutive failed or lost updates still self-heal: the span the next fire
    re-sweeps reaches back past both. A cycle's own extension also publishes past the
    next fire, so even with none missed the window has to reach a second init."""
    update_cron_job, _ = dataset.operational_kubernetes_resources("test-image-tag")
    (fire,) = _fire_minutes(update_cron_job.schedule)
    window = NoaaGefsForecast35Day05DegreeVirtualRegionJob.operational_update_window
    assert window == 3 * _INIT_TIME_FREQUENCY
    assert window > pd.Timedelta(minutes=fire) + pd.Timedelta("28h5m")


def test_cron_job_names_fit_the_kubernetes_limit(
    dataset: NoaaGefsForecast35Day05DegreeVirtualDataset,
) -> None:
    """The dataset id plus "-validate" exceeds the 52 character limit, so the names
    drop "-degree". Constructing the CronJobs at all is the check -- the field
    validator rejects a longer name -- but pin the abbreviation so it stays
    recognizable."""
    for cron_job in dataset.operational_kubernetes_resources("test-image-tag"):
        assert cron_job.name.startswith("noaa-gefs-forecast-35-day-0-5-virtual-")
        assert len(cron_job.name) <= 52
        assert cron_job.dataset_id == dataset.dataset_id


def test_validators(
    dataset: NoaaGefsForecast35Day05DegreeVirtualDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3

    _, validation_cron_job = dataset.operational_kubernetes_resources("test-image-tag")
    validation_fire = pd.Timestamp("2026-09-01") + pd.Timedelta(
        minutes=min(_fire_minutes(validation_cron_job.schedule))
    )
    # The cycle the update that this fire follows ingested. Inits are a day apart and
    # the fire is inside that day, so it is the cycle of the fire's own date.
    newest_init = validation_fire.floor(f"{whole_hours(_INIT_TIME_FREQUENCY)}h")
    assert validation_fire - newest_init == pd.Timedelta("9h55m")

    # A missed cycle would go unreported for a whole day if it were tolerated, so a
    # cycle is due as soon as the validation that follows it fires.
    current_data = next(
        v for v in validators if isinstance(v, validation.CheckCurrentData)
    )
    assert current_data.max_delay == timedelta(hours=9, minutes=50)
    assert (
        stalled_cycles_before_alerting(
            current_data.max_delay,
            validation_fire,
            newest_init,
            _INIT_TIME_FREQUENCY,
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
    # The leading tier covers the newest init, whose 840 hour extension the source is
    # still publishing; every older init must be whole.
    assert completeness.min_present_fraction == (0.57, 1.0)
    # 181 lead times of 31 members, in both the a and b products.
    files_per_lead_time = 31 * 2
    expected_files_per_init = 181 * files_per_lead_time
    allowed_missing = max(
        missing
        for missing in range(expected_files_per_init)
        if (expected_files_per_init - missing) / expected_files_per_init >= 0.57
    )
    # The newest init may be missing its whole extension past 384 hours, which is what
    # it is missing when validation fires: 76 of its 181 lead times.
    extension_files = 76 * files_per_lead_time
    assert extension_files <= allowed_missing
    # What is left over is under one member's share of the compact first stage, so a
    # member absent from that stage fails even while its init is newest.
    per_member_first_stage = 105 * 2
    assert allowed_missing < extension_files + per_member_first_stage

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


def test_manifest_splits_hold_a_similar_number_of_refs_per_array(
    dataset: NoaaGefsForecast35Day05DegreeVirtualDataset,
) -> None:
    """Every array holds one ref per (lead time, ensemble member, level) of every init
    in the split, so the split size fixes both the manifest's ref count and its bytes.
    A vertical group's arrays hold a ref per level, so they are split finer to keep
    every manifest a similar size."""
    split = dataset.icechunk_virtual_config.manifest_split
    refs_per_init = 181 * 31

    root_split = _resolved_split_size(split, "/temperature_2m")
    assert root_split == 8
    root_refs = root_split * refs_per_init
    assert root_refs == 44_888

    # The smallest manifest in the store, and still far above the 1000 refs icechunk
    # needs before it compresses ref locations.
    assert root_refs > 1000

    for group, levels, expected_split in (
        ("pressure_level", 31, 2),
        ("model_level", 4, 4),
        ("height_above_mean_sea_level", 8, 4),
    ):
        group_split = _resolved_split_size(split, f"/{group}/temperature")
        assert group_split == expected_split, group
        group_refs = group_split * refs_per_init * levels
        assert group_refs > root_refs, group
        # Well inside the 3 MiB a reader downloads to resolve any one chunk, at the
        # ~16.4 bytes/ref icechunk stores a compressed reference in.
        assert group_refs * 16.4 < 8 * 1024 * 1024, group


def test_virtual_container_matches_ref_prefix(
    dataset: NoaaGefsForecast35Day05DegreeVirtualDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == "s3://noaa-gefs-pds/"
