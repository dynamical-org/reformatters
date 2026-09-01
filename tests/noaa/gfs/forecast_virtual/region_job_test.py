from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.config_models import ROOT
from reformatters.noaa import noaa_virtual_region_job as shared_region_job_module
from reformatters.noaa.gfs.forecast_virtual.region_job import (
    NoaaGfsForecastVirtualRegionJob,
    NoaaGfsForecastVirtualSourceFileCoord,
)
from reformatters.noaa.gfs.forecast_virtual.template_config import (
    NoaaGfsForecastVirtualTemplateConfig,
)
from reformatters.noaa.gfs.virtual_region_job import NoaaGfsFileType
from reformatters.noaa.gfs.virtual_template_config import (
    PRESSURE_LEVEL_INDEX_FORMAT,
    PRESSURE_LEVELS,
)
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_grib_index import (
    grib_index_window_str,
    parse_grib_index_lines,
)
from tests.noaa.grib_index_fixtures import cached_grib_index, stub_grib_index_download

TEMPLATE_CONFIG = NoaaGfsForecastVirtualTemplateConfig()
_DATASET_ID = "noaa-gfs-forecast-virtual-test"

# Spans the whole intended archive: its first init, a few days later, the CLWMR ->
# CLMR rename era, and the present. The claims these tests pin are frozen into published
# coordinates, so the domain to sample is the archive, whose first init is
# append_dim_start rather than a date near it.
_ERAS = ("20210322", "20210325", "20230401", "20260828")
# f000 (no windowed message at all), the leads where the 6 hour bucket and the running
# total collapse onto one index line, the first lead where they separate, both forms of
# the running total's day-vs-hour window switch, and the last lead.
_LEADS = (0, 1, 3, 6, 9, 12, 24, 120, 123, 384)


def get_var(path: str) -> NoaaDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.path == path)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2021-03-23T18:00"))


def make_job(
    template_ds: xr.DataTree, data_vars: Sequence[NoaaDataVar]
) -> NoaaGfsForecastVirtualRegionJob:
    return NoaaGfsForecastVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=list(data_vars),
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


def index_url(era: str, file_type: NoaaGfsFileType, lead_hours: int) -> str:
    return (
        f"s3://noaa-gfs-bdp-pds/gfs.{era}/12/atmos/"
        f"gfs.t12z.{file_type}.0p25.f{lead_hours:03d}.idx"
    )


def index_lines(
    era: str, file_type: NoaaGfsFileType, lead_hours: int
) -> list[tuple[int, str, str, str]]:
    return parse_grib_index_lines(
        cached_grib_index(index_url(era, file_type, lead_hours), _DATASET_ID)
    )


def coords_for(
    template_ds: xr.DataTree,
    data_vars: Sequence[NoaaDataVar],
    init_times: Sequence[pd.Timestamp],
    lead_times: Sequence[pd.Timedelta],
) -> list[NoaaGfsForecastVirtualSourceFileCoord]:
    job = make_job(template_ds, data_vars)
    region_ds = xr.Dataset(
        coords={
            "init_time": pd.DatetimeIndex(list(init_times)),
            "lead_time": pd.TimedeltaIndex(list(lead_times)),
        }
    )
    return list(job.generate_source_file_coords(region_ds, list(data_vars)))


def var_levels(var: NoaaDataVar) -> list[str]:
    if var.group is not ROOT:
        return [PRESSURE_LEVEL_INDEX_FORMAT.format(level=lv) for lv in PRESSURE_LEVELS]
    return [var.internal_attrs.grib_index_level]


def test_each_lead_reads_both_products(template_ds: xr.DataTree) -> None:
    init_time = pd.Timestamp("2021-03-23T12:00")
    lead_times = pd.to_timedelta(template_ds.to_dataset().get_index("lead_time"))
    assert len(lead_times) == 209

    coords = coords_for(
        template_ds, TEMPLATE_CONFIG.data_vars, [init_time], list(lead_times)
    )

    assert len(coords) == 2 * len(lead_times)
    assert {c.file_type for c in coords} == {"pgrb2", "pgrb2b"}
    assert {c.init_time for c in coords} == {init_time}
    assert sorted({c.lead_time for c in coords}) == list(lead_times)


def test_hour_0_drops_only_the_variables_the_source_omits_there(
    template_ds: xr.DataTree,
) -> None:
    """Nine instantaneous variables share a grib element with a windowed sibling and
    ARE published at f000, so an hour-0 rule keyed on the element would drop them."""
    absent_at_hour_0 = {
        # Every windowed variable, plus five instantaneous convection/evaporation
        # diagnostics the analysis step does not produce.
        "potential_evaporation_rate_surface",
        "instantaneous_precipitation_convective_surface",
        "pressure_convective_cloud_bottom",
        "pressure_convective_cloud_top",
        "convective_cloud_cover",
    }
    present_at_hour_0 = {
        "instantaneous_categorical_snow_surface",
        "instantaneous_categorical_rain_surface",
        "instantaneous_categorical_freezing_rain_surface",
        "instantaneous_categorical_ice_pellets_surface",
        "instantaneous_total_cloud_cover_atmosphere",
        "precipitation_rate_surface",
        "low_cloud_cover",
        "medium_cloud_cover",
        "high_cloud_cover",
    }
    assert len(present_at_hour_0) == 9

    at_hour_0 = {
        var.name
        for coord in coords_for(
            template_ds,
            TEMPLATE_CONFIG.data_vars,
            [pd.Timestamp("2021-03-23T12:00")],
            [pd.Timedelta(0)],
        )
        for var in coord.data_vars
    }
    windowed = {
        v.name for v in TEMPLATE_CONFIG.data_vars if v.attrs.step_type != "instant"
    }
    assert len(windowed) == 44
    assert at_hour_0.isdisjoint(windowed | absent_at_hour_0)
    assert present_at_hour_0 <= at_hour_0
    assert (
        at_hour_0
        == {v.name for v in TEMPLATE_CONFIG.data_vars if v.name not in windowed}
        - absent_at_hour_0
    )


def test_representative_var_is_carried_only_by_its_own_product(
    template_ds: xr.DataTree,
) -> None:
    """A probe on a variable the file does not fill would never be marked ingested."""
    job = make_job(template_ds, TEMPLATE_CONFIG.data_vars)
    coords = coords_for(
        template_ds,
        TEMPLATE_CONFIG.data_vars,
        [pd.Timestamp("2021-03-23T12:00")],
        [pd.Timedelta(0), pd.Timedelta("9h")],
    )
    assert len(coords) == 4

    picked = {
        (c.file_type, c.lead_time): job.representative_var(c).name for c in coords
    }
    assert picked == {
        ("pgrb2", pd.Timedelta(0)): "temperature_2m",
        ("pgrb2", pd.Timedelta("9h")): "temperature_2m",
        ("pgrb2b", pd.Timedelta(0)): "geopotential_height_0p5pvu",
        ("pgrb2b", pd.Timedelta("9h")): "geopotential_height_0p5pvu",
    }
    for coord in coords:
        var = job.representative_var(coord)
        assert var in coord.data_vars
        assert dict(job.representative_probe_loc(coord, var)) == {
            "init_time": coord.init_time,
            "lead_time": coord.lead_time,
        }


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window reaches back over the two cycles before the one being published."""
    now = pd.Timestamp("2021-03-25T03:30")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = NoaaGfsForecastVirtualRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="init_time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )

    (job,) = jobs
    assert isinstance(job, NoaaGfsForecastVirtualRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    assert list(init_times[job.region]) == [
        pd.Timestamp("2021-03-24T12:00"),
        pd.Timestamp("2021-03-24T18:00"),
        pd.Timestamp("2021-03-25T00:00"),
    ]


@pytest.mark.slow
def test_the_archive_starts_at_the_declared_first_init() -> None:
    """append_dim_start is where the 0.25 degree archive begins, read off S3.

    Prepending to an append dim is a breaking change, so a start one cycle too late
    puts data permanently out of reach; one too early leaves a hole nothing can fill.
    """
    first_init = TEMPLATE_CONFIG.append_dim_start
    assert first_init == pd.Timestamp("2021-03-22T12:00")
    day = first_init.floor("1D")
    coords = [
        NoaaGfsForecastVirtualSourceFileCoord(
            init_time=init_time,
            lead_time=pd.Timedelta(0),
            file_type=file_type,
            data_vars=[],
        )
        for init_time in pd.date_range(day - pd.Timedelta("6h"), periods=5, freq="6h")
        for file_type in ("pgrb2", "pgrb2b")
    ]
    job = NoaaGfsForecastVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=TEMPLATE_CONFIG.get_template(first_init + pd.Timedelta("6h")),
        data_vars=TEMPLATE_CONFIG.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )

    available = job.discover_available(list(coords))

    assert {coord.init_time for coord, _ in available} == {
        first_init,
        first_init + pd.Timedelta("6h"),
    }


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
def test_the_source_publishes_every_lead_the_template_declares(era: str) -> None:
    """The 209 lead coordinate values, read back off S3 rather than taken from a note.

    The hourly-to-3-hourly step at f120 is the trap: the source publishes no f121 or
    f122, and a lead the source does not publish would be a permanently empty column.
    """
    init_time = pd.Timestamp(f"{era[:4]}-{era[4:6]}-{era[6:]}T12:00")
    lead_times = TEMPLATE_CONFIG.dimension_coordinates()["lead_time"]
    coords = [
        NoaaGfsForecastVirtualSourceFileCoord(
            init_time=init_time,
            lead_time=lead_time,
            file_type=file_type,
            data_vars=[],
        )
        for lead_time in lead_times
        for file_type in ("pgrb2", "pgrb2b")
    ]
    assert len(coords) == 418

    job = NoaaGfsForecastVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=TEMPLATE_CONFIG.get_template(init_time + pd.Timedelta("6h")),
        data_vars=TEMPLATE_CONFIG.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
    available = job.discover_available(list(coords))

    assert {id(coord) for coord, _ in available} == {id(coord) for coord in coords}


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
@pytest.mark.parametrize("lead_hours", _LEADS)
def test_every_windowed_window_string_matches_the_real_index(
    era: str, lead_hours: int
) -> None:
    """Render each windowed variable's index window string and look for it in the real
    index at that lead.

    The trap is the running total's day-vs-hour switch: it renders `0-1 day acc fcst`
    at f024 and `0-16 day acc fcst` at f384 but `0-123 hour acc fcst` at f123, so an
    hours-only matcher passes at f123 and silently matches nothing at every 24 hour
    multiple. f000 is the other direction: GFS publishes no windowed message there and
    every rendered string must match nothing.
    """
    published = Counter(
        (element, level, window)
        for file_type in ("pgrb2", "pgrb2b")
        for _, element, level, window in index_lines(era, file_type, lead_hours)
    )
    assert published
    windowed = [v for v in TEMPLATE_CONFIG.data_vars if v.attrs.step_type != "instant"]
    assert len(windowed) == 44

    for var in windowed:
        window = grib_index_window_str(var, lead_hours)
        elements = (
            var.internal_attrs.grib_element,
            *var.internal_attrs.grib_element_alternatives,
        )
        matches = sum(
            published[(element, level, window)]
            for element in elements
            for level in var_levels(var)
        )
        assert (matches > 0) == (lead_hours > 0), (var.name, window, matches)


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
@pytest.mark.parametrize("lead_hours", [9, 24, 120, 123, 384])
def test_the_running_totals_render_the_window_the_source_uses(
    era: str, lead_hours: int
) -> None:
    """Pinned separately from the sweep above so the day form is visible in the test."""
    expected = {
        9: "0-9 hour acc fcst",
        24: "0-1 day acc fcst",
        120: "0-5 day acc fcst",
        123: "0-123 hour acc fcst",
        384: "0-16 day acc fcst",
    }[lead_hours]
    bucket_expected = {
        9: "6-9 hour acc fcst",
        24: "18-24 hour acc fcst",
        120: "114-120 hour acc fcst",
        123: "120-123 hour acc fcst",
        384: "378-384 hour acc fcst",
    }[lead_hours]
    published = {
        (element, level, window)
        for _, element, level, window in index_lines(era, "pgrb2", lead_hours)
    }

    for name, element in (
        ("total_precipitation_run_total_surface", "APCP"),
        ("convective_precipitation_run_total_surface", "ACPCP"),
    ):
        assert grib_index_window_str(get_var(name), lead_hours) == expected, name
        assert (element, "surface", expected) in published, name
    for name, element in (
        ("total_precipitation_surface", "APCP"),
        ("convective_precipitation_surface", "ACPCP"),
    ):
        assert grib_index_window_str(get_var(name), lead_hours) == bucket_expected, name
        assert (element, "surface", bucket_expected) in published, name


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
@pytest.mark.parametrize(
    ("file_type", "lead_hours", "expected_refs"),
    [
        ("pgrb2", 0, 696),
        ("pgrb2", 1, 743),
        ("pgrb2", 24, 743),
        ("pgrb2", 123, 743),
        ("pgrb2", 384, 743),
        ("pgrb2b", 0, 306),
        ("pgrb2b", 9, 308),
        ("pgrb2b", 384, 308),
    ],
)
def test_every_source_message_reaches_an_array(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    era: str,
    tmp_path: Path,
    file_type: NoaaGfsFileType,
    lead_hours: int,
    expected_refs: int,
) -> None:
    """Run the whole catalog against a real index at the leads that stress it.

    A wrong idx level string, element spelling or rendered window string shows up here
    as a missing ref and a duplicated one as an extra. The counts equal the source's own
    message counts: every array position a file can fill is filled, and the leads where
    two index lines describe one message still yield one ref per position.
    """
    stub_grib_index_download(
        monkeypatch,
        shared_region_job_module,
        tmp_path,
        lambda url: cached_grib_index(url, _DATASET_ID),
    )
    init_time = pd.Timestamp(f"{era[:4]}-{era[4:6]}-{era[6:]}T12:00")
    data_vars = [
        v for v in TEMPLATE_CONFIG.data_vars if lead_hours > 0 or v.has_hour_0_values()
    ]
    assert data_vars
    job = make_job(template_ds, TEMPLATE_CONFIG.data_vars)
    coord = NoaaGfsForecastVirtualSourceFileCoord(
        init_time=init_time,
        lead_time=pd.Timedelta(hours=lead_hours),
        file_type=file_type,
        data_vars=data_vars,
    )

    refs = job.file_refs(coord, file_size=10**10)

    assert len(refs) == expected_refs
    positions = {
        (ref.data_var.path, tuple(sorted(ref.out_loc.items()))) for ref in refs
    }
    assert len(positions) == len(refs)


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
@pytest.mark.parametrize("file_type", ["pgrb2", "pgrb2b"])
def test_a_job_filtered_to_a_pressure_level_variable_probes_a_level_it_fills(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    era: str,
    tmp_path: Path,
    file_type: NoaaGfsFileType,
) -> None:
    """The two products split the isobaric coordinate, so the probe cell of a job
    carrying only a vertical-group variable has to be chosen per product. This is the
    single-variable backfill of docs/add_new_variable.md."""
    stub_grib_index_download(
        monkeypatch,
        shared_region_job_module,
        tmp_path,
        lambda url: cached_grib_index(url, _DATASET_ID),
    )
    var = get_var("pressure_level/temperature")
    job = make_job(template_ds, [var])
    coord = NoaaGfsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp(f"{era[:4]}-{era[4:6]}-{era[6:]}T12:00"),
        lead_time=pd.Timedelta("9h"),
        file_type=file_type,
        data_vars=[var],
    )

    refs = job.file_refs(coord, file_size=10**10)

    levels = {dict(ref.out_loc)["pressure_level"] for ref in refs}
    assert len(levels) == (41 if file_type == "pgrb2" else 16)
    assert dict(job.representative_probe_loc(coord, var))["pressure_level"] == (
        1000.0 if file_type == "pgrb2" else 875.0
    )
    job._assert_probe_chunk_covered(coord, refs)


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
@pytest.mark.parametrize("lead_hours", [1, 3, 6])
def test_the_bucket_and_the_run_total_share_one_index_line_at_short_leads(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    era: str,
    tmp_path: Path,
    lead_hours: int,
) -> None:
    """At leads 1-6 both variables render one window string and the source emits two
    identical messages for it. Both arrays must be filled, and neither twice."""
    duplicated = Counter(
        (element, level, window)
        for _, element, level, window in index_lines(era, "pgrb2", lead_hours)
    )
    assert [key for key, count in duplicated.items() if count > 1] == [
        ("APCP", "surface", f"0-{lead_hours} hour acc fcst"),
        ("ACPCP", "surface", f"0-{lead_hours} hour acc fcst"),
    ]

    stub_grib_index_download(
        monkeypatch,
        shared_region_job_module,
        tmp_path,
        lambda url: cached_grib_index(url, _DATASET_ID),
    )
    job = make_job(template_ds, TEMPLATE_CONFIG.data_vars)
    coord = NoaaGfsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp(f"{era[:4]}-{era[4:6]}-{era[6:]}T12:00"),
        lead_time=pd.Timedelta(hours=lead_hours),
        file_type="pgrb2",
        data_vars=TEMPLATE_CONFIG.data_vars,
    )

    refs = job.file_refs(coord, file_size=10**10)

    by_name = Counter(ref.data_var.name for ref in refs)
    offsets = {
        ref.data_var.name: ref.offset
        for ref in refs
        if ref.data_var.name.startswith(("total_precipitation", "convective_precip"))
    }
    for bucket, run_total in (
        ("total_precipitation_surface", "total_precipitation_run_total_surface"),
        (
            "convective_precipitation_surface",
            "convective_precipitation_run_total_surface",
        ),
    ):
        assert by_name[bucket] == 1
        assert by_name[run_total] == 1
        # One message, so both arrays point at the same byte range.
        assert offsets[bucket] == offsets[run_total]


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
def test_the_bucket_and_the_run_total_separate_past_lead_6(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    era: str,
    tmp_path: Path,
) -> None:
    stub_grib_index_download(
        monkeypatch,
        shared_region_job_module,
        tmp_path,
        lambda url: cached_grib_index(url, _DATASET_ID),
    )
    job = make_job(template_ds, TEMPLATE_CONFIG.data_vars)
    coord = NoaaGfsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp(f"{era[:4]}-{era[4:6]}-{era[6:]}T12:00"),
        lead_time=pd.Timedelta("9h"),
        file_type="pgrb2",
        data_vars=TEMPLATE_CONFIG.data_vars,
    )

    offsets = {
        ref.data_var.name: ref.offset
        for ref in job.file_refs(coord, file_size=10**10)
        if ref.data_var.name
        in ("total_precipitation_surface", "total_precipitation_run_total_surface")
    }

    assert len(offsets) == 2
    assert (
        offsets["total_precipitation_surface"]
        != offsets["total_precipitation_run_total_surface"]
    )
