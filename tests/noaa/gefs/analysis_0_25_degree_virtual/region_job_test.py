from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.noaa.gefs import virtual_region_job as region_job_module
from reformatters.noaa.gefs.analysis_0_25_degree_virtual.region_job import (
    NoaaGefsAnalysis025DegreeVirtualRegionJob,
    NoaaGefsAnalysis025DegreeVirtualSourceFileCoord,
)
from reformatters.noaa.gefs.analysis_0_25_degree_virtual.template_config import (
    NoaaGefsAnalysis025DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_B22_TRANSITION_DATE,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.gefs.virtual_region_job import NoaaGefsVirtualRegionJob

TEMPLATE_CONFIG = NoaaGefsAnalysis025DegreeVirtualTemplateConfig()
_HOUR_0 = pd.Timedelta(0)


def get_var(name: str) -> NoaaGefsVirtualDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.name == name)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2024-06-02T00:00"))


def make_job(
    template_ds: xr.DataTree,
    data_vars: Sequence[NoaaGefsVirtualDataVar],
    region: slice = slice(0, 1),
) -> NoaaGefsAnalysis025DegreeVirtualRegionJob:
    return NoaaGefsAnalysis025DegreeVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="time",
        region=region,
        reformat_job_name="test",
    )


def coords_at(
    template_ds: xr.DataTree,
    times: Sequence[pd.Timestamp],
    data_vars: Sequence[NoaaGefsVirtualDataVar],
) -> list[NoaaGefsAnalysis025DegreeVirtualSourceFileCoord]:
    job = make_job(template_ds, data_vars=data_vars)
    region_ds = xr.Dataset(coords={"time": pd.to_datetime(list(times))})
    return list(job.generate_source_file_coords(region_ds, data_vars))


def test_source_file_coord_url_and_out_loc() -> None:
    coord = NoaaGefsAnalysis025DegreeVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("temperature_2m")],
    )
    assert coord.get_url() == (
        "s3://noaa-gefs-pds/gefs.20240601/06/atmos/pgrb2sp25/"
        "gec00.t06z.pgrb2s.0p25.f003"
    )
    assert coord.get_index_url() == coord.get_url() + ".idx"
    assert dict(coord.out_loc()) == {"time": pd.Timestamp("2024-06-01T09:00")}
    # Analysis reads the control member only.
    assert coord.ensemble_member == 0


def test_shortest_available_lead_per_variable(template_ds: xr.DataTree) -> None:
    """An instant variable reads lead 0 on a cycle boundary and lead 3 between; a
    windowed one shifts back a cycle to lead 6 rather than reading lead 0's degenerate
    zero-length window."""
    instant = get_var("temperature_2m")
    windowed = get_var("total_precipitation_surface")
    data_vars = [instant, windowed]

    on_cycle = pd.Timestamp("2024-06-01T06:00")
    between = pd.Timestamp("2024-06-01T09:00")
    coords = coords_at(template_ds, [on_cycle, between], data_vars)
    by_key = {(c.valid_time(), c.lead_time): c for c in coords}

    assert by_key[(on_cycle, _HOUR_0)].init_time == on_cycle
    assert {v.name for v in by_key[(on_cycle, _HOUR_0)].data_vars} == {instant.name}
    assert by_key[(on_cycle, pd.Timedelta("6h"))].init_time == on_cycle - pd.Timedelta(
        "6h"
    )
    assert {v.name for v in by_key[(on_cycle, pd.Timedelta("6h"))].data_vars} == {
        windowed.name
    }
    # Between cycles both variables come from the same lead 3 file of the same cycle.
    lead_3 = pd.Timedelta("3h")
    assert {v.name for v in by_key[(between, lead_3)].data_vars} == {
        instant.name,
        windowed.name,
    }
    assert by_key[(between, lead_3)].init_time == pd.Timestamp("2024-06-01T06:00")
    assert len(coords) == 3


def test_whole_catalog_sources_every_variable_once(template_ds: xr.DataTree) -> None:
    data_vars = TEMPLATE_CONFIG.data_vars
    on_cycle = pd.Timestamp("2024-06-01T06:00")

    coords = coords_at(template_ds, [on_cycle], data_vars)

    assert {c.lead_time for c in coords} == {_HOUR_0, pd.Timedelta("6h")}
    sourced = sorted(v.name for c in coords for v in c.data_vars)
    assert sourced == sorted(v.name for v in data_vars)
    assert len(sourced) == len(set(sourced))
    for coord in coords:
        from_lead_0 = coord.lead_time == _HOUR_0
        assert all(v.has_hour_0_values() == from_lead_0 for v in coord.data_vars)


def test_a_variable_is_not_sourced_before_its_available_from(
    template_ds: xr.DataTree,
) -> None:
    """The file is still read for its other variables; only the gated one drops out."""
    late = get_var("visibility_surface")
    always = get_var("temperature_2m")
    available_from = late.internal_attrs.available_from
    assert available_from is not None
    data_vars = [late, always]

    for time, expected in (
        (available_from - pd.Timedelta("3h"), {always.name}),
        (available_from, {always.name, late.name}),
    ):
        coords = coords_at(template_ds, [time], data_vars)
        assert len(coords) == 1, time
        assert {v.name for v in coords[0].data_vars} == expected, time


def test_no_coord_reaches_back_before_the_archive_start(
    template_ds: xr.DataTree,
) -> None:
    """A windowed variable at the first time would shift to a cycle that does not exist."""
    windowed = get_var("total_precipitation_surface")
    start = TEMPLATE_CONFIG.append_dim_start

    assert coords_at(template_ds, [start], [windowed]) == []
    later = coords_at(template_ds, [start + pd.Timedelta("6h")], [windowed])
    assert [c.init_time for c in later] == [start]


@pytest.mark.parametrize(
    "time", [pd.Timestamp("2021-01-01T06:00"), pd.Timestamp("2024-06-01T06:00")]
)
def test_representative_var_is_an_era_stable_instant_variable(
    template_ds: xr.DataTree, time: pd.Timestamp
) -> None:
    """Probing a variable the source added partway through would leave every older file
    permanently un-ingestable, so the probe is always one published in every era."""
    data_vars = TEMPLATE_CONFIG.data_vars
    job = make_job(template_ds, data_vars=data_vars)
    coords = coords_at(template_ds, [time], data_vars)
    assert coords

    for coord in coords:
        rep = job.representative_var(coord)
        assert rep in coord.data_vars
        assert rep.internal_attrs.available_from is None
        # The lead 6 file carries only windowed variables, so instant is a preference
        # among era-stable candidates rather than a guarantee.
        era_stable_instant = [
            v
            for v in coord.data_vars
            if v.internal_attrs.available_from is None
            and v.attrs.step_type == "instant"
        ]
        if era_stable_instant:
            assert rep.attrs.step_type == "instant"


def test_representative_var_skips_a_variable_the_archive_added_later(
    template_ds: xr.DataTree,
) -> None:
    """Even when a later-added variable comes first in the file's variable list."""
    late = get_var("visibility_surface")
    assert late.internal_attrs.available_from is not None
    stable = get_var("temperature_2m")
    coord = NoaaGefsAnalysis025DegreeVirtualSourceFileCoord(
        init_time=pd.Timestamp("2021-01-01T06:00"),
        lead_time=_HOUR_0,
        data_vars=[late, stable],
    )
    job = make_job(template_ds, data_vars=[late, stable])

    assert job.representative_var(coord).name == stable.name


def test_representative_var_prefers_instant_over_windowed(
    template_ds: xr.DataTree,
) -> None:
    """A windowed variable can be absent where an instant one has data, so it is a
    weaker presence probe even when both are era-stable."""
    windowed = get_var("total_precipitation_surface")
    instant = get_var("relative_humidity_2m")
    assert windowed.internal_attrs.available_from is None
    coord = NoaaGefsAnalysis025DegreeVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("3h"),
        data_vars=[windowed, instant],
    )
    job = make_job(template_ds, data_vars=[windowed, instant])

    assert job.representative_var(coord).name == instant.name


def test_representative_var_falls_back_when_every_variable_was_added_later(
    template_ds: xr.DataTree,
) -> None:
    """A run filtered to one later-added variable has no era-stable choice, and the
    files it reads do carry that variable, so it must not refuse."""
    late = get_var("visibility_surface")
    coord = NoaaGefsAnalysis025DegreeVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=_HOUR_0,
        data_vars=[late],
    )
    job = make_job(template_ds, data_vars=[late])

    assert job.representative_var(coord).name == late.name


def test_file_refs_span_each_message_and_end_at_the_file_end(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each message's end byte is the next message's start; the last is the file end."""
    index = (
        "1:0:d=2024060106:TMP:2 m above ground:3 hour fcst:ENS=low-res ctl\n"
        "2:500:d=2024060106:APCP:surface:0-3 hour acc fcst:ENS=low-res ctl\n"
        "3:900:d=2024060106:RH:2 m above ground:3 hour fcst:ENS=low-res ctl\n"
    )

    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        path = tmp_path / "index.idx"
        path.write_text(index)
        return path

    monkeypatch.setattr(region_job_module, "s3_download_to_disk", fake_download)

    data_vars = [
        get_var("temperature_2m"),
        get_var("total_precipitation_surface"),
        get_var("relative_humidity_2m"),
    ]
    coord = NoaaGefsAnalysis025DegreeVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("3h"),
        data_vars=data_vars,
    )
    job = make_job(template_ds, data_vars=data_vars)

    refs = job.file_refs(coord, file_size=1200)

    assert [(r.data_var.name, r.offset, r.length) for r in refs] == [
        ("temperature_2m", 0, 500),
        ("total_precipitation_surface", 500, 400),
        ("relative_humidity_2m", 900, 300),
    ]
    assert all(
        dict(r.out_loc) == {"time": pd.Timestamp("2024-06-01T09:00")} for r in refs
    )
    assert all(r.location == coord.get_url() for r in refs)


def test_file_refs_refuses_an_index_missing_a_requested_variable(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """At lead 6 the accumulation window is 0-6, so a 0-3 line is a different message.
    Committing the file anyway would leave a NaN column nothing ever retries."""

    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        path = tmp_path / "index.idx"
        path.write_text(
            "1:0:d=2024060100:APCP:surface:0-3 hour acc fcst:ENS=low-res ctl\n"
        )
        return path

    monkeypatch.setattr(region_job_module, "s3_download_to_disk", fake_download)

    data_vars = [get_var("total_precipitation_surface")]
    coord = NoaaGefsAnalysis025DegreeVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T00:00"),
        lead_time=pd.Timedelta("6h"),
        data_vars=data_vars,
    )
    job = make_job(template_ds, data_vars=data_vars)

    with pytest.raises(AssertionError, match="has no message for"):
        job.file_refs(coord, file_size=1200)


def test_file_refs_refuses_two_element_spellings_for_one_chunk(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """grib_element_alternatives give a variable one lookup key per spelling. A file
    carrying two of them would hand the variable two refs for one chunk, writing the
    same bytes twice with no way to tell which won."""

    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        path = tmp_path / "index.idx"
        path.write_text(
            "1:0:d=2024060106:TMP:2 m above ground:3 hour fcst:ENS=low-res ctl\n"
            "2:500:d=2024060106:TMPK:2 m above ground:3 hour fcst:ENS=low-res ctl\n"
        )
        return path

    monkeypatch.setattr(region_job_module, "s3_download_to_disk", fake_download)

    temperature = get_var("temperature_2m")
    assert temperature.internal_attrs.grib_element == "TMP"
    two_spellings = replace(
        temperature,
        internal_attrs=replace(
            temperature.internal_attrs, grib_element_alternatives=("TMPK",)
        ),
    )
    coord = NoaaGefsAnalysis025DegreeVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("3h"),
        data_vars=[two_spellings],
    )
    job = make_job(template_ds, data_vars=[two_spellings])

    with pytest.raises(
        AssertionError, match=r"has two messages for temperature_2m at .*09:00"
    ):
        job.file_refs(coord, file_size=1200)


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2024-06-02T00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = (
        NoaaGefsAnalysis025DegreeVirtualRegionJob.operational_update_jobs(
            primary_store=Mock(),
            tmp_store=Path("unused-tmp.zarr"),
            get_template_fn=TEMPLATE_CONFIG.get_template,
            append_dim="time",
            all_data_vars=TEMPLATE_CONFIG.data_vars,
            reformat_job_name="test",
        )
    )

    (job,) = jobs
    assert isinstance(job, NoaaGefsAnalysis025DegreeVirtualRegionJob)
    assert job.processing_mode == "update"
    times = template_ds.to_dataset().get_index("time")
    # An 18h window of 3-hourly steps.
    assert job.region == slice(len(times) - 6, len(times))


def gate_coords(
    template_ds: xr.DataTree, times: Sequence[pd.Timestamp]
) -> list[NoaaGefsAnalysis025DegreeVirtualSourceFileCoord]:
    return coords_at(
        template_ds,
        times,
        [get_var("temperature_2m"), get_var("total_precipitation_surface")],
    )


def discover(
    job: NoaaGefsAnalysis025DegreeVirtualRegionJob,
    pending: Sequence[NoaaGefsAnalysis025DegreeVirtualSourceFileCoord],
    published: Sequence[NoaaGefsAnalysis025DegreeVirtualSourceFileCoord],
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[pd.Timedelta, pd.Timestamp]]:
    """Run the gate over `pending` with `published` listed by the source."""
    monkeypatch.setattr(
        NoaaGefsVirtualRegionJob,
        "discover_available",
        lambda self, pending: [(c, 100) for c in pending if c in published],
    )
    return sorted(
        (coord.lead_time, coord.valid_time())
        for coord, _ in job.discover_available(list(pending))
    )


def test_windowed_file_withheld_until_the_shortest_lead_file_publishes(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cycle's f006 lands hours before the next cycle's f000, so releasing it alone
    would extend time to a step carrying only the windowed half."""
    time = pd.Timestamp("2024-06-01T06:00")
    coords = gate_coords(template_ds, [time])
    (lead_0,) = [c for c in coords if c.lead_time == _HOUR_0]
    (lead_6,) = [c for c in coords if c.lead_time == pd.Timedelta("6h")]
    job = make_job(template_ds, data_vars=TEMPLATE_CONFIG.data_vars).model_copy(
        update={"ingested_through": time - pd.Timedelta("3h")}
    )

    assert discover(job, coords, [lead_6], monkeypatch) == []
    assert discover(job, coords, [lead_6, lead_0], monkeypatch) == [
        (_HOUR_0, time),
        (pd.Timedelta("6h"), time),
    ]


def test_only_the_time_holding_its_own_file_is_released(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    times = [pd.Timestamp("2024-06-01T06:00"), pd.Timestamp("2024-06-01T12:00")]
    coords = gate_coords(template_ds, times)
    published = [
        c for c in coords if c.valid_time() == times[0] or c.lead_time != _HOUR_0
    ]
    job = make_job(template_ds, data_vars=TEMPLATE_CONFIG.data_vars).model_copy(
        update={"ingested_through": times[0] - pd.Timedelta("3h")}
    )

    assert discover(job, coords, published, monkeypatch) == [
        (_HOUR_0, times[0]),
        (pd.Timedelta("6h"), times[0]),
    ]


def test_a_time_the_store_already_covers_is_never_withheld(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cycle the archive never published must not block the file beside it forever."""
    time = pd.Timestamp("2024-06-01T06:00")
    coords = gate_coords(template_ds, [time])
    (lead_6,) = [c for c in coords if c.lead_time == pd.Timedelta("6h")]
    job = make_job(template_ds, data_vars=TEMPLATE_CONFIG.data_vars).model_copy(
        update={"ingested_through": time}
    )

    assert discover(job, coords, [lead_6], monkeypatch) == [(pd.Timedelta("6h"), time)]


def test_an_empty_store_waits_for_a_shortest_lead_file(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    time = pd.Timestamp("2024-06-01T06:00")
    coords = gate_coords(template_ds, [time])
    (lead_6,) = [c for c in coords if c.lead_time == pd.Timedelta("6h")]
    job = make_job(template_ds, data_vars=TEMPLATE_CONFIG.data_vars)
    assert job.ingested_through is None

    assert discover(job, coords, [lead_6], monkeypatch) == []


_IDX_FIXTURES = sorted(
    p.name for p in (Path(__file__).parent / "idx_fixtures").glob("*.idx")
)
# An empty parametrize set skips rather than fails, which would silently retire the
# only check that compares this catalog against real archived indexes.
assert len(_IDX_FIXTURES) == 18, _IDX_FIXTURES


@pytest.mark.parametrize("fixture_name", _IDX_FIXTURES)
def test_every_requested_variable_maps_to_a_real_message(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture_name: str,
) -> None:
    """Run the real coord generation and ref building against real archived indexes.

    Catches an element, level or window string that is wrong for any of the 38
    variables, which declaration-derived tests cannot: they compare the config to
    itself. The fixtures straddle both inventory changes (MSLET at 2021-07-20T12, the
    other three at 2022-10-18T12), so a variable requested one cycle too early fails
    here rather than committing a NaN column.
    """
    fixture = Path(__file__).parent / "idx_fixtures" / fixture_name
    stem, lead_str = fixture_name.removesuffix(".idx").split("_f")
    init_time = pd.Timestamp(stem)
    lead_time = pd.Timedelta(hours=int(lead_str))
    index_text = fixture.read_text()

    # file_refs unlinks the index it is given, so hand it a copy rather than the fixture.
    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        copy = tmp_path / fixture_name
        copy.write_text(index_text)
        return copy

    monkeypatch.setattr(region_job_module, "s3_download_to_disk", fake_download)

    data_vars = TEMPLATE_CONFIG.data_vars
    coords = coords_at(template_ds, [init_time + lead_time], data_vars)
    matching = [
        c for c in coords if c.init_time == init_time and c.lead_time == lead_time
    ]
    assert len(matching) == 1, f"no coord reads {fixture_name}"
    coord = matching[0]

    last_start = max(int(line.split(":")[1]) for line in index_text.splitlines())
    job = make_job(template_ds, data_vars=data_vars)
    refs = job.file_refs(coord, file_size=last_start + 1_000_000)

    # file_refs asserts nothing was requested but absent; this pins the other direction,
    # that every requested variable got exactly one ref.
    assert sorted(r.data_var.name for r in refs) == sorted(
        v.name for v in coord.data_vars
    )
    assert len({r.data_var.name for r in refs}) == len(refs)


@pytest.mark.parametrize("hour", [0, 3, 6, 9, 12, 15, 18, 21])
def test_lead_assignment_at_every_three_hourly_time(
    template_ds: xr.DataTree, hour: int
) -> None:
    """The instant/windowed split behaves differently only at 00/06/12/18, and only
    00:00 rolls the date backwards, so every hour is asserted rather than sampled."""
    day = pd.Timestamp("2024-06-02")
    time = day + pd.Timedelta(hours=hour)
    instant = get_var("temperature_2m")
    windowed = get_var("total_precipitation_surface")

    by_var = {
        v.name: (c.init_time, c.lead_time)
        for c in coords_at(template_ds, [time], [instant, windowed])
        for v in c.data_vars
    }

    on_cycle = hour % 6 == 0
    cycle = day + pd.Timedelta(hours=hour - hour % 6)
    assert by_var[instant.name] == (cycle, pd.Timedelta(hours=hour % 6))
    assert by_var[windowed.name] == (
        (cycle - pd.Timedelta("6h"), pd.Timedelta("6h"))
        if on_cycle
        else (cycle, pd.Timedelta(hours=hour % 6))
    )
    if hour == 0:
        # The only time whose windowed source file is on the previous day.
        assert by_var[windowed.name][0] == pd.Timestamp("2024-06-01T18:00")


def test_available_from_is_judged_against_the_cycle_not_the_valid_time(
    template_ds: xr.DataTree,
) -> None:
    """Whether a variable exists is a property of the cycle that wrote the file.

    A windowed variable at a cycle boundary reads the *previous* cycle, so a valid time
    at or after a transition can still resolve to a file from before it. No shipped
    variable is both windowed and late-added, so this constructs that combination to pin
    the rule for the forecast datasets, whose longer leads reach it with real variables.
    """
    transition = GEFS_B22_TRANSITION_DATE
    windowed = get_var("total_precipitation_surface")
    assert windowed.internal_attrs.available_from is None
    late_windowed = replace(
        windowed,
        internal_attrs=replace(windowed.internal_attrs, available_from=transition),
    )

    # Valid time is exactly the transition, but a windowed variable reads the 06z cycle,
    # which predates it and does not carry the variable.
    coords = coords_at(template_ds, [transition], [late_windowed])
    assert coords == []

    # One step later the same cycle is the transition cycle itself, so it is requested.
    later = coords_at(template_ds, [transition + pd.Timedelta("3h")], [late_windowed])
    assert [(c.init_time, c.lead_time) for c in later] == [
        (transition, pd.Timedelta("3h"))
    ]
