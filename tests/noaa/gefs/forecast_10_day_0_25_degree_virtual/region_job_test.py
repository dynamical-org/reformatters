import itertools
import re
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.noaa import noaa_virtual_region_job as noaa_virtual_job_module
from reformatters.noaa.gefs.forecast_10_day_0_25_degree_virtual.region_job import (
    NoaaGefsForecast10Day025DegreeVirtualRegionJob,
)
from reformatters.noaa.gefs.forecast_10_day_0_25_degree_virtual.template_config import (
    NoaaGefsForecast10Day025DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_B22_TRANSITION_DATE,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.gefs.virtual_region_job import (
    NoaaGefsForecastVirtualSourceFileCoord,
)
from tests.noaa.grib_index_fixtures import stub_grib_source_file_reads

TEMPLATE_CONFIG = NoaaGefsForecast10Day025DegreeVirtualTemplateConfig()
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
) -> NoaaGefsForecast10Day025DegreeVirtualRegionJob:
    return NoaaGefsForecast10Day025DegreeVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=region,
        reformat_job_name="test",
    )


def coords_for(
    template_ds: xr.DataTree,
    data_vars: Sequence[NoaaGefsVirtualDataVar],
    *,
    init_times: Sequence[pd.Timestamp],
    lead_times: Sequence[pd.Timedelta],
    ensemble_members: Sequence[int],
) -> list[NoaaGefsForecastVirtualSourceFileCoord]:
    job = make_job(template_ds, data_vars=data_vars)
    region_ds = xr.Dataset(
        coords={
            "init_time": pd.to_datetime(list(init_times)),
            "lead_time": pd.to_timedelta(list(lead_times)),
            "ensemble_member": list(ensemble_members),
        }
    )
    return list(job.generate_source_file_coords(region_ds, data_vars))


@pytest.mark.parametrize(
    ("ensemble_member", "member_str"), [(0, "gec00"), (1, "gep01"), (30, "gep30")]
)
def test_source_file_coord_url_names_the_member(
    ensemble_member: int, member_str: str
) -> None:
    """The control member is published as gec00 and the perturbed ones as gepNN."""
    coord = NoaaGefsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("240h"),
        ensemble_member=ensemble_member,
        source_file_type="s",
        data_vars=[get_var("temperature_2m")],
    )
    assert coord.get_url() == (
        f"s3://noaa-gefs-pds/gefs.20240601/06/atmos/pgrb2sp25/"
        f"{member_str}.t06z.pgrb2s.0p25.f240"
    )
    assert coord.get_index_url() == coord.get_url() + ".idx"
    assert dict(coord.out_loc()) == {
        "init_time": pd.Timestamp("2024-06-01T06:00"),
        "lead_time": pd.Timedelta("240h"),
        "ensemble_member": ensemble_member,
    }


def test_every_init_lead_member_triple_gets_exactly_one_file(
    template_ds: xr.DataTree,
) -> None:
    """Enumerated rather than sampled: a lead or member dropped from the product is a
    silent NaN column, and an off-by-one in the nesting only shows at the edges."""
    data_vars = TEMPLATE_CONFIG.data_vars
    init_times = [pd.Timestamp("2024-06-01T18:00"), pd.Timestamp("2024-06-02T00:00")]
    lead_times = list(TEMPLATE_CONFIG.lead_times())
    members = list(range(31))

    coords = coords_for(
        template_ds,
        data_vars,
        init_times=init_times,
        lead_times=lead_times,
        ensemble_members=members,
    )

    triples = [(c.init_time, c.lead_time, c.ensemble_member) for c in coords]
    assert len(triples) == len(set(triples))
    assert set(triples) == set(itertools.product(init_times, lead_times, members))
    assert len(triples) == 2 * 81 * 31
    assert {c.source_file_type for c in coords} == {"s"}


def test_lead_zero_carries_only_the_instant_variables(
    template_ds: xr.DataTree,
) -> None:
    """The source publishes a windowed quantity at lead 0 with a zero length window,
    which is not the quantity the variable's comment describes."""
    data_vars = TEMPLATE_CONFIG.data_vars
    coords = coords_for(
        template_ds,
        data_vars,
        init_times=[pd.Timestamp("2024-06-01T00:00")],
        lead_times=[_HOUR_0, pd.Timedelta("3h")],
        ensemble_members=[0],
    )
    by_lead = {c.lead_time: {v.name for v in c.data_vars} for c in coords}

    instant = {v.name for v in data_vars if v.attrs.step_type == "instant"}
    assert by_lead[_HOUR_0] == instant
    assert by_lead[pd.Timedelta("3h")] == {v.name for v in data_vars}
    assert len(instant) == 23


def test_a_variable_is_not_sourced_before_its_available_from(
    template_ds: xr.DataTree,
) -> None:
    """Whether a variable exists is a property of the cycle that wrote the file, and a
    forecast reads only its own cycle, so the gate is on init_time throughout its leads."""
    late = get_var("visibility_surface")
    always = get_var("temperature_2m")
    assert late.internal_attrs.available_from == GEFS_B22_TRANSITION_DATE
    data_vars = [late, always]

    for init_time, expected in (
        (GEFS_B22_TRANSITION_DATE - pd.Timedelta("6h"), {always.name}),
        (GEFS_B22_TRANSITION_DATE, {always.name, late.name}),
    ):
        coords = coords_for(
            template_ds,
            data_vars,
            init_times=[init_time],
            # The longest lead of the earlier cycle is valid well after the transition;
            # the file it reads is still the pre-transition one.
            lead_times=[pd.Timedelta("240h")],
            ensemble_members=[0],
        )
        assert len(coords) == 1, init_time
        assert {v.name for v in coords[0].data_vars} == expected, init_time


def test_representative_var_is_an_era_stable_instant_variable(
    template_ds: xr.DataTree,
) -> None:
    data_vars = TEMPLATE_CONFIG.data_vars
    job = make_job(template_ds, data_vars=data_vars)
    coords = coords_for(
        template_ds,
        data_vars,
        init_times=[pd.Timestamp("2021-01-01T06:00")],
        lead_times=[_HOUR_0, pd.Timedelta("3h"), pd.Timedelta("240h")],
        ensemble_members=[0, 30],
    )
    assert len(coords) == 6

    for coord in coords:
        rep = job.representative_var(coord)
        assert rep in coord.data_vars
        assert rep.internal_attrs.available_from is None
        assert rep.attrs.step_type == "instant"


def test_forecast_discovery_releases_a_partly_published_init(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forecast publishes one lead at a time over ~110 minutes; withholding an init
    until every lead of every member landed would delay its first lead by the length of
    the whole run. So every file the source has published is released immediately, and
    the newest init stays ragged until the run's own polling fills it in.

    The analysis dataset deliberately does the opposite, which is why this is asserted
    on behaviour rather than on discover_available being left unoverridden.
    """
    data_vars = [get_var("temperature_2m")]
    coords = coords_for(
        template_ds,
        data_vars,
        init_times=[pd.Timestamp("2024-06-01T00:00")],
        lead_times=[pd.Timedelta("0h"), pd.Timedelta("3h"), pd.Timedelta("240h")],
        ensemble_members=[0, 30],
    )
    # The source has published the two shortest leads of every member and nothing else.
    published = [c for c in coords if c.lead_time != pd.Timedelta("240h")]
    monkeypatch.setattr(
        noaa_virtual_job_module,
        "discover_available_by_obstore_listing",
        lambda pending, **kwargs: [(c, 100) for c in pending if c in published],
    )
    job = make_job(template_ds, data_vars=data_vars)

    available = job.discover_available(list(coords))

    assert sorted((c.lead_time, c.ensemble_member) for c, _ in available) == sorted(
        (c.lead_time, c.ensemble_member) for c in published
    )


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2024-06-02T00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = (
        NoaaGefsForecast10Day025DegreeVirtualRegionJob.operational_update_jobs(
            primary_store=Mock(),
            tmp_store=Path("unused-tmp.zarr"),
            get_template_fn=TEMPLATE_CONFIG.get_template,
            append_dim="init_time",
            all_data_vars=TEMPLATE_CONFIG.data_vars,
            reformat_job_name="test",
        )
    )

    (job,) = jobs
    assert isinstance(job, NoaaGefsForecast10Day025DegreeVirtualRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    # An 18h window of 6 hourly inits.
    assert job.region == slice(len(init_times) - 3, len(init_times))


def test_file_refs_span_each_message_and_end_at_the_file_end(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = (
        "1:0:d=2024060106:TMP:2 m above ground:9 hour fcst:ENS=perturbed forecast 1\n"
        "2:500:d=2024060106:APCP:surface:6-9 hour acc fcst:ENS=perturbed forecast 1\n"
        "3:900:d=2024060106:RH:2 m above ground:9 hour fcst:ENS=perturbed forecast 1\n"
    )

    stub_grib_source_file_reads(
        monkeypatch, noaa_virtual_job_module, tmp_path, lambda _url: index
    )

    data_vars = [
        get_var("temperature_2m"),
        get_var("total_precipitation_surface"),
        get_var("relative_humidity_2m"),
    ]
    coord = NoaaGefsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("9h"),
        ensemble_member=1,
        source_file_type="s",
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
        dict(r.out_loc)
        == {
            "init_time": pd.Timestamp("2024-06-01T06:00"),
            "lead_time": pd.Timedelta("9h"),
            "ensemble_member": 1,
        }
        for r in refs
    )
    assert all(r.location == coord.get_url() for r in refs)


def test_file_refs_refuses_an_index_missing_a_requested_variable(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """At lead 9 the accumulation window is 6-9, so a 0-9 line is a different message.
    Committing the file anyway would leave a NaN column nothing ever retries."""

    stub_grib_source_file_reads(
        monkeypatch,
        noaa_virtual_job_module,
        tmp_path,
        lambda _url: (
            "1:0:d=2024060106:APCP:surface:0-9 hour acc fcst:ENS=low-res ctl\n"
        ),
        data_file_size=1200,
    )

    data_vars = [get_var("total_precipitation_surface")]
    coord = NoaaGefsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("9h"),
        ensemble_member=0,
        source_file_type="s",
        data_vars=data_vars,
    )
    job = make_job(template_ds, data_vars=data_vars)

    with pytest.raises(AssertionError, match="has no message for"):
        job.file_refs(coord, file_size=1200)


_IDX_FIXTURE_NAME = re.compile(r"^(\d{8}T\d{2})_([cp]\d{2})_f(\d{3})\.idx$")
_IDX_FIXTURES = sorted(
    p.name for p in (Path(__file__).parent / "idx_fixtures").glob("*.idx")
)
# An empty parametrize set skips rather than fails, which would silently retire the
# only check that compares this catalog against real archived indexes.
assert len(_IDX_FIXTURES) == 20, _IDX_FIXTURES
assert all(_IDX_FIXTURE_NAME.match(name) for name in _IDX_FIXTURES), _IDX_FIXTURES


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
    other three at 2022-10-18T12), cover lead 0, the 3 hour and 6 hour accumulation
    windows and the 240 hour end of the s file, and include perturbed members as well
    as the control.
    """
    fixture = Path(__file__).parent / "idx_fixtures" / fixture_name
    match = _IDX_FIXTURE_NAME.match(fixture_name)
    assert match is not None
    init_time = pd.Timestamp(match.group(1))
    ensemble_member = int(match.group(2)[1:])
    lead_time = pd.Timedelta(hours=int(match.group(3)))
    index_text = fixture.read_text()

    stub_grib_source_file_reads(
        monkeypatch, noaa_virtual_job_module, tmp_path, lambda _url: fixture
    )

    data_vars = TEMPLATE_CONFIG.data_vars
    (coord,) = coords_for(
        template_ds,
        data_vars,
        init_times=[init_time],
        lead_times=[lead_time],
        ensemble_members=[ensemble_member],
    )
    assert coord.get_url().endswith(
        f"{'gec' if ensemble_member == 0 else 'gep'}{ensemble_member:02}"
        f".t{init_time:%H}z.pgrb2s.0p25.f{int(lead_time.total_seconds() // 3600):03d}"
    )

    last_start = max(int(line.split(":")[1]) for line in index_text.splitlines())
    job = make_job(template_ds, data_vars=data_vars)
    refs = job.file_refs(coord, file_size=last_start + 1_000_000)

    # file_refs asserts nothing was requested but absent; this pins the other direction,
    # that every requested variable got exactly one ref.
    assert sorted(r.data_var.name for r in refs) == sorted(
        v.name for v in coord.data_vars
    )
    assert len({r.data_var.name for r in refs}) == len(refs)

    # Both sides above come from the same available_from filter, so together they would
    # still hold if a variable stopped being requested in an era that publishes it.
    # Count what the file itself offers instead: every message except the two kinds this
    # dataset deliberately declines -- surface geopotential height, which the s file
    # carries only at lead 0, and the degenerate zero length TMAX/TMIN window there.
    messages = [line.split(":") for line in index_text.splitlines() if line]
    declined = [
        fields
        for fields in messages
        if (fields[3] == "HGT" and fields[4] == "surface")
        or (fields[3] in ("TMAX", "TMIN") and fields[5].startswith("0-0 "))
    ]
    assert len(refs) == len(messages) - len(declined), (
        f"{fixture_name}: {len(refs)} refs for {len(messages)} messages "
        f"less {len(declined)} declined"
    )


def test_a_windowed_variable_added_late_is_gated_on_its_own_cycle(
    template_ds: xr.DataTree,
) -> None:
    """No shipped variable is both windowed and late added, so this constructs the
    combination: the gate must read init_time, never the valid time a long lead reaches."""
    windowed = get_var("total_precipitation_surface")
    assert windowed.internal_attrs.available_from is None
    late_windowed = replace(
        windowed,
        internal_attrs=replace(
            windowed.internal_attrs, available_from=GEFS_B22_TRANSITION_DATE
        ),
    )

    before = coords_for(
        template_ds,
        [late_windowed],
        init_times=[GEFS_B22_TRANSITION_DATE - pd.Timedelta("6h")],
        lead_times=[pd.Timedelta("240h")],
        ensemble_members=[0],
    )
    assert before == []

    at_transition = coords_for(
        template_ds,
        [late_windowed],
        init_times=[GEFS_B22_TRANSITION_DATE],
        lead_times=[pd.Timedelta("3h")],
        ensemble_members=[0],
    )
    assert [(c.init_time, c.lead_time) for c in at_transition] == [
        (GEFS_B22_TRANSITION_DATE, pd.Timedelta("3h"))
    ]
