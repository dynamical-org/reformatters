import itertools
import re
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.config_models import ROOT
from reformatters.noaa import noaa_virtual_region_job as noaa_virtual_job_module
from reformatters.noaa.gefs.forecast_16_day_0_5_degree_virtual.region_job import (
    NoaaGefsForecast16Day05DegreeVirtualRegionJob,
)
from reformatters.noaa.gefs.forecast_16_day_0_5_degree_virtual.template_config import (
    NoaaGefsForecast16Day05DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFSSourceFileType,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.gefs.virtual_region_job import (
    NoaaGefsForecastVirtualSourceFileCoord,
)

TEMPLATE_CONFIG = NoaaGefsForecast16Day05DegreeVirtualTemplateConfig()
_HOUR_0 = pd.Timedelta(0)


def get_var(path: str) -> NoaaGefsVirtualDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.path == path)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2024-06-02T00:00"))


def make_job(
    template_ds: xr.DataTree,
    data_vars: Sequence[NoaaGefsVirtualDataVar],
    region: slice = slice(0, 1),
) -> NoaaGefsForecast16Day05DegreeVirtualRegionJob:
    return NoaaGefsForecast16Day05DegreeVirtualRegionJob(
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
@pytest.mark.parametrize("source_file_type", ["a", "b"])
def test_source_file_coord_url_names_the_member_and_product(
    ensemble_member: int, member_str: str, source_file_type: GEFSSourceFileType
) -> None:
    """The control member is published as gec00 and the perturbed ones as gepNN; both
    0.5 degree products live beside each other under the cycle."""
    coord = NoaaGefsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("384h"),
        ensemble_member=ensemble_member,
        source_file_type=source_file_type,
        data_vars=[get_var("temperature_2m")],
    )
    assert coord.get_url() == (
        f"s3://noaa-gefs-pds/gefs.20240601/06/atmos/pgrb2{source_file_type}p5/"
        f"{member_str}.t06z.pgrb2{source_file_type}.0p50.f384"
    )
    assert coord.get_index_url() == coord.get_url() + ".idx"


def test_every_init_lead_member_triple_gets_one_file_per_product(
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

    keys = [(c.init_time, c.lead_time, c.ensemble_member, c.source_file_type) for c in coords]  # fmt: skip
    assert len(keys) == len(set(keys))
    assert set(keys) == set(
        itertools.product(init_times, lead_times, members, ("a", "b"))
    )
    assert len(keys) == 2 * 105 * 31 * 2


def test_a_pressure_variable_is_requested_from_both_products(
    template_ds: xr.DataTree,
) -> None:
    """The products split its levels, so it must be asked of both; asking only one
    would leave the other half of the column empty."""
    both = get_var("pressure_level/temperature")
    b_only = get_var("pressure_level/specific_humidity")
    a_only = get_var("maximum_temperature_2m")

    coords = coords_for(
        template_ds,
        [both, b_only, a_only],
        init_times=[pd.Timestamp("2024-06-01T00:00")],
        lead_times=[pd.Timedelta("3h")],
        ensemble_members=[0],
    )
    by_file = {c.source_file_type: {v.path for v in c.data_vars} for c in coords}
    assert by_file == {
        "a": {both.path, a_only.path},
        "b": {both.path, b_only.path},
    }


def test_lead_zero_carries_only_the_variables_the_source_publishes_there(
    template_ds: xr.DataTree,
) -> None:
    """The source publishes a windowed quantity at lead 0 with a zero length window,
    which is not the quantity the variable's comment describes, and omits three
    convective cloud fields entirely."""
    data_vars = TEMPLATE_CONFIG.data_vars
    coords = coords_for(
        template_ds,
        data_vars,
        init_times=[pd.Timestamp("2024-06-01T00:00")],
        lead_times=[_HOUR_0, pd.Timedelta("3h")],
        ensemble_members=[0],
    )
    at_lead_0 = {v.path for c in coords if c.lead_time == _HOUR_0 for v in c.data_vars}
    at_lead_3 = {
        v.path for c in coords if c.lead_time == pd.Timedelta("3h") for v in c.data_vars
    }

    assert at_lead_0 == {v.path for v in data_vars if v.has_hour_0_values()}
    assert at_lead_3 == {v.path for v in data_vars}
    assert at_lead_3 - at_lead_0 == {
        v.path for v in data_vars if not v.has_hour_0_values()
    }


def test_representative_var_is_a_root_instant_variable(
    template_ds: xr.DataTree,
) -> None:
    """The products partition every vertical group's levels, so a group variable's
    probe cell can be a level the file never carries and the filter would re-ingest
    that file forever."""
    data_vars = TEMPLATE_CONFIG.data_vars
    job = make_job(template_ds, data_vars=data_vars)
    coords = coords_for(
        template_ds,
        data_vars,
        init_times=[pd.Timestamp("2021-01-01T06:00")],
        lead_times=[_HOUR_0, pd.Timedelta("3h"), pd.Timedelta("384h")],
        ensemble_members=[0, 30],
    )
    assert len(coords) == 12

    for coord in coords:
        rep = job.representative_var(coord)
        assert rep in coord.data_vars
        assert rep.group is ROOT
        assert rep.attrs.step_type == "instant"
        assert rep.internal_attrs.available_from is None
        assert dict(job.representative_probe_loc(coord, rep)) == dict(coord.out_loc())


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2024-06-02T00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = (
        NoaaGefsForecast16Day05DegreeVirtualRegionJob.operational_update_jobs(
            primary_store=Mock(),
            tmp_store=Path("unused-tmp.zarr"),
            get_template_fn=TEMPLATE_CONFIG.get_template,
            append_dim="init_time",
            all_data_vars=TEMPLATE_CONFIG.data_vars,
            reformat_job_name="test",
        )
    )

    (job,) = jobs
    assert isinstance(job, NoaaGefsForecast16Day05DegreeVirtualRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    # An 18h window of 6 hourly inits.
    assert job.region == slice(len(init_times) - 3, len(init_times))


def test_file_refs_refuses_an_index_missing_a_requested_variable(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """At lead 9 the accumulation window is 6-9, so a 0-9 line is a different message.
    Committing the file anyway would leave a NaN column nothing ever retries."""

    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        path = tmp_path / "index.idx"
        path.write_text(
            "1:0:d=2024060106:APCP:surface:0-9 hour acc fcst:ENS=low-res ctl\n"
        )
        return path

    monkeypatch.setattr(noaa_virtual_job_module, "s3_download_to_disk", fake_download)

    data_vars = [get_var("total_precipitation_surface")]
    coord = NoaaGefsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("9h"),
        ensemble_member=0,
        source_file_type="a",
        data_vars=data_vars,
    )
    job = make_job(template_ds, data_vars=data_vars)

    with pytest.raises(AssertionError, match="has no message for"):
        job.file_refs(coord, file_size=1200)


def test_a_vertical_group_variable_fills_one_chunk_per_level(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The group's chunk is one level deep, so a level's message must land at that
    level's index rather than at the first one."""
    index = (
        "1:0:d=2024060106:TMP:850 mb:9 hour fcst:ENS=perturbed forecast 1\n"
        "2:500:d=2024060106:TMP:500 mb:9 hour fcst:ENS=perturbed forecast 1\n"
    )

    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        path = tmp_path / "index.idx"
        path.write_text(index)
        return path

    monkeypatch.setattr(noaa_virtual_job_module, "s3_download_to_disk", fake_download)

    data_vars = [get_var("pressure_level/temperature")]
    coord = NoaaGefsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("9h"),
        ensemble_member=1,
        source_file_type="a",
        data_vars=data_vars,
    )
    job = make_job(template_ds, data_vars=data_vars)

    refs = job.file_refs(coord, file_size=1200)

    assert [
        (r.data_var.path, dict(r.out_loc)["pressure_level"], r.offset, r.length)
        for r in refs
    ] == [
        ("pressure_level/temperature", 850.0, 0, 500),
        ("pressure_level/temperature", 500.0, 500, 700),
    ]


_IDX_FIXTURE_NAME = re.compile(r"^(\d{8}T\d{2})_([cp]\d{2})_([ab])_f(\d{3})\.idx$")
_IDX_FIXTURES = sorted(
    p.name for p in (Path(__file__).parent / "idx_fixtures").glob("*.idx")
)
# An empty parametrize set skips rather than fails, which would silently retire the
# only check that compares this catalog against real archived indexes.
assert len(_IDX_FIXTURES) == 18, _IDX_FIXTURES
assert all(_IDX_FIXTURE_NAME.match(name) for name in _IDX_FIXTURES), _IDX_FIXTURES

# The two messages this catalog declines wherever they appear: surface geopotential
# height, which no single product carries across the whole lead axis, and ozone at
# 125 mb, the one isobaric level outside the 31 the pressure_level dimension spans.
_DECLINED = (("HGT", "surface"), ("O3MR", "125 mb"))


@pytest.mark.parametrize("fixture_name", _IDX_FIXTURES)
def test_every_requested_variable_maps_to_a_real_message(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture_name: str,
) -> None:
    """Run the real coord generation and ref building against real archived indexes.

    Catches an element, level or window string that is wrong for any of the 268
    variables, which declaration-derived tests cannot: they compare the config to
    itself. The fixtures cover both products, both spellings of the cloud mixing ratio
    element (CLWMR before 2025-12-19, CLMR after), lead 0, the 3 and 6 hour windows,
    the 240 hour step up in lead spacing and the 384 hour end, and perturbed members
    as well as the control.
    """
    fixture = Path(__file__).parent / "idx_fixtures" / fixture_name
    match = _IDX_FIXTURE_NAME.match(fixture_name)
    assert match is not None
    init_time = pd.Timestamp(match.group(1))
    ensemble_member = int(match.group(2)[1:])
    source_file_type = match.group(3)
    lead_time = pd.Timedelta(hours=int(match.group(4)))
    index_text = fixture.read_text()

    # file_refs unlinks the index it is given, so hand it a copy rather than the fixture.
    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        copy = tmp_path / fixture_name
        copy.write_text(index_text)
        return copy

    monkeypatch.setattr(noaa_virtual_job_module, "s3_download_to_disk", fake_download)

    data_vars = TEMPLATE_CONFIG.data_vars
    (coord,) = [
        coord
        for coord in coords_for(
            template_ds,
            data_vars,
            init_times=[init_time],
            lead_times=[lead_time],
            ensemble_members=[ensemble_member],
        )
        if coord.source_file_type == source_file_type
    ]
    assert coord.get_url().endswith(
        f"{'gec' if ensemble_member == 0 else 'gep'}{ensemble_member:02}"
        f".t{init_time:%H}z.pgrb2{source_file_type}.0p50"
        f".f{int(lead_time.total_seconds() // 3600):03d}"
    )

    last_start = max(int(line.split(":")[1]) for line in index_text.splitlines())
    job = make_job(template_ds, data_vars=data_vars)
    refs = job.file_refs(coord, file_size=last_start + 1_000_000)

    # file_refs asserts nothing was requested but absent; this pins the other direction,
    # that every message the file offers was taken exactly once. Counting the file's own
    # messages rather than the requested variables keeps both sides from moving together
    # if a variable silently stopped being requested in an era that publishes it.
    messages = [line.split(":") for line in index_text.splitlines() if line]
    declined = [
        fields
        for fields in messages
        if (fields[3], fields[4]) in _DECLINED or fields[5].startswith("0-0 ")
    ]
    assert len(refs) == len(messages) - len(declined), (
        f"{fixture_name}: {len(refs)} refs for {len(messages)} messages "
        f"less {len(declined)} declined"
    )
    cells = [(r.data_var.path, tuple(sorted(dict(r.out_loc).items()))) for r in refs]
    assert len(set(cells)) == len(cells)
