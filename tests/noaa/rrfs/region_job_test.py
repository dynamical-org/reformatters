import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from reformatters.common.pydantic import replace
from reformatters.common.types import DatetimeLike
from reformatters.noaa.noaa_grib_index import parse_grib_index_lines
from reformatters.noaa.rrfs.forecast_18_hour_virtual.region_job import (
    NoaaRrfsForecast18HourVirtualRegionJob,
)
from reformatters.noaa.rrfs.forecast_18_hour_virtual.template_config import (
    NoaaRrfsForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.region_job import (
    NoaaRrfsForecast84HourVirtualRegionJob,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.template_config import (
    NoaaRrfsForecast84HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.region_job import (
    NoaaRrfsForecastSubHourlyVirtualRegionJob,
    NoaaRrfsSubHourlySourceFileCoord,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.template_config import (
    NoaaRrfsForecastSubHourlyVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.models import RrfsSourceFamily
from reformatters.noaa.rrfs.region_job import NoaaRrfsRegionJob, NoaaRrfsSourceFileCoord
from reformatters.noaa.rrfs.template_config import NoaaRrfsForecastTemplateConfig
from reformatters.noaa.rrfs_ens.forecast_virtual.region_job import (
    NoaaRrfsEnsForecastVirtualRegionJob,
    NoaaRrfsEnsSourceFileCoord,
)
from reformatters.noaa.rrfs_ens.forecast_virtual.template_config import (
    NoaaRrfsEnsForecastVirtualTemplateConfig,
)

FIXTURES = Path(__file__).parent / "fixtures"
CONFIGS = (
    NoaaRrfsForecast84HourVirtualTemplateConfig(),
    NoaaRrfsForecast18HourVirtualTemplateConfig(),
    NoaaRrfsForecastSubHourlyVirtualTemplateConfig(),
    NoaaRrfsEnsForecastVirtualTemplateConfig(),
)


def metadata_tree(config: NoaaRrfsForecastTemplateConfig) -> xr.DataTree:
    nodes: dict[str, xr.Dataset] = {
        "/": xr.Dataset(attrs={"dataset_id": config.dataset_id})
    }
    for group, levels in config._vertical_dimension_coordinates().items():
        nodes[group] = xr.Dataset(
            {
                v.name: ((group,), np.zeros(len(levels)))
                for v in config.data_vars
                if v.group == group
            },
            coords={group: levels},
        )
    return xr.DataTree.from_dict(nodes)


def make_job(config: NoaaRrfsForecastTemplateConfig) -> NoaaRrfsRegionJob:
    job_class = {
        CONFIGS[0].dataset_id: NoaaRrfsForecast84HourVirtualRegionJob,
        CONFIGS[1].dataset_id: NoaaRrfsForecast18HourVirtualRegionJob,
        CONFIGS[2].dataset_id: NoaaRrfsForecastSubHourlyVirtualRegionJob,
        CONFIGS[3].dataset_id: NoaaRrfsEnsForecastVirtualRegionJob,
    }[config.dataset_id]
    return job_class(
        tmp_store=Path("unused"),
        template_ds=metadata_tree(config),
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode="backfill",
    )


@pytest.mark.parametrize(
    "config", [CONFIGS[0], CONFIGS[2], CONFIGS[3]], ids=lambda c: c.dataset_id
)
def test_every_census_message_is_referenced_or_explicitly_accounted_for(
    config: NoaaRrfsForecastTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = make_job(config)
    pattern = "rrfsens.*/12/m001/*.idx" if config.members else "rrfs.*/00/*.idx"
    paths = [
        p for p in FIXTURES.glob(pattern) if ("subh" in p.name) == config.sub_hourly
    ]
    assert len(paths) == (2 if config.sub_hourly else 6 if config.members else 9)
    for path in paths:
        hour = int(path.name.split(".f")[1][:3])
        lead = pd.Timedelta(hours=hour)
        family = (
            "subh"
            if config.sub_hourly
            else "prslev"
            if "prslev" in path.name
            else "2dfld"
        )
        variables = [
            v
            for v in config.data_vars
            if v.internal_attrs.source_family == family and v.available_at(lead)
        ]
        init = (
            pd.Timestamp(path.parts[-4].split(".")[1]) + pd.Timedelta(hours=12)
            if config.members
            else pd.Timestamp(path.parts[-3].split(".")[1])
        )
        coord: NoaaRrfsSourceFileCoord
        if config.members:
            coord = NoaaRrfsEnsSourceFileCoord(
                init_time=init,
                lead_time=lead,
                source_family=family,
                ensemble_member=1,
                data_vars=variables,
            )
        elif config.sub_hourly:
            coord = NoaaRrfsSubHourlySourceFileCoord(
                init_time=init, lead_time=lead, data_vars=variables
            )
        else:
            coord = NoaaRrfsSourceFileCoord(
                init_time=init,
                lead_time=lead,
                source_family=family,
                data_vars=variables,
            )
        lines = parse_grib_index_lines(path)
        local = tmp_path / "index.idx"
        local.write_bytes(path.read_bytes())
        monkeypatch.setattr(
            "reformatters.noaa.noaa_virtual_region_job.s3_download_to_disk",
            lambda *a, local=local, **k: local,
        )
        monkeypatch.setattr(NoaaRrfsRegionJob, "grib_message_length_at", lambda *a: 217)
        refs = job.file_refs(coord, lines[-1][0] + 10_000_000)
        assert refs
        used_offsets = {r.offset for r in refs}
        seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
        for offset, element, level, window, selectors in lines:
            identity = (element, level, window, selectors)
            if offset not in used_offsets:
                duplicate = identity in seen and (
                    element == "TSNOWP" or "parmcat=1 parm=19" in element
                )
                zero_placeholder = (
                    "parmcat=1 parm=19" in element
                    and "-" in window
                    and len(set(window.split()[0].split("-"))) == 1
                )
                # These surface fields are effectively empty in NOAA's source as
                # of 2026-10-09; RRFS is not yet operational, so this may change.
                # specific_humidity_2m is present and remains in the dataset.
                excluded_surface_product = (
                    not config.members
                    and not config.sub_hourly
                    and level == "surface"
                    and element in {"SPFH", "VEGMIN", "VEGMAX", "PEVAP", "PEVPR"}
                )
                # These ensemble fields are effectively empty in NOAA's source
                # as of 2026-10-09; RRFS is not yet operational, so this may change.
                excluded_member_product = config.members and (
                    (element, level)
                    in {
                        ("AOTK", "entire atmosphere (considered as a single layer)"),
                        (
                            "var discipline=2 master_table=2 parmcat=4 parm=26",
                            "surface",
                        ),
                        ("WFIREPOT", "surface"),
                    }
                )
                # These fields are not usable as of 2026-10-09; RRFS is not yet
                # operational, so this may change.
                excluded_radiation_or_shear = (
                    element == "ULWRF"
                    and level == "top of atmosphere"
                    and (
                        window == "anl"
                        or re.fullmatch(r"\d+ (?:hour|min) fcst", window) is not None
                    )
                ) or (
                    element in {"VUCSH", "VVCSH"} and level == "0-1000 m above ground"
                )
                # These FFG flags are constant zero in NOAA's source as of
                # 2026-10-09; RRFS is not yet operational, so this may change.
                excluded_ffg = (
                    not config.members
                    and not config.sub_hourly
                    and family == "2dfld"
                    and element
                    in {
                        "var discipline=1 center=7 local_table=1 parmcat=1 parm=197",
                        "QPFFFG",
                    }
                    and level == "surface"
                    and selectors == ("prob >1", "prob fcst 0/1")
                    and re.fullmatch(r"\d+-\d+ (?:hour|day) acc fcst", window)
                    is not None
                )
                # This field is not usable as of 2026-10-09; RRFS is not yet
                # operational, so this may change.
                excluded_lifted_index = (
                    not config.members
                    and not config.sub_hourly
                    and family == "2dfld"
                    and element == "LFTX"
                    and level == "500-1000 mb"
                    and not selectors
                    and (
                        window == "anl"
                        or re.fullmatch(r"\d+ hour fcst", window) is not None
                    )
                )
                # This field's data duplicates PRES at cloud top as of 2026-10-13;
                # RRFS is not yet operational, so this may change.
                excluded_cloud_top_duplicate = (
                    not config.members
                    and not config.sub_hourly
                    and family == "2dfld"
                    and element == "PRES"
                    and level == "grid scale cloud top level"
                )
                retired_accumulation = (
                    family in {"2dfld", "subh"}
                    and element in {"FRZR", "FROZR"}
                    and level == "surface"
                )
                assert (
                    retired_accumulation
                    or duplicate
                    or zero_placeholder
                    or excluded_surface_product
                    or excluded_member_product
                    or excluded_radiation_or_shear
                    or excluded_ffg
                    or excluded_lifted_index
                    or excluded_cloud_top_duplicate
                ), (path, identity)
            seen.add(identity)


def test_sub_hourly_file_contributes_four_precise_leads() -> None:
    coord = NoaaRrfsSubHourlySourceFileCoord(
        init_time=pd.Timestamp("2026-09-15T00"),
        lead_time=pd.Timedelta("2h"),
        source_family="subh",
        data_vars=[],
    )
    assert coord.message_lead_times() == tuple(
        pd.Timedelta(minutes=m) for m in (75, 90, 105, 120)
    )
    assert coord.get_url().endswith(
        "rrfs.20260915/00/rrfs.t00z.2dfld.3km.subh.f002.conus.grib2"
    )


def test_member_selector_stripping_preserves_process_and_aerosol_selectors() -> None:
    coord = NoaaRrfsEnsSourceFileCoord(
        init_time=pd.Timestamp("2026-09-15T12"),
        lead_time=pd.Timedelta("6h"),
        source_family="2dfld",
        ensemble_member=5,
        data_vars=[],
    )
    assert coord.index_selectors(("ENS=+5", "process=193", "aerosol=Dust dry")) == (
        "process=193",
        "aerosol=Dust dry",
    )
    assert coord.get_url().endswith(
        "rrfsens.20260915/12/m005/rrfs.t12z.m005.2dfldnomads.3km.f006.conus.grib2"
    )
    with pytest.raises(AssertionError):
        coord.index_selectors(("ENS=+1",))


def test_missing_required_messages_warn(caplog: pytest.LogCaptureFixture) -> None:
    config = CONFIGS[0]
    var = next(v for v in config.data_vars if v.name == "temperature_2m")
    coord = NoaaRrfsSourceFileCoord(
        init_time=pd.Timestamp("2026-10-15T12"),
        lead_time=pd.Timedelta("2h"),
        source_family="2dfld",
        data_vars=[var],
    )
    with caplog.at_level("WARNING"):
        make_job(config)._check_refs_complete(coord, [])
    (warning,) = caplog.records
    assert warning.levelname == "WARNING"
    assert coord.get_url() in warning.message
    assert "GRIB messages absent from index" in warning.message
    assert "temperature_2m" in warning.message
    assert "ingesting available messages" in warning.message


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.dataset_id)
def test_post_retirement_index_contains_all_configured_messages(
    config: NoaaRrfsForecastTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    filename = (
        "rrfsens-20261015-12-m003-f001-conus.idx"
        if config.members
        else "rrfs-20261015-12-subh-f001-conus.idx"
        if config.sub_hourly
        else "rrfs-20261015-12-2dfld-f001-conus.idx"
    )
    path = FIXTURES / "missing_messages" / filename
    lead = pd.Timedelta("1h")
    job = make_job(config)
    variables = [
        v
        for v in config.data_vars
        if v.internal_attrs.source_family == ("subh" if config.sub_hourly else "2dfld")
        and v.available_at(lead)
    ]
    (coord,) = job.generate_source_file_coords(
        xr.Dataset(
            coords={
                "init_time": [pd.Timestamp("2026-10-15T12")],
                "lead_time": [lead],
                "ensemble_member": [3],
            }
        ),
        variables,
    )
    local = tmp_path / "index.idx"
    monkeypatch.setattr(
        "reformatters.noaa.noaa_virtual_region_job.s3_download_to_disk",
        lambda *a, **k: local,
    )
    monkeypatch.setattr(NoaaRrfsRegionJob, "grib_message_length_at", lambda *a: 217)
    lines = parse_grib_index_lines(path)
    assert not {"FRZR", "FROZR"} & {line[1] for line in lines}
    file_size = lines[-1][0] + 1000
    local.write_bytes(path.read_bytes())
    with caplog.at_level("WARNING"):
        refs = job.file_refs(coord, file_size)
    assert refs
    assert not any(
        v.internal_attrs.grib_element in {"FRZR", "FROZR"} for v in variables
    )
    present_paths = {ref.data_var.path for ref in refs}
    assert present_paths == {v.path for v in variables}
    assert job.representative_var(coord).path in present_paths
    expected_leads = (
        set(pd.to_timedelta([15, 30, 45, 60], unit="min"))
        if config.sub_hourly
        else {lead}
    )
    assert {(ref.data_var.path, ref.out_loc["lead_time"]) for ref in refs} == {
        (var_path, lead_time)
        for var_path in present_paths
        for lead_time in expected_leads
    }
    assert not caplog.records


def test_supported_leads_are_explicit_for_binary_run_totals() -> None:
    variables = CONFIGS[0].data_vars
    ari = next(
        v
        for v in variables
        if v.name
        == "categorical_precipitation_exceeding_2_year_average_recurrence_interval_run_total_surface"
    )
    assert [h for h in range(85) if ari.available_at(pd.Timedelta(hours=h))] == [
        1,
        3,
        6,
        12,
        24,
    ]
    assert ari.attrs.flag_values == (0, 1)
    assert ari.attrs.flag_meanings == "no yes"
    assert ari.attrs.units == "1"
    assert "guidance is unavailable" in (ari.attrs.comment or "")


def test_exact_selector_absence_and_wildcard_have_distinct_lookup_keys() -> None:
    config = CONFIGS[0]
    var = next(v for v in config.data_vars if v.name == "temperature_2m")
    job = make_job(config)
    assert ("TMP", "2 m above ground", "2 hour fcst", ()) in job._message_lookup(
        [var], 2
    )
    wildcard = replace(
        var, internal_attrs=replace(var.internal_attrs, grib_index_selectors=None)
    )
    assert ("TMP", "2 m above ground", "2 hour fcst", None) in job._message_lookup(
        [wildcard], 2
    )
    process = replace(
        var,
        internal_attrs=replace(
            var.internal_attrs, grib_index_selectors=("process=193",)
        ),
    )
    assert (
        "TMP",
        "2 m above ground",
        "2 hour fcst",
        ("process=193",),
    ) in job._message_lookup([process], 2)


@pytest.mark.parametrize("config", CONFIGS[:2], ids=lambda c: c.dataset_id)
def test_hourly_source_generation_respects_families_availability_and_horizon(
    config: NoaaRrfsForecastTemplateConfig,
) -> None:
    init = config.append_dim_start
    leads = [pd.Timedelta(0), pd.Timedelta("1h"), config.forecast_length]
    ds = xr.Dataset(
        coords={
            "init_time": [init, init + config.append_dim_frequency],
            "lead_time": leads,
        }
    )
    variables = [
        v
        for v in config.data_vars
        if v.path
        in {
            "temperature_2m",
            "total_precipitation_surface",
            "categorical_precipitation_exceeding_2_year_average_recurrence_interval_run_total_surface",
            "pressure_level/temperature",
        }
    ]
    coords = make_job(config).generate_source_file_coords(ds, variables)
    assert len(coords) == 12
    assert {(c.init_time, c.lead_time, c.source_family) for c in coords} == {
        (time, lead, family)
        for time in (init, init + config.append_dim_frequency)
        for lead in leads
        for family in ("2dfld", "prslev")
    }
    for coord in coords:
        assert "ensemble_member" not in coord.out_loc()
        assert coord.message_lead_times() == (coord.lead_time,)
        names = {v.path for v in coord.data_vars}
        assert names == (
            {"pressure_level/temperature"}
            if coord.source_family == "prslev"
            else {"temperature_2m"}
            if coord.lead_time == pd.Timedelta(0)
            else {
                "temperature_2m",
                "total_precipitation_surface",
                "categorical_precipitation_exceeding_2_year_average_recurrence_interval_run_total_surface",
            }
            if coord.lead_time == pd.Timedelta("1h")
            else {"temperature_2m", "total_precipitation_surface"}
        )
        assert coord.get_url().endswith(
            f"f{int(coord.lead_time / pd.Timedelta('1h')):03}.conus.grib2"
        )


def test_subhourly_source_generation_deduplicates_file_hours() -> None:
    config = CONFIGS[2]
    ds = xr.Dataset(
        coords={
            "init_time": [config.append_dim_start],
            "lead_time": pd.to_timedelta([15, 60, 75, 1080], unit="min"),
        }
    )
    variables = [v for v in config.data_vars if v.name == "temperature_2m"]
    coords = make_job(config).generate_source_file_coords(ds, variables)
    assert all(isinstance(c, NoaaRrfsSubHourlySourceFileCoord) for c in coords)
    assert [c.lead_time for c in coords] == list(pd.to_timedelta([1, 2, 18], unit="h"))
    assert all(len(c.message_lead_times()) == 4 for c in coords)
    assert coords[0].message_lead_times()[0] == pd.Timedelta("15min")
    assert coords[-1].message_lead_times()[-1] == config.forecast_length


def test_member_source_generation_routes_control_and_perturbed_members() -> None:
    config = CONFIGS[3]
    ds = xr.Dataset(
        coords={
            "init_time": [config.append_dim_start],
            "lead_time": [pd.Timedelta(0), config.forecast_length],
            "ensemble_member": [0, 1, 5],
        }
    )
    variables = [
        v
        for v in config.data_vars
        if v.path
        in {
            "temperature_2m",
            "total_precipitation_run_total_surface",
            "pressure_level/temperature",
        }
    ]
    coords = make_job(config).generate_source_file_coords(ds, variables)
    assert len(coords) == 12
    assert all(isinstance(c, NoaaRrfsEnsSourceFileCoord) for c in coords)
    for coord in coords:
        assert isinstance(coord, NoaaRrfsEnsSourceFileCoord)
        member = coord.ensemble_member
        assert coord.out_loc()["ensemble_member"] == member
        if member == 0:
            assert coord.get_url().startswith("s3://noaa-rrfs-ops-pds/rrfs.")
            assert coord.index_selectors(("process=193",)) == ("process=193",)
        else:
            assert f"/m{member:03}/" in coord.get_url()
            assert f".{coord.source_family}nomads." in coord.get_url()
            assert coord.index_selectors((f"ENS=+{member}", "process=193")) == (
                "process=193",
            )
        if coord.source_family == "2dfld":
            assert {v.path for v in coord.data_vars} == (
                {"temperature_2m"}
                if coord.lead_time == pd.Timedelta(0)
                else {"temperature_2m", "total_precipitation_run_total_surface"}
            )


@pytest.mark.parametrize("scheduled", [True, False])
def test_subhourly_update_selects_through_the_due_init(
    scheduled: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fire = pd.Timestamp("2026-10-02T01:15")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fire))

    def get_template(end: DatetimeLike) -> xr.DataTree:
        assert end == pd.Timestamp("2026-10-02T00:15")
        return xr.DataTree(
            xr.Dataset(
                coords={
                    "init_time": pd.date_range(
                        "2026-10-01", end, freq="1h", inclusive="left"
                    ),
                    "lead_time": pd.timedelta_range("15min", "18h", freq="15min"),
                }
            )
        )

    jobs, _ = NoaaRrfsForecastSubHourlyVirtualRegionJob.operational_update_jobs(
        MemoryStore(),
        tmp_path,
        get_template,
        "init_time",
        [],
        "test",
        job_fire_time=fire if scheduled else None,
    )
    (job,) = jobs
    assert isinstance(job, NoaaRrfsForecastSubHourlyVirtualRegionJob)
    assert job.template_ds.init_time.values[-1] == pd.Timestamp("2026-10-02T00:00")
    assert job.processing_mode == "update"
    assert job.region == slice(7, 25)


@pytest.mark.parametrize("family", ["2dfld", "prslev"])
def test_member_control_source_has_all_required_fields(
    family: RrfsSourceFamily, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = CONFIGS[3]
    job = make_job(config)
    paths = list(FIXTURES.glob(f"rrfs.*/00/*.{family}.3km.f*.idx"))
    assert paths
    for path in paths:
        lead = pd.Timedelta(hours=int(path.name.split(".f")[1][:3]))
        coord = NoaaRrfsEnsSourceFileCoord(
            init_time=pd.Timestamp(path.parts[-3].split(".")[1]),
            lead_time=lead,
            source_family=family,
            ensemble_member=0,
            data_vars=[
                v
                for v in config.data_vars
                if v.internal_attrs.source_family == family and v.available_at(lead)
            ],
        )
        local = tmp_path / path.name
        local.write_bytes(path.read_bytes())
        monkeypatch.setattr(
            "reformatters.noaa.noaa_virtual_region_job.s3_download_to_disk",
            lambda *a, local=local, **k: local,
        )
        monkeypatch.setattr(NoaaRrfsRegionJob, "grib_message_length_at", lambda *a: 217)
        lines = parse_grib_index_lines(path)
        refs = job.file_refs(coord, lines[-1][0] + 10_000_000)
        assert refs
        assert all(ref.out_loc["ensemble_member"] == 0 for ref in refs)
