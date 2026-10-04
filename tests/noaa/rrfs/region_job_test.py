from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.noaa.noaa_grib_index import parse_grib_index_lines
from reformatters.noaa.rrfs.forecast_18_hour_virtual.template_config import (
    NoaaRrfsForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.template_config import (
    NoaaRrfsForecast84HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.template_config import (
    NoaaRrfsForecastSubHourlyVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.region_job import NoaaRrfsRegionJob, NoaaRrfsSourceFileCoord
from reformatters.noaa.rrfs.template_config import NoaaRrfsForecastTemplateConfig
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
    return NoaaRrfsRegionJob(
        tmp_store=Path("unused"),
        template_ds=metadata_tree(config),
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode="backfill",
    )


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.dataset_id)
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
    assert len(paths) >= 3
    declared_deferred: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
    seen_deferred: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
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
        coord = NoaaRrfsSourceFileCoord(
            init_time=pd.Timestamp(path.parts[-4].split(".")[1])
            + pd.Timedelta(hours=12)
            if config.members
            else pd.Timestamp(path.parts[-3].split(".")[1]),
            lead_time=lead,
            source_family=family,
            ensemble_member=1 if config.members else None,
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
        instant = "anl" if hour == 0 else f"{hour} hour fcst"
        if family != "2dfld":
            deferred = set()
        elif config.members:
            deferred = {
                (
                    "2dfld",
                    "AOTK",
                    "entire atmosphere (considered as a single layer)",
                    instant,
                    ("ENS=+1",),
                ),
                (
                    "2dfld",
                    "var discipline=2 master_table=2 parmcat=4 parm=26",
                    "surface",
                    "0-0 day ave fcst"
                    if hour == 0
                    else f"{hour - 1}-{hour} hour ave fcst",
                    ("ENS=+1",),
                ),
            }
        else:
            deferred = {
                ("2dfld", "SPFH", "surface", instant, ()),
                ("2dfld", "PEVPR", "surface", instant, ()),
            }
            if hour == 0:
                deferred.update(
                    {
                        ("2dfld", "VEGMIN", "surface", "anl", ()),
                        ("2dfld", "VEGMAX", "surface", "anl", ()),
                    }
                )
            if hour > 0:
                deferred.add(
                    (
                        "2dfld",
                        "PEVAP",
                        "surface",
                        f"{hour - 1}-{hour} hour acc fcst",
                        (),
                    )
                )
        declared_deferred.update(deferred)
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
                assert (
                    duplicate or zero_placeholder or (family, *identity) in deferred
                ), (path, identity)
                if (family, *identity) in deferred:
                    seen_deferred.add((family, *identity))
            seen.add(identity)
    assert seen_deferred == declared_deferred


def test_sub_hourly_file_contributes_four_precise_leads() -> None:
    coord = NoaaRrfsSourceFileCoord(
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
    coord = NoaaRrfsSourceFileCoord(
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


def test_missing_required_messages_fail_loudly() -> None:
    config = CONFIGS[0]
    var = next(v for v in config.data_vars if v.name == "temperature_2m")
    coord = NoaaRrfsSourceFileCoord(
        init_time=pd.Timestamp("2026-09-15T00"),
        lead_time=pd.Timedelta("2h"),
        source_family="2dfld",
        data_vars=[var],
    )
    with pytest.raises(
        AssertionError, match=r"required GRIB messages absent.*temperature_2m"
    ):
        make_job(config)._check_refs_complete(coord, [])


def test_supported_leads_are_explicit_for_binary_run_totals() -> None:
    variables = CONFIGS[0].data_vars
    ffg = next(
        v
        for v in variables
        if v.name
        == "categorical_precipitation_exceeding_flash_flood_guidance_run_total_surface"
    )
    ari = next(
        v
        for v in variables
        if v.name
        == "categorical_precipitation_exceeding_2_year_average_recurrence_interval_run_total_surface"
    )
    assert [h for h in range(85) if ffg.available_at(pd.Timedelta(hours=h))] == [
        1,
        3,
        6,
        12,
    ]
    assert [h for h in range(85) if ari.available_at(pd.Timedelta(hours=h))] == [
        1,
        3,
        6,
        12,
        24,
    ]
    assert ffg.attrs.flag_values == (0, 1)
    assert ffg.attrs.flag_meanings == "no yes"
    assert ffg.attrs.units == "1"
    assert "guidance is unavailable" in (ffg.attrs.comment or "")


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
