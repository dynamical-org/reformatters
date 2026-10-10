import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.iterating import item
from reformatters.common.pydantic import replace
from reformatters.common.types import Timedelta
from reformatters.noaa.noaa_grib_index import (
    grib_index_window_str,
    parse_grib_index_lines,
)
from reformatters.noaa.refs.forecast_3_hourly_virtual.template_config import (
    NoaaRefsForecast3HourlyVirtualTemplateConfig,
)
from reformatters.noaa.refs.forecast_hourly_virtual.template_config import (
    NoaaRefsForecastHourlyVirtualTemplateConfig,
)
from reformatters.noaa.refs.models import (
    NoaaRefsDataVar,
    ProductFamily,
)
from reformatters.noaa.refs.region_job import (
    NoaaRefsRegionJob,
    NoaaRefsSourceFileCoord,
)

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = NoaaRefsForecastHourlyVirtualTemplateConfig()
CONFIGS = (CONFIG, NoaaRefsForecast3HourlyVirtualTemplateConfig())
VARIABLES = [v for config in CONFIGS for v in config.data_vars]
FAMILIES: tuple[ProductFamily, ...] = (
    "mean",
    "sprd",
    "prob",
    "pmmn",
    "lpmm",
    "avrg",
    "eas",
    "ffri",
)
INDEX_PATHS = tuple(sorted(FIXTURES.glob("refs.*/*/ensprod/*.idx")))


def index_path(family: ProductFamily, hours: int) -> Path:
    return item(
        p for p in INDEX_PATHS if p.name.split(".")[2:4] == [family, f"f{hours:02}"]
    )


@pytest.mark.parametrize(
    ("family", "lead", "expected_counts"),
    [
        ("prob", 1, {14}),
        ("prob", 43, {13}),
        ("prob", 49, {12}),
        ("prob", 54, {6}),
        ("eas", 1, {14}),
        ("eas", 43, {13, 14}),
        ("eas", 49, {12}),
        ("eas", 54, {12}),
        ("eas", 60, {6}),
        ("ffri", 1, {14}),
        ("ffri", 43, {13}),
        ("ffri", 49, {12}),
        ("ffri", 54, {6}),
    ],
)
def test_real_indexes_preserve_threshold_identity_across_member_count_bands(
    family: ProductFamily, lead: int, expected_counts: set[int]
) -> None:
    path = index_path(family, lead)
    init = pd.Timestamp(path.parts[-4].removeprefix("refs.")) + pd.Timedelta(
        hours=int(path.parts[-3])
    )
    coord = file_coord(init, family, pd.Timedelta(hours=lead))
    counts = set()
    for _, _, _, _, selectors in parse_grib_index_lines(path):
        tokens = [s for s in selectors if re.fullmatch(r"prob fcst \d+/\d+", s)]
        assert len(tokens) == 1
        counts.add(int(tokens[0].split("/")[1]))
        assert coord.index_selectors(selectors) == tuple(
            s for s in selectors if s != tokens[0]
        )
    assert counts == expected_counts


def make_job(
    data_vars: Sequence[NoaaRefsDataVar] = VARIABLES,
) -> NoaaRefsRegionJob:
    return NoaaRefsRegionJob(
        tmp_store=Path("unused"),
        template_ds=xr.DataTree(xr.Dataset(attrs={"dataset_id": CONFIG.dataset_id})),
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode="backfill",
    )


def file_coord(
    init: pd.Timestamp,
    family: ProductFamily,
    lead: Timedelta,
    data_vars: Sequence[NoaaRefsDataVar] = VARIABLES,
) -> NoaaRefsSourceFileCoord:
    return NoaaRefsSourceFileCoord(
        init_time=init,
        lead_time=lead,
        source_family=family,
        data_vars=[
            v
            for v in data_vars
            if family in v.internal_attrs.source_families and v.available_at(lead)
        ],
    )


def test_every_source_message_is_referenced_except_the_explicit_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = make_job()
    monkeypatch.setattr(NoaaRefsRegionJob, "grib_message_length_at", lambda *a: 217)
    covered = set()
    for path in INDEX_PATHS:
        family = next(f for f in FAMILIES if f == path.name.split(".")[2])
        lead = pd.Timedelta(hours=int(path.name.split(".")[3][1:]))
        coord = file_coord(
            pd.Timestamp(path.parts[-4].removeprefix("refs."))
            + pd.Timedelta(hours=int(path.parts[-3])),
            family,
            lead,
        )
        local = tmp_path / "index.idx"
        local.write_bytes(path.read_bytes())
        monkeypatch.setattr(
            "reformatters.noaa.noaa_virtual_region_job.s3_download_to_disk",
            lambda *a, local=local, **k: local,
        )
        lines = parse_grib_index_lines(path)
        refs = []
        for config in CONFIGS:
            variant_coord = file_coord(coord.init_time, family, lead, config.data_vars)
            if not variant_coord.data_vars:
                continue
            local.write_bytes(path.read_bytes())
            variant_refs = make_job(config.data_vars).file_refs(
                variant_coord, lines[-1][0] + 1000
            )
            assert {ref.data_var.path for ref in variant_refs} == {
                v.path for v in variant_coord.data_vars
            }
            assert lead in config.dimension_coordinates()["lead_time"]
            refs.extend(variant_refs)
        accounted = {
            start
            for start, element, level, _, selectors in lines
            if family == "sprd"
            and element == "HGT"
            and level == "surface"
            and selectors == ("ens spread", "process=193")
        }
        assert {ref.offset for ref in refs} == {
            line[0] for line in lines
        } - accounted, path
        assert len(refs) == len(lines) - len(accounted), path
        assert all(ref.out_loc == coord.out_loc() for ref in refs)
        if family == "prob":
            categorical_names = {
                "CRAIN": "probability_of_categorical_rain_surface",
                "CSNOW": "probability_of_categorical_snow_surface",
                "CICEP": "probability_of_categorical_ice_pellets_surface",
                "CFRZR": "probability_of_categorical_freezing_rain_surface",
            }
            assert {
                ref.data_var.name
                for ref in refs
                if ref.data_var.internal_attrs.grib_element in categorical_names
            } == set(categorical_names.values()), path
            for ref in refs:
                element = ref.data_var.internal_attrs.grib_element
                if element in categorical_names:
                    assert ref.data_var.name == categorical_names[element]
                    assert ref.data_var.internal_attrs.grib_index_selectors == (
                        "prob >=1 <0",
                    )
        for ref in refs:
            assert isinstance(ref.data_var, NoaaRefsDataVar)
            assert ("statistic" in ref.out_loc) == (family in ("mean", "sprd"))
        if family == "mean" and lead == pd.Timedelta("1h"):
            with pytest.raises(AssertionError, match="required GRIB messages absent"):
                job._check_refs_complete(
                    coord,
                    [ref for ref in refs if ref.data_var.name != "temperature_2m"],
                )
            with pytest.raises(AssertionError, match="duplicate output references"):
                job._check_refs_complete(coord, [*refs, refs[0]])
        covered.update(ref.data_var.path for ref in refs)
    assert covered == {var.path for var in VARIABLES}


def test_mean_coords_exclude_exactly_the_five_spread_only_fields() -> None:
    region = xr.Dataset(
        coords={
            "init_time": [pd.Timestamp("2026-09-15T12")],
            "lead_time": [pd.Timedelta("1h")],
        }
    )
    coords = make_job().generate_source_file_coords(region, VARIABLES)
    assert len(coords) == 8
    mean = next(c for c in coords if c.source_family == "mean")
    spread = next(c for c in coords if c.source_family == "sprd")
    assert mean.out_loc()["statistic"] == "mean"
    assert spread.out_loc()["statistic"] == "standard_deviation"
    assert {v.name for v in spread.data_vars if v.has_statistic} - {
        v.name for v in mean.data_vars if v.has_statistic
    } == {
        "maximum_updraft_helicity_5000_2000m",
        "derived_radar_reflectivity_1000m",
        "hourly_maximum_radar_reflectivity_1000m",
        "composite_reflectivity",
        "echo_top",
    }
    assert all(
        "statistic" not in c.out_loc()
        for c in coords
        if c.source_family not in ("mean", "sprd")
    )


@pytest.mark.parametrize("config", CONFIGS, ids=["hourly", "3-hourly"])
def test_source_coords_follow_cadence_and_window_starts(
    config: NoaaRefsForecastHourlyVirtualTemplateConfig
    | NoaaRefsForecast3HourlyVirtualTemplateConfig,
) -> None:
    region = xr.Dataset(
        coords={
            "init_time": [
                config.append_dim_start,
                config.append_dim_start + pd.Timedelta("6h"),
            ],
            "lead_time": config.dimension_coordinates()["lead_time"],
        }
    )
    coords = make_job(config.data_vars).generate_source_file_coords(
        region, config.data_vars
    )
    actual = {
        (coord.init_time, coord.lead_time, coord.source_family, var.path)
        for coord in coords
        for var in coord.data_vars
    }
    expected = {
        (init, lead, family, var.path)
        for init in pd.to_datetime(region.init_time.values)
        for var in config.data_vars
        for lead in var.internal_attrs.supported_lead_times
        for family in var.internal_attrs.source_families
    }
    assert actual == expected
    for hours in (3, 6):
        at_lead = [c for c in coords if c.lead_time == pd.Timedelta(hours=hours)]
        assert at_lead
        assert all(
            var.internal_attrs.window_duration is None
            or var.internal_attrs.window_duration <= coord.lead_time
            for coord in at_lead
            for var in coord.data_vars
        )


@pytest.mark.parametrize("count", [14, 13, 12, 6])
@pytest.mark.parametrize("family", ["prob", "eas", "ffri"])
def test_member_count_token_is_removed_without_losing_exact_selectors(
    count: int, family: ProductFamily
) -> None:
    coord = file_coord(pd.Timestamp("2026-10-02"), family, pd.Timedelta("60h"))
    selectors = ("prob >12.7", f"prob fcst 0/{count}", "process=197")
    assert coord.index_selectors(selectors) == ("prob >12.7", "process=197")
    with pytest.raises(AssertionError):
        coord.index_selectors(("prob >12.7", "process=197"))


@pytest.mark.parametrize(
    ("duration", "hours", "label"),
    [
        (6, 9, "3-9 hour acc fcst"),
        (24, 24, "0-1 day acc fcst"),
        (24, 36, "12-36 hour acc fcst"),
        (24, 48, "1-2 day acc fcst"),
    ],
)
def test_rolling_windows_match_real_labels(
    duration: int, hours: int, label: str
) -> None:
    variable = next(
        v
        for v in VARIABLES
        if v.internal_attrs.source_families == ("mean", "sprd")
        and v.internal_attrs.grib_element == "ASNOW"
        and v.internal_attrs.window_duration == pd.Timedelta(hours=duration)
    )
    assert grib_index_window_str(variable, hours) == label
    windows = [line[3] for line in parse_grib_index_lines(index_path("mean", hours))]
    assert label in windows, (duration, hours)
    assert variable.available_at(pd.Timedelta(hours=hours))
    assert not variable.available_at(pd.Timedelta(hours=hours + 1))


def test_missing_required_or_wrong_selector_fails() -> None:
    coord = file_coord(pd.Timestamp("2026-09-15T12"), "mean", pd.Timedelta("1h"))
    with pytest.raises(AssertionError, match="required GRIB messages absent"):
        make_job()._check_refs_complete(coord, [])
    var = next(v for v in VARIABLES if v.name == "temperature_2m")
    altered = replace(
        var,
        internal_attrs=replace(
            var.internal_attrs, grib_index_selectors=("process=193",)
        ),
    )
    assert ("TMP", "2 m above ground", "1 hour fcst", ()) in make_job()._message_lookup(
        [var], 1
    )
    assert (
        "TMP",
        "2 m above ground",
        "1 hour fcst",
        ("process=193",),
    ) in make_job()._message_lookup([altered], 1)


def test_identical_sparse_selectors_still_have_different_source_files() -> None:
    lead = pd.Timedelta("9h")
    pmmn = file_coord(pd.Timestamp("2026-09-15T12"), "pmmn", lead)
    avrg = file_coord(pmmn.init_time, "avrg", lead)
    assert pmmn.get_url() != avrg.get_url()
    pmmn_var = next(
        v
        for v in pmmn.data_vars
        if v.internal_attrs.window_duration == pd.Timedelta("3h")
    )
    avrg_var = next(
        v
        for v in avrg.data_vars
        if v.internal_attrs.window_duration == pd.Timedelta("3h")
    )
    assert (
        pmmn_var.internal_attrs.grib_index_selectors
        == avrg_var.internal_attrs.grib_index_selectors
    )
    assert pmmn_var.name != avrg_var.name


@pytest.mark.parametrize("processes", [(4, 193), (193, 4), (4,), (193,), ()])
def test_surface_height_spread_accepts_either_copy(
    processes: tuple[int, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = index_path("sprd", 1)
    lines = path.read_text().splitlines()
    originals = [line for line in lines if ":HGT:surface:" in line]
    assert len(originals) == 2
    replacements = {
        old: old.removesuffix(":process=193")
        + (":process=193" if process == 193 else "")
        for old, process in zip(originals, processes, strict=False)
    }
    local = tmp_path / "index.idx"
    local.write_text(
        "\n".join(
            replacements.get(line, line)
            for line in lines
            if line not in originals or line in replacements
        )
        + "\n"
    )
    monkeypatch.setattr(
        "reformatters.noaa.noaa_virtual_region_job.s3_download_to_disk",
        lambda *a, **k: local,
    )
    monkeypatch.setattr(NoaaRefsRegionJob, "grib_message_length_at", lambda *a: 217)
    coord = file_coord(pd.Timestamp("2026-08-13T00"), "sprd", pd.Timedelta("1h"))
    file_size = int(lines[-1].split(":")[1]) + 1000
    if not processes:
        with pytest.raises(AssertionError, match="required GRIB messages absent"):
            make_job().file_refs(coord, file_size)
        return
    refs = make_job().file_refs(coord, file_size)
    height = item(r for r in refs if r.data_var.name == "geopotential_height_surface")
    assert height.offset == int(originals[0].split(":")[1])
    assert height.out_loc["statistic"] == "standard_deviation"
    assert len(refs) == len(coord.data_vars)


def test_pmm_process_cannot_supply_arithmetic_mean() -> None:
    coord = file_coord(pd.Timestamp("2026-08-13T00"), "mean", pd.Timedelta("1h"))
    with pytest.raises(
        AssertionError, match="PMM process cannot supply an arithmetic mean"
    ):
        coord.index_selectors(("wt ens mean", "process=193"))
