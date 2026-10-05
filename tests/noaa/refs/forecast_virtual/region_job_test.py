import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.common.types import Timedelta
from reformatters.noaa.noaa_grib_index import (
    grib_index_window_str,
    parse_grib_index_lines,
)
from reformatters.noaa.refs.forecast_virtual.models import (
    NoaaRefsDataVar,
    ProductFamily,
)
from reformatters.noaa.refs.forecast_virtual.region_job import (
    NoaaRefsRegionJob,
    NoaaRefsSourceFileCoord,
)
from reformatters.noaa.refs.forecast_virtual.template_config import (
    NoaaRefsForecastVirtualTemplateConfig,
)

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = NoaaRefsForecastVirtualTemplateConfig()
VARIABLES = CONFIG.data_vars


@pytest.mark.parametrize(
    ("lead", "count", "eas_counts"),
    [
        (42, 14, {14}),
        (43, 13, {13, 14}),
        (48, 13, {13, 14}),
        (49, 12, {12}),
        (53, 12, {12}),
        (54, 6, {12}),
        (60, 6, {6}),
    ],
)
@pytest.mark.parametrize("family", ["prob", "eas", "ffri"])
def test_real_indexes_preserve_threshold_identity_across_member_count_bands(
    lead: int, count: int, eas_counts: set[int], family: ProductFamily
) -> None:
    path = (
        FIXTURES
        / f"refs.20261002/00/ensprod/refs.t00z.{family}.f{lead:02}.conus.grib2.idx"
    )
    coord = file_coord(pd.Timestamp("2026-10-02"), family, pd.Timedelta(hours=lead))
    counts = set()
    for _, _, _, _, selectors in parse_grib_index_lines(path):
        tokens = [s for s in selectors if re.fullmatch(r"prob fcst \d+/\d+", s)]
        assert len(tokens) == 1
        counts.add(int(tokens[0].split("/")[1]))
        assert coord.index_selectors(selectors) == tuple(
            s for s in selectors if s != tokens[0]
        )
    assert counts == (eas_counts if family == "eas" else {count})


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
    init: pd.Timestamp, family: ProductFamily, lead: Timedelta
) -> NoaaRefsSourceFileCoord:
    return NoaaRefsSourceFileCoord(
        init_time=init,
        lead_time=lead,
        source_family=family,
        data_vars=[
            v
            for v in VARIABLES
            if family in v.internal_attrs.source_families and v.available_at(lead)
        ],
    )


@pytest.mark.parametrize("cycle", ["20260813/00", "20260915/12", "20261002/00"])
def test_every_source_message_is_referenced_except_the_explicit_duplicate(
    cycle: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    day, hour = cycle.split("/")
    paths = sorted((FIXTURES / f"refs.{day}" / hour / "ensprod").glob("*.idx"))
    assert len(paths) == (480 if day == "20261002" else 64)
    job = make_job()
    monkeypatch.setattr(NoaaRefsRegionJob, "grib_message_length_at", lambda *a: 217)
    total_refs = 0
    duplicates = 0
    for path in paths:
        family: ProductFamily = path.name.split(".")[2]  # ty: ignore[invalid-assignment]
        lead = pd.Timedelta(hours=int(path.name.split(".")[3][1:]))
        coord = file_coord(
            pd.Timestamp(day) + pd.Timedelta(hours=int(hour)), family, lead
        )
        local = tmp_path / "index.idx"
        local.write_bytes(path.read_bytes())
        monkeypatch.setattr(
            "reformatters.noaa.noaa_virtual_region_job.s3_download_to_disk",
            lambda *a, local=local, **k: local,
        )
        lines = parse_grib_index_lines(path)
        refs = job.file_refs(coord, lines[-1][0] + 1000)
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
        if family == "mean":
            with pytest.raises(AssertionError, match="required GRIB messages absent"):
                job._check_refs_complete(
                    coord,
                    [ref for ref in refs if ref.data_var.name != "temperature_2m"],
                )
            with pytest.raises(AssertionError, match="duplicate output references"):
                job._check_refs_complete(coord, [*refs, refs[0]])
        duplicates += len(accounted)
        total_refs += len(refs)
    assert duplicates == (60 if day == "20261002" else 8)
    assert total_refs == (20349 if day == "20261002" else 2975)


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
    for family in ("mean", "sprd", "prob", "eas", "ffri"):
        path = (
            FIXTURES
            / "refs.20261002/00/ensprod"
            / f"refs.t00z.{family}.f{hours:02}.conus.grib2.idx"
        )
        windows = [line[3] for line in parse_grib_index_lines(path)]
        assert label in windows, (family, duration, hours)
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
