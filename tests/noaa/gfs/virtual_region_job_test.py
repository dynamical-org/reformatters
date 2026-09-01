from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.config_models import ROOT
from reformatters.noaa.gfs import virtual_region_job as region_job_module
from reformatters.noaa.gfs.analysis_virtual.region_job import (
    NoaaGfsAnalysisVirtualRegionJob,
    NoaaGfsAnalysisVirtualSourceFileCoord,
)
from reformatters.noaa.gfs.analysis_virtual.template_config import (
    NoaaGfsAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.gfs.virtual_region_job import (
    PGRB2_PREFERRED_MESSAGES,
    NoaaGfsFileType,
    carried_by,
)
from reformatters.noaa.gfs.virtual_template_config import (
    PRESSURE_LEVEL_INDEX_FORMAT,
    PRESSURE_LEVELS,
)
from reformatters.noaa.noaa_grib_index import parse_grib_index_lines
from tests.noaa.grib_index_fixtures import cached_grib_index, stub_grib_index_download

TEMPLATE_CONFIG = NoaaGfsAnalysisVirtualTemplateConfig()
_DATASET_ID = "noaa-gfs-analysis-virtual-test"

# Spans the whole intended archive: the first weeks, the CLWMR -> CLMR rename era, and
# the present. The oldest is one day after the 0.25 degree archive's first cycle.
_ERAS = ("20210323", "20230401", "20260828")


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


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
@pytest.mark.parametrize("lead_hours", [0, 3, 9, 120])
def test_pgrb2_preferred_messages_matches_the_real_indexes(
    era: str, lead_hours: int
) -> None:
    """The constant is exactly the (element, level) set both products publish.

    A pair missing from it would write two byte ranges into one array position; a pair
    that is not really duplicated would drop the only copy of a pgrb2b message.
    """
    assert PGRB2_PREFERRED_MESSAGES
    in_both = {
        (element, level)
        for _, element, level, _ in index_lines(era, "pgrb2", lead_hours)
    } & {
        (element, level)
        for _, element, level, _ in index_lines(era, "pgrb2b", lead_hours)
    }
    assert in_both == PGRB2_PREFERRED_MESSAGES


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
def test_pressure_level_coordinate_covers_every_isobaric_level(era: str) -> None:
    """The published coordinate is the union of both products' isobaric levels, and the
    format string renders each one's index level string exactly."""
    published = {
        PRESSURE_LEVEL_INDEX_FORMAT.format(level=level) for level in PRESSURE_LEVELS
    }
    in_source = {
        level
        for file_type in ("pgrb2", "pgrb2b")
        for _, _, level, _ in index_lines(era, file_type, 9)
        if level.endswith(" mb")
    }
    assert len(published) == 57
    assert in_source == published


def test_pressure_level_index_format_is_lossless() -> None:
    assert PRESSURE_LEVEL_INDEX_FORMAT.format(level=1000.0) == "1000 mb"
    assert PRESSURE_LEVEL_INDEX_FORMAT.format(level=0.7) == "0.7 mb"
    assert PRESSURE_LEVEL_INDEX_FORMAT.format(level=0.01) == "0.01 mb"


def _fake_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str) -> None:
    stub_grib_index_download(
        monkeypatch, region_job_module, tmp_path, lambda _url: content
    )


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2021-03-23T00:00"))


def _job(template_ds: xr.DataTree, paths: list[str]) -> NoaaGfsAnalysisVirtualRegionJob:
    return NoaaGfsAnalysisVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=[v for v in TEMPLATE_CONFIG.data_vars if v.path in paths],
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


@pytest.mark.parametrize(
    ("file_type", "expected_offsets"),
    [("pgrb2", [0, 100]), ("pgrb2b", [100])],
)
def test_pgrb2b_skips_the_messages_pgrb2_owns(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    file_type: NoaaGfsFileType,
    expected_offsets: list[int],
) -> None:
    """The duplicated CNWAT message is taken from pgrb2 and skipped in pgrb2b, while a
    message only pgrb2b carries is taken from it."""
    assert ("CNWAT", "surface") in PGRB2_PREFERRED_MESSAGES
    _fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2021032312:CNWAT:surface:anl:\n"
        "2:100:d=2021032312:TMP:305 m above mean sea level:anl:\n",
    )
    paths = ["plant_canopy_surface_water_surface", "temperature_305m_amsl"]
    job = _job(template_ds, paths)
    coord = NoaaGfsAnalysisVirtualSourceFileCoord(
        init_time=pd.Timestamp("2021-03-23T12:00"),
        lead_time=pd.Timedelta(0),
        file_type=file_type,
        data_vars=job.data_vars,
    )

    refs = job.file_refs(coord, file_size=200)

    assert [ref.offset for ref in refs] == expected_offsets


def test_a_pressure_level_message_lands_in_its_own_level(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2021032312:TMP:850 mb:anl:\n1:500:d=2021032312:TMP:0.07 mb:anl:\n",
    )
    job = _job(template_ds, ["pressure_level/temperature"])
    coord = NoaaGfsAnalysisVirtualSourceFileCoord(
        init_time=pd.Timestamp("2021-03-23T12:00"),
        lead_time=pd.Timedelta(0),
        file_type="pgrb2",
        data_vars=job.data_vars,
    )

    refs = job.file_refs(coord, file_size=1000)

    assert [(dict(ref.out_loc)["pressure_level"], ref.offset) for ref in refs] == [
        (850.0, 0),
        (0.07, 500),
    ]


def test_cloud_mixing_ratio_matches_both_element_spellings(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLWMR was respelled CLMR between the 2023-02-02 18Z and 2023-02-03 00Z cycles."""
    for element in ("CLMR", "CLWMR"):
        _fake_index(monkeypatch, tmp_path, f"1:0:d=2021032312:{element}:850 mb:anl:\n")
        job = _job(template_ds, ["pressure_level/cloud_mixing_ratio"])
        coord = NoaaGfsAnalysisVirtualSourceFileCoord(
            init_time=pd.Timestamp("2021-03-23T12:00"),
            lead_time=pd.Timedelta(0),
            file_type="pgrb2",
            data_vars=job.data_vars,
        )
        assert [ref.data_var.path for ref in job.file_refs(coord, file_size=100)] == [
            "pressure_level/cloud_mixing_ratio"
        ], element


def test_source_file_coord_url_and_out_loc() -> None:
    coord = NoaaGfsAnalysisVirtualSourceFileCoord(
        init_time=pd.Timestamp("2026-08-28T12:00"),
        lead_time=pd.Timedelta("6h"),
        file_type="pgrb2b",
        data_vars=[],
    )
    assert coord.get_url() == (
        "s3://noaa-gfs-bdp-pds/gfs.20260828/12/atmos/gfs.t12z.pgrb2b.0p25.f006"
    )
    assert coord.get_index_url() == coord.get_url() + ".idx"
    assert dict(coord.out_loc()) == {"time": pd.Timestamp("2026-08-28T18:00")}


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
@pytest.mark.parametrize(
    ("file_type", "lead_hours", "expected_refs"),
    [
        ("pgrb2", 0, 696),
        ("pgrb2", 9, 741),
        ("pgrb2b", 0, 306),
        ("pgrb2b", 9, 308),
    ],
)
def test_every_source_message_reaches_an_array(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    era: str,
    file_type: NoaaGfsFileType,
    lead_hours: int,
    expected_refs: int,
) -> None:
    """Run the whole catalog against a real index at both a lead an analysis reads.

    A wrong idx level string, element spelling or rendered window string shows up here
    as a missing ref, and a duplicated one as an extra: every message either fills
    exactly one array position or is a pgrb2b copy of a pgrb2 message.
    """
    stub_grib_index_download(
        monkeypatch,
        region_job_module,
        tmp_path,
        lambda url: cached_grib_index(url, _DATASET_ID),
    )
    data_vars = TEMPLATE_CONFIG.data_vars
    assert data_vars
    job = _job(template_ds, [v.path for v in data_vars])
    coord = NoaaGfsAnalysisVirtualSourceFileCoord(
        init_time=pd.Timestamp(f"{era[:4]}-{era[4:6]}-{era[6:]}T12:00"),
        lead_time=pd.Timedelta(hours=lead_hours),
        file_type=file_type,
        data_vars=data_vars,
    )

    refs = job.file_refs(coord, file_size=10**10)

    assert len(refs) == expected_refs
    # One byte range per array position: nothing is written twice.
    positions = {
        (ref.data_var.path, tuple(sorted(ref.out_loc.items()))) for ref in refs
    }
    assert len(positions) == len(refs)


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
def test_hour_0_overrides_match_what_f000_publishes(era: str) -> None:
    """The instantaneous variables the source drops at f000, derived from the indexes
    rather than copied from a design note.

    GFS publishes no windowed message at f000 at all, and nine instantaneous variables
    share an element with a windowed sibling and ARE published there, so a rule keyed on
    the element rather than the variable would drop them.
    """
    published_at_f000 = {
        (element, level)
        for file_type in ("pgrb2", "pgrb2b")
        for _, element, level, _ in index_lines(era, file_type, 0)
    }
    assert len(published_at_f000) > 500
    assert {window for _, _, _, window in index_lines(era, "pgrb2", 0)} == {"anl"}

    instant_vars = [
        v for v in TEMPLATE_CONFIG.data_vars if v.attrs.step_type == "instant"
    ]
    assert len(instant_vars) == 251
    for var in instant_vars:
        levels = (
            [
                PRESSURE_LEVEL_INDEX_FORMAT.format(level=level)
                for level in PRESSURE_LEVELS
            ]
            if var.group is not ROOT
            else [var.internal_attrs.grib_index_level]
        )
        elements = (
            var.internal_attrs.grib_element,
            *var.internal_attrs.grib_element_alternatives,
        )
        in_f000 = any(
            (element, level) in published_at_f000
            for element in elements
            for level in levels
        )
        assert var.has_hour_0_values() == in_f000, var.name


@pytest.mark.slow
def test_cloud_mixing_ratio_element_was_respelled_between_the_2021_and_2026_eras() -> (
    None
):
    """grib_element_alternatives exists for this one rename; both spellings are real."""
    spellings = {
        era: {
            element
            for _, element, _, _ in index_lines(era, "pgrb2", 9)
            if element in ("CLMR", "CLWMR")
        }
        for era in _ERAS
    }
    assert spellings == {
        "20210323": {"CLWMR"},
        "20230401": {"CLMR"},
        "20260828": {"CLMR"},
    }


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
@pytest.mark.parametrize("file_type", ["pgrb2", "pgrb2b"])
def test_product_membership_matches_the_real_indexes(
    era: str, file_type: NoaaGfsFileType
) -> None:
    """`carried_by` must agree with the product's real inventory for every variable.

    A false positive sends a job to a file with nothing in it for that variable, and
    lets `representative_var` probe a chunk the file never fills, which would make the
    file re-ingest forever.
    """
    published = {
        (element, level) for _, element, level, _ in index_lines(era, file_type, 9)
    }
    # A pgrb2b index's copies of the messages pgrb2 owns are skipped, so they are not
    # part of what this product supplies.
    if file_type == "pgrb2b":
        published -= PGRB2_PREFERRED_MESSAGES
    assert published

    for var in TEMPLATE_CONFIG.data_vars:
        levels = (
            [
                PRESSURE_LEVEL_INDEX_FORMAT.format(level=level)
                for level in PRESSURE_LEVELS
            ]
            if var.group is not ROOT
            else [var.internal_attrs.grib_index_level]
        )
        elements = (
            var.internal_attrs.grib_element,
            *var.internal_attrs.grib_element_alternatives,
        )
        in_product = any(
            (element, level) in published for element in elements for level in levels
        )
        assert carried_by(var, file_type) == in_product, (var.path, file_type)


@pytest.mark.slow
@pytest.mark.parametrize("era", _ERAS)
@pytest.mark.parametrize("lead_hours", [1, 3, 6])
def test_one_ref_per_position_at_the_leads_that_duplicate_accumulations(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    era: str,
    lead_hours: int,
) -> None:
    """At leads 1-6 the bucket and the running total of APCP and ACPCP render one window
    string and the index carries two identical messages for it, so the analysis's single
    bucket array matches twice."""
    index_path = cached_grib_index(index_url(era, "pgrb2", lead_hours), _DATASET_ID)
    duplicated = Counter(
        (element, level, window)
        for _, element, level, window in parse_grib_index_lines(index_path)
    )
    assert [key for key, count in duplicated.items() if count > 1] == [
        ("APCP", "surface", f"0-{lead_hours} hour acc fcst"),
        ("ACPCP", "surface", f"0-{lead_hours} hour acc fcst"),
    ]

    stub_grib_index_download(
        monkeypatch,
        region_job_module,
        tmp_path,
        lambda url: cached_grib_index(url, _DATASET_ID),
    )
    data_vars = TEMPLATE_CONFIG.data_vars
    job = _job(template_ds, [v.path for v in data_vars])
    coord = NoaaGfsAnalysisVirtualSourceFileCoord(
        init_time=pd.Timestamp(f"{era[:4]}-{era[4:6]}-{era[6:]}T12:00"),
        lead_time=pd.Timedelta(hours=lead_hours),
        file_type="pgrb2",
        data_vars=data_vars,
    )

    refs = job.file_refs(coord, file_size=10**10)

    positions = {
        (ref.data_var.path, tuple(sorted(ref.out_loc.items()))) for ref in refs
    }
    assert len(positions) == len(refs)
    assert sum(ref.data_var.name == "total_precipitation_surface" for ref in refs) == 1
