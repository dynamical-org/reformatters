import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.rtma.analysis_virtual import region_job as shared
from reformatters.noaa.rtma_ru.analysis_virtual.region_job import (
    NoaaRtmaRuAnalysisVirtualRegionJob,
    NoaaRtmaRuAnalysisVirtualSourceFileCoord,
)
from reformatters.noaa.rtma_ru.analysis_virtual.template_config import (
    NoaaRtmaRuAnalysisVirtualTemplateConfig,
)
from tests.noaa.rtma.grib_header_test import append_message, edit_header, sparse_source

CONFIG = NoaaRtmaRuAnalysisVirtualTemplateConfig()


@pytest.fixture
def job(tmp_path: Path) -> NoaaRtmaRuAnalysisVirtualRegionJob:
    return NoaaRtmaRuAnalysisVirtualRegionJob(
        tmp_store=tmp_path,
        template_ds=xr.DataTree(
            xr.Dataset(
                coords={"time": pd.date_range("2017-12-13", periods=7, freq="15min")}
            )
        ),
        data_vars=CONFIG.data_vars,
        append_dim="time",
        region=slice(0, 7),
        reformat_job_name="test",
    )


@pytest.mark.parametrize(
    "time",
    ["2017-12-13T00:00", "2017-12-13T00:15", "2020-02-29T23:45", "2020-03-01T00:00"],
)
def test_source_keys_and_locations(time: str) -> None:
    timestamp = pd.Timestamp(time)
    coord = NoaaRtmaRuAnalysisVirtualSourceFileCoord(
        time=timestamp, data_vars=CONFIG.data_vars
    )
    assert coord.get_url() == (
        f"s3://noaa-rtma-pds/rtma2p5_ru.{timestamp:%Y%m%d}/"
        f"rtma2p5_ru.t{timestamp:%H%M}z.2dvaranl_ndfd.grb2"
    )
    assert coord.out_loc() == {"time": timestamp}


@pytest.mark.parametrize(
    "names", [None, ["temperature_2m"], ["total_cloud_cover_atmosphere"]]
)
def test_onset_filters_each_selected_variable_before_generating_coords(
    job: NoaaRtmaRuAnalysisVirtualRegionJob, names: list[str] | None
) -> None:
    selected = [v for v in CONFIG.data_vars if names is None or v.name in names]
    times = pd.date_range("2020-07-27T23:45", periods=2, freq="15min")
    coords = job.generate_source_file_coords(
        xr.Dataset(coords={"time": times}), selected
    )
    if names == ["total_cloud_cover_atmosphere"]:
        assert [c.time for c in coords] == [times[1]]
    else:
        assert [c.time for c in coords] == list(times)
    for coord in coords:
        assert type(coord) is NoaaRtmaRuAnalysisVirtualSourceFileCoord
        assert "/rtma2p5_ru." in coord.get_url()
        assert [v.name for v in coord.data_vars] == [
            v.name
            for v in selected
            if coord.time >= times[1] or v.name != "total_cloud_cover_atmosphere"
        ]
    preonset = job.generate_source_file_coords(
        xr.Dataset(coords={"time": times[:1]}),
        [v for v in selected if v.name == "total_cloud_cover_atmosphere"],
    )
    assert not preonset


def test_listing_leaves_missing_quarter_hours_without_index(
    job: NoaaRtmaRuAnalysisVirtualRegionJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    coords = list(job.source_file_coords())
    available = [c for i, c in enumerate(coords) if i not in (2, 5)]
    listing = Mock(
        return_value={
            c.get_url().removeprefix(shared.SOURCE_PREFIX): 1234 for c in available
        }
    )
    monkeypatch.setattr(
        "reformatters.common.virtual_source_listing._list_objects", listing
    )
    assert job.discover_available(coords) == [(c, 1234) for c in available]
    assert listing.call_args.args[1] == ["rtma2p5_ru.20171213/"]


def test_preonset_representative_uses_an_available_field(
    job: NoaaRtmaRuAnalysisVirtualRegionJob,
) -> None:
    selected = list(reversed(CONFIG.data_vars))
    assert selected[0].name == "total_cloud_cover_atmosphere"
    coords = job.generate_source_file_coords(job.template_ds.dataset, selected)
    assert coords
    assert all(
        job.representative_var(coord).name == "cloud_ceiling_height" for coord in coords
    )


@pytest.mark.parametrize(
    "subset", [None, ["temperature_2m"], ["cloud_ceiling_height", "wind_direction_10m"]]
)
def test_refs_cover_available_selected_fields_with_exact_source_ranges(
    job: NoaaRtmaRuAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    subset: list[str] | None,
) -> None:
    selected = [v for v in CONFIG.data_vars if subset is None or v.name in subset]
    read = sparse_source(source_fixture)

    def read_bytes(url: str, *, region: str, start: int, end: int) -> bytes:
        assert url == source_fixture["url"]
        assert region == "us-east-1"
        return read(start, end)

    monkeypatch.setattr(shared, "s3_read_bytes", read_bytes)
    (coord,) = job.generate_source_file_coords(
        xr.Dataset(coords={"time": [pd.Timestamp(source_fixture["time"])]}), selected
    )
    refs = job.file_refs(coord, source_fixture["size"])
    assert {r.data_var.name for r in refs} == {v.name for v in coord.data_vars}
    for ref in refs:
        expected = next(
            m
            for m in source_fixture["messages"]
            if m["element"] == ref.data_var.internal_attrs.grib_element
        )
        assert (ref.offset, ref.length) == (expected["offset"], expected["length"])
        assert ref.out_loc == {"time": pd.Timestamp(source_fixture["time"])}


@pytest.mark.parametrize("cloud_only", [False, True])
def test_missing_selected_cloud_cover_after_onset_raises(
    job: NoaaRtmaRuAnalysisVirtualRegionJob,
    monkeypatch: pytest.MonkeyPatch,
    cloud_only: bool,
) -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/20200728t0000.json").read_text()
    )
    edit_header(fixture, 12, 118 + 10, b"\xff")
    read = sparse_source(fixture)
    monkeypatch.setattr(
        shared, "s3_read_bytes", lambda url, *, region, start, end: read(start, end)
    )
    coord = NoaaRtmaRuAnalysisVirtualSourceFileCoord(
        time=fixture["time"],
        data_vars=[
            var
            for var in CONFIG.data_vars
            if not cloud_only or var.name == "total_cloud_cover_atmosphere"
        ],
    )
    with pytest.raises(
        AssertionError, match=r"Missing selected fields.*total_cloud_cover_atmosphere"
    ):
        job._file_refs_or_skip(coord, fixture["size"])


@pytest.mark.parametrize(
    ("offset", "data", "error"),
    [
        (16 + 17, b"\x17", "reference time"),
        (118 + 17, b"\x02", "forecast time unit"),
        (118 + 18, (1).to_bytes(4), "zero-lead"),
        (37 + 30, (2145).to_bytes(4), "grid geometry"),
        (152 + 9, b"\0\x02", "simple packing"),
        (173 + 5, b"\xfe", "without bitmap"),
    ],
)
def test_invalid_selected_ru_headers_raise(
    job: NoaaRtmaRuAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    offset: int,
    data: bytes,
    error: str,
) -> None:
    edit_header(source_fixture, 0, offset, data)
    read = sparse_source(source_fixture)
    monkeypatch.setattr(
        shared, "s3_read_bytes", lambda url, *, region, start, end: read(start, end)
    )
    (coord,) = job.generate_source_file_coords(
        xr.Dataset(coords={"time": [pd.Timestamp(source_fixture["time"])]}),
        CONFIG.data_vars,
    )
    with pytest.raises(AssertionError, match=error):
        job.file_refs(coord, source_fixture["size"])


def test_duplicate_checks_and_unsupported_products_apply_only_to_selected_fields(
    job: NoaaRtmaRuAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for _ in range(2):
        extra = append_message(source_fixture, 0)
        edit_header(source_fixture, extra, 118 + 10, b"\xff")
        edit_header(source_fixture, extra, 118 + 7, b"\0\x08")
    read = sparse_source(source_fixture)
    monkeypatch.setattr(
        shared, "s3_read_bytes", lambda url, *, region, start, end: read(start, end)
    )
    (coord,) = job.generate_source_file_coords(
        xr.Dataset(coords={"time": [pd.Timestamp(source_fixture["time"])]}),
        CONFIG.data_vars,
    )
    assert len(job.file_refs(coord, source_fixture["size"])) == len(coord.data_vars)
    append_message(source_fixture, 0)
    with pytest.raises(AssertionError, match="Duplicate GRIB field"):
        job.file_refs(coord, source_fixture["size"])
