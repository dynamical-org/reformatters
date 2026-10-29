import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.noaa.rtma.analysis_virtual import region_job as module
from reformatters.noaa.rtma.analysis_virtual.region_job import (
    NoaaRtmaAnalysisVirtualRegionJob,
    NoaaRtmaAnalysisVirtualSourceFileCoord,
)
from reformatters.noaa.rtma.analysis_virtual.template_config import (
    NoaaRtmaAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.rtma_ru.analysis_virtual.region_job import (
    NoaaRtmaRuAnalysisVirtualRegionJob,
)
from reformatters.noaa.rtma_ru.analysis_virtual.template_config import (
    NoaaRtmaRuAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.urma.analysis_virtual.region_job import (
    NoaaUrmaAnalysisVirtualRegionJob,
)
from reformatters.noaa.urma.analysis_virtual.template_config import (
    NoaaUrmaAnalysisVirtualTemplateConfig,
)
from tests.noaa.rtma.grib_header_test import append_message, edit_header, sparse_source

CONFIG = NoaaRtmaAnalysisVirtualTemplateConfig()


@pytest.fixture
def job(tmp_path: Path) -> NoaaRtmaAnalysisVirtualRegionJob:
    return NoaaRtmaAnalysisVirtualRegionJob(
        tmp_store=tmp_path,
        template_ds=xr.DataTree(
            xr.Dataset(
                coords={"time": pd.date_range("2017-05-02", periods=2, freq="h")}
            )
        ),
        data_vars=CONFIG.data_vars,
        append_dim="time",
        region=slice(0, 2),
        reformat_job_name="test",
    )


@pytest.mark.parametrize(
    "time", ["2017-05-02T00:00", "2020-02-29T23:00", "2020-03-01T00:00"]
)
def test_source_keys_and_locations(time: str) -> None:
    timestamp = pd.Timestamp(time)
    coord = NoaaRtmaAnalysisVirtualSourceFileCoord(
        time=timestamp, data_vars=CONFIG.data_vars
    )
    assert (
        coord.get_url()
        == f"s3://noaa-rtma-pds/rtma2p5.{timestamp:%Y%m%d}/rtma2p5.t{timestamp:%H}z.2dvaranl_ndfd.grb2_wexp"
    )
    assert coord.out_loc() == {"time": timestamp}


def test_generate_coords_keeps_selected_subset(
    job: NoaaRtmaAnalysisVirtualRegionJob,
) -> None:
    selected = CONFIG.data_vars[2:4]
    coords = job.generate_source_file_coords(job.template_ds.dataset, selected)
    assert [coord.time for coord in coords] == list(
        pd.date_range("2017-05-02", periods=2, freq="h")
    )
    assert all(coord.data_vars == selected for coord in coords)


@pytest.mark.parametrize(
    "names", [None, ["temperature_2m"], ["total_cloud_cover_atmosphere"], []]
)
def test_generate_coords_respects_variable_availability(
    job: NoaaRtmaAnalysisVirtualRegionJob, names: list[str] | None
) -> None:
    onset = pd.Timestamp("2017-05-02T01:00")
    selected = [
        replace(var, internal_attrs=replace(var.internal_attrs, available_from=onset))
        if var.name == "total_cloud_cover_atmosphere"
        else var
        for var in CONFIG.data_vars
        if names is None or var.name in names
    ]
    coords = job.generate_source_file_coords(job.template_ds.dataset, selected)
    expected_times = (
        []
        if not selected
        else [onset]
        if names == ["total_cloud_cover_atmosphere"]
        else list(pd.date_range("2017-05-02", periods=2, freq="h"))
    )
    assert [coord.time for coord in coords] == expected_times
    for coord in coords:
        assert type(coord) is NoaaRtmaAnalysisVirtualSourceFileCoord
        assert [var.name for var in coord.data_vars] == [
            var.name
            for var in selected
            if coord.time >= onset or var.name != "total_cloud_cover_atmosphere"
        ]


def test_listing_leaves_source_gaps_without_requiring_index(
    job: NoaaRtmaAnalysisVirtualRegionJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    coords = list(
        job.generate_source_file_coords(job.template_ds.dataset, CONFIG.data_vars)
    )
    listing = Mock(
        return_value={coords[0].get_url().removeprefix(module.SOURCE_PREFIX): 1234}
    )
    monkeypatch.setattr(
        "reformatters.common.virtual_source_listing._list_objects", listing
    )
    assert job.discover_available(coords) == [(coords[0], 1234)]
    assert listing.call_args.args[1] == ["rtma2p5.20170502/"]


@pytest.mark.parametrize(
    "subset", [None, ["temperature_2m"], ["cloud_ceiling_height", "wind_direction_10m"]]
)
def test_refs_cover_selected_fields_with_exact_source_ranges(
    job: NoaaRtmaAnalysisVirtualRegionJob,
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

    monkeypatch.setattr(module, "s3_read_bytes", read_bytes)
    coord = NoaaRtmaAnalysisVirtualSourceFileCoord(
        time=source_fixture["time"], data_vars=selected
    )
    refs = job.file_refs(coord, source_fixture["size"])
    assert {ref.data_var.name for ref in refs} == {v.name for v in selected}
    for ref in refs:
        expected = next(
            m
            for m in source_fixture["messages"]
            if m["element"] == ref.data_var.internal_attrs.grib_element
        )
        assert (ref.offset, ref.length) == (expected["offset"], expected["length"])
        assert ref.out_loc == {"time": pd.Timestamp(source_fixture["time"])}


@pytest.mark.parametrize(
    ("offset", "data"), [(118 + 10, b"\xff"), (118 + 7, b"\0\x08")]
)
def test_missing_selected_field_keeps_usable_refs(
    job: NoaaRtmaAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    offset: int,
    data: bytes,
) -> None:
    edit_header(source_fixture, 12, offset, data)
    read = sparse_source(source_fixture)
    monkeypatch.setattr(
        module, "s3_read_bytes", lambda url, *, region, start, end: read(start, end)
    )
    coord = NoaaRtmaAnalysisVirtualSourceFileCoord(
        time=source_fixture["time"], data_vars=CONFIG.data_vars
    )
    refs = job.file_refs(coord, source_fixture["size"])
    assert {ref.data_var.name for ref in refs} == {
        var.name
        for var in CONFIG.data_vars
        if var.name != "total_cloud_cover_atmosphere"
    }
    assert len(refs) == len(CONFIG.data_vars) - 1


def test_new_product_template_with_selected_parameter_preserves_complete_refs(
    job: NoaaRtmaAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = append_message(source_fixture, 2)
    edit_header(source_fixture, extra, 118 + 7, b"\0\x08")
    read = sparse_source(source_fixture)
    monkeypatch.setattr(
        module, "s3_read_bytes", lambda url, *, region, start, end: read(start, end)
    )
    coord = NoaaRtmaAnalysisVirtualSourceFileCoord(
        time=source_fixture["time"], data_vars=CONFIG.data_vars
    )
    refs = job.file_refs(coord, source_fixture["size"])
    assert len(refs) == len(CONFIG.data_vars)
    assert {ref.data_var.name for ref in refs} == {var.name for var in CONFIG.data_vars}


def test_reordering_messages_does_not_change_selected_fields(
    job: NoaaRtmaAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = source_fixture["messages"]
    messages.reverse()
    offset = 0
    for message in messages:
        message["offset"] = offset
        offset += message["length"]
    read = sparse_source(source_fixture)
    monkeypatch.setattr(
        module, "s3_read_bytes", lambda url, *, region, start, end: read(start, end)
    )
    coord = NoaaRtmaAnalysisVirtualSourceFileCoord(
        time=source_fixture["time"], data_vars=CONFIG.data_vars
    )
    refs = job.file_refs(coord, source_fixture["size"])
    assert [ref.data_var.internal_attrs.grib_element for ref in refs] == [
        m["element"] for m in messages
    ]


@pytest.mark.parametrize(
    ("region_job_class", "config", "fixture_name"),
    [
        (NoaaRtmaAnalysisVirtualRegionJob, CONFIG, "rtma/fixtures/20170502.json"),
        (
            NoaaUrmaAnalysisVirtualRegionJob,
            NoaaUrmaAnalysisVirtualTemplateConfig(),
            "urma/fixtures/urma_20181203T12.json",
        ),
        (
            NoaaRtmaRuAnalysisVirtualRegionJob,
            NoaaRtmaRuAnalysisVirtualTemplateConfig(),
            "rtma_ru/fixtures/20200728t0000.json",
        ),
    ],
    ids=["rtma", "urma", "rtma-ru"],
)
@pytest.mark.parametrize("processing_mode", ["backfill", "update"])
@pytest.mark.parametrize("defect", ["header", "terminator", "duplicate"])
def test_invalid_source_does_not_block_healthy_partial_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    region_job_class: type[NoaaRtmaAnalysisVirtualRegionJob],
    config: NoaaRtmaAnalysisVirtualTemplateConfig,
    fixture_name: str,
    processing_mode: Literal["backfill", "update"],
    defect: str,
) -> None:
    source = json.loads((Path(__file__).parents[2] / fixture_name).read_text())
    selected = config.data_vars[:2]
    missing_element = selected[1].internal_attrs.grib_element
    missing_message = next(
        i for i, m in enumerate(source["messages"]) if m["element"] == missing_element
    )
    edit_header(source, missing_message, 118 + 10, b"\xff")
    time = pd.Timestamp(source["time"])
    times = pd.date_range(time - config.append_dim_frequency, time, periods=2)
    ds = xr.Dataset({selected[0].name: ("time", [0, 0])}, coords={"time": times})
    ds[selected[0].name].encoding["chunks"] = (1,)
    job = region_job_class(
        tmp_store=tmp_path,
        template_ds=xr.DataTree(ds),
        data_vars=selected,
        append_dim="time",
        region=slice(0, 2),
        reformat_job_name="test",
        processing_mode=processing_mode,
    )
    bad, healthy = job.source_file_coords()
    read = sparse_source(source)
    bad_source = deepcopy(source)
    reference = bad.time.year.to_bytes(2) + bytes(
        [bad.time.month, bad.time.day, bad.time.hour, bad.time.minute, 0]
    )
    for i in range(len(bad_source["messages"])):
        edit_header(bad_source, i, 16 + 12, reference)
    if defect == "duplicate":
        append_message(bad_source, 0)
    bad_read = sparse_source(bad_source)

    def read_bytes(url: str, *, region: str, start: int, end: int) -> bytes:
        if url == bad.get_url():
            if defect == "header":
                return b""
            if defect == "terminator" and start == source["size"] - 4:
                return b"0000"
            return bad_read(start, end)
        assert url == healthy.get_url()
        return read(start, end)

    monkeypatch.setattr(module, "s3_read_bytes", read_bytes)
    discover = Mock(return_value=[(bad, bad_source["size"]), (healthy, source["size"])])
    monkeypatch.setattr(region_job_class, "discover_available", discover)
    (batch,) = list(job.process_virtual_refs([bad, healthy]))
    assert len(batch) == 1
    coord, refs = batch[0]
    assert coord == healthy
    assert len(refs) == 1
    assert refs[0].data_var == selected[0]
    assert refs[0].location == healthy.get_url()
    assert refs[0].out_loc == healthy.out_loc()
    assert f"Skipping {bad.get_url()}: could not build virtual refs" in caplog.text
    assert {
        "header": "Truncated GRIB header range",
        "terminator": "Missing final GRIB terminator",
        "duplicate": "Duplicate GRIB field",
    }[defect] in caplog.text
    discover.assert_called_once()


def test_duplicate_selected_config_identities_remain_invariant(
    job: NoaaRtmaAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read = sparse_source(source_fixture)
    monkeypatch.setattr(
        module, "s3_read_bytes", lambda url, *, region, start, end: read(start, end)
    )
    coord = NoaaRtmaAnalysisVirtualSourceFileCoord(
        time=source_fixture["time"],
        data_vars=[CONFIG.data_vars[0]] * 2,
    )
    with pytest.raises(AssertionError, match="Duplicate selected GRIB identities"):
        job._file_refs_or_skip(coord, source_fixture["size"])
