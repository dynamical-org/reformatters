from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.rtma.analysis_virtual import region_job as shared
from reformatters.noaa.urma.analysis_virtual.region_job import (
    SOURCE_PREFIX,
    NoaaUrmaAnalysisVirtualRegionJob,
    NoaaUrmaAnalysisVirtualSourceFileCoord,
)
from reformatters.noaa.urma.analysis_virtual.template_config import (
    NoaaUrmaAnalysisVirtualTemplateConfig,
)
from tests.noaa.rtma.grib_header_test import append_message, edit_header, sparse_source

CONFIG = NoaaUrmaAnalysisVirtualTemplateConfig()
ONSET = pd.Timestamp("2018-12-03T12:00")
WAVE = CONFIG.data_vars[-1]
VARS = CONFIG.data_vars


@pytest.fixture
def job(tmp_path: Path) -> NoaaUrmaAnalysisVirtualRegionJob:
    return NoaaUrmaAnalysisVirtualRegionJob(
        tmp_store=tmp_path,
        template_ds=xr.DataTree(
            xr.Dataset(
                coords={"time": pd.date_range("2018-12-03T11:00", periods=2, freq="h")}
            )
        ),
        data_vars=VARS,
        append_dim="time",
        region=slice(0, 2),
        reformat_job_name="test",
    )


@pytest.mark.parametrize("wave_only", [False, True])
def test_wave_onset_filters_before_generating_coords(
    job: NoaaUrmaAnalysisVirtualRegionJob, wave_only: bool
) -> None:
    selected = [WAVE] if wave_only else VARS
    coords = job.generate_source_file_coords(job.template_ds.dataset, selected)
    assert [c.time for c in coords] == list(
        pd.date_range(
            "2018-12-03T12:00" if wave_only else "2018-12-03T11:00",
            periods=1 if wave_only else 2,
            freq="h",
        )
    )
    for coord in coords:
        assert type(coord) is NoaaUrmaAnalysisVirtualSourceFileCoord
        assert (
            coord.get_url()
            == f"s3://noaa-urma-pds/urma2p5.{coord.time:%Y%m%d}/urma2p5.t{coord.time:%H}z.2dvaranl_ndfd.grb2_wexp"
        )
        assert coord.out_loc() == {"time": coord.time}
        assert (WAVE in coord.data_vars) == (coord.time >= ONSET)


def test_listing_uses_urma_without_index(
    job: NoaaUrmaAnalysisVirtualRegionJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    coords = list(job.source_file_coords())
    listing = Mock(return_value={coords[0].get_url().removeprefix(SOURCE_PREFIX): 1234})
    monkeypatch.setattr(
        "reformatters.common.virtual_source_listing._list_objects", listing
    )
    assert job.discover_available(coords) == [(coords[0], 1234)]
    assert listing.call_args.args[1] == ["urma2p5.20181203/"]


@pytest.mark.parametrize(
    "subset", [None, ["significant_wave_height_surface"], ["temperature_2m"]]
)
def test_source_refs_cover_only_available_selected_fields(
    job: NoaaUrmaAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    subset: list[str] | None,
) -> None:
    selected = [v for v in VARS if subset is None or v.name in subset]
    read = sparse_source(source_fixture)
    calls: list[tuple[int, int]] = []

    def read_bytes(url: str, *, region: str, start: int, end: int) -> bytes:
        assert url == source_fixture["url"]
        assert region == "us-east-1"
        calls.append((start, end))
        return read(start, end)

    monkeypatch.setattr(shared, "s3_read_bytes", read_bytes)
    coords = job.generate_source_file_coords(
        xr.Dataset(coords={"time": [pd.Timestamp(source_fixture["time"])]}), selected
    )
    if (
        subset == ["significant_wave_height_surface"]
        and pd.Timestamp(source_fixture["time"]) < ONSET
    ):
        assert not coords
        return
    (coord,) = coords
    refs = job.file_refs(coord, source_fixture["size"])
    for extra in source_fixture["extra_ranges"]:
        assert ((extra["start"], extra["start"] + 5) in calls) == (
            WAVE in coord.data_vars
        )
    assert {r.data_var.name for r in refs} == {v.name for v in coord.data_vars}
    for ref in refs:
        message = next(
            m
            for m in source_fixture["messages"]
            if m["element"] == ref.data_var.internal_attrs.grib_element
        )
        assert (ref.offset, ref.length) == (message["offset"], message["length"])


@pytest.mark.parametrize("wave_only", [False, True])
def test_missing_wave_after_onset_keeps_usable_refs(
    job: NoaaUrmaAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    wave_only: bool,
) -> None:
    if not source_fixture["extra_ranges"]:
        pytest.skip("Source precedes bitmap onset")
    edit_header(
        source_fixture,
        next(
            i
            for i, m in enumerate(source_fixture["messages"])
            if m["element"] == "HTSGW"
        ),
        118 + 10,
        b"\xff",
    )
    read = sparse_source(source_fixture)
    monkeypatch.setattr(
        shared, "s3_read_bytes", lambda url, *, region, start, end: read(start, end)
    )
    coord = NoaaUrmaAnalysisVirtualSourceFileCoord(
        time=source_fixture["time"], data_vars=[WAVE] if wave_only else VARS
    )
    refs = job.file_refs(coord, source_fixture["size"])
    assert {ref.data_var.name for ref in refs} == (
        set() if wave_only else {var.name for var in VARS if var != WAVE}
    )
    assert len(refs) == (0 if wave_only else len(VARS) - 1)


def test_duplicate_source_field_skips_whole_file(
    job: NoaaUrmaAnalysisVirtualRegionJob,
    source_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    append_message(source_fixture, 0)
    read = sparse_source(source_fixture)
    monkeypatch.setattr(
        shared, "s3_read_bytes", lambda url, *, region, start, end: read(start, end)
    )
    (coord,) = job.generate_source_file_coords(
        xr.Dataset(coords={"time": [pd.Timestamp(source_fixture["time"])]}), VARS
    )
    assert job._file_refs_or_skip(coord, source_fixture["size"]) == []
    assert "Duplicate GRIB field" in caplog.text
