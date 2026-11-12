import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.virtual_region_job import VirtualRef
from reformatters.ecmwf.aifs_ens.forecast_virtual import region_job as module
from reformatters.ecmwf.aifs_ens.forecast_virtual.region_job import (
    EcmwfAifsEnsForecastVirtualSourceFileCoord,
)
from tests.ecmwf.aifs_ens.forecast_virtual.conftest import (
    CONFIG,
    FIXTURES,
    get_var,
    make_job,
)


def coordinates(
    init: str, leads: Sequence[int] = (0, 6), members: Sequence[int] = (0, 1, 50)
) -> xr.Dataset:
    return xr.Dataset(
        coords={
            "init_time": [pd.Timestamp(init)],
            "lead_time": pd.TimedeltaIndex([pd.Timedelta(hours=h) for h in leads]),
            "ensemble_member": list(members),
            "pressure_level": CONFIG.dimension_coordinates()["pressure_level"],
        }
    )


def serve_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixture: str
) -> None:
    def download(url: str, dataset_id: str) -> Path:
        path = tmp_path / "source.index"
        path.write_bytes((FIXTURES / fixture).read_bytes())
        return path

    monkeypatch.setattr(module, "gcs_download_to_disk", download)


@pytest.mark.parametrize(
    ("init", "new_fields"), [("2026-05-12T00", False), ("2026-05-12T06", True)]
)
def test_upgrade_boundary_does_not_request_absent_fields_or_levels(
    template: xr.DataTree, init: str, new_fields: bool
) -> None:
    job = make_job(template)
    coords = job.generate_source_file_coords(coordinates(init), CONFIG.data_vars)
    assert [(c.file_type, c.ensemble_members) for c in coords] == [
        ("cf", (0,)),
        ("pf", (1, 50)),
    ] * 2
    for coord in coords:
        params = {v.internal_attrs.grib_index_param for v in coord.data_vars}
        assert {"cp", "fscov", "vsw"}.issubset(params) == new_fields
        assert params.isdisjoint({"cp", "fscov", "vsw"}) != new_fields
        assert (10 in coord.pressure_levels) == new_fields
        assert {
            loc["pressure_level"]
            for loc in job.message_locations(
                coord, get_var("pressure_level/specific_humidity")
            )
        } == {50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000}
        assert all(
            v in coord.data_vars
            for v in CONFIG.data_vars
            if v.attrs.step_type == "accum"
            and (new_fields or v.internal_attrs.grib_index_param != "cp")
        )
        for var in CONFIG.data_vars:
            if var.internal_attrs.lead_0_only:
                assert (var in coord.data_vars) == (coord.lead_time == pd.Timedelta(0))
                if var in coord.data_vars:
                    assert {
                        loc["ensemble_member"]
                        for loc in job.message_locations(coord, var)
                    } == set(coord.ensemble_members)
                    assert (
                        coord.variable_out_loc(var)["ensemble_member"]
                        == coord.ensemble_members
                    )


@pytest.mark.parametrize("init", ["2025070200", "2026051300"])
@pytest.mark.parametrize("lead", [0, 6])
@pytest.mark.parametrize("file_type", ["cf", "pf"])
def test_real_indexes_route_every_message_to_its_member_and_level(
    template: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    init: str,
    lead: int,
    file_type: str,
) -> None:
    fixture = f"{init}_{lead}_{file_type}.index"
    iso = f"{init[:4]}-{init[4:6]}-{init[6:8]}T{init[8:]}"
    job = make_job(template)
    coord = next(
        c
        for c in job.generate_source_file_coords(
            coordinates(iso, [lead]), CONFIG.data_vars
        )
        if c.file_type == file_type
    )
    expected_url = f"gs://ecmwf-open-data/{init[:8]}/{init[8:]}z/aifs-ens/0p25/enfo/{init}0000-{lead}h-enfo-{file_type}"
    assert coord.get_url() == expected_url + ".grib2"
    assert coord.get_index_url() == expected_url + ".index"
    serve_fixture(monkeypatch, tmp_path, fixture)
    rows = [json.loads(line) for line in (FIXTURES / fixture).read_text().splitlines()]
    refs = job.file_refs(coord, max(r["_offset"] + r["_length"] for r in rows))
    count = (99 if init == "2025070200" else 108) + (4 if lead == 0 else 0)
    assert len(refs) == count * (1 if file_type == "cf" else 2)
    assert len({(r.data_var.path, tuple(r.out_loc.values())) for r in refs}) == len(
        refs
    )
    by_offset = {r.offset: r for r in refs}
    assert set(by_offset) == {row["_offset"] for row in rows}
    for row in rows:
        ref = by_offset[row["_offset"]]
        attrs = ref.data_var.internal_attrs
        assert ref.out_loc["ensemble_member"] == int(row.get("number", 0))
        assert attrs.grib_index_param == row["param"]
        assert attrs.grib_index_level_type == row["levtype"]
        assert ref.length == row["_length"]
        if row["levtype"] == "pl":
            assert ref.out_loc["pressure_level"] == int(row["levelist"])
        elif row["levtype"] == "sol":
            assert attrs.grib_index_level_value == int(row["levelist"])


def test_unrequested_out_of_bounds_index_entry_skips_whole_file(
    template: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = make_job(template, [get_var("temperature_2m")])
    coord = job.generate_source_file_coords(
        coordinates("2026-05-13", [6]), job.data_vars
    )[0]
    fixture = "2026051300_6_cf.index"
    rows = [json.loads(line) for line in (FIXTURES / fixture).read_text().splitlines()]
    size = max(row["_offset"] + row["_length"] for row in rows) - 1
    assert rows[-1]["param"] != "2t"
    serve_fixture(monkeypatch, tmp_path, fixture)
    assert job.file_refs(coord, size) == []
    assert caplog.records[-1].levelname == "WARNING"


def test_duplicate_source_selector_is_rejected(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(template, [get_var("temperature_2m")])
    coord = job.generate_source_file_coords(
        coordinates("2026-05-13", [6]), job.data_vars
    )[0]
    row = next(
        json.loads(line)
        for line in (FIXTURES / "2026051300_6_cf.index").read_text().splitlines()
        if json.loads(line)["param"] == "2t"
    )
    path = tmp_path / "duplicate.index"
    path.write_text((json.dumps(row) + "\n") * 2)
    monkeypatch.setattr(module, "gcs_download_to_disk", lambda *args: path)
    with pytest.raises(AssertionError, match="Duplicate source selector"):
        job.file_refs(coord, row["_offset"] + row["_length"])


@pytest.mark.parametrize("members", [(0,), (1, 50), (50,)])
def test_presence_probe_targets_a_member_the_file_contains(
    template: xr.DataTree, members: tuple[int, ...]
) -> None:
    job = make_job(template, [get_var("pressure_level/specific_humidity")])
    (coord,) = job.generate_source_file_coords(
        coordinates("2026-11-11", [6], members), job.data_vars
    )
    var = job.representative_var(coord)
    probe = job.representative_probe_loc(coord, var)
    assert probe["ensemble_member"] == members[0]
    assert probe in job.message_locations(coord, var)


def test_backfill_batches_bound_refs_to_400_files_without_losing_files(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(template, [get_var("temperature_2m")]).model_copy(
        update={"processing_mode": "backfill"}
    )
    coords = [
        job.generate_source_file_coords(
            coordinates("2026-11-11", [0], [0]), job.data_vars
        )[0]
        for _ in range(801)
    ]

    def discover(
        self: object, pending: list[EcmwfAifsEnsForecastVirtualSourceFileCoord]
    ) -> list[tuple[EcmwfAifsEnsForecastVirtualSourceFileCoord, int]]:
        return [(coord, 100) for coord in pending]

    def file_refs(
        self: object, coord: EcmwfAifsEnsForecastVirtualSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        return [
            VirtualRef(
                data_var=coord.data_vars[0],
                out_loc=job.representative_probe_loc(coord, coord.data_vars[0]),
                location=coord.get_url(),
                offset=0,
                length=file_size,
            )
        ]

    monkeypatch.setattr(type(job), "discover_available", discover)
    monkeypatch.setattr(type(job), "file_refs", file_refs)
    batches = list(job.process_virtual_refs(coords))
    assert [len(batch) for batch in batches] == [400, 400, 1]
    assert [id(coord) for batch in batches for coord, _ in batch] == [
        id(coord) for coord in coords
    ]
