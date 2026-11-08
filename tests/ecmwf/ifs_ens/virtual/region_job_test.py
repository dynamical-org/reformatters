import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import obstore.store
import pandas as pd
import pytest
import xarray as xr

from reformatters.ecmwf import ecmwf_virtual_source as source_module
from reformatters.ecmwf.ifs_ens import virtual_region_job as module
from reformatters.ecmwf.ifs_ens.forecast_6_day_0_1_degree_virtual.region_job import (
    EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob,
)
from reformatters.ecmwf.ifs_ens.virtual_region_job import (
    S3_FALLBACK_BEFORE,
    EcmwfIfsEnsForecastVirtualSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.virtual_template_config import (
    PRESSURE_LEVELS,
    EcmwfIfsEnsVirtualDataVar,
    available_pressure_levels,
)
from tests.ecmwf.ifs_ens.virtual.conftest import CONFIG, FIXTURES, get_var, make_job


def coordinates(
    init: str,
    leads: Sequence[int] = (0, 3, 93, 150),
    members: Sequence[int] = (0, 1, 50),
) -> xr.Dataset:
    return xr.Dataset(
        coords={
            "init_time": [pd.Timestamp(init)],
            "lead_time": pd.TimedeltaIndex([pd.Timedelta(hours=h) for h in leads]),
            "ensemble_member": list(members),
            "pressure_level": list(PRESSURE_LEVELS),
        }
    )


def source_coord(
    data_vars: Sequence[EcmwfIfsEnsVirtualDataVar],
    init: str = "2026-11-06",
    lead: int = 24,
    family: Literal["enfo", "oper"] = "enfo",
) -> EcmwfIfsEnsForecastVirtualSourceFileCoord:
    return EcmwfIfsEnsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp(init),
        lead_time=pd.Timedelta(hours=lead),
        resolution="0p25",
        family=family,
        data_vars=data_vars,
        ensemble_members=(0,)
        if family == "oper"
        else (
            (1, 50)
            if pd.Timestamp(init) >= pd.Timestamp("2026-05-12T06")
            else (0, 1, 50)
        ),
        pressure_levels=available_pressure_levels(pd.Timestamp(init)),
    )


def serve_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixture: str
) -> None:
    def download(url: str, dataset_id: str, **kwargs: object) -> Path:
        p = tmp_path / "test.index"
        p.write_bytes((FIXTURES / fixture).read_bytes())
        return p

    monkeypatch.setattr(module, "gcs_download_to_disk", download)
    monkeypatch.setattr(module, "s3_download_to_disk", download)


@pytest.mark.parametrize(
    ("init", "families"),
    [("2026-05-12T00", ["enfo"]), ("2026-05-12T06", ["enfo", "oper"])],
)
def test_control_cutover(template: xr.DataTree, init: str, families: list[str]) -> None:
    coords = make_job(template).generate_source_file_coords(
        coordinates(init, [24]), CONFIG.data_vars
    )
    assert [c.family for c in coords] == families
    assert coords[0].ensemble_members == ((0, 1, 50) if len(families) == 1 else (1, 50))
    assert all(not v.internal_attrs.control_only for v in coords[0].data_vars)
    if len(families) > 1:
        assert coords[1].ensemble_members == (0,)
        assert get_var("pressure_level/geopotential") in coords[1].data_vars
        assert "-24h-oper-fc.grib2" in coords[1].get_url()


def test_fine_resolution_file_families(template: xr.DataTree) -> None:
    job = EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob.model_validate(
        make_job(template).model_dump()
    )
    coords = job.generate_source_file_coords(
        coordinates("2026-11-06", [24]), CONFIG.data_vars
    )
    assert [(c.family, c.level_family) for c in coords] == [
        ("enfo", "sfc"),
        ("enfo", "pl"),
        ("oper", "sfc"),
        ("oper", "pl"),
    ]
    for c in coords:
        assert f"/0p1/{c.family}/" in c.get_url()
        assert c.get_url().endswith(f"-{c.level_family}.grib2")
        assert all(
            (v.group == "pressure_level") == (c.level_family == "pl")
            for v in c.data_vars
        )


def test_early_pressure_probe_and_expired_cape(template: xr.DataTree) -> None:
    job = make_job(template)
    pressure = get_var("pressure_level/temperature")
    (c,) = job.generate_source_file_coords(coordinates("2024-04-01", [24]), [pressure])
    probe = job.representative_probe_loc(c, pressure)
    assert probe["pressure_level"] == 50
    assert probe["ensemble_member"] == 1
    cape = get_var("convective_available_potential_energy_surface")
    assert job.generate_source_file_coords(coordinates("2024-11-12T06"), [cape]) == []
    assert (
        len(job.generate_source_file_coords(coordinates("2024-11-12T00", [24]), [cape]))
        == 1
    )


@pytest.mark.parametrize(
    ("init", "param", "levtype", "level"),
    [("2024-11-12T00", "stl3", "sfc", None), ("2024-11-12T06", "sot", "sol", 3)],
)
def test_soil_selector(init: str, param: str, levtype: str, level: int | None) -> None:
    assert get_var("soil_temperature_layer_3").source_selector(
        pd.Timestamp(init), pd.Timedelta("24h")
    ) == (param, levtype, level)


def test_lead_zero_contract_and_window_parameters(template: xr.DataTree) -> None:
    coords = make_job(template).generate_source_file_coords(
        coordinates("2026-11-06"), CONFIG.data_vars
    )
    for c in coords:
        if c.lead_time == pd.Timedelta(0):
            assert all(v.attrs.step_type not in {"max", "min"} for v in c.data_vars)
            assert get_var("total_precipitation_run_total_surface") in c.data_vars
            assert get_var("precipitation_rate_surface") in c.data_vars
        else:
            assert all(not v.internal_attrs.lead_0_only for v in c.data_vars)
    assert (
        get_var("wind_gust_10m").source_selector(
            pd.Timestamp("2026-11-06"), pd.Timedelta("93h")
        )[0]
        == "10fg3"
    )
    assert (
        get_var("maximum_temperature_2m").source_selector(
            pd.Timestamp("2026-11-06"), pd.Timedelta("150h")
        )[0]
        == "mx2t6"
    )


@pytest.mark.parametrize(
    ("fixture", "init", "lead", "family", "count"),
    [
        ("20240401_00_24_enfo_0p25.index", "2024-04-01", 24, "enfo", 327),
        ("20261106_0p25_enfo_24_enfo-ef.index", "2026-11-06", 24, "enfo", 354),
        ("20261106_0p25_enfo_0_enfo-ef.index", "2026-11-06", 0, "enfo", 348),
        ("20261106_0p25_enfo_150_enfo-ef.index", "2026-11-06", 150, "enfo", 354),
        ("20261106_0p25_oper_24_oper-fc.index", "2026-11-06", 24, "oper", 191),
        ("20261106_0p25_oper_0_oper-fc.index", "2026-11-06", 0, "oper", 191),
    ],
)
def test_real_source_inventory(
    template: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture: str,
    init: str,
    lead: int,
    family: str,
    count: int,
) -> None:
    job = make_job(template)
    coords = job.generate_source_file_coords(
        coordinates(init, [lead]), CONFIG.data_vars
    )
    c = next(c for c in coords if c.family == family)
    serve_fixture(monkeypatch, tmp_path, fixture)
    rows = [json.loads(line) for line in (FIXTURES / fixture).read_text().splitlines()]
    size = max(row["_offset"] + row["_length"] for row in rows)
    refs = job._file_refs_or_skip(c, size)
    assert len(refs) == count
    assert len({(r.data_var.path, tuple(r.out_loc.values())) for r in refs}) == count
    assert all(r.out_loc["ensemble_member"] in c.ensemble_members for r in refs)


def test_partial_control_block_keeps_healthy_perturbed_refs(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = "20240918_18_0_enfo_0p25.index"
    job = make_job(template)
    (c,) = job.generate_source_file_coords(
        coordinates("2024-09-18T18", [0]), CONFIG.data_vars
    )
    serve_fixture(monkeypatch, tmp_path, fixture)
    refs = job._file_refs_or_skip(c, 10_000_000_000)
    assert sum(r.out_loc["ensemble_member"] != 0 for r in refs) == 218
    assert sum(r.out_loc["ensemble_member"] == 0 for r in refs) == 81
    assert not any(
        r.out_loc["ensemble_member"] == 0 and r.data_var.name == "pressure_surface"
        for r in refs
    )
    assert any(
        r.out_loc["ensemble_member"] == 1 and r.data_var.name == "pressure_surface"
        for r in refs
    )


@pytest.mark.parametrize("init", ["2024-04-01", "2025-02-13", "2026-11-06"])
def test_fallback_cutoff(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch, init: str
) -> None:
    c = source_coord([get_var("temperature_2m")], init=init)
    calls = []

    def listing(
        pending: list[EcmwfIfsEnsForecastVirtualSourceFileCoord],
        *,
        store: obstore.store.ObjectStore,
        location_prefix: str,
        require_index: bool,
    ) -> list[tuple[EcmwfIfsEnsForecastVirtualSourceFileCoord, int]]:
        calls.append(location_prefix)
        assert require_index
        return [(p, 100) for p in pending] if location_prefix.startswith("s3") else []

    monkeypatch.setattr(source_module, "discover_available_by_obstore_listing", listing)
    found = make_job(template).discover_available([c])
    if pd.Timestamp(init) < S3_FALLBACK_BEFORE:
        assert found == [(c, 100)]
        assert c.get_url().startswith("s3://ecmwf-forecasts/")
    else:
        assert found == []
        assert calls == ["gs://ecmwf-open-data/"]


def test_gcs_hit_avoids_fallback(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = source_coord([get_var("temperature_2m")], init="2024-04-01")

    def listing(
        pending: list[EcmwfIfsEnsForecastVirtualSourceFileCoord], **kwargs: object
    ) -> list[tuple[EcmwfIfsEnsForecastVirtualSourceFileCoord, int]]:
        assert kwargs["location_prefix"] == "gs://ecmwf-open-data/"
        return [(pending[0], 100)]

    monkeypatch.setattr(source_module, "discover_available_by_obstore_listing", listing)
    assert make_job(template).discover_available([c]) == [(c, 100)]
    assert c.source == "gcs"


def test_stale_index_skipped(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = source_coord([get_var("temperature_2m")])
    serve_fixture(monkeypatch, tmp_path, "20261106_0p25_enfo_24_enfo-ef.index")
    assert make_job(template).file_refs(c, 1) == []


@pytest.mark.parametrize(
    ("family", "level_family", "lead", "count"),
    [
        ("enfo", "sfc", 24, 102),
        ("enfo", "pl", 24, 252),
        ("oper", "sfc", 24, 51),
        ("oper", "pl", 24, 140),
        ("enfo", "sfc", 0, 96),
        ("enfo", "pl", 0, 252),
        ("enfo", "sfc", 150, 102),
        ("enfo", "pl", 150, 252),
    ],
)
def test_real_fine_resolution_inventory(
    template: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    family: str,
    level_family: str,
    lead: int,
    count: int,
) -> None:
    job = EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob.model_validate(
        make_job(template).model_dump()
    )
    coord = next(
        c
        for c in job.generate_source_file_coords(
            coordinates("2026-11-06", [lead]), CONFIG.data_vars
        )
        if c.family == family and c.level_family == level_family
    )
    fixture = f"20261106_0p1_{family}_{lead}_{family}-{'ef' if family == 'enfo' else 'fc'}-{level_family}.index"
    serve_fixture(monkeypatch, tmp_path, fixture)
    assert len(job._file_refs_or_skip(coord, 100_000_000_000)) == count


def test_fine_resolution_never_falls_back_even_for_old_coord(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob.model_validate(
        make_job(template).model_dump()
    )
    coord = source_coord([get_var("temperature_2m")], init="2024-04-01").model_copy(
        update={"resolution": "0p1", "level_family": "sfc"}
    )

    def listing(
        pending: list[EcmwfIfsEnsForecastVirtualSourceFileCoord], **kwargs: object
    ) -> list[tuple[EcmwfIfsEnsForecastVirtualSourceFileCoord, int]]:
        assert kwargs["location_prefix"] == "gs://ecmwf-open-data/"
        return []

    monkeypatch.setattr(source_module, "discover_available_by_obstore_listing", listing)
    assert job.discover_available([coord]) == []


def test_s3_refs_keep_source_and_legacy_soil_selectors(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(template)
    (coord,) = job.generate_source_file_coords(
        coordinates("2024-04-01", [24]), [get_var("soil_temperature_layer_3")]
    )
    coord.fall_back_to_s3()
    serve_fixture(monkeypatch, tmp_path, "20240401_00_24_enfo_0p25.index")
    refs = job.file_refs(coord, 10_000_000_000)
    assert len(refs) == 3
    assert all(r.location.startswith("s3://ecmwf-forecasts/") for r in refs)
    assert {r.out_loc["ensemble_member"] for r in refs} == {0, 1, 50}


@pytest.mark.parametrize(
    ("processing_mode", "sizes"), [("backfill", [2, 1]), ("update", [3])]
)
def test_backfill_bounds_reference_batches_without_splitting_update_sweeps(
    template: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    processing_mode: str,
    sizes: list[int],
) -> None:
    job = make_job(template).model_copy(update={"processing_mode": processing_mode})
    monkeypatch.setattr(type(job), "backfill_files_per_batch", 2)
    coords = [
        source_coord([get_var("temperature_2m")], lead=lead) for lead in (0, 3, 6)
    ]

    def discover(
        _self: object, pending: list[EcmwfIfsEnsForecastVirtualSourceFileCoord]
    ) -> list[tuple[EcmwfIfsEnsForecastVirtualSourceFileCoord, int]]:
        return [(coord, 100) for coord in pending]

    def file_refs(
        _self: object, coord: EcmwfIfsEnsForecastVirtualSourceFileCoord, file_size: int
    ) -> list[module.VirtualRef]:
        return [
            module.VirtualRef(
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
    assert [len(batch) for batch in batches] == sizes
    yielded = [coord for batch in batches for coord, refs in batch if refs]
    assert [id(coord) for coord in yielded] == [id(coord) for coord in coords]


@pytest.mark.parametrize("canonical", [False, True])
def test_temperature_aliases_cover_all_members(
    template: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    canonical: bool,
) -> None:
    variables = [get_var("temperature_2m"), get_var("skin_temperature_surface")]
    coord = source_coord(variables, init="2024-09-17T18", lead=3).model_copy(
        update={"ensemble_members": tuple(range(51))}
    )
    rows = [
        json.loads(line)
        for line in (FIXTURES / "20240917_18_3_temperature_aliases.index")
        .read_text()
        .splitlines()
    ]
    if canonical:
        rows += [
            {
                **row,
                "param": {"mean2t": "2t", "mskt": "skt"}[row["param"]],
                "_offset": row["_offset"] + 1,
            }
            for row in rows.copy()
        ]
    index_path = tmp_path / "aliases.index"
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(module, "gcs_download_to_disk", lambda *args: index_path)
    refs = make_job(template, variables).file_refs(coord, 10_000_000_000)
    assert len(refs) == 102
    for var in variables:
        assert {
            ref.out_loc["ensemble_member"] for ref in refs if ref.data_var == var
        } == set(range(51))
    assert {ref.offset for ref in refs} == {
        row["_offset"] for row in rows[(102 if canonical else 0) :]
    }


def test_duplicate_alias_selector_fails(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    row = json.loads(
        (FIXTURES / "20240917_18_3_temperature_aliases.index")
        .read_text()
        .splitlines()[0]
    )
    path = tmp_path / "duplicate.index"
    path.write_text((json.dumps(row) + "\n") * 2)
    monkeypatch.setattr(module, "gcs_download_to_disk", lambda *args: path)
    variables = [get_var("temperature_2m"), get_var("skin_temperature_surface")]
    coord = source_coord(variables, init="2024-09-17T18", lead=3).model_copy(
        update={"ensemble_members": tuple(range(51))}
    )
    with pytest.raises(AssertionError, match="Duplicate source selector"):
        make_job(template, variables).file_refs(coord, 10_000_000_000)


def test_real_index_with_out_of_bounds_unrequested_rows_skips_file(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    var = get_var("temperature_2m")
    coord = source_coord([var], init="2024-09-18T06", lead=111)
    fixture = "20240918_06_111_enfo_0p25.index"
    rows = [json.loads(line) for line in (FIXTURES / fixture).read_text().splitlines()]
    file_size = 4472635497
    assert max(row["_offset"] + row["_length"] for row in rows) == 4472760345
    assert rows[-1]["param"] == "r"
    serve_fixture(monkeypatch, tmp_path, fixture)
    assert make_job(template, [var]).file_refs(coord, file_size) == []
