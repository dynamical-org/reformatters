from itertools import product
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pytest
import zarr

from reformatters.google.weathernext_virtual.holdback_audit import (
    audit_ancestry,
    audit_source_locations,
    forbidden_chunk_keys,
    parse_wn2_source_location,
    run_audit,
)


@pytest.mark.parametrize("spatial", [("y", "x"), ("latitude", "longitude")])
@pytest.mark.parametrize(
    "extra_dims", [(), ("statistic",), ("statistic", "pressure_level")]
)
def test_audit_all_nonspatial_dimensions(
    spatial: tuple[str, str], extra_dims: tuple[str, ...], tmp_path: Path
) -> None:
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="w")
    root.create_array(
        "init_time",
        data=np.array([0, 3600], dtype="int64"),
        dimension_names=("init_time",),
        attributes={"units": "seconds since 1970-01-01 00:00:00"},
    )
    root.create_array(
        "lead_time",
        data=np.array([3600, 7200], dtype="float64"),
        dimension_names=("lead_time",),
        attributes={"units": "seconds"},
    )
    sizes = {"statistic": 6, "pressure_level": 2}
    chunks = {"statistic": 2, "pressure_level": 1}
    root.create_array(
        "temperature",
        shape=(2, 2, 2, 2, *(sizes[dim] for dim in extra_dims)),
        chunks=(1, 1, 1, 2, *(chunks[dim] for dim in extra_dims)),
        dtype="float32",
        dimension_names=("lead_time", spatial[0], "init_time", spatial[1], *extra_dims),
    )
    planted_key = (
        "temperature/c/1/1/1/0"
        + "/2" * ("statistic" in extra_dims)
        + "/1" * ("pressure_level" in extra_dims)
    )
    session.store.set_virtual_ref(
        planted_key,
        "https://example.com/unread-chunk",
        offset=0,
        length=16,
        validate_container=False,
    )
    snapshot = session.commit("plant forbidden reference")
    cutoff = pd.Timestamp("1970-01-01T02:00")
    expected = {
        "temperature/c/1/" + str(y) + "/1/0" + "".join(f"/{i}" for i in indices)
        for y in range(2)
        for indices in product(
            *(range(sizes[dim] // chunks[dim]) for dim in extra_dims)
        )
    }
    group = zarr.open_group(repo.readonly_session(snapshot_id=snapshot).store, mode="r")
    assert {chunk.key for chunk in forbidden_chunk_keys(group, cutoff)} == expected

    result = run_audit(repo, "fixture", cutoff, snapshot, None, tmp_path)

    assert result.total_keys == len(expected)
    assert result.present_keys == 1
    assert result.present_steps == {
        (pd.Timestamp("1970-01-01T01:00"), pd.Timedelta("2h"))
    }
    assert result.max_present_valid_time == pd.Timestamp("1970-01-01T03:00")
    assert result.last_age_out == pd.Timestamp("1970-01-01T04:00")
    assert f"- Total keys probed: {len(expected)}" in result.report_path.read_text()
    assert repo.lookup_branch("main") == snapshot


@pytest.mark.parametrize("index", [0, 1, 60])
def test_source_audit_checks_underlying_url_without_fetching(
    index: int, tmp_path: Path
) -> None:

    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    group = zarr.open_group(session.store, mode="w")
    group.create_array("temperature", shape=(1,), chunks=(1,), dtype="float32")
    # The target is eligible; only the independently planted source URL is forbidden.
    location = (
        "https://wn.dynamical.org/chunks/weathernext_2_0_0/zarr/2025_to_present/"
        f"20250101_00hr_01_preds/predictions.zarr/2m_temperature/0.{index}.0.0"
    )
    session.store.set_virtual_ref(
        "temperature/c/0", location, offset=0, length=16, validate_container=False
    )
    snapshot = session.commit(
        "planted source", metadata={"publication_cutoff": "2025-01-01T06:00:00+00:00"}
    )
    if index:
        with pytest.raises(AssertionError, match="invalid source locations"):
            audit_source_locations(repo, snapshot, parse_wn2_source_location, tmp_path)
    else:
        result = audit_source_locations(
            repo, snapshot, parse_wn2_source_location, tmp_path
        )
        assert result.total_locations == 1
        assert result.peak_rss_kib > 0
    assert "Process peak RSS" in (tmp_path / "holdback_source_locations.md").read_text()


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.example/chunks/weathernext_2_0_0/zarr/2025_to_present/20250101_00hr_01_preds/predictions.zarr/2m_temperature/0.0.0.0",
        "https://wn.dynamical.org/chunks/weathernext_2_0_0/zarr/2025_to_present/20250101_01hr_01_preds/predictions.zarr/2m_temperature/0.0.0.0",
        "https://wn.dynamical.org/chunks/weathernext_2_0_0/zarr/2025_to_present/20250101_00hr_01_preds/predictions.zarr/unknown/0.0.0.0",
    ],
)
def test_source_parser_rejects_unexpected_paths(path: str) -> None:

    with pytest.raises(AssertionError):
        parse_wn2_source_location(path)


def test_recorded_audit_requires_metadata(tmp_path: Path) -> None:
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    with pytest.raises(AssertionError, match="no publication_cutoff"):
        run_audit(repo, "fixture", None, repo.lookup_branch("main"), None, tmp_path)


@pytest.mark.parametrize(("second", "passes"), [("09:10", True), ("08:59", False)])
def test_ancestry_uses_effective_cutoffs(
    second: str, passes: bool, tmp_path: Path
) -> None:

    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    for cutoff in ("09:20", second):
        session = repo.writable_session("main")
        zarr.open_group(session.store, mode="a").attrs["cutoff"] = cutoff
        session.commit(
            "cutoff", metadata={"publication_cutoff": f"2025-01-01T{cutoff}:00+00:00"}
        )
    if passes:
        audit_ancestry(repo, tmp_path)
    else:
        with pytest.raises(AssertionError, match="decreased"):
            audit_ancestry(repo, tmp_path)


def test_ancestry_reports_ahead_writer_clock(tmp_path: Path) -> None:

    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="a").attrs["clock"] = "ahead"
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta("55min")
    session.commit("ahead clock", metadata={"publication_cutoff": cutoff.isoformat()})
    report = audit_ancestry(repo, tmp_path).read_text()
    assert "False" in report
    assert "±10 minutes" in report


def test_recorded_target_audit_catches_aged_out_violation(tmp_path: Path) -> None:
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="w")
    root.create_array(
        "init_time",
        data=np.array([0], dtype="int64"),
        dimension_names=("init_time",),
        attributes={"units": "seconds since 1970-01-01"},
    )
    root.create_array(
        "lead_time",
        data=np.array([7200], dtype="float64"),
        dimension_names=("lead_time",),
        attributes={"units": "seconds"},
    )
    root.create_array(
        "temperature",
        shape=(1, 1, 1, 1),
        chunks=(1, 1, 1, 1),
        dtype="float32",
        dimension_names=("init_time", "lead_time", "y", "x"),
    )
    session.store.set_virtual_ref(
        "temperature/c/0/0/0/0",
        "https://invalid.example/no-fetch",
        offset=0,
        length=4,
        validate_container=False,
    )
    snapshot = session.commit(
        "past violation", metadata={"publication_cutoff": "1970-01-01T01:00:00+00:00"}
    )
    with pytest.raises(AssertionError, match="1 forbidden refs"):
        run_audit(repo, "fixture", None, snapshot, None, tmp_path)
    assert "1970-01-01T01:00:00+00:00" in (tmp_path / "holdback_audit.md").read_text()


@pytest.mark.parametrize("pressure", [False, True])
def test_wn2_historical_source_parser(pressure: bool) -> None:
    variable = "temperature" if pressure else "2m_temperature"
    indices = "0.15.59.0.0.0" if pressure else "0.15.59.0.0"
    init, lead = parse_wn2_source_location(
        f"https://wn.dynamical.org/chunks/weathernext_2_0_0/zarr/2022_to_2023/predictions.zarr/{variable}/{indices}"
    )
    assert init == pd.Timestamp("2022-01-01")
    assert lead == pd.Timedelta("360h")
