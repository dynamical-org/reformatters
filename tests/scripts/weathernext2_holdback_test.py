import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pytest
import typer
import zarr
from typer.testing import CliRunner

from scripts import icechunk_utils, weathernext2_holdback
from scripts.weathernext2_holdback import (
    _operational_dataset,
    _parse_cutoff,
    _run_audit,
    _run_delete,
    _run_publish,
    forbidden_chunk_keys,
)


@dataclass(frozen=True)
class HoldbackRepo:
    storage: icechunk.Storage
    repo: icechunk.Repository
    cutoff: pd.Timestamp
    init_times: pd.DatetimeIndex
    lead_times: pd.TimedeltaIndex


@pytest.fixture
def holdback_repo() -> HoldbackRepo:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="w")

    init_times = pd.date_range("2026-01-01", periods=4, freq="6h")
    lead_times = pd.timedelta_range("6h", periods=3, freq="6h")
    init_seconds = init_times.to_numpy(dtype="datetime64[s]").astype("int64")
    lead_seconds = lead_times.to_numpy(dtype="timedelta64[s]").astype("float64")

    def create_coordinates(group: zarr.Group) -> None:
        group.create_array(
            "init_time",
            data=init_seconds,
            chunks=(4,),
            attributes={
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
            },
            dimension_names=("init_time",),
        )
        group.create_array(
            "lead_time",
            data=lead_seconds,
            chunks=(3,),
            attributes={"units": "seconds"},
            dimension_names=("lead_time",),
        )

    create_coordinates(root)
    root.create_array("spatial_ref", data=np.array(0, dtype="int32"), chunks=())
    root.create_array(
        "valid_time",
        data=np.add.outer(init_seconds, lead_seconds).astype("int64"),
        chunks=(1, 1),
        dimension_names=("init_time", "lead_time"),
    )
    root_array = root.create_array(
        "temperature_2m",
        shape=(4, 2, 3, 2, 2),
        chunks=(1, 1, 1, 2, 2),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=("init_time", "ensemble_member", "lead_time", "y", "x"),
    )
    root_array[:] = 1

    pressure_group = root.create_group("pressure_level")
    create_coordinates(pressure_group)
    pressure_group.create_array(
        "spatial_ref", data=np.array(0, dtype="int32"), chunks=()
    )
    pressure_array = pressure_group.create_array(
        "temperature",
        shape=(4, 2, 3, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 1),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=(
            "init_time",
            "ensemble_member",
            "lead_time",
            "y",
            "x",
            "pressure_level",
        ),
    )
    pressure_array[:] = 2
    session.commit("create holdback test store")
    return HoldbackRepo(
        storage=storage,
        repo=repo,
        cutoff=init_times[0] + lead_times[-1],
        init_times=init_times,
        lead_times=lead_times,
    )


def _expected_keys(repo: HoldbackRepo) -> set[str]:
    expected = set()
    for init_index, init_time in enumerate(repo.init_times):
        for lead_index, lead_time in enumerate(repo.lead_times):
            if init_time + lead_time <= repo.cutoff:
                continue
            for member in range(2):
                expected.add(f"temperature_2m/c/{init_index}/{member}/{lead_index}/0/0")
                for level in range(2):
                    expected.add(
                        "pressure_level/temperature/c/"
                        f"{init_index}/{member}/{lead_index}/0/0/{level}"
                    )
    return expected


def _utc_iso(value: pd.Timestamp) -> str:
    return value.tz_localize("UTC").isoformat()


def _delete_keys(store: icechunk.IcechunkStore, keys: set[str]) -> None:
    async def delete() -> None:
        await asyncio.gather(*(store.delete(key) for key in keys))

    asyncio.run(delete())


def test_forbidden_chunk_keys_are_exhaustive_and_cutoff_is_inclusive(
    holdback_repo: HoldbackRepo,
) -> None:
    store = holdback_repo.repo.readonly_session(branch="main").store
    group = zarr.open_group(store, mode="r")

    chunks = list(forbidden_chunk_keys(group, holdback_repo.cutoff))

    assert {chunk.key for chunk in chunks} == _expected_keys(holdback_repo)
    assert {chunk.var_path for chunk in chunks} == {
        "temperature_2m",
        "pressure_level/temperature",
    }
    assert len(chunks) == 36
    assert not any(
        chunk.init_time + chunk.lead_time == holdback_repo.cutoff for chunk in chunks
    )
    assert not any(
        chunk.init_time == holdback_repo.init_times[2]
        and chunk.lead_time == holdback_repo.lead_times[0]
        for chunk in chunks
    )


def test_forbidden_chunk_keys_reject_sharded_arrays(
    holdback_repo: HoldbackRepo,
) -> None:
    session = holdback_repo.repo.writable_session("main")
    root = zarr.open_group(session.store, mode="r+")
    root.create_array(
        "sharded",
        shape=(4, 2, 3, 2, 2),
        chunks=(1, 1, 1, 1, 1),
        shards=(1, 2, 1, 2, 2),
        dtype="float32",
        dimension_names=("init_time", "ensemble_member", "lead_time", "y", "x"),
    )
    session.commit("add unsupported sharded array")
    store = holdback_repo.repo.readonly_session(branch="main").store
    group = zarr.open_group(store, mode="r")

    with pytest.raises(NotImplementedError, match="sharded arrays are unsupported"):
        list(forbidden_chunk_keys(group, holdback_repo.cutoff))


def test_parse_cutoff_requires_offset_and_normalizes_to_naive_utc() -> None:
    with pytest.raises(typer.BadParameter, match="must include a UTC offset"):
        _parse_cutoff("2026-01-01T00:00:00")

    result = _parse_cutoff("2026-01-01T01:30:00+01:30")

    assert result == pd.Timestamp("2026-01-01T00:00:00")
    assert result.tz is None


def test_delete_and_publish_require_explicit_cutoff() -> None:
    runner = CliRunner()

    delete_result = runner.invoke(
        weathernext2_holdback.app,
        ["delete", "google-weathernext2-forecast-operational-virtual"],
    )
    publish_result = runner.invoke(
        weathernext2_holdback.app,
        [
            "publish",
            "google-weathernext2-forecast-operational-virtual",
            "--from-snapshot",
            "snapshot-id",
        ],
    )

    assert delete_result.exit_code == 2
    assert "--cutoff" in delete_result.output
    assert publish_result.exit_code == 2
    assert "--cutoff" in publish_result.output


def test_destructive_commands_reject_non_operational_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_lookup(_dataset_id: str) -> None:
        pytest.fail("registry lookup must follow the exact dataset guard")

    monkeypatch.setattr(
        weathernext2_holdback, "resolve_virtual_dataset", unexpected_lookup
    )

    with pytest.raises(typer.BadParameter, match="destructive commands only support"):
        _operational_dataset("google-weathernext2-forecast-historical-virtual")


def test_audit_reports_present_refs_timing_and_exits_one(
    holdback_repo: HoldbackRepo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        weathernext2_holdback,
        "_icechunk_storage",
        lambda _url: holdback_repo.storage,
    )
    output_dir = tmp_path / "audit"

    with pytest.raises(typer.Exit) as exc_info:
        weathernext2_holdback.audit(
            "https://example.com/test.icechunk",
            cutoff=_utc_iso(holdback_repo.cutoff),
            output=output_dir,
        )

    assert exc_info.value.exit_code == 1
    max_valid_time = holdback_repo.init_times[-1] + holdback_repo.lead_times[-1]
    report = (output_dir / "holdback_audit.md").read_text()
    assert "- Total keys probed: 36" in report
    assert "- Keys present: 36" in report
    assert "- Steps with any present ref: 6" in report
    assert "- Inits with any present ref: 3" in report
    assert f"- Maximum present valid time: `{_utc_iso(max_valid_time)}`" in report
    assert (
        f"- Last age-out: `{_utc_iso(max_valid_time + pd.Timedelta(hours=1))}`"
        in report
    )
    assert "temperature_2m" in report
    assert "pressure_level/temperature" in report


def test_audit_finds_sparse_nonrepresentative_ref(
    holdback_repo: HoldbackRepo, tmp_path: Path
) -> None:
    target_init_index = 2
    target_lead_index = 2
    target_key = (
        f"pressure_level/temperature/c/{target_init_index}/1/{target_lead_index}/0/0/1"
    )
    session = holdback_repo.repo.writable_session("main")
    _delete_keys(session.store, _expected_keys(holdback_repo) - {target_key})
    sparse_snapshot = session.commit("leave one nonrepresentative forbidden ref")

    result = _run_audit(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        sparse_snapshot,
        None,
        tmp_path,
    )

    expected_valid_time = (
        holdback_repo.init_times[target_init_index]
        + holdback_repo.lead_times[target_lead_index]
    )
    assert result.present_keys == 1
    assert result.max_present_valid_time == expected_valid_time
    assert result.last_age_out == expected_valid_time + pd.Timedelta(hours=1)
    report = result.report_path.read_text()
    assert f"- Maximum present valid time: `{_utc_iso(expected_valid_time)}`" in report


def test_audit_uses_explicit_snapshot_read_only(
    holdback_repo: HoldbackRepo, tmp_path: Path
) -> None:
    audited_snapshot = holdback_repo.repo.lookup_branch("main")
    session = holdback_repo.repo.writable_session("main")
    _delete_keys(session.store, _expected_keys(holdback_repo))
    new_main = session.commit("remove refs from later main")
    branches_before = holdback_repo.repo.list_branches()

    result = _run_audit(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        audited_snapshot,
        None,
        tmp_path,
    )

    assert result.present_keys == 36
    assert holdback_repo.repo.lookup_branch("main") == new_main
    assert holdback_repo.repo.list_branches() == branches_before


def test_audit_rejects_branch_and_snapshot(
    holdback_repo: HoldbackRepo, tmp_path: Path
) -> None:
    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        weathernext2_holdback.audit(
            "https://example.com/test.icechunk",
            cutoff=_utc_iso(holdback_repo.cutoff),
            snapshot=holdback_repo.repo.lookup_branch("main"),
            branch="main",
            output=tmp_path,
        )


def test_delete_dry_run_leaves_repo_unchanged(
    holdback_repo: HoldbackRepo, tmp_path: Path
) -> None:
    main_before = holdback_repo.repo.lookup_branch("main")
    branches_before = holdback_repo.repo.list_branches()
    ancestry_before = [
        snapshot.id for snapshot in holdback_repo.repo.ancestry(branch="main")
    ]

    result = _run_delete(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        tmp_path,
        force=False,
    )

    assert result.present_keys == 36
    assert holdback_repo.repo.lookup_branch("main") == main_before
    assert holdback_repo.repo.list_branches() == branches_before
    assert [
        snapshot.id for snapshot in holdback_repo.repo.ancestry(branch="main")
    ] == ancestry_before


def test_delete_force_only_changes_purge_branch(
    holdback_repo: HoldbackRepo, tmp_path: Path
) -> None:
    initial_snapshot = holdback_repo.repo.lookup_branch("main")

    delete_result = _run_delete(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        tmp_path / "delete",
        force=True,
        root_split=2,
        pressure_split=1,
    )

    assert delete_result.present_keys == 36
    assert holdback_repo.repo.lookup_branch("main") == initial_snapshot
    purge_tip = holdback_repo.repo.lookup_branch("holdback-purge")
    purge_commits = []
    for snapshot in holdback_repo.repo.ancestry(branch="holdback-purge"):
        if snapshot.id == initial_snapshot:
            break
        purge_commits.append(snapshot)
    assert len(purge_commits) == 5
    assert all(snapshot.message.startswith("Remove ") for snapshot in purge_commits)

    main_result = _run_audit(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        None,
        "main",
        tmp_path / "main-audit",
    )
    assert main_result.present_keys == 36

    resumed_result = _run_delete(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        tmp_path / "resumed-delete",
        force=True,
        root_split=2,
        pressure_split=1,
    )
    assert resumed_result.present_keys == 0
    assert holdback_repo.repo.lookup_branch("holdback-purge") == purge_tip


def test_publish_audits_then_atomically_moves_main(
    holdback_repo: HoldbackRepo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_snapshot = holdback_repo.repo.lookup_branch("main")
    _run_delete(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        tmp_path / "delete",
        force=True,
        root_split=2,
        pressure_split=1,
    )
    purge_tip = holdback_repo.repo.lookup_branch("holdback-purge")
    audited = False
    real_run_audit = weathernext2_holdback._run_audit

    def recording_audit(
        repo: icechunk.Repository,
        store_label: str,
        cutoff: pd.Timestamp,
        snapshot_id: str | None,
        branch: str | None,
        output_dir: Path,
    ) -> weathernext2_holdback.HoldbackAudit:
        nonlocal audited
        result = real_run_audit(
            repo, store_label, cutoff, snapshot_id, branch, output_dir
        )
        audited = True
        return result

    monkeypatch.setattr(weathernext2_holdback, "_run_audit", recording_audit)

    result = _run_publish(
        holdback_repo.repo,
        "test.icechunk",
        initial_snapshot,
        holdback_repo.cutoff,
        tmp_path / "publish",
        force=True,
    )

    assert audited
    assert result.present_keys == 0
    assert holdback_repo.repo.lookup_branch("main") == purge_tip
    assert "holdback-purge" not in holdback_repo.repo.list_branches()

    fresh_store = holdback_repo.repo.readonly_session(branch="main").store
    fresh_group = zarr.open_group(fresh_store, mode="r")
    root_array = fresh_group["temperature_2m"]
    pressure_array = fresh_group["pressure_level/temperature"]
    assert isinstance(root_array, zarr.Array)
    assert isinstance(pressure_array, zarr.Array)
    for init_index, init_time in enumerate(holdback_repo.init_times):
        for lead_index, lead_time in enumerate(holdback_repo.lead_times):
            forbidden = init_time + lead_time > holdback_repo.cutoff
            root_values = root_array[init_index, :, lead_index, :, :]
            pressure_values = pressure_array[init_index, :, lead_index, :, :, :]
            if forbidden:
                assert np.isnan(root_values).all()
                assert np.isnan(pressure_values).all()
            else:
                assert np.all(root_values == 1)
                assert np.all(pressure_values == 2)


def test_publish_dry_run_leaves_branches_unchanged(
    holdback_repo: HoldbackRepo, tmp_path: Path
) -> None:
    initial_snapshot = holdback_repo.repo.lookup_branch("main")
    _run_delete(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        tmp_path / "delete",
        force=True,
        root_split=2,
        pressure_split=1,
    )
    purge_tip = holdback_repo.repo.lookup_branch("holdback-purge")

    result = _run_publish(
        holdback_repo.repo,
        "test.icechunk",
        initial_snapshot,
        holdback_repo.cutoff,
        tmp_path / "publish",
        force=False,
    )

    assert result.present_keys == 0
    assert holdback_repo.repo.lookup_branch("main") == initial_snapshot
    assert holdback_repo.repo.lookup_branch("holdback-purge") == purge_tip


def test_publish_refuses_if_main_moved_before_audit(
    holdback_repo: HoldbackRepo, tmp_path: Path
) -> None:
    initial_snapshot = holdback_repo.repo.lookup_branch("main")
    _run_delete(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        tmp_path / "delete",
        force=True,
        root_split=2,
        pressure_split=1,
    )
    main_session = holdback_repo.repo.writable_session("main")
    main_group = zarr.open_group(main_session.store, mode="r+")
    main_group.attrs["moved"] = True
    moved_snapshot = main_session.commit("move main")

    with pytest.raises(typer.BadParameter, match=f"main is at {moved_snapshot}"):
        _run_publish(
            holdback_repo.repo,
            "test.icechunk",
            initial_snapshot,
            holdback_repo.cutoff,
            tmp_path / "publish",
            force=True,
        )


def test_publish_cas_fails_if_main_advances_after_audit(
    holdback_repo: HoldbackRepo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_snapshot = holdback_repo.repo.lookup_branch("main")
    _run_delete(
        holdback_repo.repo,
        "test.icechunk",
        holdback_repo.cutoff,
        tmp_path / "delete",
        force=True,
        root_split=2,
        pressure_split=1,
    )
    purge_tip = holdback_repo.repo.lookup_branch("holdback-purge")
    real_run_audit = weathernext2_holdback._run_audit

    def advancing_audit(
        repo: icechunk.Repository,
        store_label: str,
        cutoff: pd.Timestamp,
        snapshot_id: str | None,
        branch: str | None,
        output_dir: Path,
    ) -> weathernext2_holdback.HoldbackAudit:
        result = real_run_audit(
            repo, store_label, cutoff, snapshot_id, branch, output_dir
        )
        session = holdback_repo.repo.writable_session("main")
        root = zarr.open_group(session.store, mode="r+")
        root.attrs["advanced"] = True
        session.commit("advance main during publish audit")
        return result

    monkeypatch.setattr(weathernext2_holdback, "_run_audit", advancing_audit)

    with pytest.raises(icechunk.ConflictError):
        _run_publish(
            holdback_repo.repo,
            "test.icechunk",
            initial_snapshot,
            holdback_repo.cutoff,
            tmp_path / "publish",
            force=True,
        )

    assert holdback_repo.repo.lookup_branch("main") != purge_tip
    assert holdback_repo.repo.lookup_branch("holdback-purge") == purge_tip


def test_publish_refuses_branch_unrelated_to_expected_main(
    holdback_repo: HoldbackRepo, tmp_path: Path
) -> None:
    old_main = holdback_repo.repo.lookup_branch("main")
    holdback_repo.repo.create_branch("holdback-purge", old_main)
    main_session = holdback_repo.repo.writable_session("main")
    main_group = zarr.open_group(main_session.store, mode="r+")
    main_group.attrs["new-main"] = True
    expected_main = main_session.commit("advance expected main")

    with pytest.raises(typer.BadParameter, match="does not descend"):
        _run_publish(
            holdback_repo.repo,
            "test.icechunk",
            expected_main,
            holdback_repo.cutoff,
            tmp_path,
            force=True,
        )

    assert holdback_repo.repo.lookup_branch("main") == expected_main
    assert holdback_repo.repo.lookup_branch("holdback-purge") == old_main


def test_icechunk_utils_parses_k8s_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], icechunk_utils.AuthMode, str]] = []

    def fake_count_snapshots(
        repos: list[str], *, auth: icechunk_utils.AuthMode, k8s_secret: str
    ) -> None:
        calls.append((repos, auth, k8s_secret))

    monkeypatch.setattr(icechunk_utils, "count_snapshots", fake_count_snapshots)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "icechunk_utils.py",
            "--repo",
            "s3://bucket/repo.icechunk",
            "--k8s-secret",
            "weathernext2-storage-options-key",
            "count",
        ],
    )

    icechunk_utils.main()

    assert calls == [
        (
            ["s3://bucket/repo.icechunk"],
            "secret",
            "weathernext2-storage-options-key",
        )
    ]
