import dataclasses
import multiprocessing
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
import zarr.storage
from zarr.abc.store import Store

from reformatters.common import replica_mirror as mirror
from reformatters.common import storage, validation
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.zarr import copy_data_var


@pytest.fixture(autouse=True)
def native_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    original = storage._repository_config_and_credentials

    def config(
        virtual_config: storage.IcechunkVirtualConfig | None,
    ) -> tuple[icechunk.RepositoryConfig, dict[str, Any] | None]:
        result, credentials = original(virtual_config)
        result.inline_chunk_threshold_bytes = 0
        return result, credentials

    monkeypatch.setattr(storage, "_repository_config_and_credentials", config)


def stores() -> tuple[StoreFactory, icechunk.Repository]:
    sf = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="primary", format=DatasetFormat.ICECHUNK
        ),
        replica_storage_configs=(
            StorageConfig(base_path="replica", format=DatasetFormat.ZARR3),
        ),
        dataset_id="mirror-test",
        template_config_version="v1",
        replica_handoff=True,
    )
    return sf, sf.icechunk_repos(sort="primary-first")[0][1]


def dataset(start: str, days: int) -> xr.Dataset:
    times = pd.date_range(start, periods=days)
    ds = xr.Dataset(
        {"temperature": ("init_time", np.arange(days, dtype="float32"))},
        coords={
            "init_time": times,
            "ingested_forecast_length": (
                "init_time",
                np.full(days, np.timedelta64("NaT", "ns")),
            ),
        },
    )
    ds["temperature"].encoding.update(chunks=(1,), shards=(1,))
    for name in ds.coords:
        ds[name].encoding.update(
            chunks=(5840,),
            dtype="float64",
            units="seconds since 1970-01-01" if name == "init_time" else "seconds",
        )
    return ds


def initialize(sf: StoreFactory, repo: icechunk.Repository) -> str:
    ds = dataset("2024-04-01", 3)
    session = repo.writable_session("main")
    ds.to_zarr(session.store, mode="w", consolidated=False)
    snapshot = session.commit("baseline")
    ds.to_zarr(sf.replica_stores(writable=True)[0], mode="w", consolidated=False)
    return snapshot


def activate(sf: StoreFactory, repo: icechunk.Repository) -> str:
    snapshot = initialize(sf, repo)
    mirror.begin_handoff(sf, "handoff")
    mirror.adopt(
        sf,
        snapshot,
        inventory_sha256=mirror.adoption_dry_run(sf, snapshot)["inventory_sha256"],
        rehearsal="local test",
        max_read_bytes=100_000,
        drained_writers="all jobs terminal, no pods",
    )
    return snapshot


def test_pending_marker_blocks_direct_writers() -> None:
    sf, repo = stores()
    initialize(sf, repo)
    assert sf.replica_mode() == "direct"
    mirror.begin_handoff(sf, "handoff")
    with pytest.raises(AssertionError, match="handoff"):
        sf.assert_direct_replica_writes()
    with pytest.raises(AssertionError, match="pending"):
        sf.primary_store(writable=True)


def test_mirror_copies_published_append_and_retries() -> None:
    sf, repo = stores()
    activate(sf, repo)
    session = repo.writable_session("main")
    dataset("2024-04-01", 5).isel(init_time=slice(3, None)).to_zarr(
        session.store, append_dim="init_time", consolidated=False
    )
    target = session.commit("two new days")
    mirror.mirror_published(sf, target)
    mirror.mirror_published(sf, target)
    actual = xr.open_zarr(
        sf.replica_stores()[0], chunks=None, decode_timedelta=True, consolidated=False
    )
    xr.testing.assert_equal(actual, dataset("2024-04-01", 5))
    assert mirror.read_state(sf)["snapshot"] == target


def test_epoch_non_aligned_coordinates_and_nat() -> None:
    sf, repo = stores()
    before = activate(sf, repo)
    repo.create_branch("shift", before)
    session = repo.writable_session("shift")
    ds = dataset("2024-03-30", 5)
    ds["temperature"].values[2:] = np.arange(3)
    ds.to_zarr(session.store, mode="w", consolidated=False)
    after = session.commit("shift")
    mirror.prepare_epoch(
        sf,
        before,
        after,
        inventory_sha256=mirror.epoch_dry_run(sf, before, after)["inventory_sha256"],
        max_read_bytes=100_000,
    )
    repo.reset_branch("main", after, from_snapshot_id=before)
    mirror.mirror_published(sf, after)
    actual = xr.open_zarr(
        sf.replica_stores()[0], chunks=None, decode_timedelta=True, consolidated=False
    )
    xr.testing.assert_equal(actual, dataset("2024-04-01", 3))
    assert zarr.open_array(sf.replica_stores()[0], path="init_time").chunks == (5840,)


def test_unexplained_reset_stops_without_recopy() -> None:
    sf, repo = stores()
    initial = activate(sf, repo)
    session = repo.writable_session("main")
    dataset("2024-04-01", 4).isel(init_time=slice(3, None)).to_zarr(
        session.store, append_dim="init_time", consolidated=False
    )
    new = session.commit("append")
    mirror.mirror_published(sf, new)
    repo.reset_branch("main", initial, from_snapshot_id=new)
    with pytest.raises(AssertionError, match="ancestr"):
        mirror.mirror_published(sf, initial)


def test_lock_never_steals_existing_owner() -> None:
    sf, repo = stores()
    activate(sf, repo)
    with mirror.ownership(sf), pytest.raises(FileExistsError), mirror.ownership(sf):
        pytest.fail("second owner")


def append_days(repo: icechunk.Repository, days: int) -> str:
    session = repo.writable_session("main")
    current = zarr.open_array(session.store, path="init_time").shape[0]
    dataset("2024-04-01", days).isel(init_time=slice(current, None)).to_zarr(
        session.store, append_dim="init_time", consolidated=False
    )
    return session.commit("append")


def test_adoption_default_budget_refuses_before_payload_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, repo = stores()
    initial = initialize(sf, repo)
    mirror.begin_handoff(sf, "pending")
    report = mirror.adoption_dry_run(sf, initial)
    assert report["residual_read_bytes"] > 0

    def forbidden_read(*args: object, **kwargs: object) -> None:
        pytest.fail("payload read before budget check")

    monkeypatch.setattr(mirror, "_bytes", forbidden_read)
    with pytest.raises(AssertionError, match="budget"):
        mirror.adopt(
            sf,
            initial,
            inventory_sha256=report["inventory_sha256"],
            rehearsal="test",
            drained_writers="test",
        )
    assert sf.replica_mode() == "pending"


def test_adoption_rejects_equal_length_corruption() -> None:
    sf, repo = stores()
    initial = initialize(sf, repo)
    destination = sf.replica_stores(writable=True)[0]
    key = "temperature/c/1"
    value = mirror._bytes(destination, key)
    assert value is not None
    mirror._put(destination, key, bytes([value[0] ^ 1]) + value[1:])
    mirror.begin_handoff(sf, "pending")
    report = mirror.adoption_dry_run(sf, initial)
    with pytest.raises(AssertionError, match="Encoded shard mismatch"):
        mirror.adopt(
            sf,
            initial,
            inventory_sha256=report["inventory_sha256"],
            rehearsal="test",
            drained_writers="test",
            max_read_bytes=report["residual_read_bytes"],
        )
    assert sf.replica_mode() == "pending"


@pytest.mark.parametrize("mutation", ["missing", "extra", "truncated"])
def test_adoption_rejects_bad_inventory(mutation: str) -> None:
    sf, repo = stores()
    initial = initialize(sf, repo)
    destination = sf.replica_stores(writable=True)[0]
    value = mirror._bytes(destination, "temperature/c/1")
    assert value is not None
    if mutation == "missing":
        mirror._put(destination, "temperature/c/1", None)
    elif mutation == "extra":
        mirror._put(destination, "temperature/c/99", value)
    else:
        mirror._put(destination, "temperature/c/1", value[:-1])
    with pytest.raises(AssertionError, match="mismatch"):
        mirror.adoption_dry_run(sf, initial)


def test_pending_retry_finishes_before_new_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, repo = stores()
    initial = activate(sf, repo)
    first = append_days(repo, 4)
    original = mirror._put

    def interrupted(store: Store, key: str, value: bytes | None) -> None:
        if key == "temperature/c/3":
            raise RuntimeError("interrupted data copy")
        original(store, key, value)

    monkeypatch.setattr(mirror, "_put", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        mirror.mirror_published(sf, first)
    assert mirror.read_state(sf)["snapshot"] == initial
    assert mirror.read_state(sf)["pending"] == first
    assert (
        xr.open_zarr(sf.replica_stores()[0], consolidated=False).sizes["init_time"] == 3
    )
    monkeypatch.setattr(mirror, "_put", original)
    second = append_days(repo, 5)
    mirror.mirror_published(sf, second)
    assert mirror.read_state(sf)["snapshot"] == second
    actual = xr.open_zarr(
        sf.replica_stores()[0], consolidated=False, decode_timedelta=True
    )
    xr.testing.assert_equal(actual, dataset("2024-04-01", 5))
    mirror.mirror_published(sf, first)
    assert mirror.read_state(sf)["snapshot"] == second


def shifted_branch(repo: icechunk.Repository, before: str) -> str:
    repo.create_branch("shifted", before)
    session = repo.writable_session("shifted")
    array = zarr.open_array(session.store, path="temperature")
    array.resize((5,))
    session.shift_array("/temperature", [2])
    coords = dataset("2024-03-30", 5).drop_vars("temperature")
    memory = zarr.storage.MemoryStore()
    coords.to_zarr(memory, mode="w", consolidated=False)
    for name in coords.coords:
        mirror._put(
            session.store,
            f"{name}/zarr.json",
            mirror._bytes(memory, f"{name}/zarr.json"),
        )
        for key in mirror.sync(mirror._keys(memory, f"{name}/c")):
            mirror._put(session.store, key, mirror._bytes(memory, key))
    return session.commit("native reference shift")


def test_native_epoch_no_replica_reads_and_explicit_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, repo = stores()
    before = activate(sf, repo)
    after = shifted_branch(repo, before)

    def no_replica_reads(*args: object, **kwargs: object) -> None:
        pytest.fail("epoch accessed replica")

    with monkeypatch.context() as patch:
        patch.setattr(StoreFactory, "replica_stores", no_replica_reads)
        report = mirror.epoch_dry_run(sf, before, after)
        assert report["residual_read_bytes"] == 0
        mirror.prepare_epoch(
            sf, before, after, inventory_sha256=report["inventory_sha256"]
        )
        with pytest.raises(AssertionError, match="awaits publication"):
            mirror.mirror_published(sf, before)
    assert mirror.read_state(sf)["intent"]["after"] == after
    daily = append_days(repo, 4)
    with pytest.raises(AssertionError, match="intent"):
        mirror.mirror_published(sf, daily)
    mirror.cancel_epoch(
        sf, before=before, after=after, evidence="publisher stopped after CAS failure"
    )
    mirror.mirror_published(sf, daily)
    assert mirror.read_state(sf)["snapshot"] == daily


def test_epoch_crash_recovery_non_aligned_append_and_rollback() -> None:
    sf, repo = stores()
    before = activate(sf, repo)
    after = shifted_branch(repo, before)
    report = mirror.epoch_dry_run(sf, before, after)
    mirror.prepare_epoch(sf, before, after, inventory_sha256=report["inventory_sha256"])
    repo.reset_branch("main", after, from_snapshot_id=before)
    with pytest.raises(AssertionError, match="cannot be cancelled"):
        mirror.cancel_epoch(sf, before=before, after=after, evidence="test")
    mirror.retry_mirror(sf)
    assert mirror.read_state(sf)["snapshot"] == after
    session = repo.writable_session("main")
    ds = dataset("2024-03-30", 6).isel(init_time=slice(5, None))
    ds["temperature"].values[:] = 3
    ds.to_zarr(session.store, append_dim="init_time", consolidated=False)
    daily = session.commit("shifted append")
    mirror.mirror_published(sf, daily)
    actual = xr.open_zarr(
        sf.replica_stores()[0], consolidated=False, decode_timedelta=True
    )
    xr.testing.assert_equal(actual, dataset("2024-04-01", 4))
    assert mirror._bytes(sf.replica_stores()[0], "ingested_forecast_length/c/0") is None
    repo.create_branch("rollback", before)
    rollback_session = repo.writable_session("rollback")
    dataset("2024-04-01", 4).isel(init_time=slice(3, None)).to_zarr(
        rollback_session.store, append_dim="init_time", consolidated=False
    )
    rollback = rollback_session.commit("retained daily append")
    report = mirror.epoch_dry_run(sf, daily, rollback)
    mirror.prepare_epoch(
        sf,
        daily,
        rollback,
        inventory_sha256=report["inventory_sha256"],
        max_read_bytes=report["residual_read_bytes"],
    )
    repo.reset_branch("main", rollback, from_snapshot_id=daily)
    mirror.retry_mirror(sf)
    assert mirror.read_state(sf)["snapshot"] == rollback
    xr.testing.assert_equal(
        xr.open_zarr(sf.replica_stores()[0], consolidated=False, decode_timedelta=True),
        dataset("2024-04-01", 4),
    )


def test_marker_read_error_fails_before_writable_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _ = stores()

    def fail_read(*args: object, **kwargs: object) -> None:
        raise OSError("marker unavailable")

    monkeypatch.setattr(StoreFactory, "read_coordination_file", fail_read)
    with pytest.raises(OSError, match="marker unavailable"):
        sf.primary_store(writable=True)


def test_direct_batch_rechecks_mid_job_marker(tmp_path: Path) -> None:
    sf, repo = stores()
    initialize(sf, repo)
    destination = sf.replica_stores(writable=True)
    primary = sf.primary_store(writable=True)
    template = dataset("2024-04-01", 3)
    mirror.begin_handoff(sf, "mid-job")
    with pytest.raises(AssertionError, match="Direct replica writes blocked"):
        copy_data_var(
            "temperature",
            slice(0, 1),
            template,
            "init_time",
            tmp_path,
            primary,
            destination,
            replica_write_guard=sf.assert_direct_replica_writes,
        )


def test_activated_validation_uses_timestamps_and_detects_staleness() -> None:
    primary = dataset("2024-03-30", 5)
    replica = primary.sel(init_time=slice("2024-04-01", None))
    context = validation.ValidationContext(
        store=zarr.storage.MemoryStore(),
        ds=replica,
        primary_ds=primary,
        append_dim="init_time",
    )
    assert not validation.CheckReplicaMatchesPrimary().check(context).passed
    assert validation.CheckReplicaTimestampIntersection().check(context).passed
    stale = dataclasses.replace(context, ds=replica.isel(init_time=slice(None, -1)))
    assert not validation.CheckReplicaTimestampIntersection().check(stale).passed


def _claim_lock(base: str, result_path: str) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(storage, "_LOCAL_ZARR_STORE_BASE_PATH", base)
    sf, _ = stores()
    try:
        sf.create_coordination_file("replica-mirror", "lock.json", b"child")
    except FileExistsError:
        Path(result_path).write_text("blocked")
    else:
        Path(result_path).write_text("acquired")


def test_lock_exclusion_across_processes(tmp_path: Path) -> None:
    sf, _ = stores()
    result = tmp_path / "child-result"
    with mirror.ownership(sf):
        process = multiprocessing.get_context("spawn").Process(
            target=_claim_lock, args=(str(tmp_path), str(result))
        )
        process.start()
        process.join(timeout=30)
        assert process.exitcode == 0
        assert result.read_text() == "blocked"
