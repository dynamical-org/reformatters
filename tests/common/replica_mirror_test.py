import dataclasses
import json
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
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.implementations.memory import MemoryFileSystem
from zarr.abc.store import Store

from reformatters.common import replica_mirror as mirror
from reformatters.common import storage, validation
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.zarr import copy_data_var


@pytest.fixture(autouse=True)
def native_chunks(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    original = storage._repository_config_and_credentials

    def config(
        virtual_config: storage.IcechunkVirtualConfig | None,
    ) -> tuple[icechunk.RepositoryConfig, dict[str, Any] | None]:
        result, credentials = original(virtual_config)
        result.inline_chunk_threshold_bytes = getattr(request, "param", 0)
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


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("after_write", [False, True])
@pytest.mark.parametrize(
    "crash_key",
    [
        "init_time/c/0",
        "ingested_forecast_length/zarr.json",
        "init_time/zarr.json",
        "temperature/zarr.json",
        "zarr.json",
    ],
)
def test_replay_partial_coordinate_metadata(
    monkeypatch: pytest.MonkeyPatch, reverse: bool, after_write: bool, crash_key: str
) -> None:
    sf, repo = stores()
    activate(sf, repo)
    target = append_days(repo, 5)
    mapping = mirror._mapping

    def ordered_mapping(
        source: Store, destination: Store, *, checkpoint_length: int | None = None
    ) -> tuple[dict[str, zarr.Array], dict[str, zarr.Array], set[str], int, int]:
        arrays, replicas, coords, offset, length = mapping(
            source, destination, checkpoint_length=checkpoint_length
        )
        return (
            dict(sorted(arrays.items(), reverse=reverse)),
            replicas,
            coords,
            offset,
            length,
        )

    monkeypatch.setattr(mirror, "_mapping", ordered_mapping)
    original = mirror._put

    def interrupted(store: Store, key: str, value: bytes | None) -> None:
        crash = isinstance(store, zarr.storage.LocalStore) and key == crash_key
        if crash and not after_write:
            raise InterruptedError("metadata crash")
        original(store, key, value)
        if crash:
            raise InterruptedError("metadata crash")

    monkeypatch.setattr(mirror, "_put", interrupted)
    with pytest.raises(InterruptedError, match="metadata crash"):
        mirror.mirror_published(sf, target)
    assert mirror.read_state(sf)["pending"] == target
    monkeypatch.setattr(mirror, "_put", original)
    mirror.retry_mirror(sf)
    mirror.retry_mirror(sf)
    assert mirror.read_state(sf)["snapshot"] == target
    assert mirror.read_state(sf)["pending"] is None
    for consolidated in (False, True):
        with xr.open_zarr(
            sf.replica_stores()[0], consolidated=consolidated, decode_timedelta=True
        ) as actual:
            xr.testing.assert_equal(actual, dataset("2024-04-01", 5))
            assert np.isnat(actual.ingested_forecast_length.values).all()


@pytest.mark.parametrize("corruption", ["nat", "old-label", "short", "long", "schema"])
def test_pending_replay_rejects_corruption(corruption: str) -> None:
    sf, repo = stores()
    activate(sf, repo)
    target = append_days(repo, 5)
    state = mirror.read_state(sf)
    state["pending"] = target
    mirror._save(sf, state)
    destination = sf.replica_stores(writable=True, mirror=True)[0]
    if corruption in {"nat", "old-label"}:
        array = zarr.open_array(destination, path="init_time", mode="r+")
        array[1] = np.nan if corruption == "nat" else array[0]
    else:
        metadata = json.loads(
            mirror._bytes(destination, "temperature/zarr.json") or b""
        )
        if corruption == "schema":
            metadata["attributes"]["units"] = "corrupt"
        else:
            metadata["shape"] = [2 if corruption == "short" else 6]
        mirror._put(destination, "temperature/zarr.json", json.dumps(metadata).encode())
    with pytest.raises(AssertionError, match=r"labels|daily grid|bounds|layout drift"):
        mirror.retry_mirror(sf)
    assert mirror.read_state(sf) == state


def test_abort_pending_handoff_records_audit_before_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, repo = stores()
    initialize(sf, repo)
    mirror.begin_handoff(sf, "handoff")
    delete = StoreFactory.delete_coordination_file

    def checked_delete(self: StoreFactory, job: str, key: str) -> None:
        if key == "state.json":
            audits = self.read_all_coordination_files(job, "aborted")
            assert len(audits) == 1
            audit = json.loads(audits[0])
            assert audit["state"] == mirror.read_state(self)
            assert audit["drained_writers"] == "all writers terminal and absent"
            assert audit["operator_evidence"] == "operator approves return to direct"
            assert pd.Timestamp(audit["aborted_at"]).utcoffset() == pd.Timedelta(0)
        delete(self, job, key)

    monkeypatch.setattr(StoreFactory, "delete_coordination_file", checked_delete)
    mirror.abort_handoff(
        sf,
        drained_writers="all writers terminal and absent",
        operator_evidence="operator approves return to direct",
    )
    assert sf.replica_mode() == "direct"
    sf.primary_store(writable=True)
    sf.assert_direct_replica_writes()


@pytest.mark.parametrize(
    "failure", ["audit", "removal", "lock", "active", "drain", "operator"]
)
def test_abort_handoff_fails_closed(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    sf, repo = stores()
    if failure == "active":
        activate(sf, repo)
    else:
        initialize(sf, repo)
        mirror.begin_handoff(sf, "handoff")
    state = mirror.read_state(sf)
    create = StoreFactory.create_coordination_file
    delete = StoreFactory.delete_coordination_file

    def create_failure(self: StoreFactory, job: str, key: str, data: bytes) -> None:
        if key.startswith("aborted/"):
            raise OSError("audit failure")
        create(self, job, key, data)

    def delete_failure(self: StoreFactory, job: str, key: str) -> None:
        if key == "state.json":
            raise OSError("removal failure")
        delete(self, job, key)

    if failure == "audit":
        monkeypatch.setattr(StoreFactory, "create_coordination_file", create_failure)
    elif failure == "removal":
        monkeypatch.setattr(StoreFactory, "delete_coordination_file", delete_failure)
    elif failure == "lock":
        create(sf, mirror._JOB, "lock.json", b"another owner")
    with pytest.raises((AssertionError, OSError)):
        mirror.abort_handoff(
            sf,
            drained_writers="  " if failure == "drain" else "drained",
            operator_evidence="" if failure == "operator" else "approved",
        )
    assert mirror.read_state(sf) == state
    if failure != "active":
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


def test_adoption_rejects_changed_reviewed_digest() -> None:
    sf, repo = stores()
    snapshot = initialize(sf, repo)
    mirror.begin_handoff(sf, "pending")
    with pytest.raises(AssertionError, match="inventory changed"):
        mirror.adopt(
            sf,
            snapshot,
            inventory_sha256="0" * 64,
            rehearsal="test",
            drained_writers="test",
            max_read_bytes=100_000,
        )
    assert sf.replica_mode() == "pending"


def test_mirror_replays_deleted_shard() -> None:
    sf, repo = stores()
    activate(sf, repo)
    session = repo.writable_session("main")
    mirror._put(session.store, "temperature/c/1", None)
    snapshot = session.commit("remove invalid shard")
    mirror.mirror_published(sf, snapshot)
    assert mirror._bytes(sf.replica_stores()[0], "temperature/c/1") is None
    assert mirror.read_state(sf)["snapshot"] == snapshot


def test_lock_recovery_requires_utc_dead_writer_attestation() -> None:
    sf, repo = stores()
    initialize(sf, repo)
    sf.create_coordination_file("replica-mirror", "lock.json", b"owner")
    evidence = mirror.DeadWriterAttestation(
        job_name="mirror-job",
        terminal_state="Failed",
        pods_absent_checked_at="2026-09-30T12:00:00",
        confirmed_by="operator",
    )
    with pytest.raises(AssertionError, match="UTC"):
        mirror.recover_abandoned_lock(sf, "owner", evidence=evidence)
    assert sf.read_coordination_file("replica-mirror", "lock.json") == b"owner"
    evidence = evidence.model_copy(
        update={"pods_absent_checked_at": "2026-09-30T12:00:00Z"}
    )
    mirror.recover_abandoned_lock(sf, "owner", evidence=evidence)
    assert sf.read_coordination_file("replica-mirror", "lock.json") is None
    assert (
        sf.read_coordination_file("replica-mirror", "recovery/owner.json") is not None
    )


def test_object_size_never_falls_back_to_payload_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs = AsyncFileSystemWrapper(fs=MemoryFileSystem(), asynchronous=True)
    destination = zarr.storage.FsspecStore(fs, path="/replica")

    async def absent_size(path: str, **kwargs: object) -> dict[str, object]:
        return {"name": path, "type": "file"}

    async def no_get(*args: object, **kwargs: object) -> bytes:
        pytest.fail("object-size fallback read payload")

    monkeypatch.setattr(fs, "_info", absent_size)
    monkeypatch.setattr(fs, "_cat_file", no_get)
    with pytest.raises(AssertionError, match="size"):
        mirror._object_size(destination, "temperature/c/1")


@pytest.mark.parametrize("native_chunks", [10_000], indirect=True)
def test_inline_adoption_budget_counts_only_replica_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, repo = stores()
    snapshot = initialize(sf, repo)
    mirror.begin_handoff(sf, "inline")
    report = mirror.adoption_dry_run(sf, snapshot)
    assert all(entry["source_reference"]["kind"] == 3 for entry in report["entries"])
    assert report["residual_read_bytes"] == sum(
        entry["length"] for entry in report["entries"]
    )
    original = mirror._bytes

    def no_source_get(store: Store, key: str) -> bytes | None:
        assert not isinstance(store, mirror.IcechunkStore), (
            "Inline bytes are in manifests"
        )
        return original(store, key)

    monkeypatch.setattr(mirror, "_bytes", no_source_get)
    mirror.adopt(
        sf,
        snapshot,
        inventory_sha256=report["inventory_sha256"],
        rehearsal="test",
        drained_writers="test",
        max_read_bytes=report["residual_read_bytes"],
    )
    after = shifted_branch(repo, snapshot)
    epoch = mirror.epoch_dry_run(sf, snapshot, after)
    assert epoch["residual_read_bytes"] == 0
    mirror.prepare_epoch(
        sf, snapshot, after, inventory_sha256=epoch["inventory_sha256"]
    )
