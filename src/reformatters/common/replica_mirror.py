"""Fixed-origin ENS replica handoff and post-publication mirroring."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Literal, NamedTuple
from uuid import uuid4

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import zarr.storage
from icechunk.store import IcechunkStore
from pydantic import Field
from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.storage import StoreFactory
from reformatters.common.template_utils import ignore_consolidated_metadata_spec_warning

_JOB = "replica-mirror"
_ORIGIN = pd.Timestamp("2024-04-01")
_DIM = "init_time"


class DeadWriterAttestation(FrozenBaseModel):
    job_name: str = Field(min_length=1)
    terminal_state: Literal["Complete", "Failed", "Deleted"]
    pods_absent_checked_at: str = Field(min_length=1)
    confirmed_by: str = Field(min_length=1)


class ChunkReference(NamedTuple):
    key: str
    kind: int
    path: str
    offset: int
    length: int
    inline: bytes | None

    def identity(self) -> tuple[int, str, int, int, bytes | None]:
        return self.kind, self.path, self.offset, self.length, self.inline

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "offset": self.offset,
            "length": self.length,
            "inline_sha256": hashlib.sha256(self.inline).hexdigest()
            if self.inline is not None
            else None,
        }


async def _references(
    store: IcechunkStore, names: set[str], offset: int
) -> dict[str, ChunkReference]:
    result = {}
    for name in sorted(names):
        async for (
            coords,
            kinds,
            paths,
            offsets,
            lengths,
            inline,
        ) in store.array_chunk_iterator(name):
            for i, index in enumerate(coords):
                key = f"{name}/c/" + "/".join(map(str, index))
                mapped = _mapped_key(key, offset)
                if mapped is not None:
                    kind = int(kinds[i])
                    assert kind in {1, 3}, "ENS mirror requires native or inline chunks"
                    result[mapped] = ChunkReference(
                        key,
                        kind,
                        paths[i],
                        int(offsets[i]),
                        int(lengths[i]),
                        inline.get(i),
                    )
    return result


def _digest(report: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()


def _verify_residual(
    report: dict[str, Any],
    source: Store,
    destination: Store,
    *,
    max_read_bytes: int,
) -> None:
    assert max_read_bytes >= 0
    assert report["residual_read_bytes"] <= max_read_bytes, (
        f"Verification needs {report['residual_read_bytes']} residual bytes; budget is {max_read_bytes}"
    )
    for entry in report["entries"]:
        if entry["tier"] != "residual":
            continue
        left = (
            bytes.fromhex(entry["source_inline"])
            if entry["source_inline"] is not None
            else _bytes(source, entry["source_key"])
        )
        right = (
            bytes.fromhex(entry["destination_inline"])
            if entry["destination_inline"] is not None
            else _bytes(destination, entry["destination_key"])
        )
        assert left is not None, "Source shard disappeared during verification"
        assert right is not None, "Replica shard disappeared during verification"
        assert len(left) == len(right) == entry["length"]
        assert left == right, f"Encoded shard mismatch: {entry['key']}"
        entry.update(tier="byte-verified", sha256=hashlib.sha256(left).hexdigest())
    report["verified_at"] = pd.Timestamp.now(tz="UTC").isoformat()


def read_state(factory: StoreFactory) -> dict[str, Any]:
    raw = factory.read_coordination_file(_JOB, "state.json")
    assert raw is not None, "Mirror requires verified handoff"
    return json.loads(raw)


def _save(factory: StoreFactory, state: dict[str, Any]) -> None:
    factory.write_coordination_file(_JOB, "state.json", json.dumps(state).encode())


@contextmanager
def ownership(factory: StoreFactory) -> Iterator[None]:
    owner = uuid4().hex.encode()
    factory.create_coordination_file(_JOB, "lock.json", owner)
    try:
        yield
    finally:
        assert factory.read_coordination_file(_JOB, "lock.json") == owner
        factory.delete_coordination_file(_JOB, "lock.json")


def recover_abandoned_lock(
    factory: StoreFactory, owner: str, *, evidence: DeadWriterAttestation
) -> None:
    checked = pd.Timestamp(evidence.pods_absent_checked_at)
    assert checked.tzinfo is not None, "Pod-absence check timestamp must be UTC-aware"
    assert checked.utcoffset() == pd.Timedelta(0), "Pod-absence check must use UTC"
    assert factory.read_coordination_file(_JOB, "lock.json") == owner.encode()
    factory.write_coordination_file(
        _JOB, f"recovery/{owner}.json", evidence.model_dump_json().encode()
    )
    factory.delete_coordination_file(_JOB, "lock.json")


def begin_handoff(factory: StoreFactory, handoff_id: str) -> None:
    assert factory.replica_handoff
    assert handoff_id
    with ownership(factory):
        factory.create_coordination_file(
            _JOB,
            "state.json",
            json.dumps(
                {
                    "mode": "pending",
                    "handoff_id": handoff_id,
                    "replica_urls": factory.replica_urls(),
                    "origin": _ORIGIN.isoformat(),
                }
            ).encode(),
        )


def abort_handoff(
    factory: StoreFactory, *, drained_writers: str, operator_evidence: str
) -> None:
    assert drained_writers.strip(), "Drained-writer evidence is required"
    assert operator_evidence.strip(), "Operator evidence is required"
    with ownership(factory):
        assert factory.replica_mode() == "pending", "Only pending handoff can abort"
        state = read_state(factory)
        factory.create_coordination_file(
            _JOB,
            f"aborted/{uuid4().hex}.json",
            json.dumps(
                {
                    "state": state,
                    "drained_writers": drained_writers,
                    "operator_evidence": operator_evidence,
                    "aborted_at": pd.Timestamp.now(tz="UTC").isoformat(),
                }
            ).encode(),
        )
        factory.delete_coordination_file(_JOB, "state.json")


def _repo(factory: StoreFactory) -> icechunk.Repository:
    return dict(factory.icechunk_repos(sort="primary-first"))["primary"]


def _published(repo: icechunk.Repository, snapshot: str) -> None:
    assert any(info.id == snapshot for info in repo.ancestry(branch="main")), (
        "Mirror target was not published on current main ancestry"
    )


def _bytes(store: Store, key: str) -> bytes | None:
    value = sync(store.get(key, default_buffer_prototype()))
    return None if value is None else value.to_bytes()


def _put(store: Store, key: str, value: bytes | None) -> None:
    if value is None:
        sync(store.delete(key))
    else:
        sync(store.set(key, default_buffer_prototype().buffer.from_bytes(value)))


async def _keys(store: Store, prefix: str = "") -> list[str]:
    array_prefix = prefix.split("/c", 1)[0] if "/c" in prefix else prefix
    return [
        key async for key in store.list_prefix(array_prefix) if key.startswith(prefix)
    ]


def _metadata(array: zarr.Array) -> dict[str, Any]:
    return array.metadata.to_dict()


def _contract(metadata: dict[str, Any]) -> str:
    metadata = deepcopy(metadata)
    dims = metadata["dimension_names"]
    metadata["shape"] = [
        size if dim != _DIM else None
        for dim, size in zip(dims, metadata["shape"], strict=True)
    ]
    metadata["attributes"].pop("statistics_approximate", None)
    return json.dumps(metadata, sort_keys=True)


def _mapping(
    source: Store, destination: Store, *, checkpoint_length: int | None = None
) -> tuple[dict[str, zarr.Array], dict[str, zarr.Array], set[str], int, int]:
    src = zarr.open_group(source, mode="r", use_consolidated=False)
    dst = zarr.open_group(destination, mode="r", use_consolidated=False)
    _assert_root_contract(src, dst)
    assert not list(src.groups()), "ENS mirror requires root arrays"
    assert not list(dst.groups()), "ENS mirror requires root arrays"
    arrays, replicas = dict(src.arrays()), dict(dst.arrays())
    assert arrays.keys() == replicas.keys(), "Mirror schema drift"
    with xr.open_zarr(
        source, chunks=None, decode_timedelta=True, consolidated=False
    ) as ds:
        coordinates = set(ds.coords)
        times = pd.DatetimeIndex(ds[_DIM].values)
    time_array = replicas[_DIM]
    replica_times = pd.DatetimeIndex(
        xr.decode_cf(
            xr.Dataset({_DIM: ((_DIM,), time_array[:], dict(time_array.attrs))})
        )[_DIM].values
    )
    assert len(times)
    assert len(replica_times)
    assert not times.hasnans, "Invalid source init_time labels"
    assert not replica_times.hasnans, "Invalid replica init_time labels"
    assert replica_times[0] == _ORIGIN, "Replica origin changed"
    assert times.equals(pd.date_range(times[0], periods=len(times), freq="D")), (
        "Source is not a daily grid"
    )
    assert replica_times.equals(
        pd.date_range(_ORIGIN, periods=len(replica_times), freq="D")
    ), "Replica is not a daily grid"
    offset = int((_ORIGIN - times[0]) / pd.Timedelta(days=1))
    assert 0 <= offset < len(times)
    assert times[offset] == _ORIGIN
    length = len(times) - offset
    assert length >= len(replica_times), "Mirror cannot retract replica coverage"
    for name, array in arrays.items():
        metadata = _metadata(array)
        dims = metadata["dimension_names"]
        if _DIM in dims:
            size = replicas[name].shape[dims.index(_DIM)]
            if checkpoint_length is None:
                assert size == len(replica_times), "Inconsistent replica coverage"
            else:
                assert checkpoint_length <= size <= length, (
                    f"Replica coverage outside checkpoint/target bounds: {name}"
                )
        assert _contract(metadata) == _contract(_metadata(replicas[name])), (
            f"Mirror encoding/layout drift: {name}"
        )
        assert metadata["chunk_key_encoding"] == {
            "name": "default",
            "configuration": {"separator": "/"},
        }, "Unsupported chunk-key encoding"
        if name not in coordinates:
            assert metadata["dimension_names"][0] == _DIM
            assert metadata["chunk_grid"]["configuration"]["chunk_shape"][0] == 1, (
                "Raw mirror requires one timestamp per data shard"
            )
    return arrays, replicas, coordinates, offset, length


def _assert_root_contract(source: zarr.Group, destination: zarr.Group) -> None:
    left, right = dict(source.attrs), dict(destination.attrs)
    for name in ("time_domain", "attribution"):
        left.pop(name, None)
        right.pop(name, None)
    assert left == right, "Mirror root schema drift"


def _mapped_key(key: str, offset: int) -> str | None:
    parts = key.split("/")
    assert parts[1] == "c"
    index = int(parts[2]) - offset
    if index < 0:
        return None
    parts[2] = str(index)
    return "/".join(parts)


def _verify_inventory(source: Store, destination: Store) -> tuple[str, int]:
    arrays, replicas, coordinates, offset, length = _mapping(source, destination)
    assert replicas[_DIM].shape[0] == length, "Adoption requires matching coverage"
    digest = hashlib.sha256()
    count = 0
    for name, array in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(_contract(_metadata(array)).encode())
        if name in coordinates:
            values = np.asarray(array[...])
            dims = _metadata(array)["dimension_names"]
            if _DIM in dims:
                selection = [slice(None)] * len(dims)
                selection[dims.index(_DIM)] = slice(offset, None)
                values = values[tuple(selection)]
            np.testing.assert_equal(values, replicas[name][...])
            digest.update(values.tobytes())
        else:
            expected = {
                mapped
                for key in sync(_keys(source, f"{name}/c/"))
                if (mapped := _mapped_key(key, offset)) is not None
            }
            actual = set(sync(_keys(destination, f"{name}/c/")))
            assert actual == expected, f"Implicit fill/key mismatch: {name}"
            digest.update(json.dumps(sorted(expected)).encode())
            count += len(expected)
    return digest.hexdigest(), count


def _object_size(store: Store, key: str) -> int:
    if isinstance(store, zarr.storage.LocalStore):
        return (store.root / key).stat().st_size
    assert isinstance(store, zarr.storage.FsspecStore)
    info = sync(store.fs._info(f"{store.path}/{key}"))  # noqa: SLF001
    assert isinstance(info.get("size"), int), "Metadata-only object size required"
    return info["size"]


def adoption_dry_run(factory: StoreFactory, snapshot: str) -> dict[str, Any]:
    repo = _repo(factory)
    _published(repo, snapshot)
    source = repo.readonly_session(snapshot_id=snapshot).store
    replicas = factory.replica_stores()
    assert len(replicas) == 1
    destination = replicas[0]
    assert isinstance(
        destination, (zarr.storage.LocalStore, zarr.storage.FsspecStore)
    ), "Adoption requires metadata-only object size support"
    digest, count = _verify_inventory(source, destination)
    arrays, _, coordinates, offset, _ = _mapping(source, destination)
    references = sync(_references(source, set(arrays) - coordinates, offset))
    entries: list[dict[str, Any]] = []
    for key, ref in sorted(references.items()):
        size = _object_size(destination, key)
        assert size == ref.length, f"Shard length mismatch: {key}"
        entries.append(
            {
                "key": key,
                "source_key": ref.key,
                "destination_key": key,
                "source_reference": ref.manifest(),
                "source_inline": ref.inline.hex() if ref.inline is not None else None,
                "destination_inline": None,
                "length": size,
                "read_bytes": size + (size if ref.inline is None else 0),
                "tier": "residual",
            }
        )
    report = {
        "kind": "adoption",
        "snapshot": snapshot,
        "replica_urls": factory.replica_urls(),
        "labels_layout_keys_sha256": digest,
        "shard_count": count,
        "residual_shard_count": len(entries),
        "residual_read_bytes": sum(entry["read_bytes"] for entry in entries),
        "entries": entries,
    }
    report["inventory_sha256"] = _digest(report)
    return report


def adopt(
    factory: StoreFactory,
    snapshot: str,
    *,
    inventory_sha256: str,
    rehearsal: str,
    drained_writers: str,
    max_read_bytes: int = 0,
) -> None:
    assert rehearsal
    assert drained_writers
    with ownership(factory):
        state = read_state(factory)
        assert state["mode"] == "pending"
        repo = _repo(factory)
        assert repo.lookup_branch("main") == snapshot, (
            "Adoption snapshot is no longer main"
        )
        report = adoption_dry_run(factory, snapshot)
        assert report["inventory_sha256"] == inventory_sha256, (
            "Adoption inventory changed"
        )
        _verify_residual(
            report,
            repo.readonly_session(snapshot_id=snapshot).store,
            factory.replica_stores()[0],
            max_read_bytes=max_read_bytes,
        )
        assert repo.lookup_branch("main") == snapshot, "Main advanced during adoption"
        proof_sha256 = _digest(report)
        factory.write_coordination_file(
            _JOB, f"proofs/{proof_sha256}.json", json.dumps(report).encode()
        )
        state.update(
            mode="active",
            snapshot=snapshot,
            evidence=proof_sha256,
            rehearsal_attestation=rehearsal,
            drained_writers_attestation=drained_writers,
        )
        _save(factory, state)


def _retained(
    store: IcechunkStore,
) -> tuple[dict[str, zarr.Array], dict[str, np.ndarray], dict[str, ChunkReference]]:
    group = zarr.open_group(store, mode="r", use_consolidated=False)
    arrays = dict(group.arrays())
    with xr.open_zarr(
        store, chunks=None, decode_timedelta=True, consolidated=False
    ) as ds:
        names = set(ds.coords)
        times = pd.DatetimeIndex(ds[_DIM].values)
    assert times.equals(pd.date_range(times[0], periods=len(times), freq="D"))
    offset = int((_ORIGIN - times[0]) / pd.Timedelta(days=1))
    assert offset >= 0
    assert times[offset] == _ORIGIN
    coordinates = {}
    for name in names:
        array = arrays[name]
        values = np.asarray(array[...])
        dims = _metadata(array)["dimension_names"]
        if _DIM in dims:
            selection = [slice(None)] * len(dims)
            selection[dims.index(_DIM)] = slice(offset, None)
            values = values[tuple(selection)]
        coordinates[name] = values
    for name in set(arrays) - names:
        metadata = _metadata(arrays[name])
        assert metadata["dimension_names"][0] == _DIM
        assert metadata["chunk_grid"]["configuration"]["chunk_shape"][0] == 1
    return arrays, coordinates, sync(_references(store, set(arrays) - names, offset))


def epoch_dry_run(factory: StoreFactory, before: str, after: str) -> dict[str, Any]:
    repo = _repo(factory)
    old = repo.readonly_session(snapshot_id=before).store
    new = repo.readonly_session(snapshot_id=after).store
    _assert_root_contract(
        zarr.open_group(old, mode="r", use_consolidated=False),
        zarr.open_group(new, mode="r", use_consolidated=False),
    )
    old_arrays, old_coords, old_refs = _retained(old)
    new_arrays, new_coords, new_refs = _retained(new)
    assert old_arrays.keys() == new_arrays.keys(), "Epoch schema drift"
    assert old_coords.keys() == new_coords.keys()
    for name in old_arrays:
        assert _contract(_metadata(old_arrays[name])) == _contract(
            _metadata(new_arrays[name])
        ), f"Epoch encoding/layout drift: {name}"
    for name in old_coords:
        np.testing.assert_equal(old_coords[name], new_coords[name])
    assert old_refs.keys() == new_refs.keys(), "Epoch implicit fill/key mismatch"
    entries: list[dict[str, Any]] = []
    for key, left in sorted(old_refs.items()):
        right = new_refs[key]
        assert left.length == right.length, f"Epoch shard length mismatch: {key}"
        same = left.identity() == right.identity()
        entries.append(
            {
                "key": key,
                "source_key": left.key,
                "destination_key": right.key,
                "source_reference": left.manifest(),
                "destination_reference": right.manifest(),
                "source_inline": left.inline.hex() if left.inline is not None else None,
                "destination_inline": right.inline.hex()
                if right.inline is not None
                else None,
                "length": left.length,
                "read_bytes": (left.length if left.inline is None else 0)
                + (right.length if right.inline is None else 0),
                "tier": "reference-identical" if same else "residual",
            }
        )
    report = {
        "kind": "epoch",
        "before": before,
        "after": after,
        "entries": entries,
        "residual_shard_count": sum(entry["tier"] == "residual" for entry in entries),
        "residual_read_bytes": sum(
            entry["read_bytes"] for entry in entries if entry["tier"] == "residual"
        ),
    }
    report["inventory_sha256"] = _digest(report)
    return report


def prepare_epoch(
    factory: StoreFactory,
    before: str,
    after: str,
    *,
    inventory_sha256: str,
    max_read_bytes: int = 0,
) -> None:
    with ownership(factory):
        assert factory.replica_mode() == "active"
        state = read_state(factory)
        assert not state.get("intent")
        assert not state.get("pending")
        assert state["snapshot"] == before
        repo = _repo(factory)
        assert repo.lookup_branch("main") == before
        report = epoch_dry_run(factory, before, after)
        assert report["inventory_sha256"] == inventory_sha256, "Epoch inventory changed"
        _verify_residual(
            report,
            repo.readonly_session(snapshot_id=before).store,
            repo.readonly_session(snapshot_id=after).store,
            max_read_bytes=max_read_bytes,
        )
        assert repo.lookup_branch("main") == before, (
            "Main advanced during epoch verification"
        )
        proof_sha256 = _digest(report)
        factory.write_coordination_file(
            _JOB, f"proofs/{proof_sha256}.json", json.dumps(report).encode()
        )
        state["intent"] = {
            "before": before,
            "after": after,
            "checkpoint": before,
            "proof_sha256": proof_sha256,
        }
        _save(factory, state)


def cancel_epoch(
    factory: StoreFactory, *, before: str, after: str, evidence: str
) -> None:
    assert evidence, "Confirm the publisher is stopped before cancelling its intent"
    with ownership(factory):
        state = read_state(factory)
        assert state["intent"]["before"] == before
        assert state["intent"]["after"] == after
        repo = _repo(factory)
        actual = repo.lookup_branch("main")
        assert not any(
            info.id == after for info in repo.ancestry(snapshot_id=actual)
        ), "Published epoch cannot be cancelled"
        state["cancelled_intent"] = {
            **state.pop("intent"),
            "cancellation": evidence,
            "observed_main": actual,
        }
        _save(factory, state)


def _reconcile_epoch(repo: icechunk.Repository, state: dict[str, Any]) -> None:
    intent = state.get("intent")
    if intent is None:
        return
    actual = repo.lookup_branch("main")
    if actual == intent["before"]:
        raise AssertionError("Mirror epoch awaits publication or explicit cancellation")
    elif actual == intent["after"] or any(
        info.id == intent["after"] for info in repo.ancestry(snapshot_id=actual)
    ):
        assert state["snapshot"] == intent["checkpoint"]
        state["snapshot"] = intent["after"]
        del state["intent"]
    else:
        raise AssertionError("Main does not match mirror epoch intent")


def _copy_coordinates_and_metadata(
    source: Store, destination: Store, *, checkpoint_length: int
) -> None:
    arrays, replicas, coordinates, offset, length = _mapping(
        source, destination, checkpoint_length=checkpoint_length
    )
    stage = zarr.storage.MemoryStore()
    root = zarr.open_group(
        destination, mode="r", use_consolidated=False
    ).metadata.to_dict()
    root.pop("consolidated_metadata", None)
    _put(stage, "zarr.json", json.dumps(root).encode())
    for name, array in arrays.items():
        metadata = _metadata(replicas[name])
        metadata["shape"] = list(metadata["shape"])
        dims = metadata["dimension_names"]
        if _DIM in dims:
            metadata["shape"][dims.index(_DIM)] = length
        _put(stage, f"{name}/zarr.json", json.dumps(metadata).encode())
        if name in coordinates:
            values = np.asarray(array[...])
            if _DIM in dims:
                selection = [slice(None)] * len(dims)
                selection[dims.index(_DIM)] = slice(offset, None)
                values = values[tuple(selection)]
            output = zarr.open_array(
                stage, path=name, mode="r+", config={"write_empty_chunks": False}
            )
            output[...] = values
            prefix = f"{name}/c"
            keys = set(sync(_keys(stage, prefix)))
            for key in keys:
                _put(destination, key, _bytes(stage, key))
            for key in set(sync(_keys(destination, prefix))) - keys:
                _put(destination, key, None)
    with ignore_consolidated_metadata_spec_warning():
        zarr.consolidate_metadata(stage)
    for name in arrays:
        _put(destination, f"{name}/zarr.json", _bytes(stage, f"{name}/zarr.json"))
    _put(destination, "zarr.json", _bytes(stage, "zarr.json"))


def mirror_published(factory: StoreFactory, snapshot: str) -> None:
    with ownership(factory):
        state = read_state(factory)
        if state.get("pending") and state["pending"] != snapshot:
            _mirror_published(factory, state["pending"])
        _mirror_published(factory, snapshot)


def retry_mirror(factory: StoreFactory) -> None:
    mirror_published(factory, _repo(factory).lookup_branch("main"))


def _mirror_published(factory: StoreFactory, snapshot: str) -> None:
    assert factory.replica_mode() == "active"
    state = read_state(factory)
    repo = _repo(factory)
    _reconcile_epoch(repo, state)
    _save(factory, state)
    _published(repo, snapshot)
    baseline = state["snapshot"]
    _published(repo, baseline)
    if any(
        info.id == snapshot for info in repo.ancestry(snapshot_id=baseline)
    ) and not state.get("pending"):
        return
    assert any(info.id == baseline for info in repo.ancestry(snapshot_id=snapshot)), (
        "Unexplained non-ancestral mirror change"
    )
    if state.get("pending"):
        assert state["pending"] == snapshot, (
            "Retry pending mirror target before advancing"
        )
    source = repo.readonly_session(snapshot_id=snapshot).store
    with (
        xr.open_zarr(
            source, chunks=None, consolidated=False, decode_timedelta=True
        ) as current,
        xr.open_zarr(
            repo.readonly_session(snapshot_id=baseline).store,
            chunks=None,
            consolidated=False,
            decode_timedelta=True,
        ) as previous,
    ):
        assert current[_DIM].values[0] == previous[_DIM].values[0], (
            "Origin changed without a mirror epoch"
        )
        checkpoint_length = int(
            (previous[_DIM].values >= _ORIGIN.to_datetime64()).sum()
        )
    replicas = factory.replica_stores(writable=True, mirror=True)
    assert len(replicas) == 1
    destination = replicas[0]
    arrays, replica_arrays, coordinates, offset, _ = _mapping(
        source,
        destination,
        checkpoint_length=checkpoint_length if state.get("pending") else None,
    )
    if not state.get("pending"):
        assert replica_arrays[_DIM].shape == (checkpoint_length,), (
            "Replica coverage differs from checkpoint"
        )
    state["pending"] = snapshot
    _save(factory, state)
    if baseline != snapshot:
        diff = repo.diff(from_snapshot_id=baseline, to_snapshot_id=snapshot)
        assert not any(
            (
                diff.new_arrays,
                diff.deleted_arrays,
                diff.new_groups,
                diff.deleted_groups,
                diff.moved_nodes,
            )
        ), "Mirror schema drift"
        for path, indices in diff.updated_chunks.items():
            name = path.lstrip("/")
            if name in coordinates:
                continue
            assert name in arrays
            for index in indices:
                key = f"{name}/c/" + "/".join(map(str, index))
                mapped = _mapped_key(key, offset)
                if mapped is not None:
                    _put(destination, mapped, _bytes(source, key))
    _copy_coordinates_and_metadata(
        source, destination, checkpoint_length=checkpoint_length
    )
    state.update(snapshot=snapshot, pending=None)
    _save(factory, state)
