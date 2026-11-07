import argparse
import asyncio
import hashlib
import json
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import zarr.buffer
import zarr.storage
from pydantic import BaseModel, ConfigDict, Field
from zarr.abc.store import Store
from zarr.core.sync import sync

import reformatters
import scripts
from reformatters.common.logging import get_logger
from reformatters.common.zarr import sync_to_store
from scripts.icechunk_utils import open_repo

log = get_logger(__name__)
DATASET_ID = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
DIMS = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Artifact(Model):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    def checked_path(self, root: Path) -> Path:
        path = (root / self.path).resolve()
        require(path.is_relative_to(root.resolve()), "Artifact escapes input directory")
        require(path.is_file(), f"Missing artifact: {self.path}")
        require(
            file_digest(path) == self.sha256, f"Artifact hash mismatch: {self.path}"
        )
        return path


class ExpectedField(Model):
    member: int = Field(ge=0)
    lead_hours: int = Field(ge=0)
    values: Artifact
    source_evidence: Artifact


class Unit(Model):
    variable: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    init: str
    requests: Artifact
    fields: tuple[ExpectedField, ...] = Field(min_length=1)


class Manifest(Model):
    version: Literal[1]
    dataset_id: Literal["ecmwf-ifs-ens-forecast-15-day-0-25-degree"]
    repository: str
    base_snapshot: str
    protected_branch: str
    candidate_branch: str
    runtime_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    metadata_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    prerequisites: Artifact
    units: tuple[Unit, ...] = Field(min_length=1)


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime_digest(root: Path) -> str:
    paths = [root / "uv.lock", root / "pyproject.toml"]
    paths.extend(sorted((root / "src").rglob("*.py")))
    return object_digest({str(p.relative_to(root)): file_digest(p) for p in paths})


def read_bytes(store: Store, key: str) -> bytes:
    data = sync(store.get(key, prototype=zarr.buffer.default_buffer_prototype()))
    if data is None:
        raise ValueError(f"Missing store key: {key}")
    return data.to_bytes()


async def keys(store: Store) -> list[str]:
    return [key async for key in store.list()]


def open_tree(store: Store) -> xr.DataTree:
    return xr.open_datatree(
        store,  # ty: ignore[invalid-argument-type]
        engine="zarr",
        chunks={},
        decode_timedelta=True,
        consolidated=False,
    )


def metadata_bytes(store: Store, tree: xr.DataTree) -> dict[str, bytes]:
    require(tree.groups == ("/",), "Only a single root group is supported")
    coordinates = set(tree.coords)
    return {
        key: read_bytes(store, key)
        for key in sorted(asyncio.run(keys(store)))
        if key.endswith("zarr.json") or key.split("/", 1)[0] in coordinates
    }


def metadata_digest(metadata: Mapping[str, bytes]) -> str:
    return object_digest(
        {key: hashlib.sha256(value).hexdigest() for key, value in metadata.items()}
    )


def clone_metadata(metadata: Mapping[str, bytes], directory: Path) -> None:
    directory.mkdir(exist_ok=False)
    for key, value in metadata.items():
        path = directory / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)


def resolve_units(manifest: Manifest, tree: xr.DataTree) -> list[tuple[Unit, int]]:
    require(tree.attrs.get("dataset_id") == manifest.dataset_id, "Dataset mismatch")
    require(tree.groups == ("/",), "Nested groups are not supported")
    pairs: set[tuple[str, pd.Timestamp]] = set()
    result = []
    for unit in manifest.units:
        init = pd.Timestamp(unit.init)
        require(init.tzinfo is None, "Init must be naive UTC")
        pair = (unit.variable, init)
        require(pair not in pairs, "Duplicate variable/init unit")
        pairs.add(pair)
        indices = np.flatnonzero(tree.init_time.values == init.to_datetime64())
        require(len(indices) == 1, f"Init must resolve exactly once: {unit.init}")
        require(unit.variable in tree.data_vars, f"Unknown variable: {unit.variable}")
        array = tree[unit.variable]
        require(tuple(array.dims) == DIMS, "Unexpected variable dimensions")
        require(array.dtype == np.dtype("float32"), "Only float32 is supported")
        require(array.encoding["shards"][0] == 1, "Init shard width must equal one")
        seen = set()
        for field in unit.fields:
            coordinate = (field.member, field.lead_hours)
            require(coordinate not in seen, "Duplicate permitted field")
            seen.add(coordinate)
            require(
                np.count_nonzero(array.ensemble_member.values == field.member) == 1,
                "Unknown member",
            )
            require(
                np.count_nonzero(
                    array.lead_time.values == np.timedelta64(field.lead_hours, "h")
                )
                == 1,
                "Unknown lead",
            )
        result.append((unit, int(indices[0])))
    return result


def equivalent(
    left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    require(
        left.shape == right.shape and left.dtype == right.dtype,
        "Array shape/dtype mismatch",
    )
    return (left.view(np.uint32) == right.view(np.uint32)) | (
        np.isnan(left) & np.isnan(right)
    )


def compare_unit(
    before: xr.DataTree,
    after: xr.DataTree,
    unit: Unit,
    index: int,
    input_root: Path,
) -> dict[str, int]:
    original = before[unit.variable].isel(init_time=index)
    candidate = after[unit.variable].isel(init_time=index)
    require(original.shape == candidate.shape, "Staged shape mismatch")
    fields = {}
    for field in unit.fields:
        path = field.values.checked_path(input_root)
        evidence = json.loads(
            field.source_evidence.checked_path(input_root).read_text()
        )
        identity = {
            "status": "pass",
            "independent": True,
            "init": unit.init,
            "variable": unit.variable,
            "member": field.member,
            "lead_hours": field.lead_hours,
            "values_sha256": field.values.sha256,
        }
        require(
            all(evidence.get(key) == value for key, value in identity.items()),
            "Independent source evidence is not bound to this field",
        )
        require(
            isinstance(evidence.get("source_message_sha256s"), list)
            and bool(evidence["source_message_sha256s"])
            and all(
                isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
                for digest in evidence["source_message_sha256s"]
            ),
            "Missing source message identities",
        )
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        require(
            values.shape == original.shape[-2:] and values.dtype == np.dtype("float32"),
            "Expected source field shape/dtype mismatch",
        )
        fields[field.member, field.lead_hours] = values
    changed = total = outside = 0
    for latitude in range(0, original.sizes["latitude"], 32):
        for longitude in range(0, original.sizes["longitude"], 320):
            selection = {
                "latitude": slice(latitude, latitude + 32),
                "longitude": slice(longitude, longitude + 320),
            }
            left = np.asarray(original.isel(selection).values)
            right = np.asarray(candidate.isel(selection).values)
            same = equivalent(left, right)
            allowed = np.zeros(same.shape, dtype=bool)
            for lead_index, lead in enumerate(original.lead_time.values):
                hours = int(lead / np.timedelta64(1, "h"))
                for member_index, member in enumerate(original.ensemble_member.values):
                    expected = fields.get((int(member), hours))
                    if expected is not None:
                        allowed[lead_index, member_index] = True
                        require(
                            bool(
                                equivalent(
                                    right[lead_index, member_index],
                                    expected[
                                        latitude : latitude + 32,
                                        longitude : longitude + 320,
                                    ],
                                ).all()
                            ),
                            f"Independent source mismatch: {unit.variable} m{member} L{hours}",
                        )
            require(
                bool(same[~allowed].all()),
                f"Unexpected change outside permitted fields: {unit.variable}",
            )
            changed += int(np.count_nonzero(~same))
            total += same.size
            outside += int(np.count_nonzero(~allowed))
    require(total == original.size, "Incomplete full-unit comparison")
    require(
        total - outside
        == len(fields) * original.sizes["latitude"] * original.sizes["longitude"],
        "Incomplete permitted-field comparison",
    )
    return {"compared_cells": total, "changed_cells": changed, "outside_cells": outside}


def allowed_keys(tree: xr.DataTree, unit: Unit, index: int) -> set[str]:
    array = tree[unit.variable]
    shards = array.encoding["shards"]
    require(len(shards) == len(DIMS), "Missing shard layout")
    require(
        shards[1] >= array.sizes["lead_time"]
        and shards[2] >= array.sizes["ensemble_member"],
        "Lead/member sharding is not supported",
    )
    return {
        f"{unit.variable}/c/{index}/0/0/{lat}/{lon}"
        for lat in range((array.sizes["latitude"] + shards[3] - 1) // shards[3])
        for lon in range((array.sizes["longitude"] + shards[4] - 1) // shards[4])
    }


def changed_shards(
    before: xr.DataTree, after: xr.DataTree, unit: Unit, index: int
) -> set[str]:
    array = before[unit.variable]
    shards = array.encoding["shards"]
    changed = set()
    for key in allowed_keys(before, unit, index):
        lat, lon = map(int, key.split("/")[-2:])
        selection = {
            "init_time": index,
            "latitude": slice(lat * shards[3], (lat + 1) * shards[3]),
            "longitude": slice(lon * shards[4], (lon + 1) * shards[4]),
        }
        for latitude in range(
            lat * shards[3], min((lat + 1) * shards[3], array.sizes["latitude"]), 32
        ):
            selection["latitude"] = slice(
                latitude, min(latitude + 32, (lat + 1) * shards[3])
            )
            for longitude in range(
                lon * shards[4],
                min((lon + 1) * shards[4], array.sizes["longitude"]),
                320,
            ):
                selection["longitude"] = slice(
                    longitude, min(longitude + 320, (lon + 1) * shards[4])
                )
                left = np.asarray(array.isel(selection).values)
                right = np.asarray(after[unit.variable].isel(selection).values)
                if not equivalent(left, right).all():
                    changed.add(key)
                    break
            if key in changed:
                break
    return changed


def validate_diff(diff: icechunk.Diff, expected: set[str]) -> None:
    require(
        not any(
            (
                diff.new_groups,
                diff.new_arrays,
                diff.deleted_groups,
                diff.deleted_arrays,
                diff.updated_groups,
                diff.updated_arrays,
                diff.moved_nodes,
            )
        ),
        "Metadata or structure changed",
    )
    actual = {
        f"{path.lstrip('/')}/c/{'/'.join(map(str, coordinate))}"
        for path, coordinates in diff.updated_chunks.items()
        for coordinate in coordinates
    }
    require(
        actual == expected, "Transaction changed-key set differs from validated shards"
    )


def validate_refs(repo: icechunk.Repository, manifest: Manifest) -> None:
    require(
        manifest.protected_branch != "main",
        "Protected branch must be distinct from main",
    )
    name = manifest.candidate_branch
    require(
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)),
        "Invalid candidate branch name",
    )
    require(
        name not in {"main", manifest.protected_branch}, "Forbidden candidate branch"
    )
    require(
        name not in repo.list_branches(),
        "Candidate branch already exists; investigate before retry",
    )
    require(
        repo.lookup_branch(manifest.protected_branch) == manifest.base_snapshot,
        "Protected branch moved",
    )


def verify_inputs(manifest: Manifest, root: Path, code_root: Path) -> None:
    require(sys.flags.optimize == 0, "Optimized Python is not allowed")
    require(
        runtime_digest(code_root) == manifest.runtime_sha256,
        "Runtime source/lock hash mismatch",
    )
    for module in (reformatters, scripts):
        require(
            module.__file__ is not None
            and Path(module.__file__)
            .resolve()
            .is_relative_to((code_root / "src").resolve()),
            "Imported runtime does not match hashed source tree",
        )
    locked = tomllib.loads((code_root / "uv.lock").read_text())
    for package in locked["package"]:
        if package["name"] in {
            "numpy",
            "pandas",
            "xarray",
            "zarr",
            "icechunk",
            "rasterio",
            "numba",
            "numcodecs",
            "rioxarray",
            "affine",
        }:
            require(
                version(package["name"]) == package["version"],
                f"Installed dependency differs from lock: {package['name']}",
            )
    receipt = json.loads(manifest.prerequisites.checked_path(root).read_text())
    expected = {
        "status": "pass",
        "base_snapshot": manifest.base_snapshot,
        "runtime_sha256": manifest.runtime_sha256,
        "units_sha256": object_digest(
            [u.model_dump(mode="json") for u in manifest.units]
        ),
    }
    require(
        all(receipt.get(key) == value for key, value in expected.items()),
        "Prerequisite receipt binding mismatch",
    )
    for gate in (
        "selector_parity",
        "config_parity",
        "decoder_transform_parity",
        "full_field_source_confirmation",
        "retention_review",
    ):
        require(receipt.get(gate) is True, f"Missing reviewed prerequisite: {gate}")
        evidence = receipt.get("evidence", {}).get(gate)
        require(evidence is not None, f"Missing measurement artifact: {gate}")
        Artifact.model_validate(evidence).checked_path(root)
    for unit in manifest.units:
        unit.requests.checked_path(root)
        for field in unit.fields:
            field.values.checked_path(root)
            field.source_evidence.checked_path(root)


def write_record(path: Path, record: object) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    require(not path.exists(), "Evidence output already exists")
    temporary.rename(path)


def run(
    repo: icechunk.Repository,
    manifest: Manifest,
    input_root: Path,
    work: Path,
    regenerate: Callable[..., dict[str, Any]],
    *,
    commit: bool,
) -> dict[str, Any]:
    require(sys.flags.optimize == 0, "Optimized Python is not allowed")
    validate_refs(repo, manifest)
    work.mkdir(exist_ok=False)
    try:
        return run_in_work_directory(
            repo, manifest, input_root, work, regenerate, commit=commit
        )
    except BaseException as error:
        write_record(
            work / "STOP.json",
            {
                "status": "stopped",
                "error_type": type(error).__name__,
                "message": str(error),
                "automatic_retry_allowed": False,
                "base_snapshot": manifest.base_snapshot,
                "candidate_branch": manifest.candidate_branch,
                "manifest_sha256": object_digest(manifest.model_dump(mode="json")),
            },
        )
        raise


def run_in_work_directory(  # noqa: PLR0915
    repo: icechunk.Repository,
    manifest: Manifest,
    input_root: Path,
    work: Path,
    regenerate: Callable[..., dict[str, Any]],
    *,
    commit: bool,
) -> dict[str, Any]:
    baseline_store = repo.readonly_session(snapshot_id=manifest.base_snapshot).store
    before = open_tree(baseline_store)
    metadata = metadata_bytes(baseline_store, before)
    require(
        metadata_digest(metadata) == manifest.metadata_sha256,
        "Baseline metadata/coordinate hash mismatch",
    )
    resolved = resolve_units(manifest, before)
    stage_path = work / "stage"
    clone_metadata(metadata, stage_path)
    stage = zarr.storage.LocalStore(stage_path)
    changed: set[str] = set()
    records = []
    for number, (unit, index) in enumerate(resolved):
        unit_dir = work / f"unit-{number:02d}"
        unit_dir.mkdir()
        temporary = unit_dir / "temporary"
        clone_metadata(metadata, temporary)
        receipt = regenerate(
            before,
            unit.variable,
            index,
            temporary,
            stage,
            unit.requests.checked_path(input_root),
        )
        require(
            receipt.get("requests_sha256") == unit.requests.sha256,
            "Consumed request roster hash mismatch",
        )
        after = open_tree(stage)
        require(
            metadata_bytes(stage, after) == metadata,
            "Staging metadata/coordinates changed",
        )
        comparison = compare_unit(before, after, unit, index, input_root)
        require(
            comparison["changed_cells"] > 0,
            "Unit has no correction; investigate stale scope",
        )
        expected_keys = allowed_keys(before, unit, index)
        actual_keys = set(asyncio.run(keys(stage))) - set(metadata)
        require(
            actual_keys
            <= set().union(
                *(allowed_keys(before, u, i) for u, i in resolved[: number + 1])
            ),
            "Unexpected staging data key",
        )
        require(
            expected_keys <= actual_keys,
            "Staging omitted a shard, including possible fill-only output",
        )
        changed.update(changed_shards(before, after, unit, index))
        record = {
            "unit": unit.model_dump(mode="json"),
            "init_index": index,
            "processing": receipt,
            "comparison": comparison,
        }
        write_record(unit_dir / "verified.json", record)
        records.append(record)
    stage_hashes = {
        key: hashlib.sha256(read_bytes(stage, key)).hexdigest()
        for key in sorted(changed)
    }
    result: dict[str, Any] = {
        "status": "staged",
        "base_snapshot": manifest.base_snapshot,
        "manifest_sha256": object_digest(manifest.model_dump(mode="json")),
        "changed_shards": stage_hashes,
        "units": records,
    }
    write_record(work / "precommit.json", result)
    validate_refs(repo, manifest)
    if commit:
        repo.create_branch(manifest.candidate_branch, manifest.base_snapshot)
        session = repo.writable_session(manifest.candidate_branch)
        require(session.snapshot_id == manifest.base_snapshot, "Candidate base changed")
        for key, digest in stage_hashes.items():
            value = read_bytes(stage, key)
            require(
                hashlib.sha256(value).hexdigest() == digest,
                "Staged shard changed after validation",
            )
            sync_to_store(session.store, key, value)
        validate_diff(session.status(), changed)
        require(
            repo.lookup_branch(manifest.protected_branch) == manifest.base_snapshot,
            "Protected branch moved before commit",
        )
        snapshot = session.commit(
            f"Verified retained repair {result['manifest_sha256']}"
        )
        write_record(
            work / "committed.json",
            {"snapshot": snapshot, "candidate": manifest.candidate_branch},
        )
        ancestry = list(islice(repo.ancestry(snapshot_id=snapshot), 2))
        require(
            len(ancestry) >= 2 and ancestry[1].id == manifest.base_snapshot,
            "Unexpected commit parent",
        )
        validate_diff(
            repo.diff(from_snapshot_id=manifest.base_snapshot, to_snapshot_id=snapshot),
            changed,
        )
        committed_store = repo.readonly_session(snapshot_id=snapshot).store
        committed = open_tree(committed_store)
        require(
            metadata_bytes(committed_store, committed) == metadata,
            "Committed metadata/coordinates changed",
        )
        for unit, index in resolved:
            compare_unit(before, committed, unit, index, input_root)
        require(
            repo.lookup_branch(manifest.protected_branch) == manifest.base_snapshot,
            "Protected branch moved after commit",
        )
        require(
            repo.lookup_branch(manifest.candidate_branch) == snapshot,
            "Candidate moved after commit",
        )
        result.update(status="committed-and-verified", snapshot=snapshot)
    write_record(work / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--auth", choices=("anonymous", "env", "secret"), default="anonymous"
    )
    parser.add_argument("--k8s-secret", default="")
    parser.add_argument("--commit-to-new-branch", action="store_true")
    args = parser.parse_args()
    require(
        file_digest(args.manifest) == args.manifest_sha256, "Manifest hash mismatch"
    )
    manifest = Manifest.model_validate_json(args.manifest.read_text())
    root = args.manifest.resolve().parent
    verify_inputs(manifest, root, Path(__file__).resolve().parents[2])
    require(
        urlparse(manifest.repository).scheme == "s3",
        "Only explicit S3 repositories are supported",
    )
    repo = open_repo(manifest.repository, args.auth, args.k8s_secret)
    from scripts.retained_ifs_repair_stage import regenerate_unit  # noqa: PLC0415

    result = run(
        repo,
        manifest,
        root,
        args.work_dir,
        regenerate_unit,
        commit=args.commit_to_new_branch,
    )

    log.info(f"Retained repair result: {result['status']}")


if __name__ == "__main__":
    main()
