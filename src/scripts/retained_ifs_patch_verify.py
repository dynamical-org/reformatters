import argparse
import asyncio
import hashlib
import json
import sys
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import icechunk
import numpy as np
import xarray as xr
import zarr
import zarr.buffer
import zarr.storage
from zarr.abc.store import Store
from zarr.core.sync import sync

from reformatters.common.logging import get_logger

log = get_logger(__name__)
BASE = "VZSXDB55KG2P774ZHT8G"
PROTECTED = "ifs-ens-full-20261102T215837Z-187dd836"
REPOSITORY = "s3://dynamical-ecmwf-ifs-ens/ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk"
SCOPE = {
    ("pressure_reduced_to_mean_sea_level", "2023-10-08", 30, 48),
    ("precipitation_surface", "2023-11-05", 45, 111),
    ("precipitation_surface", "2023-11-05", 45, 114),
    ("wind_v_10m", "2023-12-30", 11, 3),
    ("pressure_surface", "2024-01-17", 25, 90),
    ("temperature_2m", "2024-02-03", 35, 90),
    ("categorical_precipitation_type_surface", "2025-10-13", 3, 258),
}
EXPECTED_HASHES = {
    (
        "precipitation_surface",
        "2023-11-05T00:00:00",
        45,
        111,
    ): "a116e297d2295874b725946b08446265b116adf51a41d4b5ed65258a6f08e45b",
    (
        "precipitation_surface",
        "2023-11-05T00:00:00",
        45,
        114,
    ): "7a89d4bd372cfe12b80ead10b199623e6366cc0609f1e8e8cc9f87ac0b594d74",
    (
        "pressure_reduced_to_mean_sea_level",
        "2023-10-08T00:00:00",
        30,
        48,
    ): "bbbcfc689e8767f857078bc670ef63f6f0aee6dae2bba946b3595768ad7ee52d",
    (
        "wind_v_10m",
        "2023-12-30T00:00:00",
        11,
        3,
    ): "8f408ec60d323221c8f7d6b6a253d3d79bcdd4afc05acef3d5ac4cddf4623c7d",
    (
        "pressure_surface",
        "2024-01-17T00:00:00",
        25,
        90,
    ): "5f7bf886234e2d39c3b6b77ea7681b26ddcbc8495d4c4def63a2daa674b4f3db",
    (
        "temperature_2m",
        "2024-02-03T00:00:00",
        35,
        90,
    ): "c7a7a3e2625df2350cd03db52a315dc1d545844304b0dccf919579d17142a183",
    (
        "categorical_precipitation_type_surface",
        "2025-10-13T00:00:00",
        3,
        258,
    ): "773ea6ded9e4c9f2f50ab14a2f2ec08ee672ba083c64f992c47dfbe102415829",
}
SCIENCE_SHA256 = "12ef8c70444f6146c4d3f312568407f980f7afaeb544d07c78e21b2ae84792f3"
DIMS = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def raw_digest(values: np.ndarray) -> str:
    check(values.dtype == np.dtype("<f4"), "Expected little-endian float32")
    return digest(np.ascontiguousarray(values).tobytes())


def read(store: Store, key: str) -> bytes:
    value = sync(store.get(key, prototype=zarr.buffer.default_buffer_prototype()))
    check(value is not None, f"Missing key {key}")
    assert value is not None
    return value.to_bytes()


async def list_keys(store: Store) -> set[str]:
    return {key async for key in store.list()}


def validate_manifest(manifest: dict[str, Any]) -> None:
    check(sys.flags.optimize == 0, "Optimized Python is forbidden")
    check(manifest["version"] == 2, "Unexpected manifest version")
    check(
        manifest["dataset_id"] == "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "Unexpected dataset",
    )
    check(
        manifest["scientific_manifest_sha256"] == SCIENCE_SHA256,
        "Scientific manifest pin mismatch",
    )
    check(manifest["base_snapshot"] == BASE, "Unexpected base snapshot")
    check(manifest["protected_branch"] == PROTECTED, "Unexpected protected branch")
    check(manifest["repository"] == REPOSITORY, "Unexpected repository")
    check(
        manifest["candidate_branch"] not in {PROTECTED, "main"}, "Forbidden candidate"
    )
    fields = [
        (u["variable"], u["init"][:10], f["member"], f["lead_hours"])
        for u in manifest["units"]
        for f in u["fields"]
    ]
    check(len(fields) == 7 and set(fields) == SCOPE, "Unexpected repair scope")
    expected = {
        (u["variable"], u["init"], f["member"], f["lead_hours"]): f["expected_sha256"]
        for u in manifest["units"]
        for f in u["fields"]
    }
    check(expected == EXPECTED_HASHES, "Frozen expected fields or full init mismatch")
    check(len(manifest["units"]) == 6, "Expected six complete units")
    check(bool(manifest["approved_shards"]), "Missing approved shard hashes")


def metadata(store: Store) -> dict[str, bytes]:
    names = asyncio.run(list_keys(store))
    coordinate_names = set(
        xr.open_zarr(
            store, chunks=None, consolidated=False, decode_timedelta=True
        ).coords
    )
    return {
        key: read(store, key)
        for key in sorted(names)
        if key.endswith("zarr.json") or key.split("/")[0] in coordinate_names
    }


def metadata_sha256(values: dict[str, bytes]) -> str:
    return digest(
        json.dumps(
            {key: digest(value) for key, value in values.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def diff_keys(diff: icechunk.Diff) -> set[str]:
    check(
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
        "Repository metadata or structure changed",
    )
    return {
        f"{path.lstrip('/')}/c/{'/'.join(map(str, position))}"
        for path, positions in diff.updated_chunks.items()
        for position in positions
    }


def compare_shard(
    left: np.ndarray,
    right: np.ndarray,
    allowed_positions: set[tuple[int, int]],
) -> tuple[int, int]:
    check(left.shape == right.shape and left.dtype == right.dtype, "Shape/dtype drift")
    check(left.ndim == 4 and left.dtype == np.dtype("<f4"), "Invalid decoded shard")
    changed = outside = 0
    for lead in range(left.shape[0]):
        for member in range(left.shape[1]):
            differences = np.count_nonzero(
                left[lead, member].view(np.uint32)
                != right[lead, member].view(np.uint32)
            )
            changed += int(differences)
            if (lead, member) not in allowed_positions:
                check(differences == 0, "Outside-mask uint32 change")
                outside += left.shape[2] * left.shape[3]
    return changed, outside


def coordinate_index(values: np.ndarray, target: object) -> int:
    matches = np.flatnonzero(values == target)
    check(len(matches) == 1, f"Coordinate must match exactly once: {target}")
    return int(matches[0])


def verify_unit(
    before: zarr.Group,
    after: zarr.Group,
    after_store: Store,
    unit: dict[str, Any],
    approved: dict[str, Any],
) -> dict[str, Any]:
    variable = unit["variable"]
    left_array = before[variable]
    right_array = after[variable]
    assert isinstance(left_array, zarr.Array)
    assert isinstance(right_array, zarr.Array)
    check(left_array.shape == right_array.shape, "Unit shape changed")
    check(left_array.shards is not None, "Missing shards")
    assert left_array.shards is not None
    shard = left_array.shards
    check(
        shard[0] == 1 and shard[1:3] == left_array.shape[1:3],
        "Unsupported shard layout",
    )
    coords = xr.open_zarr(
        before.store, chunks=None, consolidated=False, decode_timedelta=True
    )
    index = coordinate_index(coords.init_time.values, np.datetime64(unit["init"]))
    fields = {
        (
            coordinate_index(
                coords.lead_time.values, np.timedelta64(f["lead_hours"], "h")
            ),
            coordinate_index(coords.ensemble_member.values, f["member"]),
        ): f
        for f in unit["fields"]
    }
    expected_outputs = {
        position: np.empty(left_array.shape[-2:], dtype="<f4") for position in fields
    }
    total = changed = outside = 0
    seen_keys = set()
    all_keys = set()
    for y in range(0, left_array.shape[3], shard[3]):
        for x in range(0, left_array.shape[4], shard[4]):
            ys, xs = slice(y, y + shard[3]), slice(x, x + shard[4])
            selection = (index, slice(None), slice(None), ys, xs)
            left = np.asarray(left_array[selection])
            right = np.asarray(right_array[selection])
            delta, outside_count = compare_shard(left, right, set(fields))
            changed += delta
            outside += outside_count
            total += left.size
            key = f"{variable}/c/{index}/0/0/{y // shard[3]}/{x // shard[4]}"
            all_keys.add(key)
            if delta:
                check(key in approved, f"Unapproved changed shard {key}")
            if key in approved:
                check(
                    digest(read(after_store, key)) == approved[key]["encoded_sha256"],
                    f"Encoded shard mismatch: {key}",
                )
                check(
                    raw_digest(right) == approved[key]["decoded_sha256"],
                    f"Decoded shard mismatch: {key}",
                )
                seen_keys.add(key)
            for position, output in expected_outputs.items():
                output[ys, xs] = right[position]
            log.info(f"Verified {key}: {delta} changed cells")
    for position, output in expected_outputs.items():
        check(
            raw_digest(output) == fields[position]["expected_sha256"],
            f"Correct-source field mismatch: {variable} {position}",
        )
    check(total == int(np.prod(left_array.shape[1:])), "Incomplete full-unit coverage")
    check(changed > 0, "No corrected cells in unit")
    return {
        "variable": variable,
        "init": unit["init"],
        "compared_cells": total,
        "outside_cells": outside,
        "changed_cells": changed,
        "checked_keys": sorted(seen_keys),
        "all_keys": sorted(all_keys),
    }


def verify(
    repo: icechunk.Repository, manifest: dict[str, Any], snapshot: str
) -> dict[str, Any]:
    validate_manifest(manifest)
    check(repo.lookup_branch(PROTECTED) == BASE, "Protected ref moved")
    check(
        repo.lookup_branch(manifest["candidate_branch"]) == snapshot,
        "Candidate ref mismatch",
    )
    ancestry = list(islice(repo.ancestry(snapshot_id=snapshot), 2))
    check(
        len(ancestry) == 2 and ancestry[0].id == snapshot and ancestry[1].id == BASE,
        "Candidate is not one commit above base",
    )
    check(
        diff_keys(repo.diff(from_snapshot_id=BASE, to_snapshot_id=snapshot))
        == set(manifest["approved_shards"]),
        "Repository changed-key set mismatch",
    )
    before_store = repo.readonly_session(snapshot_id=BASE).store
    after_store = repo.readonly_session(snapshot_id=snapshot).store
    original_metadata = metadata(before_store)
    check(
        metadata_sha256(original_metadata) == manifest["metadata_sha256"],
        "Base metadata hash mismatch",
    )
    check(metadata(after_store) == original_metadata, "Metadata/coordinates changed")
    before = zarr.open_group(before_store, mode="r")
    after = zarr.open_group(after_store, mode="r")
    units = [
        verify_unit(before, after, after_store, u, manifest["approved_shards"])
        for u in manifest["units"]
    ]
    check(
        {key for u in units for key in u["checked_keys"]}
        == set(manifest["approved_shards"]),
        "Approved key coverage incomplete",
    )
    check(
        repo.lookup_branch(PROTECTED) == BASE, "Protected ref moved during verification"
    )
    check(
        repo.lookup_branch(manifest["candidate_branch"]) == snapshot,
        "Candidate ref moved during verification",
    )
    return {
        "status": "pass",
        "snapshot": snapshot,
        "base_snapshot": BASE,
        "candidate_branch": manifest["candidate_branch"],
        "units": units,
    }


def verify_stage(
    repo: icechunk.Repository, manifest: dict[str, Any], stage: Path
) -> dict[str, Any]:
    validate_manifest(manifest)
    check(repo.lookup_branch(PROTECTED) == BASE, "Protected ref moved")
    before_store = repo.readonly_session(snapshot_id=BASE).store
    after_store = zarr.storage.LocalStore(stage, read_only=True)
    original_metadata = metadata(before_store)
    check(
        metadata_sha256(original_metadata) == manifest["metadata_sha256"],
        "Base metadata hash mismatch",
    )
    check(
        metadata(after_store) == original_metadata,
        "Staged metadata/coordinates changed",
    )
    before = zarr.open_group(before_store, mode="r")
    after = zarr.open_group(after_store, mode="r")
    units = [
        verify_unit(before, after, after_store, u, manifest["approved_shards"])
        for u in manifest["units"]
    ]
    check(
        {key for u in units for key in u["checked_keys"]}
        == set(manifest["approved_shards"]),
        "Approved key coverage incomplete",
    )
    check(
        repo.lookup_branch(PROTECTED) == BASE, "Protected ref moved during verification"
    )
    check(
        asyncio.run(list_keys(after_store))
        == set(original_metadata) | {key for unit in units for key in unit["all_keys"]},
        "Unexpected or missing local stage keys",
    )
    return {
        "status": "pass-local-stage",
        "base_snapshot": BASE,
        "units": units,
        "approved_shards": manifest["approved_shards"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently verify a retained repair using read-only sessions"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot")
    mode.add_argument("--stage-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "Output already exists")
    body = args.manifest.read_bytes()
    check(digest(body) == args.manifest_sha256, "Manifest hash mismatch")
    manifest = json.loads(body)
    validate_manifest(manifest)
    url = urlparse(manifest["repository"])
    repo = icechunk.Repository.open(
        icechunk.s3_storage(
            bucket=url.netloc, prefix=url.path.strip("/"), anonymous=True
        )
    )
    zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
    result = (
        verify_stage(repo, manifest, args.stage_dir)
        if args.stage_dir
        else verify(repo, manifest, args.snapshot)
    )
    result["manifest_sha256"] = args.manifest_sha256
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    log.info(f"PASS independent verification: {args.snapshot or args.stage_dir}")


if __name__ == "__main__":
    main()
