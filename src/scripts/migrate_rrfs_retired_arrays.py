"""Plan or apply the authorized RRFS/REFS retired-array deletion on one repository."""

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import icechunk
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata

from reformatters.common.logging import get_logger
from scripts.icechunk_utils import open_repo

log = get_logger(__name__)

RETIRED_ARRAYS: dict[str, tuple[str, ...]] = {
    "noaa-rrfs-forecast-18-hour-virtual": (
        "frozen_precipitation_run_total_surface",
        "freezing_rain_run_total_surface",
        "frozen_precipitation_surface",
        "freezing_rain_surface",
    ),
    "noaa-rrfs-forecast-84-hour-virtual": (
        "frozen_precipitation_run_total_surface",
        "freezing_rain_run_total_surface",
        "frozen_precipitation_surface",
        "freezing_rain_surface",
    ),
    "noaa-rrfs-forecast-sub-hourly-virtual": (
        "frozen_precipitation_run_total_surface",
        "freezing_rain_run_total_surface",
    ),
    "noaa-rrfs-ens-forecast-virtual": (
        "freezing_rain_run_total_surface",
        "freezing_rain_surface",
    ),
    "noaa-refs-forecast-hourly-virtual": (
        "neighborhood_probability_of_freezing_rain_1_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_1_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_1_hour_surface_above_6p35",
        "composite_reflectivity",
    ),
    "noaa-refs-forecast-3-hourly-virtual": (
        "neighborhood_probability_of_freezing_rain_3_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_3_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_3_hour_surface_above_6p35",
        "neighborhood_probability_of_freezing_rain_6_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_6_hour_surface_above_12p7",
        "neighborhood_probability_of_freezing_rain_6_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_6_hour_surface_above_6p35",
        "neighborhood_probability_of_freezing_rain_12_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_12_hour_surface_above_12p7",
        "neighborhood_probability_of_freezing_rain_12_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_12_hour_surface_above_6p35",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_0p254",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_12p7",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_2p54",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_25p4",
        "neighborhood_probability_of_freezing_rain_24_hour_surface_above_6p35",
    ),
}
MIGRATION = "rrfs-retired-arrays-v1"


@dataclass(frozen=True)
class MigrationResult:
    status: Literal["planned", "applied", "already-applied"]
    snapshot: str
    historical_tag: str
    deleted_arrays: tuple[str, ...]


async def _read_json(store: icechunk.IcechunkStore, key: str) -> dict[str, Any]:
    value = await store.get(key, default_buffer_prototype())
    assert value is not None, f"Required metadata is missing: {key}"
    return json.loads(value.to_bytes())


async def _validate_schema(
    store: icechunk.IcechunkStore, dataset: str, paths: tuple[str, ...]
) -> tuple[dict[str, Any], bool]:
    root = await _read_json(store, "zarr.json")
    assert root["zarr_format"] == 3, "Expected Zarr v3"
    assert root["node_type"] == "group", "Expected a root group"
    assert root["attributes"]["dataset_id"] == dataset, "Dataset identity mismatch"
    consolidated = root.get("consolidated_metadata")
    if consolidated is not None:
        assert consolidated["kind"] == "inline", "Expected inline consolidated metadata"
    for path in paths:
        metadata = await _read_json(store, f"{path}/zarr.json")
        assert metadata["zarr_format"] == 3, f"Expected Zarr v3: {path}"
        assert metadata["node_type"] == "array", f"Expected an array: {path}"
        array = ArrayV3Metadata.from_dict(metadata)
        dims = ("init_time", "lead_time", "y", "x")
        if dataset == "noaa-rrfs-ens-forecast-virtual":
            dims = ("init_time", "ensemble_member", "lead_time", "y", "x")
        elif path == "composite_reflectivity":
            dims = ("init_time", "statistic", "lead_time", "y", "x")
        assert array.dimension_names == dims, f"Unexpected dimensions for {path}"
        if consolidated is not None:
            assert consolidated["metadata"].get(path) == metadata, (
                f"Consolidated metadata disagrees for {path}"
            )
            del consolidated["metadata"][path]
    return root, consolidated is not None


def _assert_deletion_diff(
    diff: icechunk.Diff, paths: tuple[str, ...], consolidated: bool
) -> None:
    assert diff.deleted_arrays == {f"/{path}" for path in paths}, (
        "Migration must delete exactly the authorized arrays"
    )
    assert diff.updated_groups == ({"/"} if consolidated else set())
    assert not diff.new_arrays
    assert not diff.new_groups
    assert not diff.deleted_groups
    assert not diff.updated_arrays
    assert not diff.updated_chunks
    assert not diff.moved_nodes


async def _delete_arrays(
    store: icechunk.IcechunkStore,
    paths: tuple[str, ...],
    root: dict[str, Any],
    consolidated: bool,
) -> None:
    for path in paths:
        # Deleting an Icechunk array node also removes its chunk references.
        await store.delete(f"{path}/zarr.json")
    if consolidated:
        await store.set(
            "zarr.json",
            default_buffer_prototype().buffer.from_bytes(json.dumps(root).encode()),
        )


def migrate(
    repo: icechunk.Repository,
    dataset: str,
    expected_snapshot: str,
    *,
    apply: bool = False,
) -> MigrationResult:
    """Require the reviewed head, or recognize its tagged, direct migration result."""
    assert dataset in RETIRED_ARRAYS, f"Dataset is not authorized: {dataset}"
    paths = RETIRED_ARRAYS[dataset]
    tag = f"{MIGRATION}-before-{dataset}-{expected_snapshot}"
    marker = {
        "migration": MIGRATION,
        "dataset": dataset,
        "expected_snapshot": expected_snapshot,
        "deleted_arrays": list(paths),
    }
    head = repo.lookup_branch("main")
    if tag in repo.list_tags():
        assert repo.lookup_tag(tag) == expected_snapshot, "Historical tag mismatch"
    original = repo.readonly_session(snapshot_id=expected_snapshot)
    root, consolidated = asyncio.run(_validate_schema(original.store, dataset, paths))
    if head != expected_snapshot:
        info = repo.lookup_snapshot(head)
        assert info.parent_id == expected_snapshot, (
            f"Stale head: expected {expected_snapshot}, found {head}"
        )
        assert info.metadata == marker, f"Stale head: {head} is not this migration"
        assert tag in repo.list_tags(), "Completed migration lacks historical tag"
        _assert_deletion_diff(
            repo.diff(from_snapshot_id=expected_snapshot, to_snapshot_id=head),
            paths,
            consolidated,
        )
        current = repo.readonly_session(snapshot_id=head)
        assert asyncio.run(_read_json(current.store, "zarr.json")) == root
        return MigrationResult("already-applied", head, tag, paths)

    result = MigrationResult("planned", head, tag, paths)
    if not apply:
        return result

    session = repo.writable_session("main")
    assert session.snapshot_id == expected_snapshot, (
        "Head changed before session creation"
    )
    if tag not in repo.list_tags():
        repo.create_tag(tag, expected_snapshot)
    assert repo.lookup_tag(tag) == expected_snapshot, "Historical tag mismatch"
    assert repo.lookup_branch("main") == expected_snapshot, "Head changed after tagging"
    asyncio.run(_delete_arrays(session.store, paths, root, consolidated))
    _assert_deletion_diff(session.status(), paths, consolidated)
    # Icechunk's compare-and-swap rejects a concurrent commit; never rebase.
    snapshot = session.commit(
        f"{MIGRATION}: retire {len(paths)} arrays from {dataset}", metadata=marker
    )
    return MigrationResult("applied", snapshot, tag, paths)


def open_repository(uri: str, *, auth_from_env: bool) -> icechunk.Repository:
    parsed = urlparse(uri)
    assert not parsed.query, "Repository URI cannot have query"
    assert not parsed.fragment, "Repository URI cannot have fragment"
    if parsed.scheme == "file":
        assert parsed.netloc in ("", "localhost"), "Expected a local file URI"
        assert parsed.path.startswith("/"), "Expected an absolute file URI"
        assert not auth_from_env, "Environment authentication is only for S3"
        return icechunk.Repository.open(
            icechunk.local_filesystem_storage(unquote(parsed.path))
        )
    assert parsed.scheme == "s3", "Expected an s3:// or file:/// repository"
    assert parsed.netloc, "Expected an S3 bucket"
    assert parsed.path.strip("/"), "Expected an explicit S3 prefix"
    return open_repo(uri, auth="env" if auth_from_env else "anonymous")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(RETIRED_ARRAYS))
    parser.add_argument("--repo", required=True, help="Explicit S3 or local file URI")
    parser.add_argument("--expected-snapshot", required=True)
    parser.add_argument("--auth-from-env", action="store_true")
    parser.add_argument(
        "--apply", action="store_true", help="Apply (default: read-only plan)"
    )
    args = parser.parse_args()
    assert (
        not args.apply or args.auth_from_env or urlparse(args.repo).scheme == "file"
    ), "S3 apply requires explicit --auth-from-env"
    repo = open_repository(args.repo, auth_from_env=args.auth_from_env)
    result = migrate(repo, args.dataset, args.expected_snapshot, apply=args.apply)
    log.info(f"{result.status}: repo={args.repo} snapshot={result.snapshot}")
    log.info(f"Historical tag: {result.historical_tag}")
    for path in result.deleted_arrays:
        log.info(f"Retired array: {path}")


if __name__ == "__main__":
    main()
