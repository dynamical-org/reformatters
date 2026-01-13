#!/usr/bin/env python3
"""
Zarr V3 to Icechunk Migration Manager

A single-file script for migrating Zarr V3 datasets to Icechunk with support for:
- Metadata initialization (init mode)
- Distributed data migration (migrate mode)
- Kubernetes Job generation (generate-k8s mode)

Usage:
    python zarr_to_icechunk.py --mode init --source <uri> --dest <uri>
    python zarr_to_icechunk.py --mode migrate --source <uri> --dest <uri> --variable <path>
    python zarr_to_icechunk.py --mode generate-k8s --source <uri> --dest <uri> --variable <path>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import icechunk
import obstore
import xarray as xr
import zarr.buffer
from icechunk import BasicConflictSolver, VersionSelection
from zarr.storage import ObjectStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 5
MAX_RETRY_ATTEMPTS = 6
CHUNK_SIZE_LIMIT_MB = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Zarr V3 datasets to Icechunk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["init", "migrate", "generate-k8s"],
        required=True,
        help="Operation mode",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Source Zarr V3 URI (s3:// or file://)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        help="Destination Icechunk URI (s3:// or file://)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        help="Variable path to migrate (for migrate/generate-k8s modes)",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Limit number of shards to process (for testing)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent chunk operations (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--source-region",
        type=str,
        default="us-west-2",
        help="AWS region for source S3 bucket",
    )
    parser.add_argument(
        "--dest-region",
        type=str,
        default="us-west-2",
        help="AWS region for destination S3 bucket",
    )
    return parser.parse_args()


def retry_with_backoff[T](
    func: Callable[[], T],
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base_delay: float = 1.0,
) -> T:
    """Execute a function with exponential backoff retry."""
    last_exception: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return func()
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001
            last_exception = e
            if attempt < max_attempts - 1:
                delay = base_delay * (2**attempt) * (0.5 + random.random())  # noqa: S311
                log.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
    raise last_exception if last_exception else AssertionError("unreachable")


async def retry_with_backoff_async[T](
    coro_func: Callable[[], Coroutine[Any, Any, T]],
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base_delay: float = 1.0,
) -> T:
    """Execute an async function with exponential backoff retry."""
    last_exception: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await coro_func()
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001
            last_exception = e
            if attempt < max_attempts - 1:
                delay = base_delay * (2**attempt) * (0.5 + random.random())  # noqa: S311
                log.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
    raise last_exception if last_exception else AssertionError("unreachable")


def create_source_store(uri: str, region: str) -> obstore.store.ObjectStore:
    """Create an obstore ObjectStore for the source Zarr."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        store = obstore.store.S3Store.from_url(
            f"s3://{bucket}",
            region=region,
            skip_signature=True,
            client_options={
                "connect_timeout": "10 seconds",
                "timeout": "120 seconds",
            },
        )
        return store
    elif parsed.scheme == "file" or not parsed.scheme:
        return obstore.store.LocalStore.from_url(uri)
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")


def get_source_path(uri: str) -> str:
    """Extract the path from the source URI."""
    parsed = urlparse(uri)
    return parsed.path.lstrip("/")


def create_icechunk_storage(uri: str, region: str) -> icechunk.Storage:
    """Create Icechunk storage for the destination."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return icechunk.s3_storage(
            bucket=bucket,
            prefix=prefix,
            region=region,
        )
    elif parsed.scheme == "file" or not parsed.scheme:
        path = parsed.path if parsed.path else uri
        return icechunk.local_filesystem_storage(path)
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")


def read_source_file(store: obstore.store.ObjectStore, path: str) -> bytes:
    """Read a file from the source store."""
    return retry_with_backoff(lambda: obstore.get(store, path).bytes().to_bytes())


async def read_source_file_async(store: obstore.store.ObjectStore, path: str) -> bytes:
    """Read a file from the source store asynchronously."""

    async def _get() -> bytes:
        result = await obstore.get_async(store, path)
        data = await result.bytes_async()
        return data.to_bytes()

    return await retry_with_backoff_async(_get)


def list_source_keys(store: obstore.store.ObjectStore, prefix: str) -> list[str]:
    """List all keys under a prefix in the source store."""
    keys: list[str] = []
    for batch in obstore.list(store, prefix=prefix):
        keys.extend(obj["path"] for obj in batch)
    return keys


async def list_source_keys_async(
    store: obstore.store.ObjectStore, prefix: str
) -> list[str]:
    """List all keys under a prefix in the source store asynchronously."""
    keys: list[str] = []
    async for batch in obstore.list(store, prefix=prefix):
        keys.extend(obj["path"] for obj in batch)
    return keys


def mode_init(args: argparse.Namespace) -> None:  # noqa: PLR0915
    """Initialize Icechunk repository and copy metadata."""
    assert args.source is not None
    assert args.dest is not None

    log.info(f"Initializing Icechunk repository at {args.dest}")
    log.info(f"Source Zarr: {args.source}")

    source_store = create_source_store(args.source, args.source_region)
    source_path = get_source_path(args.source)

    # Open source zarr with xarray to identify coordinates
    log.info("Opening source zarr with xarray to identify coordinates")
    parsed = urlparse(args.source)
    if parsed.scheme == "s3":
        # Create store from full URI (including path) for xarray
        zarr_store = ObjectStore(
            obstore.store.S3Store.from_url(
                args.source,
                region=args.source_region,
                skip_signature=True,
                client_options={
                    "connect_timeout": "10 seconds",
                    "timeout": "120 seconds",
                },
            )
        )
    elif parsed.scheme == "file" or not parsed.scheme:
        zarr_store = args.source
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")

    source_ds = xr.open_zarr(zarr_store, chunks=None)
    coord_names = set(source_ds.coords.keys())
    log.info(f"Identified coordinates: {sorted(coord_names)}")
    source_ds.close()

    dest_storage = create_icechunk_storage(args.dest, args.dest_region)

    repo = icechunk.Repository.open_or_create(dest_storage)
    session = repo.writable_session("main")
    icechunk_store = session.store

    root_zarr_path = f"{source_path}/zarr.json" if source_path else "zarr.json"
    log.info(f"Reading root zarr.json from {root_zarr_path}")
    root_metadata = read_source_file(source_store, root_zarr_path)
    log.info(f"Root metadata size: {len(root_metadata)} bytes")

    root_metadata_dict = json.loads(root_metadata)

    buffer = zarr.buffer.default_buffer_prototype().buffer.from_bytes(root_metadata)
    asyncio.run(icechunk_store.set("zarr.json", buffer))
    log.info("Wrote zarr.json to Icechunk")

    consolidated = root_metadata_dict.get("consolidated_metadata", {})
    metadata_entries = consolidated.get("metadata", {})

    for var_name, var_metadata in metadata_entries.items():
        var_zarr_json = json.dumps(var_metadata).encode()
        key = f"{var_name}/zarr.json"
        var_buffer = zarr.buffer.default_buffer_prototype().buffer.from_bytes(
            var_zarr_json
        )
        asyncio.run(icechunk_store.set(key, var_buffer))
        log.info(f"Wrote {key} to Icechunk")

        # Copy coordinate data if this is a coordinate
        if var_name in coord_names and var_metadata.get("node_type") == "array":
            coord_chunks_prefix = (
                f"{source_path}/{var_name}/c/" if source_path else f"{var_name}/c/"
            )
            try:
                chunk_keys = list_source_keys(source_store, coord_chunks_prefix)
                if not chunk_keys:
                    log.warning(f"No chunk files found for coordinate {var_name}")
                if len(chunk_keys) > 1:
                    raise RuntimeError(
                        f"Expected exactly one chunk for coords, found {len(chunk_keys)} for {var_name}"
                    )
                else:
                    for chunk_key in chunk_keys:
                        # Remove source_path prefix to get relative path
                        dest_key = (
                            chunk_key.removeprefix(source_path + "/")
                            if source_path
                            else chunk_key
                        )
                        coord_data = read_source_file(source_store, chunk_key)
                        coord_buffer = (
                            zarr.buffer.default_buffer_prototype().buffer.from_bytes(
                                coord_data
                            )
                        )
                        asyncio.run(icechunk_store.set(dest_key, coord_buffer))
                    log.info(
                        f"Copied {len(chunk_keys)} chunk(s) for coordinate {var_name}"
                    )
            except FileNotFoundError:
                log.warning(f"Coordinate data not found for {var_name}")
            except Exception as e:  # noqa: BLE001
                log.warning(f"Could not copy coordinate data for {var_name}: {e}")

    conflict_solver = BasicConflictSolver(on_chunk_conflict=VersionSelection.Fail)
    snapshot_id = session.commit(
        "Initial metadata import from Zarr V3", rebase_with=conflict_solver
    )
    log.info(f"Initial commit successful. Snapshot ID: {snapshot_id}")


async def migrate_variable_async(
    source_store: obstore.store.ObjectStore,
    source_path: str,
    icechunk_store: icechunk.IcechunkStore,
    variable: str,
    limit_shards: int | None,
    concurrency: int,
) -> int:
    """Migrate chunks for a specific variable asynchronously."""
    semaphore = asyncio.Semaphore(concurrency)
    var_prefix = f"{source_path}/{variable}/c/" if source_path else f"{variable}/c/"

    log.info(f"Listing chunks under {var_prefix}")
    all_keys = await list_source_keys_async(source_store, var_prefix)
    log.info(f"Found {len(all_keys)} chunk keys")

    if not all_keys:
        raise RuntimeError(f"No chunks found for variable {variable}")

    # For sharded zarr, each computed key is a shard
    if limit_shards is not None and limit_shards < len(all_keys):
        all_keys = all_keys[:limit_shards]
        log.info(f"Limited to {limit_shards} shards")

    chunks_copied = 0
    failed_chunks: list[str] = []

    async def copy_chunk(key: str) -> bool:
        nonlocal chunks_copied
        async with semaphore:
            try:
                data = await read_source_file_async(source_store, key)
                dest_key = key.removeprefix(source_path + "/" if source_path else "")
                chunk_buffer = zarr.buffer.default_buffer_prototype().buffer.from_bytes(
                    data
                )
                await icechunk_store.set(dest_key, chunk_buffer)
                chunks_copied += 1
                if chunks_copied % 10 == 0:
                    log.info(f"Copied {chunks_copied}/{len(all_keys)} chunks")
                return True
            except Exception as e:  # noqa: BLE001
                log.error(f"Failed to copy chunk {key}: {e}")
                failed_chunks.append(key)
                return False

    tasks = [copy_chunk(key) for key in all_keys]
    await asyncio.gather(*tasks)

    if failed_chunks:
        log.error(f"Failed to copy {len(failed_chunks)} chunks")
        for chunk in failed_chunks[:10]:
            log.error(f"  - {chunk}")
        if len(failed_chunks) > 10:
            log.error(f"  ... and {len(failed_chunks) - 10} more")

    return chunks_copied


def mode_migrate(args: argparse.Namespace) -> None:
    """Migrate data chunks for a specific variable."""
    assert args.source is not None
    assert args.dest is not None
    assert args.variable is not None

    log.info(f"Migrating variable {args.variable}")
    log.info(f"Source: {args.source}")
    log.info(f"Destination: {args.dest}")

    source_store = create_source_store(args.source, args.source_region)
    source_path = get_source_path(args.source)

    dest_storage = create_icechunk_storage(args.dest, args.dest_region)

    repo = icechunk.Repository.open(dest_storage)
    session = repo.writable_session("main")
    icechunk_store = session.store

    chunks_copied = asyncio.run(
        migrate_variable_async(
            source_store=source_store,
            source_path=source_path,
            icechunk_store=icechunk_store,
            variable=args.variable,
            limit_shards=args.limit_shards,
            concurrency=args.concurrency,
        )
    )

    log.info(f"Copied {chunks_copied} chunks for variable {args.variable}")

    if chunks_copied > 0:
        conflict_solver = BasicConflictSolver(on_chunk_conflict=VersionSelection.Fail)
        snapshot_id: str = retry_with_backoff(
            lambda: session.commit(
                f"Migrated variable {args.variable}",
                rebase_with=conflict_solver,
            )
        )
        log.info(f"Commit successful. Snapshot ID: {snapshot_id}")
    else:
        log.warning("No chunks copied, skipping commit")


def mode_generate_k8s(args: argparse.Namespace) -> None:
    """Generate Kubernetes Job JSON for migration."""
    assert args.source is not None
    assert args.dest is not None
    assert args.variable is not None

    script_path = Path(__file__)
    script_content = script_path.read_text()

    script_escaped = script_content.replace("'", "'\"'\"'")

    migrate_cmd = (
        f"python3 /tmp/migrate.py "
        f"--mode migrate "
        f"--source '{args.source}' "
        f"--dest '{args.dest}' "
        f"--variable '{args.variable}' "
        f"--source-region '{args.source_region}' "
        f"--dest-region '{args.dest_region}' "
        f"--concurrency {args.concurrency}"
    )

    if args.limit_shards is not None:
        migrate_cmd += f" --limit-shards {args.limit_shards}"

    bootstrap_script = f"""#!/bin/bash
set -e
pip install obstore icechunk

cat > /tmp/migrate.py << 'SCRIPT_EOF'
{script_escaped}
SCRIPT_EOF

{migrate_cmd}
"""

    job_name = f"zarr-icechunk-migrate-{args.variable.replace('/', '-').replace('_', '-')[:40]}"

    job_spec: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name},
        "spec": {
            "backoffLimit": 3,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "migrator",
                            "image": "python:3.13-slim",
                            "command": ["/bin/bash", "-c", bootstrap_script],
                            "resources": {
                                "requests": {
                                    "cpu": "8",
                                    "memory": "32Gi",
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "32Gi",
                                },
                            },
                            "env": [
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                            ],
                        }
                    ],
                    "restartPolicy": "Never",
                    "nodeSelector": {
                        "eks.amazonaws.com/compute-type": "auto",
                    },
                }
            },
            "ttlSecondsAfterFinished": 86400,
        },
    }

    print(json.dumps(job_spec, indent=2))  # noqa: T201


def main() -> None:
    args = parse_args()

    if args.mode == "init":
        if not args.source or not args.dest:
            log.error("--source and --dest are required for init mode")
            sys.exit(1)
        mode_init(args)
    elif args.mode == "migrate":
        if not args.source or not args.dest or not args.variable:
            log.error("--source, --dest, and --variable are required for migrate mode")
            sys.exit(1)
        mode_migrate(args)
    elif args.mode == "generate-k8s":
        if not args.source or not args.dest or not args.variable:
            log.error(
                "--source, --dest, and --variable are required for generate-k8s mode"
            )
            sys.exit(1)
        mode_generate_k8s(args)
    else:
        log.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
