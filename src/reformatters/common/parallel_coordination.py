"""Worker coordination for parallel writes across Kubernetes indexed jobs.

See docs/parallel_processing.md for the overall design.
"""

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import icechunk
import pandas as pd
import xarray as xr
import zarr
import zarr.buffer
import zarr.core.sync
from pydantic import TypeAdapter
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common import storage, template_utils
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob, SourceFileResult
from reformatters.common.storage import StoreFactory
from reformatters.common.zarr import copy_zarr_metadata, sync_to_store

log = get_logger(__name__)

_WORKER_RESULTS_ADAPTER: TypeAdapter[dict[str, list[SourceFileResult]]] = TypeAdapter(
    dict[str, list[SourceFileResult]]
)


def dump_worker_results_json(
    worker_results: Mapping[str, Sequence[SourceFileResult]],
) -> bytes:
    return _WORKER_RESULTS_ADAPTER.dump_json(
        {k: list(v) for k, v in worker_results.items()}
    )


class SetupInfo(TypedDict, total=False):
    repo_snapshots: dict[str, str]


def parallel_setup(
    store_factory: StoreFactory,
    *,
    is_first: bool,
    workers_total: int,
    reformat_job_name: str,
    branch_name: str,
    template_ds: xr.DataTree,
    tmp_store: Path,
    icechunk_repos: list[tuple[str, icechunk.Repository]],
    consolidated: bool,
    overwrite: bool = False,
) -> SetupInfo:
    if is_first:
        template_utils.write_metadata(template_ds, tmp_store, consolidated=consolidated)

        # On retry, reuse snapshots from the prior attempt's ready.json so
        # original_snapshot stays stable. This keeps finalize's
        # from_snapshot_id check correct if main was written externally
        # between the first attempt and the retry.
        existing_setup = store_factory.read_all_coordination_files(
            reformat_job_name, "setup"
        )
        setup_info: SetupInfo = json.loads(existing_setup[0]) if existing_setup else {}

        if overwrite:
            # Overwrite backfill: workers commit straight to main (no temp branch),
            # so write template metadata — refreshed attrs, new arrays, an explicit
            # extension — directly into the live stores now. skip_unchanged makes a
            # chunks-only backfill a metadata no-op, and store-written coordinate
            # values are preserved.
            template_utils.write_template_metadata_to_stores(
                template_ds,
                store_factory,
                tmp_store,
                commit_message="Write template metadata",
            )
            if icechunk_repos:
                store_factory.persist_virtual_config()
        elif icechunk_repos:
            # Icechunk: create temp branch, write full metadata, commit on branch.
            # This expands the dataset on the temp branch while readers on "main" are unaffected.
            # Branch name is deterministic so worker 0 retries reuse the same branch.
            repo_snapshots = setup_info.setdefault("repo_snapshots", {})
            for role, repo in icechunk_repos:
                snapshot = repo_snapshots.setdefault(role, repo.lookup_branch("main"))
                if branch_name in repo.list_branches():
                    # Branch already exists from a previous worker 0 attempt — reuse it.
                    log.info(f"Branch {branch_name} already exists on {role}, reusing")
                else:
                    repo.create_branch(branch_name, snapshot)
            # Copy metadata from local tmp_store to icechunk stores,
            # expanding dimensions by writing updated zarr.json and coordinate arrays.
            ic_stores = [
                repo.writable_session(branch_name).store
                for _role, repo in icechunk_repos
            ]
            for ic_store in ic_stores:
                copy_zarr_metadata(template_ds, tmp_store, ic_store)
            storage.commit_if_icechunk(
                "Expand dataset",
                ic_stores[0],
                ic_stores[1:],
            )
            # Persist virtual chunk containers so repo stays in sync with in-code config
            store_factory.persist_virtual_config()
        # Zarr v3 non-overwrite: do NOT expand (readers would see empty holes)

        if workers_total > 1:
            store_factory.write_coordination_file(
                reformat_job_name,
                "setup/ready.json",
                json.dumps(setup_info).encode(),
            )
        return setup_info

    # Poll until worker 0 completes setup. Rely on kubernetes pod_active_deadline for timeout.
    if workers_total > 1:
        setup_files = store_factory.read_all_coordination_files(
            reformat_job_name, "setup"
        )
        while not setup_files:
            log.info("Waiting for worker 0 to complete setup...")
            time.sleep(5)
            setup_files = store_factory.read_all_coordination_files(
                reformat_job_name, "setup"
            )
        return json.loads(setup_files[0])

    return SetupInfo()


def wait_for_workers(
    store_factory: StoreFactory, reformat_job_name: str, workers_total: int
) -> None:
    # Poll until all workers write their results file. Backfills can have
    # many thousands of workers, so count (cheap ls) rather than read.
    # Rely on kubernetes pod_active_deadline for timeout.
    if workers_total <= 1:
        return
    while (
        store_factory.count_coordination_files(reformat_job_name, "results")
        < workers_total
    ):
        log.info("Waiting for all workers to complete...")
        time.sleep(10)


def collect_results(
    store_factory: StoreFactory, reformat_job_name: str, workers_total: int
) -> Mapping[str, Sequence[SourceFileResult]]:
    wait_for_workers(store_factory, reformat_job_name, workers_total)
    result_files = store_factory.read_all_coordination_files(
        reformat_job_name, "results"
    )

    merged: dict[str, list[SourceFileResult]] = {}
    for data in result_files:
        for var_name, coords in _WORKER_RESULTS_ADAPTER.validate_json(data).items():
            merged.setdefault(var_name, []).extend(coords)
    return merged


def finalize(
    store_factory: StoreFactory,
    *,
    all_jobs: Sequence[RegionJob[Any, Any]],
    merged_results: Mapping[str, Sequence[SourceFileResult]],
    reformat_job_name: str,
    branch_name: str,
    template_ds: xr.DataTree,
    tmp_store: Path,
    setup_info: SetupInfo,
    workers_total: int,
    update_template_with_results: bool,
    consolidated: bool,
) -> None:
    if update_template_with_results:
        assert len(all_jobs) > 0
        updated_template = all_jobs[0].update_template_with_results(merged_results)
    else:
        updated_template = template_ds
    # Ensure tmp_store has written metadata. Virtual workers (besides worker 0)
    # do not otherwise write to tmp_store.
    template_utils.write_metadata(
        updated_template, tmp_store, consolidated=consolidated
    )

    now = pd.Timestamp.now(tz="UTC")
    commit_message = f"Update at {now.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    # Icechunk: write final (possibly trimmed) metadata on temp branch, commit, reset main.
    # Uses copy_zarr_metadata (not write_metadata) because the zarr arrays already exist
    # on the temp branch from parallel_setup — we only need to update the metadata files.
    # Process replicas before primary so primary (which drives future work) is last to update.
    if branch_name != "main":
        replicas_first = store_factory.icechunk_repos(sort="primary-last")
        # First pass: commit final metadata and publish each repo — a fast branch
        # reset when main hasn't moved, else replay onto the moved main.
        diverged_roles: list[str] = []
        for role, repo in replicas_first:
            original_snapshot = setup_info.get("repo_snapshots", {}).get(role)
            current_main = repo.lookup_branch("main")
            if current_main != original_snapshot:
                if branch_name not in repo.list_branches() or _snapshot_is_on_branch(
                    repo, branch_name, current_main, original_snapshot
                ):
                    log.info(f"{role}: main already reset by a previous attempt")
                    continue
                if update_template_with_results and original_snapshot is not None:
                    # Another job (an overwrite backfill) committed to main while
                    # this update ran. The committed branch can't be rebased, so
                    # replay its writes onto the moved main instead.
                    _replay_branch_onto_main(
                        repo,
                        role,
                        branch_name,
                        original_snapshot,
                        updated_template,
                        tmp_store,
                    )
                    continue
                log.error(
                    f"{role}: main moved past this job's starting snapshot; "
                    f"branch {branch_name} will not be published"
                )
                diverged_roles.append(role)
                continue
            session = repo.writable_session(branch_name)
            copy_zarr_metadata(
                updated_template, tmp_store, session.store, icechunk_only=True
            )
            new_snapshot = session.commit(
                commit_message, rebase_with=icechunk.ConflictDetector()
            )
            repo.reset_branch("main", new_snapshot, from_snapshot_id=original_snapshot)
        if diverged_roles:
            # Leave the temp branches and coordination files in place for inspection.
            raise RuntimeError(
                f"main moved during this backfill on {diverged_roles}; its writes "
                f"remain unpublished on branch {branch_name}. Re-run the backfill "
                "to reprocess and publish."
            )
        # Second pass: clean up temp branches.
        for _role, repo in replicas_first:
            if branch_name in repo.list_branches():
                repo.delete_branch(branch_name)

    # Zarr v3: copy metadata now (makes data visible to readers).
    # Only needed for updates where metadata was deferred during setup.
    # For backfills, metadata is written before workers start.
    if update_template_with_results:
        zarr3_primary = store_factory.primary_store(writable=True)
        zarr3_replicas = store_factory.replica_stores(writable=True)
        copy_zarr_metadata(
            updated_template,
            tmp_store,
            zarr3_primary,
            replica_stores=zarr3_replicas,
            zarr3_only=True,
        )

    if workers_total > 1:
        store_factory.clear_coordination_files(reformat_job_name)


def _snapshot_is_on_branch(
    repo: icechunk.Repository,
    branch_name: str,
    snapshot_id: str,
    stop_snapshot_id: str | None,
) -> bool:
    """Whether snapshot_id is among the branch's commits since stop_snapshot_id
    (the snapshot the branch was created from)."""
    for snap in repo.ancestry(branch=branch_name):
        if snap.id == snapshot_id:
            return True
        if snap.id == stop_snapshot_id:
            return False
    return False


def _replay_branch_onto_main(
    repo: icechunk.Repository,
    role: str,
    branch_name: str,
    original_snapshot: str,
    updated_template: xr.DataTree,
    tmp_store: Path,
) -> None:
    """Publish an update's temp branch onto a main that moved while the update ran.

    Copies the branch's data-variable chunk bytes (from repo.diff) and the final
    template metadata onto a fresh main session, then commits with a rebase that
    keeps this update's version of any conflicting chunk — the update wins, the
    concurrent writer (an overwrite backfill) loses only the overlapping chunks.
    Bounded by one update's writes, and idempotent: a crashed replay left nothing
    visible, and a completed one is detected by its commit message.
    """
    replay_message = f"Replay {branch_name}"
    for snap in repo.ancestry(branch="main"):
        if snap.message == replay_message:
            log.info(f"{role}: {branch_name} already replayed onto main")
            return
        if snap.id == original_snapshot:
            break

    branch_tip = repo.lookup_branch(branch_name)
    diff = repo.diff(from_snapshot_id=original_snapshot, to_snapshot_id=branch_tip)
    assert not diff.deleted_arrays, f"cannot replay deletions: {diff.deleted_arrays}"
    assert not diff.deleted_groups, f"cannot replay deletions: {diff.deleted_groups}"
    branch_store = repo.readonly_session(snapshot_id=branch_tip).store
    branch_group = zarr.open_group(store=branch_store, mode="r")
    session = repo.writable_session("main")

    # Metadata and coordinate values come from the final updated template (branch
    # coord chunks predate update_template_with_results). Coordinate values the
    # template leaves entirely null are store-written state main already has.
    copy_zarr_metadata(
        updated_template,
        tmp_store,
        session.store,
        icechunk_only=True,
        exclude_coord_value_chunks=template_utils.store_written_coords(
            updated_template
        ),
    )
    coordinate_paths = {
        f"{node.path.rstrip('/')}/{name}"
        for node in updated_template.subtree
        for name in node.to_dataset(inherit=False).coords
    }
    prototype = zarr.buffer.default_buffer_prototype()
    replayed_chunks = 0
    for path, chunk_indices in diff.updated_chunks.items():
        if path in coordinate_paths:
            continue
        array = branch_group[path.lstrip("/")]
        assert isinstance(array, zarr.Array)
        metadata = array.metadata
        assert isinstance(metadata, ArrayV3Metadata)
        encode_chunk_key = metadata.chunk_key_encoding.encode_chunk_key
        for chunk_index in chunk_indices:
            key = f"{path.lstrip('/')}/{encode_chunk_key(tuple(chunk_index))}"
            chunk_bytes = zarr.core.sync.sync(
                branch_store.get(key, prototype=prototype)
            )
            assert chunk_bytes is not None, f"missing chunk {key} on {branch_name}"
            sync_to_store(session.store, key, chunk_bytes.to_bytes())
            replayed_chunks += 1

    session.commit(
        replay_message,
        rebase_with=icechunk.BasicConflictSolver(
            on_chunk_conflict=icechunk.VersionSelection.UseOurs
        ),
    )
    log.info(
        f"{role}: main moved during this update; replayed {replayed_chunks} chunks "
        f"from {branch_name} onto main"
    )
