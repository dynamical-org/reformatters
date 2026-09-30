"""Worker coordination for parallel writes across Kubernetes indexed jobs.

See docs/parallel_processing.md for the overall design.
"""

import json
import re
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import icechunk
import pandas as pd
import xarray as xr
from pydantic import TypeAdapter

from reformatters.common import storage, template_utils
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob, SourceFileResult
from reformatters.common.storage import StoreFactory
from reformatters.common.zarr import copy_zarr_metadata

log = get_logger(__name__)

_RESULT_FILE_PATTERN = re.compile(r"worker-(?P<worker_index>\d+)\.json")

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
    branch_name: str
    origin: str
    template_identity: str
    append_dim: str


def pin_operational_update(
    store_factory: StoreFactory,
    *,
    is_first: bool,
    reformat_job_name: str,
    append_dim: str,
    template_identity: str,
) -> SetupInfo:
    record = store_factory.read_coordination_file(reformat_job_name, "pin/source.json")
    if is_first and record is None:
        repos = store_factory.icechunk_repos(sort="primary-first")
        snapshots = {role: repo.lookup_branch("main") for role, repo in repos}
        primary = dict(repos)["primary"]
        with xr.open_datatree(
            primary.readonly_session(snapshot_id=snapshots["primary"]).store,  # ty: ignore[invalid-argument-type]
            engine="zarr",
            chunks=None,
            decode_timedelta=True,
        ) as existing:
            origin = pd.Timestamp(existing.coords[append_dim].values[0]).isoformat()
        info = SetupInfo(
            repo_snapshots=snapshots,
            branch_name=f"_job_{reformat_job_name}",
            origin=origin,
            template_identity=template_identity,
            append_dim=append_dim,
        )
        assert all(
            info["branch_name"] not in repo.list_branches() for _, repo in repos
        ), "Job branch exists without a durable pin; use a new job identity"
        store_factory.create_coordination_file(
            reformat_job_name, "pin/source.json", json.dumps(info).encode()
        )
        record = json.dumps(info).encode()
    while record is None:
        log.info("Waiting for worker 0 to pin the published snapshot...")
        time.sleep(5)
        record = store_factory.read_coordination_file(
            reformat_job_name, "pin/source.json"
        )
    info = SetupInfo(**json.loads(record))
    assert info["template_identity"] == template_identity, (
        "Worker template identity differs from durable pin"
    )
    return info


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
    exclude_coord_value_chunks: Collection[str] = (),
    pinned_setup: SetupInfo | None = None,
) -> SetupInfo:
    if pinned_setup is not None and not is_first:
        ready = store_factory.read_coordination_file(
            reformat_job_name, "setup/ready.json"
        )
        while ready is None:
            time.sleep(5)
            ready = store_factory.read_coordination_file(
                reformat_job_name, "setup/ready.json"
            )
        assert json.loads(ready) == pinned_setup
        return pinned_setup
    if is_first:
        if pinned_setup is not None:
            ready = store_factory.read_coordination_file(
                reformat_job_name, "setup/ready.json"
            )
            if ready is not None:
                assert json.loads(ready) == pinned_setup
                return pinned_setup
        template_utils.write_metadata(template_ds, tmp_store, consolidated=consolidated)

        # On retry, reuse snapshots from the prior attempt's ready.json so
        # original_snapshot stays stable. This keeps finalize's
        # from_snapshot_id check correct if main was written externally
        # between the first attempt and the retry.
        existing_setup = store_factory.read_all_coordination_files(
            reformat_job_name, "setup"
        )
        setup_info: SetupInfo = (
            json.loads(existing_setup[0]) if existing_setup else (pinned_setup or {})
        )
        if pinned_setup is not None:
            assert setup_info == pinned_setup

        # Icechunk: create temp branch, write full metadata, commit on branch.
        # This expands the dataset on the temp branch while readers on "main" are unaffected.
        # Branch name is deterministic so worker 0 retries reuse the same branch.
        if icechunk_repos:
            _expand_job_branches(
                store_factory,
                icechunk_repos,
                setup_info,
                pinned_setup,
                branch_name,
                template_ds,
                tmp_store,
                exclude_coord_value_chunks,
            )
        # Zarr v3: do NOT expand (readers would see empty holes)

        if workers_total > 1 or pinned_setup is not None:
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


def _expand_job_branches(
    store_factory: StoreFactory,
    icechunk_repos: list[tuple[str, icechunk.Repository]],
    setup_info: SetupInfo,
    pinned_setup: SetupInfo | None,
    branch_name: str,
    template_ds: xr.DataTree,
    tmp_store: Path,
    exclude_coord_value_chunks: Collection[str],
) -> None:
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
        repo.writable_session(branch_name).store for _role, repo in icechunk_repos
    ]
    for ic_store in ic_stores:
        if pinned_setup is not None:
            with xr.open_datatree(
                ic_store,  # ty: ignore[invalid-argument-type]
                engine="zarr",
                chunks=None,
                decode_timedelta=True,
            ) as existing:
                template_utils.assert_append_labels_match(
                    template_ds, existing, pinned_setup["append_dim"]
                )
        copy_zarr_metadata(
            template_ds,
            tmp_store,
            ic_store,
            exclude_coord_value_chunks=exclude_coord_value_chunks,
        )
    storage.commit_if_icechunk(
        "Expand dataset",
        ic_stores[0],
        ic_stores[1:],
    )
    # Persist virtual chunk containers so repo stays in sync with in-code config
    store_factory.persist_virtual_config()


def wait_for_workers(
    store_factory: StoreFactory, reformat_job_name: str, workers_total: int
) -> None:
    # Poll result filenames for every worker index without reading file contents.
    # Rely on kubernetes pod_active_deadline for timeout.
    if workers_total <= 1:
        return
    while True:
        result_files = store_factory.list_coordination_files(
            reformat_job_name, "results"
        )
        reported_workers = {
            int(match["worker_index"])
            for result_file in result_files
            if (match := _RESULT_FILE_PATTERN.fullmatch(result_file)) is not None
        }
        missing_workers = sorted(set(range(workers_total)) - reported_workers)
        if not missing_workers:
            return
        log.info(
            f"Waiting for {len(missing_workers)} of {workers_total} workers to "
            f"complete; missing worker indexes: {missing_workers[:50]}"
        )
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


def recover_primary_publication(
    factory: StoreFactory, job_name: str, setup: SetupInfo
) -> bool:
    receipt = factory.read_coordination_file(job_name, "publication/primary.json")
    if receipt is not None:
        return True
    repo = dict(factory.icechunk_repos(sort="primary-first"))["primary"]
    current = repo.lookup_branch("main")
    if current == setup["repo_snapshots"]["primary"]:
        return False
    prepared = factory.read_coordination_file(
        job_name, "publication/prepared-primary.json"
    )
    if prepared is not None and json.loads(prepared)["snapshot"] == current:
        factory.write_coordination_file(job_name, "publication/primary.json", prepared)
        return True
    raise RuntimeError(
        "main moved during this job; positional writes cannot be retried"
    )


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
    publish_zarr3_metadata: bool | None = None,
    exclude_coord_value_chunks: Collection[str] = (),
) -> str | None:
    published_snapshot: str | None = None
    replicas_first: list[tuple[str, icechunk.Repository]] = []
    if publish_zarr3_metadata is None:
        publish_zarr3_metadata = update_template_with_results
    if update_template_with_results:
        assert len(all_jobs) > 0
        updated_template = all_jobs[0].update_template_with_results(merged_results)
        pinned = setup_info.get("repo_snapshots", {}).get("primary")
        if pinned is not None and "origin" in setup_info:
            repo = dict(store_factory.icechunk_repos(sort="primary-first"))["primary"]
            existing = xr.open_datatree(
                repo.readonly_session(snapshot_id=pinned).store,  # ty: ignore[invalid-argument-type]
                engine="zarr",
                chunks=None,
                decode_timedelta=True,
            )
        else:
            existing = store_factory.open_primary_datatree()
        template_utils.assert_no_append_dim_retraction(
            updated_template, existing, all_jobs[0].append_dim
        )
    else:
        updated_template = template_ds
    # Ensure tmp_store has written metadata. Virtual workers (besides worker 0)
    # do not otherwise write to tmp_store.
    template_utils.write_metadata(
        updated_template, tmp_store, consolidated=consolidated
    )

    now = pd.Timestamp.now(tz="UTC")
    commit_message = f"Update at {now.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    if branch_name != "main":
        replicas_first = store_factory.icechunk_repos(sort="primary-last")
        published_snapshot = _publish_icechunk(
            store_factory,
            replicas_first,
            reformat_job_name,
            branch_name,
            setup_info,
            template_ds,
            updated_template,
            tmp_store,
            setup_info.get("append_dim", "time"),
            commit_message,
            exclude_coord_value_chunks,
        )
    if publish_zarr3_metadata:
        primary_store = store_factory.primary_store(writable=True)
        replica_stores = store_factory.replica_stores(writable=True)
        assert published_snapshot is None or (
            dict(store_factory.icechunk_repos(sort="primary-first"))[
                "primary"
            ].lookup_branch("main")
            == published_snapshot
        ), "Primary advanced before plain-Zarr metadata publication"
        copy_zarr_metadata(
            updated_template,
            tmp_store,
            primary_store,
            replica_stores=replica_stores,
            zarr3_only=True,
            replica_write_guard=store_factory.assert_direct_replica_writes
            if store_factory.replica_handoff
            else None,
            skip_unchanged=not update_template_with_results,
            exclude_coord_value_chunks=exclude_coord_value_chunks,
        )

    if branch_name != "main":
        # Second pass: clean up temp branches.
        for _role, repo in replicas_first:
            if branch_name in repo.list_branches():
                repo.delete_branch(branch_name)

    if "origin" in setup_info:
        store_factory.write_coordination_file(
            reformat_job_name,
            "publication/complete.json",
            json.dumps({"snapshot": published_snapshot}).encode(),
        )
    elif workers_total > 1:
        store_factory.clear_coordination_files(reformat_job_name)
    return published_snapshot


def _already_published(
    factory: StoreFactory,
    job_name: str,
    role: str,
    repo: icechunk.Repository,
    branch: str,
    current: str,
    original: str | None,
    setup: SetupInfo,
) -> bool:
    if branch in repo.list_branches():
        return _snapshot_is_on_branch(repo, branch, current, original)
    if "origin" not in setup:
        return True
    record = factory.read_coordination_file(job_name, f"publication/{role}.json")
    return record is not None and json.loads(record)["snapshot"] == current


def _publish_icechunk(
    store_factory: StoreFactory,
    replicas_first: list[tuple[str, icechunk.Repository]],
    reformat_job_name: str,
    branch_name: str,
    setup_info: SetupInfo,
    template_ds: xr.DataTree,
    updated_template: xr.DataTree,
    tmp_store: Path,
    append_dim: str,
    commit_message: str,
    exclude_coord_value_chunks: Collection[str],
) -> str | None:
    published_snapshot = None
    # First pass: commit final metadata and reset main on each repo.
    diverged_roles: list[str] = []
    for role, repo in replicas_first:
        original_snapshot = setup_info.get("repo_snapshots", {}).get(role)
        current_main = repo.lookup_branch("main")
        if current_main != original_snapshot:
            if _already_published(
                store_factory,
                reformat_job_name,
                role,
                repo,
                branch_name,
                current_main,
                original_snapshot,
                setup_info,
            ):
                log.info(f"{role}: main already reset by a previous attempt")
                if role == "primary":
                    published_snapshot = current_main
                continue
            # Another job (e.g. an operational update) moved main while this job
            # was running. Icechunk can't merge a committed branch onto the moved
            # main, so this job's writes cannot be published without discarding
            # the other job's; the other job wins and this one fails.
            log.error(
                f"{role}: main moved past this job's starting snapshot; "
                f"branch {branch_name} will not be published"
            )
            diverged_roles.append(role)
            continue
        session = repo.writable_session(branch_name)
        if "origin" in setup_info:
            with xr.open_datatree(
                session.store,  # ty: ignore[invalid-argument-type]
                engine="zarr",
                chunks=None,
                decode_timedelta=True,
            ) as existing:
                template_utils.assert_append_labels_match(
                    template_ds, existing, append_dim
                )
        copy_zarr_metadata(
            updated_template,
            tmp_store,
            session.store,
            icechunk_only=True,
            exclude_coord_value_chunks=exclude_coord_value_chunks,
        )
        new_snapshot = session.commit(commit_message)
        if "origin" in setup_info:
            store_factory.write_coordination_file(
                reformat_job_name,
                f"publication/prepared-{role}.json",
                json.dumps({"snapshot": new_snapshot}).encode(),
            )
        repo.reset_branch("main", new_snapshot, from_snapshot_id=original_snapshot)
        if role == "primary":
            published_snapshot = new_snapshot
            assert repo.lookup_branch("main") == new_snapshot, (
                "Main changed immediately after publication"
            )
        if "origin" in setup_info:
            store_factory.write_coordination_file(
                reformat_job_name,
                f"publication/{role}.json",
                json.dumps({"snapshot": new_snapshot}).encode(),
            )
    if diverged_roles:
        # Leave the temp branches and coordination files in place for inspection.
        raise RuntimeError(
            f"main moved during this job on {diverged_roles}; its writes remain "
            f"unpublished on branch {branch_name}. Another job (e.g. an "
            "operational update) published concurrently and wins; re-run this "
            "job to reprocess and publish."
        )
    return published_snapshot


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
