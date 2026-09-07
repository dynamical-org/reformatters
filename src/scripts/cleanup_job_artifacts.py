"""Remove stale Icechunk branches and worker-coordination objects.

Lists artifacts and the Kubernetes job state before making changes. This is a dry run
unless given ``--apply``.

    DYNAMICAL_ENV=prod uv run src/scripts/cleanup_job_artifacts.py <dataset-id>
    DYNAMICAL_ENV=prod uv run src/scripts/cleanup_job_artifacts.py <dataset-id> \
        --job <kubernetes-job-name> --apply
"""

import argparse
import sys
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, NoReturn

import icechunk

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import kubernetes
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.logging import get_logger
from reformatters.common.storage import StoreFactory

log = get_logger(__name__)

_JOB_BRANCH_PREFIX = "_job_"


@dataclass(frozen=True)
class BranchArtifact:
    branch: str
    job_name: str
    tip_snapshot: str
    snapshot_count: int
    snapshots_not_on_main: int
    tip_age: timedelta


def _refuse(message: str) -> NoReturn:
    sys.exit(f"Refusing to run: {message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_id")
    parser.add_argument(
        "--job",
        action="append",
        dest="jobs",
        help="Restrict cleanup to this Kubernetes job name. Repeatable.",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = _resolve(args.dataset_id)
    cleanup_job_artifacts(
        dataset.store_factory,
        job_names=args.jobs,
        apply=args.apply,
    )


def cleanup_job_artifacts(
    store_factory: StoreFactory,
    *,
    job_names: Collection[str] | None = None,
    apply: bool = False,
) -> None:
    log.info(f"Store: {store_factory.primary_url()}")
    repo = store_factory.icechunk_primary_repo()
    selected_jobs = None if job_names is None else frozenset(job_names)
    branches = _branch_artifacts(repo, selected_jobs)
    coordination = {
        job_name: count
        for job_name, count in store_factory.coordination_file_counts().items()
        if selected_jobs is None or job_name in selected_jobs
    }
    artifact_jobs = sorted({a.job_name for a in branches} | coordination.keys())

    if not artifact_jobs:
        log.info("No matching job artifacts found.")
        return

    active_jobs = {
        job_name for job_name in artifact_jobs if kubernetes.job_is_active(job_name)
    }
    _log_plan(branches, coordination, active_jobs)

    if not apply:
        log.info("Dry run. Pass --apply to delete stale artifacts.")
        return
    if active_jobs:
        _refuse(f"Kubernetes job(s) still active: {', '.join(sorted(active_jobs))}")

    for artifact in branches:
        _delete_job_branch(repo, artifact.branch)
    for job_name in coordination:
        store_factory.clear_coordination_files(job_name)

    remaining_branches = set(repo.list_branches())
    assert not {artifact.branch for artifact in branches} & remaining_branches
    remaining_coordination = store_factory.coordination_file_counts()
    assert not coordination.keys() & remaining_coordination.keys()
    log.info(
        f"Deleted {len(branches)} branch(es) and "
        f"{len(coordination)} coordination prefix(es); verified removed."
    )


def _branch_artifacts(
    repo: icechunk.Repository, selected_jobs: frozenset[str] | None
) -> list[BranchArtifact]:
    main_snapshots = {snapshot.id for snapshot in repo.ancestry(branch="main")}
    now = datetime.now(UTC)
    artifacts: list[BranchArtifact] = []
    for branch in sorted(repo.list_branches()):
        if not branch.startswith(_JOB_BRANCH_PREFIX):
            continue
        job_name = branch.removeprefix(_JOB_BRANCH_PREFIX)
        if selected_jobs is not None and job_name not in selected_jobs:
            continue
        ancestry = list(repo.ancestry(branch=branch))
        assert ancestry, f"branch {branch} has no snapshots"
        tip = ancestry[0]
        artifacts.append(
            BranchArtifact(
                branch=branch,
                job_name=job_name,
                tip_snapshot=tip.id,
                snapshot_count=len(ancestry),
                snapshots_not_on_main=sum(
                    snapshot.id not in main_snapshots for snapshot in ancestry
                ),
                tip_age=now - tip.written_at,
            )
        )
    return artifacts


def _log_plan(
    branches: Collection[BranchArtifact],
    coordination: dict[str, int],
    active_jobs: set[str],
) -> None:
    for artifact in branches:
        state = "active, keep" if artifact.job_name in active_jobs else "stale, delete"
        log.info(
            f"Branch {artifact.branch}: job={artifact.job_name} state={state} "
            f"tip={artifact.tip_snapshot} snapshots={artifact.snapshot_count} "
            f"not-on-main={artifact.snapshots_not_on_main} "
            f"tip-age={_format_age(artifact.tip_age)}"
        )
    for job_name, count in sorted(coordination.items()):
        state = "active, keep" if job_name in active_jobs else "stale, delete"
        log.info(
            f"Internal _internal/{job_name}/: job={job_name} state={state} "
            f"objects={count}"
        )


def _delete_job_branch(repo: icechunk.Repository, branch: str) -> None:
    if branch == "main" or not branch.startswith(_JOB_BRANCH_PREFIX):
        _refuse(f"branch {branch!r} is not a disposable job branch")
    repo.delete_branch(branch)


def _format_age(age: timedelta) -> str:
    seconds = max(0, int(age.total_seconds()))
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes = seconds // 60
    return f"{days}d {hours}h {minutes}m"


def _resolve(dataset_id: str) -> DynamicalDataset[Any, Any]:
    dataset = next((d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id), None)
    if dataset is None:
        known = ", ".join(d.dataset_id for d in DYNAMICAL_DATASETS)
        sys.exit(f"Unknown dataset id {dataset_id!r}. Known: {known}")
    return dataset


if __name__ == "__main__":
    main()
