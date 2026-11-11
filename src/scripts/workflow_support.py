"""Resolve workflow targets and require deployment of the selected commit and cluster."""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from reformatters.common.logging import get_logger
from reformatters.deployments import targets

log = get_logger(__name__)


def parse_staging_branch(branch: str) -> tuple[str, str]:
    match = re.fullmatch(r"stage/([a-z0-9][a-z0-9-]*)/v([a-z0-9][a-z0-9.-]*)", branch)
    assert match, f"Expected stage/{{dataset_id}}/v{{version}}, got {branch!r}"
    return match[1], match[2]


def require_enabled(cluster: targets.Cluster) -> None:
    assert cluster.enabled, (
        f"{cluster.name} deployment is disabled; no workload was launched."
    )


def deployment_matrix() -> None:
    clusters = targets.clusters()
    assert clusters[0].enabled, "The primary cluster must be enabled"
    matrix = {
        "include": [
            {"cluster": cluster.name, "environment": cluster.environment}
            for cluster in clusters
            if cluster.enabled
        ]
    }
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"matrix={json.dumps(matrix)}\n")
        output.write(f"primary_environment={clusters[0].environment}\n")


def prepare(args: argparse.Namespace) -> None:
    outputs = {}
    if args.staging_branch is not None:
        dataset_id, version = parse_staging_branch(args.staging_branch)
        outputs.update(dataset_id=dataset_id, version=version)
        cluster = targets.cluster_for_resource(dataset_id)
    elif args.dataset_id is not None:
        cluster = targets.cluster_for_resource(args.dataset_id)
    elif args.cronjob_name is not None:
        cluster = targets.cluster_for_cronjob(args.cronjob_name)
    else:
        cluster = targets.get_cluster(args.cluster)
    if args.backfill:
        assert os.environ["GITHUB_REF"] == "refs/heads/main", (
            "Backfills must run from main; the worker image is built by main's deploy."
        )
    if args.backfill or args.require_enabled:
        require_enabled(cluster)
    if args.cronjob_name is not None and args.require_enabled:
        assert not cluster.suspended, (
            f"{cluster.name} is policy-suspended; no workload was launched."
        )
    outputs.update(cluster=cluster.name, environment=cluster.environment)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.writelines(f"{key}={value}\n" for key, value in outputs.items())


def github_pages(endpoint: str) -> list[dict[str, Any]]:
    result = subprocess.run(  # noqa: S603
        ["gh", "api", "--paginate", "--slurp", endpoint],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def workflow_runs(
    repository: str, workflow: str, sha: str, *, created_after: str | None = None
) -> list[dict[str, Any]]:
    query = {"per_page": "100", "branch": "main"}
    if created_after is None:
        query.update(head_sha=sha, event="push")
    else:
        query.update(created=f">={created_after}", event="workflow_run")
    return [
        run
        for page in github_pages(
            f"repos/{repository}/actions/workflows/{workflow}/runs?{urlencode(query)}"
        )
        for run in page["workflow_runs"]
    ]


def select_run(
    runs: list[dict[str, Any]], sha: str, repository: str, event: str
) -> dict[str, Any] | None:
    candidates = [
        run
        for run in runs
        if (
            run["display_title"] == f"Deploy Operational Updates ({sha})"
            if event == "workflow_run"
            else run["head_sha"] == sha
        )
        and run["event"] == event
        and run["head_branch"] == "main"
        and run["head_repository"]["full_name"] == repository
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda run: (run["id"], run["run_attempt"]))


def deployment_succeeded(
    run: dict[str, Any], jobs: list[dict[str, Any]], cluster: str
) -> bool:
    name = f"Deploy ({cluster})"
    selected = [job for job in jobs if job["name"] == name]
    assert len(selected) <= 1, f"Ambiguous deployment job {name!r} for this run attempt"
    if not selected:
        assert run["status"] != "completed", f"Deployment has no {name!r} job"
        return False
    job = selected[0]
    if job["status"] != "completed":
        return False
    assert job["conclusion"] == "success", (
        f"{name} concluded {job['conclusion']!r}. Fix that target's deployment "
        "and rerun the backfill; no workload was launched."
    )
    return True


def deployment_jobs(
    repository: str, run: dict[str, Any], cluster: str
) -> list[dict[str, Any]]:
    name = f"Deploy ({cluster})"
    for attempt in range(run["run_attempt"], 0, -1):
        jobs = [
            job
            for page in github_pages(
                f"repos/{repository}/actions/runs/{run['id']}/attempts/"
                f"{attempt}/jobs?per_page=100"
            )
            for job in page["jobs"]
        ]
        if any(job["name"] == name for job in jobs):
            return jobs
        if run["status"] != "completed":
            return []
    return []


def wait_for_deploy(cluster: str) -> None:
    assert os.environ["GITHUB_REF"] == "refs/heads/main", "Backfills must run from main"
    require_enabled(targets.get_cluster(cluster))
    repository, sha = os.environ["GH_REPO"], os.environ["GITHUB_SHA"]
    for _ in range(60):
        quality = select_run(
            workflow_runs(repository, "code-quality.yml", sha), sha, repository, "push"
        )
        if quality is not None and quality["status"] == "completed":
            assert quality["conclusion"] == "success", (
                f"Code Quality concluded {quality['conclusion']!r} for {sha}; it cannot deploy."
            )
            deploy = select_run(
                workflow_runs(
                    repository,
                    "deploy-operational-updates.yml",
                    sha,
                    created_after=quality["created_at"],
                ),
                sha,
                repository,
                "workflow_run",
            )
            if deploy is not None:
                jobs = deployment_jobs(repository, deploy, cluster)
                if deployment_succeeded(deploy, jobs, cluster):
                    log.info(f"Deploy ({cluster}) succeeded for {sha}")
                    return
        log.info(f"Waiting for Deploy ({cluster}) of {sha}")
        time.sleep(30)
    raise TimeoutError(f"Timed out waiting for Deploy ({cluster}) of {sha}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-id")
    source.add_argument("--cronjob-name")
    source.add_argument("--staging-branch")
    source.add_argument(
        "--cluster", choices=[cluster.name for cluster in targets.clusters()]
    )
    prep.add_argument("--backfill", action="store_true")
    prep.add_argument("--require-enabled", action="store_true")
    wait = commands.add_parser("wait-for-deploy")
    wait.add_argument(
        "cluster", choices=[cluster.name for cluster in targets.clusters()]
    )
    commands.add_parser("matrix")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "matrix":
        deployment_matrix()
    else:
        wait_for_deploy(args.cluster)


if __name__ == "__main__":
    main()
