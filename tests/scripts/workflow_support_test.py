import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from yaml.resolver import BaseResolver

from reformatters.deployments import targets
from scripts import workflow_support as workflows

REPOSITORY = "dynamical/reformatters"
SHA = "a" * 40
WEST = "aws-us-west-2"
EAST = "aws-us-east-1"
WORKFLOWS = Path(__file__).parents[2] / ".github/workflows"


def run_record(**overrides: object) -> dict[str, Any]:
    return {
        "id": 10,
        "run_attempt": 1,
        "head_sha": SHA,
        "display_title": f"Deploy Operational Updates ({SHA})",
        "event": "workflow_run",
        "head_branch": "main",
        "head_repository": {"full_name": REPOSITORY},
        "status": "completed",
        "conclusion": "failure",
        "created_at": "2026-11-11T00:00:00Z",
        **overrides,
    }


def job_record(cluster: str, **overrides: object) -> dict[str, Any]:
    return {
        "name": f"Deploy ({cluster})",
        "status": "completed",
        "conclusion": "success",
        **overrides,
    }


@pytest.mark.parametrize("cluster", [WEST, EAST])
def test_gate_uses_selected_job_despite_sibling_and_run_failure(cluster: str) -> None:
    sibling = EAST if cluster == WEST else WEST
    assert workflows.deployment_succeeded(
        run_record(),
        [job_record(cluster), job_record(sibling, conclusion="failure")],
        cluster,
    )


@pytest.mark.parametrize("conclusion", ["failure", "skipped", "cancelled", "timed_out"])
def test_gate_rejects_selected_job_even_if_overall_run_succeeded(
    conclusion: str,
) -> None:
    with pytest.raises(AssertionError, match=conclusion):
        workflows.deployment_succeeded(
            run_record(conclusion="success"),
            [job_record(WEST), job_record(EAST, conclusion=conclusion)],
            EAST,
        )


def test_gate_waits_for_selected_job() -> None:
    assert not workflows.deployment_succeeded(
        run_record(status="in_progress"),
        [job_record(WEST, status="in_progress", conclusion=None)],
        WEST,
    )
    assert not workflows.deployment_succeeded(run_record(status="queued"), [], WEST)


def test_gate_rejects_missing_or_ambiguous_job() -> None:
    with pytest.raises(AssertionError, match="has no"):
        workflows.deployment_succeeded(run_record(), [job_record(EAST)], WEST)
    with pytest.raises(AssertionError, match="Ambiguous"):
        workflows.deployment_succeeded(
            run_record(), [job_record(WEST), job_record(WEST)], WEST
        )


def test_trigger_sha_comes_from_exact_run_title_when_main_advanced() -> None:
    valid = run_record(head_sha="b" * 40)
    wrong_title = run_record(
        id=20, display_title=f"Deploy Operational Updates ({'b' * 40})"
    )
    assert (
        workflows.select_run([valid, wrong_title], SHA, REPOSITORY, "workflow_run")
        == valid
    )
    assert workflows.select_run([wrong_title], SHA, REPOSITORY, "workflow_run") is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"event": "workflow_dispatch"},
        {"head_branch": "feature"},
        {"head_repository": {"full_name": "fork/reformatters"}},
        {"display_title": f"Deploy Operational Updates ({SHA}) extra"},
    ],
)
def test_gate_rejects_wrong_provenance(overrides: dict[str, Any]) -> None:
    assert (
        workflows.select_run([run_record(**overrides)], SHA, REPOSITORY, "workflow_run")
        is None
    )


def test_quality_requires_exact_sha_and_latest_attempt() -> None:
    first = run_record(event="push", conclusion="success")
    latest = run_record(event="push", run_attempt=2, conclusion="failure")
    unrelated = run_record(event="push", id=20, head_sha="b" * 40)
    assert (
        workflows.select_run([first, latest, unrelated], SHA, REPOSITORY, "push")
        == latest
    )


def test_deploy_query_does_not_filter_by_default_branch_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = Mock(return_value=[{"workflow_runs": [run_record()]}])
    monkeypatch.setattr(workflows, "github_pages", api)
    assert workflows.workflow_runs(
        REPOSITORY,
        "deploy-operational-updates.yml",
        SHA,
        created_after="2026-11-11T00:00:00Z",
    ) == [run_record()]
    query = parse_qs(urlsplit(api.call_args.args[0]).query)
    assert "head_sha" not in query
    assert query["created"] == [">=2026-11-11T00:00:00Z"]
    assert query["event"] == ["workflow_run"]


def test_wait_uses_current_run_attempt_and_selected_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("GH_REPO", REPOSITORY)
    runs = Mock(
        side_effect=[
            [run_record(event="push", conclusion="success")],
            [run_record(run_attempt=3, head_sha="b" * 40)],
        ]
    )
    pages = Mock(
        return_value=[
            {"jobs": [job_record(WEST), job_record(EAST, conclusion="failure")]}
        ]
    )
    monkeypatch.setattr(workflows, "workflow_runs", runs)
    monkeypatch.setattr(workflows, "github_pages", pages)
    workflows.wait_for_deploy(WEST)
    assert "/runs/10/attempts/3/jobs?" in pages.call_args.args[0]


def test_disabled_target_fails_before_api_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setattr(
        targets,
        "get_cluster",
        Mock(return_value=targets.Cluster("third", [], "spot", enabled=False)),
    )
    api = Mock()
    monkeypatch.setattr(workflows, "workflow_runs", api)
    with pytest.raises(AssertionError, match="third deployment is disabled"):
        workflows.wait_for_deploy("third")
    api.assert_not_called()


def test_quality_failure_prevents_deploy_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("GH_REPO", REPOSITORY)
    runs = Mock(return_value=[run_record(event="push", conclusion="failure")])
    monkeypatch.setattr(workflows, "workflow_runs", runs)
    with pytest.raises(AssertionError, match="Code Quality concluded 'failure'"):
        workflows.wait_for_deploy(WEST)
    assert runs.call_count == 1


@pytest.mark.parametrize("branch", ["stage/model/v1.2", "stage/model-virtual/v2-rc.1"])
def test_staging_branch_parses(branch: str) -> None:
    dataset_id, version = workflows.parse_staging_branch(branch)
    assert branch == f"stage/{dataset_id}/v{version}"


@pytest.mark.parametrize(
    "branch",
    [
        "main",
        "stage/x/1",
        "stage/x/v1/extra",
        "stage/x/v1\ncluster=bad",
        "stage/$(id)/v1",
    ],
)
def test_staging_branch_rejects_unsafe_input(branch: str) -> None:
    with pytest.raises(AssertionError, match="Expected stage"):
        workflows.parse_staging_branch(branch)


def prepare_args(**overrides: object) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            "dataset_id": None,
            "cronjob_name": None,
            "staging_branch": None,
            "cluster": None,
            "backfill": False,
            "require_enabled": False,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "source", ["dataset_id", "cronjob_name", "staging_branch", "cluster"]
)
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("mutation", ["read", "require_enabled", "backfill"])
def test_prepare_uses_target_metadata(
    source: str,
    enabled: bool,
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    resource = Mock(dataset_id="model")
    resource.operational_kubernetes_resources.return_value = [
        argparse.Namespace(name="model")
    ]
    cluster = targets.Cluster(
        "third", [resource], "spot", environment="third-env", enabled=enabled
    )
    monkeypatch.setattr(targets, "clusters", lambda: (cluster,))
    args = prepare_args(
        backfill=mutation == "backfill", require_enabled=mutation == "require_enabled"
    )
    setattr(
        args,
        source,
        {"staging_branch": "stage/model/v1.2", "cluster": "third"}.get(source, "model"),
    )
    if not enabled and mutation != "read":
        with pytest.raises(AssertionError, match="third deployment is disabled"):
            workflows.prepare(args)
        assert not output.exists()
    else:
        workflows.prepare(args)
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        assert values["cluster"] == "third"
        assert values["environment"] == "third-env"
        if source == "staging_branch":
            assert values["dataset_id"] == "model"
            assert values["version"] == "1.2"


def test_backfill_requires_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature")
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(
        targets, "get_cluster", Mock(return_value=targets.Cluster("third", [], "spot"))
    )
    with pytest.raises(AssertionError, match="Backfills must run from main"):
        workflows.prepare(prepare_args(cluster="third", backfill=True))
    assert not output.exists()


def test_matrix_uses_ordered_enabled_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        targets,
        "clusters",
        lambda: (
            targets.Cluster("third", [], "spot", environment="build-env"),
            targets.Cluster(
                "disabled", [], "spot", environment="disabled-env", enabled=False
            ),
            targets.Cluster("fourth", [], "spot", environment="other-env"),
        ),
    )
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    workflows.deployment_matrix()
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["primary_environment"] == "build-env"
    assert json.loads(values["matrix"]) == {
        "include": [
            {"cluster": "third", "environment": "build-env"},
            {"cluster": "fourth", "environment": "other-env"},
        ]
    }


def test_matrix_rejects_disabled_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        targets,
        "clusters",
        lambda: (
            targets.Cluster("third", [], "spot", enabled=False),
            targets.Cluster("fourth", [], "spot"),
        ),
    )
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    with pytest.raises(AssertionError, match="primary cluster must be enabled"):
        workflows.deployment_matrix()
    assert not output.exists()


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    pairs = loader.construct_pairs(node)
    assert len({key for key, _ in pairs}) == len(pairs), "Duplicate YAML key"
    return dict(pairs)


UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_workflow(name: str) -> dict[str, Any]:
    return yaml.load((WORKFLOWS / name).read_text(), Loader=UniqueKeyLoader)  # noqa: S506


@pytest.mark.parametrize(
    "name",
    [
        "deploy-operational-updates.yml",
        "deploy-staging.yml",
        "manual-backfill.yml",
        "manual-create-job-from-cronjob.yml",
        "manual-get-jobs.yml",
        "manual-get-pods.yml",
    ],
)
def test_workflows_have_unique_keys_and_pinned_actions(name: str) -> None:
    workflow = load_workflow(name)
    for job in workflow["jobs"].values():
        assert job["steps"]
        for step in job["steps"]:
            if "uses" in step:
                if step["uses"].startswith("./"):
                    assert (
                        WORKFLOWS.parent.parent / step["uses"] / "action.yml"
                    ).exists()
                    continue
                _, sha = step["uses"].split("@")
                assert len(sha) == 40
                int(sha, 16)


def test_setup_composite_has_explicit_shells() -> None:
    action = load_workflow("../actions/setup-project/action.yml")
    for step in action["runs"]["steps"]:
        if "run" in step:
            assert step["shell"] == "bash"


def test_operational_deploy_preserves_trigger_and_target_isolation() -> None:
    workflow = load_workflow("deploy-operational-updates.yml")
    assert (
        workflow["run-name"]
        == "Deploy Operational Updates (${{ github.event.workflow_run.head_sha }})"
    )
    jobs = workflow["jobs"]
    assert jobs["deploy"]["name"] == "Deploy (${{ matrix.cluster }})"
    assert jobs["deploy"]["strategy"]["fail-fast"] is False
    for predicate in (
        "conclusion == 'success'",
        "event == 'push'",
        "head_repository.full_name == github.repository",
        "head_branch == 'main'",
    ):
        assert predicate in jobs["prepare"]["if"]
    for job in jobs.values():
        checkout = next(
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"]["ref"] == "${{ github.event.workflow_run.head_sha }}"


@pytest.mark.parametrize("failure", ["identity", "push", "digest", None])
def test_publication_preflight_and_digest_gate(
    failure: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps = load_workflow("deploy-operational-updates.yml")["jobs"]["deploy"]["steps"]
    start = next(
        i for i, step in enumerate(steps) if "verify-cluster" in step.get("run", "")
    )
    assert start < next(
        i for i, step in enumerate(steps) if "depot push" in step.get("run", "")
    )
    command = "set -euo pipefail\n" + "\n".join(
        step["run"] for step in steps[start:] if "run" in step
    )
    script = tmp_path / "publish.sh"
    script.write_text(command)
    for name, body in {
        "uv": 'printf "uv %s\\n" "$*" >> "$CALLS"; if [[ "$*" == *verify-cluster* && "$FAILURE" == identity ]]; then exit 1; fi',
        "depot": 'printf "depot %s\\n" "$*" >> "$CALLS"; if [[ "$FAILURE" == push ]]; then exit 1; fi',
        "aws": 'printf "%s\\n" "$DEST_DIGEST"',
    }.items():
        binary = tmp_path / name
        binary.write_text("#!/bin/bash\nset -euo pipefail\n" + body + "\n")
        binary.chmod(0o755)
    digest = "sha256:" + "a" * 64
    for key, value in {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALLS": str(tmp_path / "calls"),
        "CLUSTER": "third",
        "DEPOT_PROJECT": "shared",
        "DOCKER_REPOSITORY": "registry/repo",
        "IMAGE_SHA": SHA,
        "BUILD_ID": "saved-build",
        "IMAGE_DIGEST": digest,
        "DEST_DIGEST": "sha256:" + "b" * 64 if failure == "digest" else digest,
        "FAILURE": failure or "",
    }.items():
        monkeypatch.setenv(key, value)
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(script)], capture_output=True, text=True, check=False
    )
    assert (result.returncode == 0) == (failure is None), result.stderr
    calls = (tmp_path / "calls").read_text()
    assert ("depot push" in calls) == (failure != "identity")
    assert ("uv run main deploy --cluster third" in calls) == (failure is None)
    if failure is None:
        assert (
            f"depot push --project shared -t registry/repo:{SHA} saved-build" in calls
        )
        assert f"--docker-image registry/repo:{SHA}@{digest}" in calls


@pytest.mark.parametrize(
    "name",
    [
        "deploy-operational-updates.yml",
        "deploy-staging.yml",
        "manual-backfill.yml",
        "manual-create-job-from-cronjob.yml",
        "manual-get-jobs.yml",
        "manual-get-pods.yml",
    ],
)
def test_resolution_has_no_environment_credentials(name: str) -> None:
    prep = load_workflow(name)["jobs"]["prepare"]
    assert "environment" not in prep
    assert "secrets." not in json.dumps(prep)
    assert prep.get("permissions", {"contents": "read"}) == {"contents": "read"}
    assert not any(
        step.get("uses", "").startswith("aws-actions/") for step in prep["steps"]
    )


def test_backfill_gate_precedes_submission() -> None:
    steps = load_workflow("manual-backfill.yml")["jobs"]["backfill"]["steps"]
    gate = next(
        i for i, step in enumerate(steps) if "wait-for-deploy" in step.get("run", "")
    )
    submission = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Start backfill (SEE LOGS)"
    )
    assert gate < submission


@pytest.mark.parametrize("suspended", [True, False])
def test_manual_cron_checks_cluster_policy(
    suspended: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = targets.Cluster("third", [], "spot", suspended=suspended)
    monkeypatch.setattr(targets, "cluster_for_cronjob", Mock(return_value=cluster))
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    args = prepare_args(cronjob_name="example-update", require_enabled=True)
    if suspended:
        with pytest.raises(AssertionError, match="third is policy-suspended"):
            workflows.prepare(args)
        assert not output.exists()
    else:
        workflows.prepare(args)
        assert output.exists()


def test_partial_rerun_retains_omitted_target_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = Mock(
        side_effect=[
            {"jobs": [job_record(EAST)]},
            {"jobs": [job_record(WEST), job_record(EAST, conclusion="failure")]},
        ]
    )
    monkeypatch.setattr(workflows, "github_pages", lambda endpoint: [pages(endpoint)])
    run = run_record(run_attempt=2)
    jobs = workflows.deployment_jobs(REPOSITORY, run, WEST)
    assert workflows.deployment_succeeded(run, jobs, WEST)
    assert "/attempts/2/jobs" in pages.call_args_list[0].args[0]
    assert "/attempts/1/jobs" in pages.call_args_list[1].args[0]


def test_rerun_target_failure_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = Mock(return_value=[{"jobs": [job_record(WEST, conclusion="failure")]}])
    monkeypatch.setattr(workflows, "github_pages", pages)
    run = run_record(run_attempt=2)
    jobs = workflows.deployment_jobs(REPOSITORY, run, WEST)
    with pytest.raises(AssertionError, match="failure"):
        workflows.deployment_succeeded(run, jobs, WEST)
    assert pages.call_count == 1


def test_in_progress_rerun_waits_for_target_instead_of_using_previous_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = Mock(return_value=[{"jobs": []}])
    monkeypatch.setattr(workflows, "github_pages", pages)
    run = run_record(run_attempt=2, status="in_progress")
    assert workflows.deployment_jobs(REPOSITORY, run, WEST) == []
    assert pages.call_count == 1
