import argparse
import hashlib
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


@pytest.mark.parametrize("enabled", ["", "false", "True", "1"])
def test_disabled_east_fails_before_api_access(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("AWS_US_EAST_1_DEPLOY_ENABLED", enabled)
    api = Mock()
    monkeypatch.setattr(workflows, "workflow_runs", api)
    with pytest.raises(AssertionError, match="AWS_US_EAST_1_DEPLOY_ENABLED=true"):
        workflows.wait_for_deploy(EAST)
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


@pytest.mark.parametrize("cluster", [WEST, EAST])
def test_prep_resolves_environment_without_credentials(
    cluster: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    resolve = Mock(return_value=cluster)
    monkeypatch.setattr(workflows, "resolve_target", resolve)
    workflows.prepare(
        argparse.Namespace(
            staging_branch="stage/model/v1.2",
            dataset_id=None,
            cronjob_name=None,
            backfill=False,
            require_enabled=False,
        )
    )
    resolve.assert_called_once_with("--dataset-id", "model")
    assert output.read_text() == (
        f"dataset_id=model\nversion=1.2\ncluster={cluster}\nenvironment={workflows.CLUSTER_ENVIRONMENTS[cluster]}\n"
    )


def test_resolver_passes_input_as_one_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    command = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=f"{WEST}\n"))
    monkeypatch.setattr(workflows.subprocess, "run", command)
    value = "bad; touch /tmp/should-not-exist"
    assert workflows.resolve_target("--cronjob-name", value) == WEST
    assert command.call_args.args[0] == [
        "uv",
        "run",
        "main",
        "resolve-cluster",
        "--cronjob-name",
        value,
    ]
    assert command.call_args.kwargs["check"]
    assert not command.call_args.kwargs.get("shell")


def test_resolver_fails_closed_on_unknown_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflows.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, stdout="unknown\n")),
    )
    with pytest.raises(AssertionError, match="Unexpected resolved cluster"):
        workflows.resolve_target("--dataset-id", "model")


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
                _, sha = step["uses"].split("@")
                assert len(sha) == 40
                int(sha, 16)


def test_operational_jobs_are_independent_and_east_skips_when_disabled() -> None:
    workflow = load_workflow("deploy-operational-updates.yml")
    assert (
        workflow["run-name"]
        == "Deploy Operational Updates (${{ github.event.workflow_run.head_sha }})"
    )
    jobs = workflow["jobs"]
    for cluster, job_id, environment in [
        (WEST, "deploy-west", "prod"),
        (EAST, "deploy-east", "prod-us-east-1"),
    ]:
        job = jobs[job_id]
        assert job["name"] == f"Deploy ({cluster})"
        assert job["needs"] == "build"
        assert job["environment"] == environment
        assert job["env"]["CLUSTER"] == cluster
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert (
                    step["with"]["ref"] == "${{ github.event.workflow_run.head_sha }}"
                )
        assert '--cluster "$CLUSTER"' in job["steps"][-1]["run"]
    assert jobs["deploy-east"]["if"] == "vars.AWS_US_EAST_1_DEPLOY_ENABLED == 'true'"
    assert "if" not in jobs["deploy-west"]
    build = jobs["build"]
    assert all(
        predicate in build["if"]
        for predicate in [
            "conclusion == 'success'",
            "event == 'push'",
            "head_repository.full_name == github.repository",
            "head_branch == 'main'",
        ]
    )
    steps = build["steps"]
    verify = next(
        i for i, step in enumerate(steps) if "verify-cluster" in step.get("run", "")
    )
    builds = [
        i
        for i, step in enumerate(steps)
        if step.get("uses", "").startswith("depot/build-push-action@")
    ]
    assert len(builds) == 1
    assert verify < builds[0]
    assert steps[builds[0]]["with"]["platforms"] == "linux/amd64,linux/arm64"
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["continue-on-error"] is True
    assert upload["with"]["name"] == "shared-image"
    assert upload["with"]["overwrite"] is True
    east_steps = jobs["deploy-east"]["steps"]
    download = next(
        step
        for step in east_steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    )
    assert download["with"]["name"] == upload["with"]["name"]
    assert not any(step.get("uses", "").startswith("depot/") for step in east_steps)
    east_verify = next(
        i
        for i, step in enumerate(east_steps)
        if "verify-cluster" in step.get("run", "")
    )
    east_copy = next(
        i
        for i, step in enumerate(east_steps)
        if "copy_workflow_image.sh" in step.get("run", "")
    )
    assert east_verify < east_copy


@pytest.mark.parametrize(
    ("name", "job_id"),
    [
        ("manual-backfill.yml", "backfill"),
        ("manual-create-job-from-cronjob.yml", "create-job"),
        ("deploy-staging.yml", "deploy"),
    ],
)
def test_dataset_workflows_resolve_before_environment_credentials(
    name: str, job_id: str
) -> None:
    jobs = load_workflow(name)["jobs"]
    prep = jobs["prepare"]
    assert "environment" not in prep
    assert prep["permissions"] == {"contents": "read"}
    assert "secrets." not in json.dumps(prep)
    assert not any(
        step.get("uses", "").startswith("aws-actions/") for step in prep["steps"]
    )
    job = jobs[job_id]
    assert "prepare" in job["needs"]
    assert job["environment"] == "${{ needs.prepare.outputs.environment }}"
    if job_id == "backfill":
        gate = next(
            i
            for i, step in enumerate(job["steps"])
            if "wait-for-deploy" in step.get("run", "")
        )
        credentials = next(
            i
            for i, step in enumerate(job["steps"])
            if step.get("uses", "").startswith("aws-actions/configure-aws-credentials@")
        )
        assert gate < credentials
    if job_id == "deploy":
        assert "AWS_US_EAST_1_DEPLOY_ENABLED == 'true'" in job["if"]
        verify = next(
            i
            for i, step in enumerate(job["steps"])
            if "verify-cluster" in step.get("run", "")
        )
        build = next(
            i
            for i, step in enumerate(job["steps"])
            if step.get("uses", "").startswith("depot/build-push-action@")
        )
        assert verify < build


@pytest.mark.parametrize(
    ("name", "job_id"),
    [
        ("manual-create-job-from-cronjob.yml", "create-job"),
        ("manual-get-jobs.yml", "get-jobs"),
        ("manual-get-pods.yml", "get-pods"),
    ],
)
def test_raw_kubectl_has_preflight_and_explicit_namespace(
    name: str, job_id: str
) -> None:
    job = load_workflow(name)["jobs"][job_id]
    verified = False
    for step in job["steps"]:
        command = step.get("run", "")
        if "verify-cluster" in command:
            verified = True
        if "kubectl " in command:
            assert verified
            assert "kubectl --namespace default " in command
    assert verified


@pytest.mark.parametrize("name", ["manual-get-jobs.yml", "manual-get-pods.yml"])
def test_generic_reads_expose_clusters_and_environment_selection(name: str) -> None:
    workflow = load_workflow(name)
    assert workflow["on"]["workflow_dispatch"]["inputs"]["cluster"]["options"] == [
        WEST,
        EAST,
    ]
    job = next(iter(workflow["jobs"].values()))
    assert (
        job["environment"]
        == "${{ inputs.cluster == 'aws-us-east-1' && 'prod-us-east-1' || 'prod' }}"
    )


@pytest.mark.parametrize("wrong_digest", [False, True])
def test_copy_preserves_digest_and_cleans_secure_auth_files(
    wrong_digest: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = '{"manifests":["amd64","arm64"]}'
    digest = "sha256:" + hashlib.sha256(manifest.encode()).hexdigest()
    image_dir = tmp_path / "shared-image"
    image_dir.mkdir()
    source = "111111111111.dkr.ecr.us-west-2.amazonaws.com/reformatters/main"
    target = "222222222222.dkr.ecr.us-east-1.amazonaws.com/reformatters/main"
    (image_dir / "repository").write_text(source)
    (image_dir / "digest").write_text(digest)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "aws").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\nprintf fake-password\n'
    )
    (binaries / "skopeo").write_text("""#!/bin/bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CALLS"
case "$1" in
  login)
    cat >/dev/null
    [[ "$(stat -c %a "$(dirname "$3")")" == 700 ]]
    printf '{}' > "$3"
    [[ "$(stat -c %a "$3")" == 600 ]]
    ;;
  copy)
    while [[ "$1" != --digestfile ]]; do shift; done
    printf '%s' "$COPIED_DIGEST" > "$2"
    ;;
  inspect) printf '%s' "$MANIFEST" ;;
esac
""")
    for binary in binaries.iterdir():
        binary.chmod(0o755)
    calls = tmp_path / "calls"
    monkeypatch.setenv("PATH", f"{binaries}:{os.environ['PATH']}")
    for key, value in {
        "RUNNER_TEMP": str(tmp_path),
        "IMAGE_SHA": SHA,
        "IMAGE_DIGEST": digest,
        "DOCKER_REPOSITORY": target,
        "CALLS": str(calls),
        "MANIFEST": manifest,
        "COPIED_DIGEST": "sha256:" + "0" * 64 if wrong_digest else digest,
    }.items():
        monkeypatch.setenv(key, value)
    result = subprocess.run(
        ["/bin/bash", "src/scripts/copy_workflow_image.sh"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode != 0) == wrong_digest, result.stderr
    commands = calls.read_text()
    assert "copy --all --preserve-digests --src-authfile" in commands
    assert f"docker://{source}@{digest} docker://{target}:{SHA}" in commands
    assert "get-login-password --region us-west-2" in commands
    assert "get-login-password --region us-east-1" in commands
    assert "fake-password" not in commands
    assert not list(tmp_path.glob("ecr-auth.*"))


def test_manual_cron_requires_enabled_before_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("AWS_US_EAST_1_DEPLOY_ENABLED", "false")
    monkeypatch.setattr(workflows, "resolve_target", Mock(return_value=EAST))
    with pytest.raises(AssertionError, match="deployment is disabled"):
        workflows.prepare(
            argparse.Namespace(
                staging_branch=None,
                dataset_id=None,
                cronjob_name="example-update",
                backfill=False,
                require_enabled=True,
            )
        )
    assert not output.exists()


def test_wait_times_out_without_a_matching_deploy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("GH_REPO", REPOSITORY)
    monkeypatch.setattr(workflows, "workflow_runs", Mock(return_value=[]))
    sleep = Mock()
    monkeypatch.setattr(workflows.time, "sleep", sleep)
    with pytest.raises(TimeoutError, match=f"Deploy \\({WEST}\\) of {SHA}"):
        workflows.wait_for_deploy(WEST)
    assert sleep.call_count == 60
    sleep.assert_called_with(30)


@pytest.mark.parametrize("suspended", ["true", "false", ""])
def test_manual_cron_rejects_suspended_before_create(
    suspended: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = load_workflow("manual-create-job-from-cronjob.yml")["jobs"]["create-job"]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Submit job (SEE LOGS)"
    )
    assert "--require-enabled" in job_prepare_command(
        "manual-create-job-from-cronjob.yml"
    )
    script = tmp_path / "submit.sh"
    script.write_text(command.replace("${{ github.repository }}", REPOSITORY))
    binary = tmp_path / "kubectl"
    binary.write_text("""#!/bin/bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CALLS"
if [[ "$*" == *'get cronjob'* ]]; then printf '%s' "$SUSPENDED"; fi
""")
    binary.chmod(0o755)
    calls = tmp_path / "calls"
    for key, value in {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALLS": str(calls),
        "SUSPENDED": suspended,
        "CRONJOB_NAME": "example-update",
        "USERNAME": "operator",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }.items():
        monkeypatch.setenv(key, value)
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(script)], capture_output=True, text=True, check=False
    )
    assert (result.returncode != 0) == (suspended == "true"), result.stderr
    commands = calls.read_text().splitlines()
    assert (
        commands[0]
        == "--namespace default get cronjob example-update -o jsonpath={.spec.suspend}"
    )
    assert any("create job" in line for line in commands) == (suspended != "true")
    if suspended == "true":
        assert "is suspended" in result.stdout


def job_prepare_command(name: str) -> str:
    return next(
        step["run"]
        for step in load_workflow(name)["jobs"]["prepare"]["steps"]
        if "workflow_support.py prepare" in step.get("run", "")
    )


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
