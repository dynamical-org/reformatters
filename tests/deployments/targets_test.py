from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from reformatters.__main__ import DYNAMICAL_DATASETS, app
from reformatters.common import deploy
from reformatters.common.kubernetes import Job
from reformatters.deployments import targets


def test_ownership_is_explicit_and_includes_archivers() -> None:
    for cluster in targets.clusters():
        for resource in cluster.resources:
            assert (
                targets.cluster_for_resource(resource.dataset_id).name == cluster.name
            )
    with pytest.raises(AssertionError, match="Expected one cluster owner"):
        targets.cluster_for_resource("unknown")
    with pytest.raises(AssertionError, match="Unknown or duplicate cluster"):
        targets.get_cluster("unknown")


def test_duplicate_ownership_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    west = targets.get_cluster("aws-us-west-2")
    monkeypatch.setattr(targets, "clusters", lambda: (west, west))
    with pytest.raises(AssertionError, match="Expected one cluster owner"):
        targets.cluster_for_resource(west.resources[0].dataset_id)


@pytest.mark.parametrize("initially_suspended", [False, True])
def test_east_policy_with_real_dataset(initially_suspended: bool) -> None:
    dataset = next(
        d for d in DYNAMICAL_DATASETS if d.dataset_id.endswith("0-25-degree-fast")
    )
    east = targets.Cluster("aws-us-east-1", [dataset], "on-demand", suspended=True)
    original = dataset.operational_kubernetes_resources("image")[0]
    original.suspend = initially_suspended
    configured = east.configure_job(original)
    rendered = configured.as_kubernetes_object()
    assert rendered["spec"]["suspend"] is True
    job_spec = rendered["spec"]["jobTemplate"]["spec"]
    assert "suspend" not in job_spec
    assert (
        job_spec["template"]["spec"]["nodeSelector"]["karpenter.sh/capacity-type"]
        == "on-demand"
    )
    assert original.suspend == initially_suspended
    west = targets.Cluster("aws-us-west-2", [dataset], "spot")
    assert west.configure_job(original).suspend == initially_suspended
    job = Job(
        command=["backfill"],
        image="image",
        dataset_id=dataset.dataset_id,
        cpu="1.7",
        memory="7G",
        workers_total=1,
        parallelism=1,
    )
    assert east.configure_job(job).capacity_type == "on-demand"
    assert "suspend" not in east.configure_job(job).as_kubernetes_object()["spec"]


def test_identity_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = Mock()
    api = Mock()
    api.read_namespaced_config_map.return_value.data = {"cluster": "aws-us-west-2"}
    monkeypatch.setattr(targets.config, "load_kube_config", loader)
    monkeypatch.setattr(targets.client, "CoreV1Api", lambda: api)
    with pytest.raises(AssertionError, match="does not identify"):
        targets.verify_cluster(targets.get_cluster("aws-us-east-1"))
    loader.assert_called_once_with()
    api.read_namespaced_config_map.assert_called_once_with(
        "reformatters-cluster-identity", "default"
    )
    targets.verify_cluster(targets.get_cluster("aws-us-west-2"))


def test_empty_cluster_does_not_build_or_contact_kubernetes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify = Mock(side_effect=AssertionError("Unexpected Kubernetes access"))
    build = Mock(side_effect=AssertionError("Unexpected image build"))
    monkeypatch.setattr(targets, "verify_cluster", verify)
    monkeypatch.setattr(deploy.docker, "build_and_push_image", build)
    result = CliRunner().invoke(app, ["deploy", "--cluster", "aws-us-east-1"])
    assert result.exit_code == 0, result.exception
    verify.assert_not_called()
    build.assert_not_called()


def test_wrong_owner_fails_before_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    verify = Mock(side_effect=AssertionError("Unexpected Kubernetes access"))
    monkeypatch.setattr(targets, "verify_cluster", verify)
    result = CliRunner().invoke(
        app,
        [
            "deploy",
            "--cluster",
            "aws-us-east-1",
            "--dataset-id",
            DYNAMICAL_DATASETS[0].dataset_id,
        ],
    )
    assert isinstance(result.exception, AssertionError)
    assert "not assigned" in str(result.exception)
    verify.assert_not_called()


def test_preflight_precedes_deploy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        targets, "verify_cluster", Mock(side_effect=AssertionError("wrong cluster"))
    )
    deploy_resources = Mock()
    monkeypatch.setattr(deploy, "deploy_operational_resources", deploy_resources)
    result = CliRunner().invoke(app, ["deploy", "--cluster", "aws-us-west-2"])
    assert str(result.exception) == "wrong cluster"
    deploy_resources.assert_not_called()


def test_east_cli_deploy_uses_policy_after_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = next(
        d for d in DYNAMICAL_DATASETS if d.dataset_id.endswith("0-25-degree-fast")
    )
    east = targets.Cluster("aws-us-east-1", [dataset], "on-demand", suspended=True)
    monkeypatch.setattr(targets, "clusters", lambda: (east,))
    assert targets.cluster_for_resource(dataset.dataset_id).name == "aws-us-east-1"
    cron = dataset.operational_kubernetes_resources("image")[0]
    assert targets.cluster_for_cronjob(cron.name).name == "aws-us-east-1"
    monkeypatch.setattr(targets, "verify_cluster", Mock())
    applied = Mock()
    monkeypatch.setattr(deploy, "_apply_resources", applied)
    monkeypatch.setattr(deploy, "trigger_role_targets", lambda _: set())
    monkeypatch.setattr(deploy, "create_trigger_bindings", Mock())
    result = CliRunner().invoke(
        app, ["deploy", "--cluster", east.name, "--docker-image", "image"]
    )
    assert result.exit_code == 0, result.exception
    rendered = next(
        item for item in applied.call_args_list[0].args[0] if item["kind"] == "CronJob"
    )
    assert rendered["spec"]["suspend"] is True
    assert (
        rendered["spec"]["jobTemplate"]["spec"]["template"]["spec"]["nodeSelector"][
            "karpenter.sh/capacity-type"
        ]
        == "on-demand"
    )
    applied.reset_mock()
    deploy.deploy_operational_resources(
        east.resources,
        "image",
        cluster=east,
        cronjob_transform=lambda job: job.model_copy(update={"suspend": False}),
    )
    rendered = next(
        item for item in applied.call_args_list[0].args[0] if item["kind"] == "CronJob"
    )
    assert rendered["spec"]["suspend"] is True
