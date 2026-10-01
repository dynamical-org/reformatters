"""Exercises the generated trigger admission policies against a real kube-apiserver.

Needs the envtest kube-apiserver and etcd binaries. Set
REFORMATTERS_ADMISSION_TEST_ASSETS to a directory containing `kube-apiserver`
and `etcd` (anywhere beneath it), e.g. the extracted
https://github.com/kubernetes-sigs/controller-tools/releases/download/envtest-v1.36.0/envtest-v1.36.0-linux-amd64.tar.gz
Also needs `openssl`. Without the assets every test is skipped. Run with `-n0`.
"""

import copy
import json
import os
import shutil
import socket
import ssl
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from kubernetes import client

from reformatters.__main__ import DYNAMICAL_DATASETS, OPERATIONAL_ARCHIVERS
from reformatters.common import kubernetes as reformatters_kubernetes
from reformatters.common import kubernetes_security
from reformatters.common.kubernetes import (
    SERVICE_ACCOUNT,
    ReformatCronJob,
    ValidationCronJob,
)
from reformatters.common.kubernetes_security import (
    _JOB_FIELDS,
    CANARY_ANNOTATION,
    trigger_admission_binding,
    trigger_admission_resources,
)

ASSETS_ENV_VAR = "REFORMATTERS_ADMISSION_TEST_ASSETS"
NAMESPACE = "dynamical-triggers"
PSA_NAMESPACE = "psa-restricted"
ADMIN_TOKEN = "admin-token"  # noqa: S105
TRIGGER_TOKEN = "trigger-token"  # noqa: S105
DEPLOY_TOKEN = "deploy-token"  # noqa: S105
EMPTY_TOKEN = "empty-token"  # noqa: S105
EMPTY_NAMESPACE = "empty-triggers"
POLICY_PREFIX = f"{NAMESPACE}-{SERVICE_ACCOUNT}"
NAME_LABEL = "dynamical.org/cronjob-name"
UID_LABEL = "dynamical.org/cronjob-uid"

UPDATE = "test-dataset-update"
VALIDATE = "test-dataset-validate"
MINIMAL = "minimal-template"
WORK_QUEUE = "work-queue-template"
MULTI = "multi-container-template"
ABSENT = "absent-template"  # approved by the policy but no CronJob exists

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get(ASSETS_ENV_VAR), reason=f"{ASSETS_ENV_VAR} not set"
    ),
]


class ApiServer:
    def __init__(self, port: int, ca_cert: Path, kubectl: Path) -> None:
        self.port = port
        self.kubectl = kubectl
        self.http = httpx.Client(
            base_url=f"https://127.0.0.1:{port}",
            verify=ssl.create_default_context(cafile=str(ca_cert)),
            timeout=30,
        )
        self.ca_cert = ca_cert

    def request(
        self, method: str, path: str, token: str, body: dict[str, Any] | None = None
    ) -> httpx.Response:
        return self.http.request(
            method,
            path,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )

    def admin(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> httpx.Response:
        return self.request(method, path, ADMIN_TOKEN, body)

    def create(self, path: str, body: dict[str, Any]) -> httpx.Response:
        response = self.admin("POST", path, body)
        assert response.status_code == 201, response.text
        return response


def _find_binary(assets: Path, name: str) -> Path:
    matches = [path for path in assets.rglob(name) if path.is_file()]
    assert matches, f"{name} not found under {assets}"
    return matches[0]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _openssl(*args: str) -> None:
    subprocess.run(["openssl", *args], check=True, capture_output=True)  # noqa: S603, S607


@pytest.fixture(scope="module")
def apiserver(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ApiServer]:
    assets = Path(os.environ[ASSETS_ENV_VAR])
    if shutil.which("openssl") is None:
        pytest.skip("openssl not available")
    work = tmp_path_factory.mktemp("apiserver")
    _openssl("genrsa", "-out", str(work / "sa.key"), "2048")
    _openssl(
        "rsa", "-in", str(work / "sa.key"), "-pubout", "-out", str(work / "sa.pub")
    )
    _openssl(
        *("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1"),
        *("-subj", "/CN=localhost", "-addext", "subjectAltName=IP:127.0.0.1"),
        *("-keyout", str(work / "tls.key"), "-out", str(work / "tls.crt")),
    )
    (work / "tokens.csv").write_text(
        f'{ADMIN_TOKEN},admin,admin-uid,"system:masters"\n'
        f'{DEPLOY_TOKEN},binding-deployer,deploy-uid,"system:authenticated"\n'
        f"{TRIGGER_TOKEN},system:serviceaccount:{NAMESPACE}:{SERVICE_ACCOUNT},trigger-uid,"
        f'"system:serviceaccounts,system:serviceaccounts:{NAMESPACE}"\n'
        f"{EMPTY_TOKEN},system:serviceaccount:{EMPTY_NAMESPACE}:{SERVICE_ACCOUNT},empty-uid,"
        f'"system:serviceaccounts,system:serviceaccounts:{EMPTY_NAMESPACE}"\n'
    )
    etcd_port, etcd_peer_port, api_port = _free_port(), _free_port(), _free_port()
    etcd_url = f"http://127.0.0.1:{etcd_port}"
    server: ApiServer | None = None
    processes: list[subprocess.Popen[bytes]] = []
    logs = []
    try:
        for command, log_name in (
            (
                [
                    str(_find_binary(assets, "etcd")),
                    f"--data-dir={work / 'etcd'}",
                    f"--listen-client-urls={etcd_url}",
                    f"--advertise-client-urls={etcd_url}",
                    f"--listen-peer-urls=http://127.0.0.1:{etcd_peer_port}",
                ],
                "etcd.log",
            ),
            (
                [
                    str(_find_binary(assets, "kube-apiserver")),
                    f"--etcd-servers={etcd_url}",
                    f"--secure-port={api_port}",
                    "--bind-address=127.0.0.1",
                    f"--cert-dir={work / 'certs'}",
                    f"--tls-cert-file={work / 'tls.crt'}",
                    f"--tls-private-key-file={work / 'tls.key'}",
                    f"--token-auth-file={work / 'tokens.csv'}",
                    "--authorization-mode=RBAC",
                    "--service-account-issuer=https://kubernetes.default.svc",
                    f"--service-account-key-file={work / 'sa.pub'}",
                    f"--service-account-signing-key-file={work / 'sa.key'}",
                    "--service-cluster-ip-range=10.96.0.0/16",
                    "--allow-privileged=true",
                    "--enable-admission-plugins=ValidatingAdmissionPolicy,PodSecurity",
                ],
                "apiserver.log",
            ),
        ):
            log_file = (work / log_name).open("wb")
            logs.append(log_file)
            processes.append(
                subprocess.Popen(command, stdout=log_file, stderr=log_file)  # noqa: S603
            )

        server = ApiServer(api_port, work / "tls.crt", _find_binary(assets, "kubectl"))
        deadline = time.monotonic() + 90
        while True:
            assert all(p.poll() is None for p in processes), (
                work / "apiserver.log"
            ).read_text()[-2000:]
            try:
                if server.admin("GET", "/readyz").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, "kube-apiserver did not become ready"
            time.sleep(0.5)
        yield server
    finally:
        for process in reversed(processes):  # kube-apiserver needs etcd to shut down
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log_file in logs:
            log_file.close()
        if server is not None:
            server.http.close()


def _metadata(name: str, namespace: str | None = None) -> dict[str, Any]:
    return {"name": name, **({"namespace": namespace} if namespace else {})}


def _plain_cronjob(name: str, job_spec_extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {"name": name},
        "spec": {
            "schedule": "0 * * * *",
            "jobTemplate": {
                "spec": {
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [{"name": "c", "image": "busybox"}],
                        }
                    },
                    **job_spec_extra,
                }
            },
        },
    }


def _multi_container_cronjob(name: str) -> dict[str, Any]:
    cronjob = _plain_cronjob(name, {})
    pod = cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    pod["containers"] = [
        {"name": "a", "image": "busybox", "env": [{"name": "X", "value": "1"}]},
        {"name": "b", "image": "busybox"},
    ]
    return cronjob


def _resource_path(resource: dict[str, Any]) -> str:
    return {
        "ValidatingAdmissionPolicy": "/apis/admissionregistration.k8s.io/v1/validatingadmissionpolicies",
        "ValidatingAdmissionPolicyBinding": "/apis/admissionregistration.k8s.io/v1/validatingadmissionpolicybindings",
    }[resource["kind"]]


def clone_job(cronjob: dict[str, Any], job_name: str) -> dict[str, Any]:
    """The Job body `create_job_from_cronjob` submits for a stored CronJob."""
    template = cronjob["spec"]["jobTemplate"]
    metadata = template.get("metadata", {})
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "labels": {
                **metadata.get("labels", {}),
                NAME_LABEL: cronjob["metadata"]["name"],
                UID_LABEL: cronjob["metadata"]["uid"],
            },
            "annotations": {
                **metadata.get("annotations", {}),
                "cronjob.kubernetes.io/instantiate": "manual",
            },
        },
        "spec": copy.deepcopy(template["spec"]),
    }


class Cluster:
    def __init__(self, server: ApiServer) -> None:
        self.server = server
        self.cronjobs: dict[str, dict[str, Any]] = {}
        self._job_count = 0

    def new_job_name(self) -> str:
        self._job_count += 1
        return f"job-t{self._job_count}"

    def trigger_create(
        self,
        job: dict[str, Any],
        token: str = TRIGGER_TOKEN,
        dry_run: bool = False,
    ) -> httpx.Response:
        return self.server.request(
            "POST",
            f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs{'?dryRun=All' if dry_run else ''}",
            token,
            job,
        )

    def job_for(self, cronjob_name: str) -> dict[str, Any]:
        return clone_job(self.cronjobs[cronjob_name], self.new_job_name())

    def assert_allowed(
        self,
        job: dict[str, Any],
        token: str = TRIGGER_TOKEN,
        dry_run: bool = False,
    ) -> None:
        response = self.trigger_create(job, token, dry_run)
        assert response.status_code == 201, response.text

    def assert_denied(
        self,
        job: dict[str, Any],
        policy: str = "",
        token: str = TRIGGER_TOKEN,
        dry_run: bool = False,
    ) -> None:
        response = self.trigger_create(job, token, dry_run)
        assert response.status_code in (403, 422), response.text
        assert (
            f"ValidatingAdmissionPolicy '{POLICY_PREFIX}-{policy}" in response.text
        ), f"denied by something other than the trigger policies: {response.text}"


@pytest.fixture(scope="module")
def cluster(apiserver: ApiServer) -> Cluster:
    for namespace, labels in (
        (NAMESPACE, {}),
        (PSA_NAMESPACE, {"pod-security.kubernetes.io/enforce": "restricted"}),
    ):
        apiserver.create(
            "/api/v1/namespaces",
            {"metadata": {**_metadata(namespace), "labels": labels}},
        )
        apiserver.create(
            f"/api/v1/namespaces/{namespace}/serviceaccounts",
            {"metadata": _metadata("default")},
        )
    apiserver.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/roles",
        {
            "metadata": _metadata("trigger"),
            "rules": [
                {
                    "apiGroups": ["batch"],
                    "resources": ["cronjobs"],
                    "resourceNames": [
                        UPDATE,
                        VALIDATE,
                        MINIMAL,
                        WORK_QUEUE,
                        MULTI,
                        ABSENT,
                    ],
                    "verbs": ["get", "trigger"],
                },
                {
                    "apiGroups": ["batch"],
                    "resources": ["jobs"],
                    "verbs": ["create", "get"],
                },
            ],
        },
    )
    apiserver.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/rolebindings",
        {
            "metadata": _metadata("trigger"),
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "trigger",
            },
            "subjects": [
                {
                    "kind": "User",
                    "name": f"system:serviceaccount:{NAMESPACE}:{SERVICE_ACCOUNT}",
                }
            ],
        },
    )

    cluster = Cluster(apiserver)
    for body in (
        ReformatCronJob(
            name=UPDATE,
            image="busybox:1",
            dataset_id="test-dataset",
            cpu="1",
            memory="1G",
            schedule="0 * * * *",
            shared_memory="1G",
            secret_names=["source-credentials"],
            service_account_name="default",
        ),
        ValidationCronJob(
            name=VALIDATE,
            image="busybox:2",
            dataset_id="test-dataset",
            cpu="1",
            memory="1G",
            schedule="30 * * * *",
        ),
        _plain_cronjob(MINIMAL, {}),
        _plain_cronjob(WORK_QUEUE, {"parallelism": 3}),
        _multi_container_cronjob(MULTI),
    ):
        stored = apiserver.create(
            f"/apis/batch/v1/namespaces/{NAMESPACE}/cronjobs",
            body if isinstance(body, dict) else body.as_kubernetes_object(),
        ).json()
        cluster.cronjobs[stored["metadata"]["name"]] = stored

    names = [UPDATE, VALIDATE, MINIMAL, WORK_QUEUE, MULTI, ABSENT]
    for resource in [
        *trigger_admission_resources(NAMESPACE),
        *(trigger_admission_binding(NAMESPACE, name) for name in names),
    ]:
        apiserver.create(_resource_path(resource), resource)

    # Policies become active asynchronously; wait until a forged Job is denied
    forged = cluster.job_for(UPDATE)
    forged["metadata"]["labels"][UID_LABEL] = "forged"
    deadline = time.monotonic() + 60
    while True:
        response = cluster.trigger_create(
            {
                **forged,
                "metadata": {**forged["metadata"], "name": cluster.new_job_name()},
            }
        )
        if response.status_code != 201:
            break
        assert time.monotonic() < deadline, "admission policies never became active"
        time.sleep(0.5)
    return cluster


@pytest.mark.parametrize("cronjob_name", [UPDATE, VALIDATE, MINIMAL, WORK_QUEUE, MULTI])
def test_exact_copy_of_each_cronjob_template_is_admitted(
    cluster: Cluster, cronjob_name: str
) -> None:
    cluster.assert_allowed(cluster.job_for(cronjob_name))


def test_create_job_from_cronjob_is_admitted(
    cluster: Cluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    def load_test_config() -> None:
        configuration = client.Configuration()
        configuration.host = f"https://127.0.0.1:{cluster.server.port}"
        configuration.ssl_ca_cert = str(cluster.server.ca_cert)
        configuration.api_key = {"authorization": f"Bearer {TRIGGER_TOKEN}"}
        client.Configuration.set_default(configuration)

    monkeypatch.setattr(
        reformatters_kubernetes.config, "load_incluster_config", load_test_config
    )
    monkeypatch.setenv("POD_NAMESPACE", NAMESPACE)
    # create_job_from_cronjob reads the CronJob, which the trigger role can't; grant it
    cluster.server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/roles",
        {
            "metadata": _metadata("read-cronjobs"),
            "rules": [
                {"apiGroups": ["batch"], "resources": ["cronjobs"], "verbs": ["get"]},
                {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["get"]},
            ],
        },
    )
    cluster.server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/rolebindings",
        {
            "metadata": _metadata("read-cronjobs"),
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "read-cronjobs",
            },
            "subjects": [
                {
                    "kind": "User",
                    "name": f"system:serviceaccount:{NAMESPACE}:{SERVICE_ACCOUNT}",
                }
            ],
        },
    )
    try:
        assert reformatters_kubernetes.create_job_from_cronjob(UPDATE, "from-code")
    finally:
        client.Configuration.set_default(client.Configuration())


def _mutate(
    function: Callable[[dict[str, Any]], None],
) -> Callable[[dict[str, Any]], None]:
    return function


def _pod(job: dict[str, Any]) -> dict[str, Any]:
    return job["spec"]["template"]["spec"]


def _container(job: dict[str, Any]) -> dict[str, Any]:
    return _pod(job)["containers"][0]


def _set(target: Callable[[dict[str, Any]], dict[str, Any]], key: str, value: object):  # noqa: ANN202
    def apply(job: dict[str, Any]) -> None:
        target(job)[key] = value

    return apply


def _spec(job: dict[str, Any]) -> dict[str, Any]:
    return job["spec"]


def _labels(job: dict[str, Any]) -> dict[str, Any]:
    return job["metadata"]["labels"]


def _annotations(job: dict[str, Any]) -> dict[str, Any]:
    return job["metadata"]["annotations"]


def _template_metadata(job: dict[str, Any]) -> dict[str, Any]:
    return job["spec"]["template"].setdefault("metadata", {})


def _drop(target: Callable[[dict[str, Any]], dict[str, Any]], key: str):  # noqa: ANN202
    return lambda job: target(job).pop(key)


def _reverse_env(job: dict[str, Any]) -> None:
    _container(job)["env"].reverse()


def _append_env(job: dict[str, Any]) -> None:
    _container(job)["env"].append({"name": "EXTRA", "value": "1"})


def _swap_secret_env(job: dict[str, Any]) -> None:
    env = next(
        e
        for e in _container(job)["env"]
        if "valueFrom" in e and "secretKeyRef" in e["valueFrom"]
    )
    env["valueFrom"]["secretKeyRef"]["name"] = "other-secret"


def _swap_secret_volume(job: dict[str, Any]) -> None:
    volume = next(v for v in _pod(job)["volumes"] if "secret" in v)
    volume["secret"]["secretName"] = "other-secret"


def _extra_volume_mount(job: dict[str, Any]) -> None:
    _container(job)["volumeMounts"].append(
        {"name": "ephemeral-vol", "mountPath": "/extra"}
    )


def _reverse_volume_mounts(job: dict[str, Any]) -> None:
    _container(job)["volumeMounts"].reverse()


def _extra_container(job: dict[str, Any]) -> None:
    _pod(job)["containers"].append({"name": "sidecar", "image": "busybox"})


def _reverse_containers(job: dict[str, Any]) -> None:
    _pod(job)["containers"].reverse()


def _init_container(job: dict[str, Any]) -> None:
    _pod(job)["initContainers"] = [{"name": "init", "image": "busybox"}]


def _privileged(job: dict[str, Any]) -> None:
    _container(job)["securityContext"] = {"privileged": True}


def _resources(job: dict[str, Any]) -> None:
    _container(job)["resources"]["requests"]["cpu"] = "64"


def _forged_owner(job: dict[str, Any]) -> None:
    job["metadata"]["ownerReferences"] = [
        {"apiVersion": "batch/v1", "kind": "CronJob", "name": UPDATE, "uid": "forged"}
    ]


def _generate_name(job: dict[str, Any]) -> None:
    job["metadata"]["generateName"] = "gen-"


def _manual_selector(job: dict[str, Any]) -> None:
    job["spec"]["manualSelector"] = True
    job["spec"]["selector"] = {"matchLabels": {"x": "y"}}
    _template_metadata(job)["labels"] = {"x": "y"}


def _pod_failure_policy(job: dict[str, Any]) -> None:
    job["spec"]["podFailurePolicy"] = {
        "rules": [
            {
                "action": "Ignore",
                "onPodConditions": [{"type": "Other", "status": "True"}],
            }
        ]
    }


NEGATIVE_MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    # forged or missing identity labels
    "missing-name-label": _drop(_labels, NAME_LABEL),
    "missing-uid-label": _drop(_labels, UID_LABEL),
    "missing-all-labels": lambda job: job["metadata"].pop("labels"),
    "forged-uid-label": _set(_labels, UID_LABEL, "forged-uid"),
    "unapproved-name-label": _set(_labels, NAME_LABEL, "not-approved"),
    "other-approved-name-label": _set(_labels, NAME_LABEL, VALIDATE),
    "extra-label": _set(_labels, "extra", "1"),
    "missing-instantiate-annotation": _drop(
        _annotations, "cronjob.kubernetes.io/instantiate"
    ),
    "extra-annotation": _set(_annotations, "extra", "1"),
    "owner-reference": _forged_owner,
    "finalizer": lambda job: job["metadata"].__setitem__("finalizers", ["a/b"]),
    "generate-name": _generate_name,
    # pod template
    "image": _set(_container, "image", "attacker/image"),
    "command": _set(_container, "command", ["sh", "-c", "id"]),
    "args": _set(_container, "args", ["--extra"]),
    "env-value": lambda job: _container(job)["env"][0].__setitem__("value", "changed"),
    "env-extra": _append_env,
    "env-reordered": _reverse_env,
    "env-secret-ref": _swap_secret_env,
    "secret-volume": _swap_secret_volume,
    "service-account": _set(_pod, "serviceAccountName", "other"),
    "automount-token": _set(_pod, "automountServiceAccountToken", True),
    "extra-container": _extra_container,
    "init-container": _init_container,
    "host-network": _set(_pod, "hostNetwork", True),
    "host-pid": _set(_pod, "hostPID", True),
    "privileged": _privileged,
    "resources": _resources,
    "extra-volume-mount": _extra_volume_mount,
    "volume-mounts-reordered": _reverse_volume_mounts,
    "node-selector": _set(_pod, "nodeName", "some-node"),
    "priority-class": _set(_pod, "priorityClassName", "system-node-critical"),
    "host-aliases": _set(_pod, "hostAliases", [{"ip": "10.0.0.1", "hostnames": ["x"]}]),
    "share-process-namespace": _set(_pod, "shareProcessNamespace", True),
    "tolerations": _set(_pod, "tolerations", [{"operator": "Exists"}]),
    "runtime-class": _set(_pod, "runtimeClassName", "other"),
    "template-annotation": lambda job: (
        _template_metadata(job).setdefault("annotations", {}).__setitem__("extra", "1")
    ),
    "template-label": lambda job: (
        _template_metadata(job).setdefault("labels", {}).__setitem__("extra", "1")
    ),
    # job spec
    "ttl": _set(_spec, "ttlSecondsAfterFinished", 1),
    "active-deadline": _set(_spec, "activeDeadlineSeconds", 1),
    "backoff-limit-per-index": _set(_spec, "backoffLimitPerIndex", 100),
    "parallelism": _set(_spec, "parallelism", 50),
    "suspend": _set(_spec, "suspend", True),
    "managed-by": _set(_spec, "managedBy", "example.com/other"),
    "pod-failure-policy": _pod_failure_policy,
    "manual-selector": _manual_selector,
}


@pytest.mark.parametrize("mutation", NEGATIVE_MUTATIONS)
def test_modified_production_job_is_denied(cluster: Cluster, mutation: str) -> None:
    job = cluster.job_for(UPDATE)
    NEGATIVE_MUTATIONS[mutation](job)
    cluster.assert_denied(job)


def test_reordered_containers_are_denied(cluster: Cluster) -> None:
    cluster.assert_allowed(cluster.job_for(MULTI))
    job = cluster.job_for(MULTI)
    _reverse_containers(job)
    cluster.assert_denied(job)


def test_job_without_a_cronjob_template_is_denied(cluster: Cluster) -> None:
    job = cluster.job_for(MINIMAL)
    job["metadata"]["labels"] = {"unrelated": "1"}
    cluster.assert_denied(job)


def test_job_for_approved_name_with_missing_cronjob_is_denied(cluster: Cluster) -> None:
    job = cluster.job_for(MINIMAL)
    job["metadata"]["labels"][NAME_LABEL] = ABSENT
    cluster.assert_denied(job)


def test_work_queue_job_omitting_completions_is_admitted(cluster: Cluster) -> None:
    job = cluster.job_for(WORK_QUEUE)
    assert "completions" not in job["spec"]
    assert job["spec"]["parallelism"] == 3
    cluster.assert_allowed(job)


def test_work_queue_job_cannot_add_completions(cluster: Cluster) -> None:
    job = cluster.job_for(WORK_QUEUE)
    job["spec"]["completions"] = 3
    cluster.assert_denied(job)


def test_minimal_job_cannot_change_a_defaulted_field(cluster: Cluster) -> None:
    for field, value in (("backoffLimit", 0), ("completions", 2), ("suspend", True)):
        job = cluster.job_for(MINIMAL)
        job["spec"][field] = value
        cluster.assert_denied(job)


def test_new_job_spec_fields_need_policy_review(cluster: Cluster) -> None:
    """A JobSpec field added by a newer Kubernetes must be reviewed before the policy is trusted."""
    schema = cluster.server.admin("GET", "/openapi/v3/apis/batch/v1").json()
    api_fields = set(
        schema["components"]["schemas"]["io.k8s.api.batch.v1.JobSpec"]["properties"]
    )
    assert api_fields == {*_JOB_FIELDS, "selector", "template"}, (
        "Kubernetes JobSpec changed; review new fields for execution or privilege controls"
    )


def test_unknown_job_spec_key_is_pruned_by_the_api_server(cluster: Cluster) -> None:
    """The API server drops unknown keys before admission, so they cannot smuggle in behavior."""
    job = cluster.job_for(UPDATE)
    job["spec"]["futureField"] = {"a": 1}
    response = cluster.trigger_create(job)
    assert response.status_code == 201
    assert "futureField" not in response.json()["spec"]


def test_restricted_namespace_rejects_privileged_pod_and_accepts_compliant_pod(
    cluster: Cluster,
) -> None:
    pods_path = f"/api/v1/namespaces/{PSA_NAMESPACE}/pods"
    privileged = {
        "metadata": _metadata("privileged"),
        "spec": {
            "containers": [
                {
                    "name": "c",
                    "image": "busybox",
                    "securityContext": {"privileged": True},
                }
            ]
        },
    }
    response = cluster.server.admin("POST", pods_path, privileged)
    assert response.status_code == 403
    assert "violates PodSecurity" in response.text

    compliant = {
        "metadata": _metadata("compliant"),
        "spec": {
            "securityContext": {
                "runAsNonRoot": True,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "c",
                    "image": "busybox",
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                }
            ],
        },
    }
    cluster.server.create(pods_path, compliant)


def _canary(job: dict[str, Any]) -> dict[str, Any]:
    job["metadata"]["annotations"][CANARY_ANNOTATION] = "true"
    return job


def test_all_registered_pod_templates_pass_restricted_admission(
    cluster: Cluster,
) -> None:
    cluster.server.create(
        f"/api/v1/namespaces/{PSA_NAMESPACE}/serviceaccounts",
        {"metadata": {"name": SERVICE_ACCOUNT}},
    )
    for resource in [*DYNAMICAL_DATASETS, *OPERATIONAL_ARCHIVERS]:
        try:
            cronjobs = resource.operational_kubernetes_resources("test-image")
        except NotImplementedError:
            continue
        for cronjob in cronjobs:
            template = cronjob.as_kubernetes_object()["spec"]["jobTemplate"]["spec"][
                "template"
            ]
            pod = {"apiVersion": "v1", "kind": "Pod", **template}
            pod["metadata"] = {**template.get("metadata", {}), "name": cronjob.name}
            response = cluster.server.admin(
                "POST", f"/api/v1/namespaces/{PSA_NAMESPACE}/pods?dryRun=All", pod
            )
            assert response.status_code == 201, (cronjob.name, response.text)


def test_dry_run_canary_from_non_trigger_identity_is_checked(cluster: Cluster) -> None:

    unlabeled = _canary(cluster.job_for(UPDATE))
    unlabeled["metadata"]["labels"] = {}
    cluster.assert_denied(unlabeled, "targets", token=ADMIN_TOKEN, dry_run=True)

    altered = _canary(cluster.job_for(UPDATE))
    _container(altered)["image"] = "attacker/image"
    cluster.assert_denied(altered, "clone", token=ADMIN_TOKEN, dry_run=True)

    cluster.assert_allowed(
        _canary(cluster.job_for(UPDATE)), token=ADMIN_TOKEN, dry_run=True
    )


def test_canary_annotation_is_not_ignored_outside_dry_run(cluster: Cluster) -> None:
    cluster.assert_denied(_canary(cluster.job_for(UPDATE)), "clone")


def test_non_trigger_identity_without_canary_annotation_is_not_checked(
    cluster: Cluster,
) -> None:
    altered = cluster.job_for(UPDATE)
    _container(altered)["image"] = "attacker/image"
    cluster.assert_allowed(altered, token=ADMIN_TOKEN, dry_run=True)


@pytest.fixture
def kubectl_against_apiserver(
    cluster: Cluster, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_run = subprocess.run
    empty_kubeconfig = tmp_path / "kubeconfig"
    empty_kubeconfig.write_text("apiVersion: v1\nkind: Config\n")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        assert command[0] == "/usr/bin/kubectl"
        server = cluster.server
        return real_run(
            [
                str(server.kubectl),
                f"--kubeconfig={empty_kubeconfig}",
                f"--server=https://127.0.0.1:{server.port}",
                f"--certificate-authority={server.ca_cert}",
                f"--token={ADMIN_TOKEN}",
                *command[1:],
            ],
            **kwargs,
        )

    monkeypatch.setattr(kubernetes_security.subprocess, "run", run)


def test_verify_trigger_admission_accepts_deployed_templates(
    cluster: Cluster, kubectl_against_apiserver: None
) -> None:
    assert kubernetes_security.verify_trigger_admission(
        NAMESPACE, [UPDATE, VALIDATE, MINIMAL]
    ) == {UPDATE, VALIDATE, MINIMAL}


def test_verify_trigger_admission_reports_missing_cronjob_by_clone_policy(
    cluster: Cluster, kubectl_against_apiserver: None
) -> None:
    assert kubernetes_security.verify_trigger_admission(NAMESPACE, [ABSENT]) == set()


def test_missing_parameter_denial_names_the_clone_policy(cluster: Cluster) -> None:
    job = _canary(cluster.job_for(MINIMAL))
    job["metadata"]["labels"][NAME_LABEL] = ABSENT
    job["metadata"]["labels"][UID_LABEL] = "missing"
    cluster.assert_denied(job, "clone", token=ADMIN_TOKEN, dry_run=True)


def test_non_trigger_non_dry_run_jobs_are_outside_policy(cluster: Cluster) -> None:
    job = cluster.job_for(UPDATE)
    _container(job)["image"] = "different-image"
    cluster.assert_allowed(job, token=ADMIN_TOKEN)


def test_trigger_cannot_take_a_scheduled_job_name(cluster: Cluster) -> None:
    job = cluster.job_for(UPDATE)
    job["metadata"]["name"] = f"{UPDATE}-29843978"
    cluster.assert_denied(job, "clone")


def test_operator_verifies_installed_policy_identity(
    cluster: Cluster,
    kubectl_against_apiserver: None,
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner  # noqa: PLC0415

    from reformatters.__main__ import app  # noqa: PLC0415

    resources = trigger_admission_resources(NAMESPACE)
    bundle = tmp_path / "admission.json"
    bundle.write_text(json.dumps({"items": resources}))
    runner = CliRunner()
    result = runner.invoke(
        app, ["verify-admission", str(bundle), "--namespace", NAMESPACE]
    )
    assert result.exit_code == 0, result.exception
    policy = resources[0]
    path = f"{_resource_path(policy)}/{policy['metadata']['name']}"
    live = cluster.server.admin("GET", path).json()
    original = copy.deepcopy(live)
    live["spec"]["matchConditions"][0]["expression"] = "true"
    assert cluster.server.admin("PUT", path, live).status_code == 200
    unapproved = cluster.job_for(UPDATE)
    unapproved["metadata"]["labels"] = {}

    def wait_for_guard(*, denied: bool) -> None:
        deadline = time.monotonic() + 30
        while True:
            response = cluster.trigger_create(unapproved, ADMIN_TOKEN, dry_run=True)
            if denied and response.status_code == 422:
                assert f"{POLICY_PREFIX}-targets" in response.text, response.text
                return
            if not denied and response.status_code == 201:
                return
            assert time.monotonic() < deadline, response.text
            time.sleep(0.1)

    try:
        wait_for_guard(denied=True)
        result = runner.invoke(
            app, ["verify-admission", str(bundle), "--namespace", NAMESPACE]
        )
        assert result.exit_code != 0
        assert "differs from bundle" in str(result.exception)
    finally:
        original["metadata"]["resourceVersion"] = cluster.server.admin(
            "GET", path
        ).json()["metadata"]["resourceVersion"]
        assert cluster.server.admin("PUT", path, original).status_code == 200
        wait_for_guard(denied=False)


def test_empty_target_bundle_denies_all_trigger_jobs_and_verifies(
    cluster: Cluster,
    kubectl_against_apiserver: None,
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner  # noqa: PLC0415

    from reformatters.__main__ import app  # noqa: PLC0415

    server = cluster.server
    server.create("/api/v1/namespaces", {"metadata": _metadata(EMPTY_NAMESPACE)})
    server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{EMPTY_NAMESPACE}/roles",
        {
            "metadata": _metadata("create-jobs"),
            "rules": [
                {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]}
            ],
        },
    )
    server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{EMPTY_NAMESPACE}/rolebindings",
        {
            "metadata": _metadata("create-jobs"),
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "create-jobs",
            },
            "subjects": [
                {
                    "kind": "User",
                    "name": f"system:serviceaccount:{EMPTY_NAMESPACE}:{SERVICE_ACCOUNT}",
                }
            ],
        },
    )
    resources = trigger_admission_resources(EMPTY_NAMESPACE)
    for resource in resources:
        server.create(_resource_path(resource), resource)

    cronjob = server.create(
        f"/apis/batch/v1/namespaces/{EMPTY_NAMESPACE}/cronjobs",
        _plain_cronjob("unapproved", {}),
    ).json()
    job = clone_job(cronjob, "empty-target-canary")
    path = f"/apis/batch/v1/namespaces/{EMPTY_NAMESPACE}/jobs?dryRun=All"
    deadline = time.monotonic() + 30
    while True:
        response = server.request("POST", path, EMPTY_TOKEN, job)
        if f"{EMPTY_NAMESPACE}-{SERVICE_ACCOUNT}-targets" in response.text:
            assert response.status_code in (403, 422), response.text
            break
        assert response.status_code == 201, response.text
        assert time.monotonic() < deadline, "empty-target guard never became active"
        time.sleep(0.1)

    kubernetes_security.verify_trigger_admission(EMPTY_NAMESPACE, [])
    bundle = tmp_path / "empty-admission.json"
    bundle.write_text(json.dumps({"items": resources}))
    result = CliRunner().invoke(
        app, ["verify-admission", str(bundle), "--namespace", EMPTY_NAMESPACE]
    )
    assert result.exit_code == 0, result.exception


def test_two_namespace_bundles_do_not_conflict_and_identity_cannot_cross(
    cluster: Cluster,
) -> None:
    other = "other-triggers"
    cluster.server.create("/api/v1/namespaces", {"metadata": {"name": other}})
    for resource in [
        *trigger_admission_resources(other),
        trigger_admission_binding(other, MINIMAL),
    ]:
        cluster.server.create(_resource_path(resource), resource)
    cronjob = cluster.server.create(
        f"/apis/batch/v1/namespaces/{other}/cronjobs", _plain_cronjob(MINIMAL, {})
    ).json()
    path = f"/apis/batch/v1/namespaces/{other}/jobs?dryRun=All"
    job = _canary(clone_job(cronjob, "other-canary"))
    altered = copy.deepcopy(job)
    _container(altered)["image"] = "different-image"
    deadline = time.monotonic() + 30
    while cluster.server.admin("POST", path, altered).status_code == 201:
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert cluster.server.admin("POST", path, job).status_code == 201
    cluster.assert_allowed(
        _canary(cluster.job_for(UPDATE)), token=ADMIN_TOKEN, dry_run=True
    )
    cluster.server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{other}/roles",
        {
            "metadata": {"name": "create"},
            "rules": [
                {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]}
            ],
        },
    )
    cluster.server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{other}/rolebindings",
        {
            "metadata": {"name": "create"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "create",
            },
            "subjects": [
                {
                    "kind": "User",
                    "name": f"system:serviceaccount:{NAMESPACE}:{SERVICE_ACCOUNT}",
                }
            ],
        },
    )
    response = cluster.server.request("POST", path, TRIGGER_TOKEN, job)
    assert response.status_code in (403, 422), response.text
    assert f"{POLICY_PREFIX}-targets" in response.text


def test_missing_staging_binding_fails_verification(
    cluster: Cluster,
    kubectl_against_apiserver: None,
) -> None:
    binding = trigger_admission_binding(NAMESPACE, MINIMAL)
    path = _resource_path(binding)
    response = cluster.server.admin("DELETE", f"{path}/{binding['metadata']['name']}")
    assert response.status_code == 200, response.text
    job = _canary(cluster.job_for(MINIMAL))
    _container(job)["image"] = "attacker/image"
    deadline = time.monotonic() + 30
    try:
        while (
            cluster.trigger_create(job, token=ADMIN_TOKEN, dry_run=True).status_code
            != 201
        ):
            assert time.monotonic() < deadline
            time.sleep(0.1)
        with pytest.raises(AssertionError, match="Admission allowed canary"):
            kubernetes_security.verify_trigger_admission(NAMESPACE, [MINIMAL])
    finally:
        cluster.server.create(path, binding)


def test_read_only_cronjob_grant_cannot_bypass_guard(cluster: Cluster) -> None:
    server = cluster.server
    server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/roles",
        {
            "metadata": {"name": "read-every-cronjob"},
            "rules": [
                {
                    "apiGroups": ["batch"],
                    "resources": ["cronjobs"],
                    "verbs": ["get", "list", "watch"],
                }
            ],
        },
    )
    server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/rolebindings",
        {
            "metadata": {"name": "read-every-cronjob"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "read-every-cronjob",
            },
            "subjects": [
                {"kind": "Group", "name": f"system:serviceaccounts:{NAMESPACE}"}
            ],
        },
    )
    cronjob = server.create(
        f"/apis/batch/v1/namespaces/{NAMESPACE}/cronjobs",
        _plain_cronjob("foreign-cronjob", {}),
    ).json()
    deadline = time.monotonic() + 15
    while (
        server.request(
            "GET",
            f"/apis/batch/v1/namespaces/{NAMESPACE}/cronjobs/foreign-cronjob",
            TRIGGER_TOKEN,
        ).status_code
        != 200
    ):
        assert time.monotonic() < deadline
        time.sleep(0.1)
    job = clone_job(cronjob, "foreign-job")
    cluster.assert_denied(job, "targets")
    cluster.assert_denied(_canary(job), "targets", dry_run=True)


def test_binding_is_probed_before_target_is_authorized(
    cluster: Cluster, kubectl_against_apiserver: None
) -> None:
    server = cluster.server
    name = "newly-deployed-cronjob"
    body = _plain_cronjob(name, {})
    body["spec"]["suspend"] = True
    cronjob = server.create(
        f"/apis/batch/v1/namespaces/{NAMESPACE}/cronjobs", body
    ).json()
    job = clone_job(cronjob, "newly-deployed-job")
    cluster.assert_denied(job, "targets", dry_run=True)
    assert kubernetes_security.create_trigger_bindings(NAMESPACE, [name])
    assert not kubernetes_security.create_trigger_bindings(NAMESPACE, [name])
    altered = _canary(copy.deepcopy(job))
    _container(altered)["image"] = "attacker/image"
    kubernetes_security.verify_trigger_admission(NAMESPACE, [name])
    cluster.assert_denied(job, "targets", dry_run=True)
    server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/roles",
        {
            "metadata": {"name": name},
            "rules": [
                {
                    "apiGroups": ["batch"],
                    "resources": ["cronjobs"],
                    "resourceNames": [name],
                    "verbs": ["get", "trigger"],
                }
            ],
        },
    )
    server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/rolebindings",
        {
            "metadata": {"name": name},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": name,
            },
            "subjects": [
                {
                    "kind": "User",
                    "name": f"system:serviceaccount:{NAMESPACE}:{SERVICE_ACCOUNT}",
                }
            ],
        },
    )
    deadline = time.monotonic() + 15
    while cluster.trigger_create(job, dry_run=True).status_code != 201:
        assert time.monotonic() < deadline
        time.sleep(0.1)
    cluster.assert_denied(altered, "clone", dry_run=True)


def test_binding_deployer_permissions_are_create_get_only(cluster: Cluster) -> None:
    role, namespace_role = list(
        yaml.safe_load_all(Path("deploy/trigger-binding-deployer.yaml").read_text())
    )
    server = cluster.server
    server.create("/apis/rbac.authorization.k8s.io/v1/clusterroles", role)
    server.create(
        "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings",
        {
            "metadata": {"name": "binding-deployer"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": role["metadata"]["name"],
            },
            "subjects": [{"kind": "User", "name": "binding-deployer"}],
        },
    )
    namespace_role["metadata"]["namespace"] = NAMESPACE
    server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/roles",
        namespace_role,
    )
    server.create(
        f"/apis/rbac.authorization.k8s.io/v1/namespaces/{NAMESPACE}/rolebindings",
        {
            "metadata": {"name": "binding-deployer"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": namespace_role["metadata"]["name"],
            },
            "subjects": [{"kind": "User", "name": "binding-deployer"}],
        },
    )
    binding = trigger_admission_binding(NAMESPACE, MINIMAL)
    binding["metadata"]["name"] += "-permission-test"
    path = _resource_path(binding)
    deadline = time.monotonic() + 15
    while True:
        response = server.request("POST", path, DEPLOY_TOKEN, binding)
        if response.status_code == 201:
            break
        assert response.status_code == 403, response.text
        assert time.monotonic() < deadline, response.text
        time.sleep(0.1)
    existing = response.json()
    target = f"{path}/{binding['metadata']['name']}"
    assert server.request("GET", target, DEPLOY_TOKEN).status_code == 200
    assert server.request("GET", path, DEPLOY_TOKEN).status_code == 403
    assert server.request("PUT", target, DEPLOY_TOKEN, existing).status_code == 403
    assert (
        server.http.patch(
            target,
            headers={
                "Authorization": f"Bearer {DEPLOY_TOKEN}",
                "Content-Type": "application/merge-patch+json",
            },
            json={"spec": {"validationActions": ["Warn"]}},
        ).status_code
        == 403
    )
    assert server.request("DELETE", target, DEPLOY_TOKEN).status_code == 403
    assert (
        server.request(
            "POST",
            "/apis/admissionregistration.k8s.io/v1/validatingadmissionpolicies",
            DEPLOY_TOKEN,
            trigger_admission_resources(NAMESPACE)[0],
        ).status_code
        == 403
    )
