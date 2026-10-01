import copy
import json
import subprocess
import time
import uuid
from collections.abc import Sequence
from typing import Any

from reformatters.common.kubernetes import SERVICE_ACCOUNT, CronJob

CANARY_ANNOTATION = "dynamical.org/admission-canary"

_JOB_FIELDS = (
    "activeDeadlineSeconds",
    "backoffLimit",
    "backoffLimitPerIndex",
    "completionMode",
    "completions",
    "managedBy",
    "manualSelector",
    "maxFailedIndexes",
    "parallelism",
    "podFailurePolicy",
    "podReplacementPolicy",
    "successPolicy",
    "suspend",
    "ttlSecondsAfterFinished",
)
_METADATA_FIELDS = (
    "name",
    "generateName",
    "namespace",
    "uid",
    "resourceVersion",
    "generation",
    "creationTimestamp",
    "deletionTimestamp",
    "deletionGracePeriodSeconds",
    "ownerReferences",
    "finalizers",
    "managedFields",
)


def _same_optional_field(actual: str, expected: str, field: str) -> str:
    return (
        f"has({actual}.{field}) == has({expected}.{field}) && "
        f"(!has({actual}.{field}) || {actual}.{field} == {expected}.{field})"
    )


def _same_map(
    actual: str,
    expected: str,
    overrides: dict[str, str] | None = None,
    ignored: str = "[]",
) -> str:
    overrides = overrides or {}
    keys = f"({json.dumps(list(overrides))} + {ignored})"
    values = " && ".join(
        f"{json.dumps(key)} in {actual} && {actual}[{json.dumps(key)}] == {value}"
        for key, value in overrides.items()
    )
    return " && ".join(
        part
        for part in (
            f"{actual}.all(k, k in {keys} || (k in {expected} && {actual}[k] == {expected}[k]))",
            f"{expected}.all(k, k in {keys} || (k in {actual} && {actual}[k] == {expected}[k]))",
            values,
        )
        if part
    )


def trigger_admission_resources(namespace: str) -> list[dict[str, Any]]:
    assert namespace
    prefix = f"{namespace}-{SERVICE_ACCOUNT}"
    identity = f"system:serviceaccount:{namespace}:{SERVICE_ACCOUNT}"
    constraints = {
        "matchPolicy": "Equivalent",
        "namespaceSelector": {},
        "objectSelector": {},
        "resourceRules": [
            {
                "apiGroups": ["batch"],
                "apiVersions": ["v1"],
                "operations": ["CREATE"],
                "resources": ["jobs"],
                "scope": "Namespaced",
            }
        ],
    }
    conditions = [
        {
            "name": "trigger-identity",
            "expression": (
                f"request.userInfo.username == {json.dumps(identity)} || "
                f"(request.namespace == {json.dumps(namespace)} && "
                "has(request.dryRun) && request.dryRun && has(object.metadata.annotations) && "
                f"{json.dumps(CANARY_ANNOTATION)} in object.metadata.annotations && "
                f"object.metadata.annotations[{json.dumps(CANARY_ANNOTATION)}] == 'true')"
            ),
        }
    ]
    target_key = "dynamical.org/cronjob-name"
    uid_key = "dynamical.org/cronjob-uid"
    guard_name = f"{prefix}-targets"
    clone_name = f"{prefix}-clone"
    resources = [
        {
            "apiVersion": "admissionregistration.k8s.io/v1",
            "kind": "ValidatingAdmissionPolicy",
            "metadata": {"name": guard_name},
            "spec": {
                "failurePolicy": "Fail",
                "matchConstraints": constraints,
                "matchConditions": conditions,
                "validations": [
                    {
                        "expression": (
                            f"request.namespace == {json.dumps(namespace)} && "
                            "has(object.metadata.labels) && "
                            f"{json.dumps(target_key)} in object.metadata.labels && "
                            f"(request.userInfo.username != {json.dumps(identity)} || "
                            f"authorizer.serviceAccount({json.dumps(namespace)}, {json.dumps(SERVICE_ACCOUNT)})"
                            ".group('batch').resource('cronjobs')"
                            f".namespace({json.dumps(namespace)})"
                            f".name(object.metadata.labels[{json.dumps(target_key)}])"
                            ".check('trigger').allowed())"
                        ),
                        "message": "Trigger Jobs must select a registered CronJob template",
                    }
                ],
            },
        },
        {
            "apiVersion": "admissionregistration.k8s.io/v1",
            "kind": "ValidatingAdmissionPolicyBinding",
            "metadata": {"name": guard_name},
            "spec": {
                "policyName": guard_name,
                "validationActions": ["Deny"],
            },
        },
    ]
    variables = [
        {
            "name": "jobLabels",
            "expression": "has(object.metadata.labels) ? object.metadata.labels : {}",
        },
        {
            "name": "sourceLabels",
            "expression": "has(params.spec.jobTemplate.metadata) && has(params.spec.jobTemplate.metadata.labels) ? params.spec.jobTemplate.metadata.labels : {}",
        },
        {
            "name": "jobAnnotations",
            "expression": "has(object.metadata.annotations) ? object.metadata.annotations : {}",
        },
        {
            "name": "sourceAnnotations",
            "expression": "has(params.spec.jobTemplate.metadata) && has(params.spec.jobTemplate.metadata.annotations) ? params.spec.jobTemplate.metadata.annotations : {}",
        },
        {
            "name": "podLabels",
            "expression": "has(object.spec.template.metadata.labels) ? object.spec.template.metadata.labels : {}",
        },
        {
            "name": "sourcePodLabels",
            "expression": "has(params.spec.jobTemplate.spec.template.metadata.labels) ? params.spec.jobTemplate.spec.template.metadata.labels : {}",
        },
    ]
    expressions = [
        "params != null",
        f"object.metadata.labels[{json.dumps(target_key)}] == params.metadata.name",
        f"object.metadata.labels[{json.dumps(uid_key)}] == params.metadata.uid",
        "!has(object.metadata.generateName) || object.metadata.generateName == ''",
        "!object.metadata.name.matches('.*-[0-9]+$')",
        "!has(object.metadata.ownerReferences) || size(object.metadata.ownerReferences) == 0",
        "!has(object.metadata.finalizers) || size(object.metadata.finalizers) == 0",
        _same_map(
            "variables.jobLabels",
            "variables.sourceLabels",
            {target_key: "params.metadata.name", uid_key: "params.metadata.uid"},
        ),
        _same_map(
            "variables.jobAnnotations",
            "variables.sourceAnnotations",
            {"cronjob.kubernetes.io/instantiate": "'manual'"},
            ignored=f"(request.dryRun == true ? [{json.dumps(CANARY_ANNOTATION)}] : [])",
        ),
        "object.spec.template.spec == params.spec.jobTemplate.spec.template.spec",
        _same_optional_field(
            "object.spec.template.metadata",
            "params.spec.jobTemplate.spec.template.metadata",
            "annotations",
        ),
        _same_map(
            "variables.podLabels",
            "variables.sourcePodLabels",
            {
                "batch.kubernetes.io/controller-uid": "object.metadata.uid",
                "controller-uid": "object.metadata.uid",
                "batch.kubernetes.io/job-name": "object.metadata.name",
                "job-name": "object.metadata.name",
            },
        ),
        "!object.spec.manualSelector",
        "object.spec.selector.matchLabels == {'batch.kubernetes.io/controller-uid': object.metadata.uid}",
        "!has(object.spec.selector.matchExpressions) || size(object.spec.selector.matchExpressions) == 0",
    ]
    defaults = {
        "backoffLimit": "has(params.spec.jobTemplate.spec.backoffLimitPerIndex) ? 2147483647 : 6",
        "parallelism": "1",
        "completionMode": "'NonIndexed'",
        "suspend": "false",
        "podReplacementPolicy": "has(params.spec.jobTemplate.spec.podFailurePolicy) ? 'Failed' : 'TerminatingOrFailed'",
        "manualSelector": "false",
    }
    for field in _JOB_FIELDS:
        expected = f"params.spec.jobTemplate.spec.{field}"
        if field == "completions":
            expressions.append(
                f"has({expected}) ? object.spec.completions == {expected} : "
                "(has(params.spec.jobTemplate.spec.parallelism) ? "
                "!has(object.spec.completions) : object.spec.completions == 1)"
            )
            continue
        expressions.append(
            f"object.spec.{field} == (has({expected}) ? {expected} : ({defaults[field]}))"
            if field in defaults
            else _same_optional_field(
                "object.spec", "params.spec.jobTemplate.spec", field
            )
        )
    expressions.extend(
        _same_optional_field(
            "object.spec.template.metadata",
            "params.spec.jobTemplate.spec.template.metadata",
            field,
        )
        for field in _METADATA_FIELDS
    )
    for field in ("containers", "initContainers"):
        actual = f"object.spec.template.spec.{field}"
        expected = f"params.spec.jobTemplate.spec.template.spec.{field}"
        expressions.extend(
            [
                f"!has({actual}) || {actual}.map(c, c.name) == {expected}.map(c, c.name)",
                (
                    f"!has({actual}) || {actual}.all(c, {expected}.exists(p, c.name == p.name && "
                    "(!has(c.env) || c.env.map(e, e.name) == p.env.map(e, e.name)) && "
                    "(!has(c.volumeMounts) || c.volumeMounts.map(v, v.mountPath) == p.volumeMounts.map(v, v.mountPath))))"
                ),
            ]
        )
    resources.append(
        {
            "apiVersion": "admissionregistration.k8s.io/v1",
            "kind": "ValidatingAdmissionPolicy",
            "metadata": {"name": clone_name},
            "spec": {
                "failurePolicy": "Fail",
                "paramKind": {"apiVersion": "batch/v1", "kind": "CronJob"},
                "matchConstraints": constraints,
                "matchConditions": conditions,
                "variables": variables,
                "validations": [
                    {
                        "expression": expression,
                        "message": "Trigger Job must exactly copy the approved CronJob template",
                    }
                    for expression in expressions
                ],
            },
        }
    )
    return resources


def trigger_admission_binding(namespace: str, cronjob_name: str) -> dict[str, Any]:
    prefix = f"{namespace}-{SERVICE_ACCOUNT}"
    return {
        "apiVersion": "admissionregistration.k8s.io/v1",
        "kind": "ValidatingAdmissionPolicyBinding",
        "metadata": {"name": f"{prefix}-{cronjob_name}"},
        "spec": {
            "policyName": f"{prefix}-clone",
            "validationActions": ["Deny"],
            "paramRef": {
                "name": cronjob_name,
                "namespace": namespace,
                "parameterNotFoundAction": "Deny",
            },
            "matchResources": {
                "matchPolicy": "Equivalent",
                "namespaceSelector": {
                    "matchLabels": {"kubernetes.io/metadata.name": namespace}
                },
                "objectSelector": {
                    "matchLabels": {"dynamical.org/cronjob-name": cronjob_name}
                },
            },
        },
    }


def create_trigger_bindings(namespace: str, cronjob_names: Sequence[str]) -> bool:
    created = False
    for name in cronjob_names:
        binding = trigger_admission_binding(namespace, name)
        response = subprocess.run(
            ["/usr/bin/kubectl", "create", "-f", "-", "-o", "json"],
            input=json.dumps(binding),
            text=True,
            capture_output=True,
            check=False,
        )
        created |= response.returncode == 0
        if response.returncode:
            assert "(AlreadyExists)" in response.stderr, response.stderr
            response = subprocess.run(  # noqa: S603
                [
                    "/usr/bin/kubectl",
                    "get",
                    "validatingadmissionpolicybinding",
                    binding["metadata"]["name"],
                    "-o",
                    "json",
                ],
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            )
        assert json.loads(response.stdout)["spec"] == binding["spec"], (
            f"Existing admission binding for {name} differs; requires administrator review"
        )

    return created


def trigger_role_targets(namespace: str) -> set[str]:
    response = subprocess.run(  # noqa: S603
        [
            "/usr/bin/kubectl",
            "get",
            "role",
            SERVICE_ACCOUNT,
            "--namespace",
            namespace,
            "--ignore-not-found",
            "-o",
            "json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    if not response.stdout.strip():
        return set()
    rules = json.loads(response.stdout)["rules"]
    targets = set()
    for rule in rules:
        if "cronjobs" in rule["resources"] and "trigger" in rule["verbs"]:
            assert rule.get("resourceNames"), (
                "Trigger CronJob permissions must name their targets"
            )
            targets.update(rule["resourceNames"])
    return targets


def verify_trigger_admission(namespace: str, cronjob_names: Sequence[str]) -> set[str]:
    """Verify admission and return the targets whose CronJobs still exist."""
    prefix = f"{namespace}-{SERVICE_ACCOUNT}"

    def probe(job: dict[str, Any], denied_by: str | None) -> None:
        deadline = time.monotonic() + 15
        while True:
            job["metadata"]["name"] = (
                f"reformatters-admission-canary-t{uuid.uuid4().hex[:12]}"
            )
            response = subprocess.run(  # noqa: S603
                [
                    "/usr/bin/kubectl",
                    "create",
                    "--namespace",
                    namespace,
                    "--dry-run=server",
                    "-f",
                    "-",
                    "-o",
                    "json",
                ],
                input=json.dumps(job),
                text=True,
                capture_output=True,
                check=False,
            )
            if (
                (denied_by is not None and response.returncode != 0)
                or (
                    denied_by is None
                    and (
                        response.returncode == 0
                        or f"{prefix}-clone" not in response.stderr
                    )
                )
                or time.monotonic() >= deadline
            ):
                break
            time.sleep(0.5)
        if denied_by is None:
            assert response.returncode == 0, response.stderr
        else:
            assert response.returncode != 0, f"Admission allowed canary for {denied_by}"
            assert denied_by in response.stderr, (
                f"Admission did not deny canary through {denied_by}: {response.stderr}"
            )

    if not cronjob_names:
        probe(
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "labels": {},
                    "annotations": {CANARY_ANNOTATION: "true"},
                },
                "spec": {
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "worker",
                                    "image": "invalid.example/admission-canary:never-run",
                                }
                            ],
                        }
                    }
                },
            },
            f"{prefix}-targets",
        )
        return set()

    live_targets = set()
    for name in cronjob_names:
        response = subprocess.run(  # noqa: S603
            [
                "/usr/bin/kubectl",
                "get",
                "cronjob",
                name,
                "--namespace",
                namespace,
                "--ignore-not-found",
                "-o",
                "json",
            ],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        )
        source = json.loads(response.stdout) if response.stdout.strip() else None
        template = (
            source["spec"]["jobTemplate"]
            if source
            else {
                "spec": {
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "worker",
                                    "image": "invalid.example/missing-template",
                                }
                            ],
                        }
                    }
                }
            }
        )
        metadata = template.get("metadata", {})
        job: dict[str, Any] = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "labels": {
                    **metadata.get("labels", {}),
                    "dynamical.org/cronjob-name": name,
                    "dynamical.org/cronjob-uid": source["metadata"]["uid"]
                    if source
                    else "missing",
                },
                "annotations": {
                    **metadata.get("annotations", {}),
                    "cronjob.kubernetes.io/instantiate": "manual",
                    CANARY_ANNOTATION: "true",
                },
            },
            "spec": copy.deepcopy(template["spec"]),
        }
        unlabeled = copy.deepcopy(job)
        unlabeled["metadata"]["labels"] = {}
        probe(unlabeled, f"{prefix}-targets")
        altered = copy.deepcopy(job)
        altered["spec"]["template"]["spec"]["containers"][0]["image"] = (
            "invalid.example/admission-canary:never-run"
        )
        probe(altered, f"{prefix}-clone")
        probe(job, None if source else f"{prefix}-clone")
        if source:
            live_targets.add(name)
    return live_targets


def trigger_deployment_resources(
    reformat_jobs: Sequence[CronJob],
    authorized_targets: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = sorted(set(authorized_targets) | {job.name for job in reformat_jobs})
    return (
        [
            {
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": {"name": SERVICE_ACCOUNT},
            },
            *[reformat_job.as_kubernetes_object() for reformat_job in reformat_jobs],
        ],
        [
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": {"name": SERVICE_ACCOUNT},
                "rules": [
                    *(
                        [
                            {
                                "apiGroups": ["batch"],
                                "resources": ["cronjobs"],
                                "resourceNames": targets,
                                "verbs": ["get", "trigger"],
                            }
                        ]
                        if targets
                        else []
                    ),
                    {
                        "apiGroups": ["batch"],
                        "resources": ["jobs"],
                        "verbs": ["create", "get"],
                    },
                ],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "RoleBinding",
                "metadata": {"name": SERVICE_ACCOUNT},
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "Role",
                    "name": SERVICE_ACCOUNT,
                },
                "subjects": [{"kind": "ServiceAccount", "name": SERVICE_ACCOUNT}],
            },
        ],
    )
