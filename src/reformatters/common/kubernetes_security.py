import copy
import json
import subprocess
from collections.abc import Sequence
from typing import Any

from reformatters.common.kubernetes import SERVICE_ACCOUNT

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


def trigger_admission_resources(
    namespace: str, cronjob_names: Sequence[str]
) -> list[dict[str, Any]]:
    assert namespace
    assert cronjob_names
    names = sorted(set(cronjob_names))
    prefix = f"{namespace}-{SERVICE_ACCOUNT}"
    identity = f"system:serviceaccount:{namespace}:{SERVICE_ACCOUNT}"
    constraints = {
        "resourceRules": [
            {
                "apiGroups": ["batch"],
                "apiVersions": ["v1"],
                "operations": ["CREATE"],
                "resources": ["jobs"],
                "scope": "Namespaced",
            }
        ]
    }
    conditions = [
        {
            "name": "trigger-identity",
            "expression": (
                f"request.userInfo.username == {json.dumps(identity)} || "
                "(request.dryRun == true && has(object.metadata.annotations) && "
                f"{json.dumps(CANARY_ANNOTATION)} in object.metadata.annotations && "
                f"object.metadata.annotations[{json.dumps(CANARY_ANNOTATION)}] == 'true')"
            ),
        }
    ]
    namespace_match = {
        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": namespace}}
    }
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
                            f"object.metadata.labels[{json.dumps(target_key)}] in {json.dumps(names)}"
                        ),
                        "message": "Trigger Jobs must select an approved CronJob template",
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
        f"object.spec.all(k, k in {json.dumps([*_JOB_FIELDS, 'selector', 'template'])})",
        f"object.metadata.labels[{json.dumps(target_key)}] == params.metadata.name",
        f"object.metadata.labels[{json.dumps(uid_key)}] == params.metadata.uid",
        "!has(params.spec.suspend) || !params.spec.suspend",
        "!has(object.metadata.generateName) || object.metadata.generateName == ''",
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
    resources.extend(
        {
            "apiVersion": "admissionregistration.k8s.io/v1",
            "kind": "ValidatingAdmissionPolicyBinding",
            "metadata": {"name": f"{prefix}-{name}"},
            "spec": {
                "policyName": clone_name,
                "validationActions": ["Deny"],
                "paramRef": {
                    "name": name,
                    "namespace": namespace,
                    "parameterNotFoundAction": "Deny",
                },
                "matchResources": {
                    **namespace_match,
                    "objectSelector": {"matchLabels": {target_key: name}},
                },
            },
        }
        for name in names
    )
    return resources


def verify_trigger_admission(
    namespace: str, templates: Sequence[dict[str, Any]], *, require_params: bool
) -> None:
    assert templates, "No approved trigger templates to verify"
    prefix = f"{namespace}-{SERVICE_ACCOUNT}"

    def probe(job: dict[str, Any], denied_by: str | None) -> None:
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
        if denied_by is None:
            assert response.returncode == 0, response.stderr
        else:
            assert response.returncode != 0, f"Admission allowed canary for {denied_by}"
            assert denied_by in response.stderr, (
                f"Admission did not deny canary through {denied_by}: {response.stderr}"
            )

    for desired in templates:
        name = desired["metadata"]["name"]
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
            capture_output=True,
            check=True,
        )
        source = json.loads(response.stdout) if response.stdout.strip() else None
        assert source is not None or not require_params, f"Missing CronJob {name}"
        template = (source or desired)["spec"]["jobTemplate"]
        metadata = template.get("metadata", {})
        job: dict[str, Any] = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "reformat-admission-canary",
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
        if source is not None and not source["spec"].get("suspend", False):
            probe(job, None)
