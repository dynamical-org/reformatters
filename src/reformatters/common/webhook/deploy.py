"""Kubernetes manifests for the always-on wxopticon webhook receiver.

Rendered as plain dicts and applied alongside the operational cronjobs by
`reformatters.common.deploy`. See docs/webhooks.md.
"""

from typing import Any

_APP = "wxopticon-webhook"
_SECRET_NAME = "wxopticon-webhook"  # noqa: S105  key: WXOPTICON_WEBHOOK_SECRET
_PORT = 8080

# Deploy-time infrastructure — confirm before first deploy (see docs/webhooks.md).
RECEIVER_HOST = "webhooks.dynamical.org"
# ACM certificate ARN for RECEIVER_HOST; the ALB serves HTTPS (wxopticon requires it).
ACM_CERTIFICATE_ARN = ""


def webhook_receiver_resources(image_tag: str) -> list[dict[str, Any]]:
    labels = {"app": _APP}
    service_account = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": _APP},
    }
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": _APP},
        "rules": [
            {
                "apiGroups": ["batch"],
                "resources": ["cronjobs"],
                "verbs": ["get"],
            },
            {
                "apiGroups": ["batch"],
                "resources": ["jobs"],
                "verbs": ["create", "get"],
            },
        ],
    }
    role_binding = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {"name": _APP},
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": _APP,
        },
        "subjects": [{"kind": "ServiceAccount", "name": _APP}],
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": _APP, "labels": labels},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "serviceAccountName": _APP,
                    "nodeSelector": {"eks.amazonaws.com/compute-type": "auto"},
                    "containers": [
                        {
                            "name": "receiver",
                            "image": image_tag,
                            "command": [
                                "python",
                                "src/reformatters/__main__.py",
                                "serve-webhooks",
                            ],
                            "ports": [{"containerPort": _PORT}],
                            "env": [
                                {"name": "DYNAMICAL_ENV", "value": "prod"},
                                {
                                    "name": "WXOPTICON_WEBHOOK_SECRET",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": _SECRET_NAME,
                                            "key": "WXOPTICON_WEBHOOK_SECRET",
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "0.1", "memory": "512Mi"}
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/healthz", "port": _PORT}
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/healthz", "port": _PORT}
                            },
                        }
                    ],
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": _APP, "labels": labels},
        "spec": {
            "selector": labels,
            "ports": [{"port": 80, "targetPort": _PORT}],
        },
    }
    ingress = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {
            "name": _APP,
            "annotations": {
                "alb.ingress.kubernetes.io/scheme": "internet-facing",
                "alb.ingress.kubernetes.io/target-type": "ip",
                "alb.ingress.kubernetes.io/listen-ports": '[{"HTTPS":443}]',
                "alb.ingress.kubernetes.io/certificate-arn": ACM_CERTIFICATE_ARN,
            },
        },
        "spec": {
            "ingressClassName": "alb",
            "rules": [
                {
                    "host": RECEIVER_HOST,
                    "http": {
                        "paths": [
                            {
                                "path": "/",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": _APP,
                                        "port": {"number": 80},
                                    }
                                },
                            }
                        ]
                    },
                }
            ],
        },
    }
    return [service_account, role, role_binding, deployment, service, ingress]
