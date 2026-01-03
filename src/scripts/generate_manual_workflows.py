#!/usr/bin/env python3
"""Generate GitHub Actions workflow files for manual Kubernetes operations.

This script generates workflow files with dropdowns containing all available cronjobs.
It should be run automatically via pre-commit hook to keep workflows in sync with code.
"""

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.kubernetes import CronJob
from reformatters.common.logging import get_logger

log = get_logger(__name__)

MANUAL_K8S_GITHUB_ENVIRONMENT = "prod"


def get_all_cronjob_names() -> list[str]:
    """Extract all CronJob names from DYNAMICAL_DATASETS."""
    cronjob_names: list[str] = []

    for dataset in DYNAMICAL_DATASETS:
        try:
            resources = dataset.operational_kubernetes_resources(
                "placeholder-image-tag"
            )
            cronjob_names.extend(
                resource.name for resource in resources if isinstance(resource, CronJob)
            )
        except NotImplementedError:
            continue

    return sorted(cronjob_names)


def generate_create_job_workflow(cronjob_names: list[str]) -> dict[str, Any]:
    """Generate the workflow for creating jobs from cronjobs."""
    return {
        "name": "Manual: Create Job from CronJob",
        "on": {
            "workflow_dispatch": {
                "inputs": {
                    "cronjob_name": {
                        "description": "CronJob to create a job from",
                        "required": True,
                        "type": "choice",
                        "options": cronjob_names,
                    }
                }
            }
        },
        "concurrency": {
            "group": "k8s-manual-${{ github.actor }}-${{ github.run_id }}",
            "cancel-in-progress": False,
        },
        "permissions": {"id-token": "write", "contents": "read"},
        "jobs": {
            "create-job": {
                "name": "Create Job",
                "runs-on": "ubuntu-24.04",
                "environment": MANUAL_K8S_GITHUB_ENVIRONMENT,
                "steps": [
                    {
                        "uses": "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
                        "with": {"sparse-checkout": "."},
                    },
                    {
                        "name": "Configure AWS Credentials",
                        "uses": "aws-actions/configure-aws-credentials@e3dd6a429d7300a6a4c196c26e071d42e0343502",
                        "with": {
                            "role-to-assume": "${{ secrets.AWS_ROLE_TO_ASSUME }}",
                            "aws-region": "${{ secrets.AWS_REGION }}",
                        },
                    },
                    {
                        "name": "Install kubectl",
                        "uses": "azure/setup-kubectl@901a10e89ea615cf61f57ac05cecdf23e7de06d8",
                        "with": {"version": "latest"},
                    },
                    {
                        "name": "Update kubeconfig",
                        "run": "aws eks update-kubeconfig --name ${{ secrets.EKS_CLUSTER_NAME }} --region ${{ secrets.AWS_REGION }}",
                    },
                    {
                        "name": "Generate job name and create job",
                        "run": """
#!/bin/bash
set -euo pipefail

CRONJOB_NAME="${{ github.event.inputs.cronjob_name }}"
USERNAME="${{ github.actor }}"
RANDOM_CHARS=$(openssl rand -hex 1)

# Generate job name with length limits
if [ ${#CRONJOB_NAME} -gt 50 ]; then
  CRONJOB_TRUNCATED="${CRONJOB_NAME:0:50}"
else
  CRONJOB_TRUNCATED="${CRONJOB_NAME}"
fi

if [ ${#USERNAME} -gt 8 ]; then
  USERNAME_TRUNCATED="${USERNAME:0:8}"
else
  USERNAME_TRUNCATED="${USERNAME}"
fi

JOB_NAME="${CRONJOB_TRUNCATED}-${USERNAME_TRUNCATED}-${RANDOM_CHARS}"

echo "Creating job: ${JOB_NAME} from cronjob: ${CRONJOB_NAME}"
kubectl create job --from=cronjob/${CRONJOB_NAME} ${JOB_NAME}

echo "✅ Job created successfully!"
echo ""
echo "Check job status with:"
echo "  gh workflow run manual-get-jobs.yml"
echo ""
echo "Or view in GitHub Actions: https://github.com/${{ github.repository }}/actions"
""".strip(),
                    },
                ],
            }
        },
    }


def generate_get_jobs_workflow() -> dict[str, Any]:
    """Generate the workflow for getting jobs."""
    return {
        "name": "Manual: Get Jobs",
        "on": {"workflow_dispatch": {}},
        "permissions": {"id-token": "write", "contents": "read"},
        "jobs": {
            "get-jobs": {
                "name": "Get Jobs",
                "runs-on": "ubuntu-24.04",
                "environment": MANUAL_K8S_GITHUB_ENVIRONMENT,
                "steps": [
                    {
                        "uses": "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
                        "with": {"sparse-checkout": "."},
                    },
                    {
                        "name": "Configure AWS Credentials",
                        "uses": "aws-actions/configure-aws-credentials@e3dd6a429d7300a6a4c196c26e071d42e0343502",
                        "with": {
                            "role-to-assume": "${{ secrets.AWS_ROLE_TO_ASSUME }}",
                            "aws-region": "${{ secrets.AWS_REGION }}",
                        },
                    },
                    {
                        "name": "Install kubectl",
                        "uses": "azure/setup-kubectl@901a10e89ea615cf61f57ac05cecdf23e7de06d8",
                        "with": {"version": "latest"},
                    },
                    {
                        "name": "Update kubeconfig",
                        "run": "aws eks update-kubeconfig --name ${{ secrets.EKS_CLUSTER_NAME }} --region ${{ secrets.AWS_REGION }}",
                    },
                    {
                        "name": "Get jobs",
                        "run": "kubectl get jobs --sort-by=.metadata.creationTimestamp",
                    },
                ],
            }
        },
    }


def generate_get_pods_workflow() -> dict[str, Any]:
    """Generate the workflow for getting pods."""
    return {
        "name": "Manual: Get Pods",
        "on": {"workflow_dispatch": {}},
        "permissions": {"id-token": "write", "contents": "read"},
        "jobs": {
            "get-pods": {
                "name": "Get Pods",
                "runs-on": "ubuntu-24.04",
                "environment": MANUAL_K8S_GITHUB_ENVIRONMENT,
                "steps": [
                    {
                        "uses": "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
                        "with": {"sparse-checkout": "."},
                    },
                    {
                        "name": "Configure AWS Credentials",
                        "uses": "aws-actions/configure-aws-credentials@e3dd6a429d7300a6a4c196c26e071d42e0343502",
                        "with": {
                            "role-to-assume": "${{ secrets.AWS_ROLE_TO_ASSUME }}",
                            "aws-region": "${{ secrets.AWS_REGION }}",
                        },
                    },
                    {
                        "name": "Install kubectl",
                        "uses": "azure/setup-kubectl@901a10e89ea615cf61f57ac05cecdf23e7de06d8",
                        "with": {"version": "latest"},
                    },
                    {
                        "name": "Update kubeconfig",
                        "run": "aws eks update-kubeconfig --name ${{ secrets.EKS_CLUSTER_NAME }} --region ${{ secrets.AWS_REGION }}",
                    },
                    {
                        "name": "Get pods",
                        "run": "kubectl get pods --sort-by=.metadata.creationTimestamp",
                    },
                ],
            }
        },
    }


def write_workflow(
    workflow: dict[str, Any], filename: str, workflows_dir: Path
) -> None:
    """Write a workflow dict to a YAML file."""
    output_path = workflows_dir / filename

    # Use yaml.dump with proper formatting
    yaml_content = yaml.dump(
        workflow,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=1000,  # Prevent line wrapping
    )

    # Add header comment
    header = """# This file is auto-generated by src/scripts/generate_manual_workflows.py
# Do not edit manually - changes will be overwritten by pre-commit hook

"""

    output_path.write_text(header + yaml_content)
    log.info(f"Generated: {output_path}")


def main() -> None:
    """Generate all manual workflow files."""
    workflows_dir = Path(__file__).parent.parent.parent / ".github" / "workflows"
    assert workflows_dir.exists(), (
        f"Workflows directory does not exist: {workflows_dir}"
    )

    # Get all cronjob names from datasets
    cronjob_names = get_all_cronjob_names()

    if not cronjob_names:
        log.warning("No cronjobs found in DYNAMICAL_DATASETS")
        return

    log.info(f"Found {len(cronjob_names)} cronjobs:")
    for name in cronjob_names:
        log.info(f"  - {name}")

    # Generate workflow files
    workflows = [
        (
            generate_create_job_workflow(cronjob_names),
            "manual-create-job-from-cronjob.yml",
        ),
        (generate_get_jobs_workflow(), "manual-get-jobs.yml"),
        (generate_get_pods_workflow(), "manual-get-pods.yml"),
    ]

    for workflow, filename in workflows:
        write_workflow(workflow, filename, workflows_dir)

    log.info("All workflow files generated successfully!")


if __name__ == "__main__":
    main()
