#!/usr/bin/env python3
"""Generate GitHub Actions workflow files for manual Kubernetes operations.

This script generates workflow files with dropdowns containing all available cronjobs.
It should be run automatically via prek hook to keep workflows in sync with code.
"""

from pathlib import Path
from typing import Any

import yaml

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.logging import get_logger

log = get_logger(__name__)


class LiteralString(str):
    """String that should be represented as a literal block scalar in YAML."""

    __slots__ = ()


def literal_string_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Represent LiteralString as a literal block scalar (|) in YAML."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(LiteralString, literal_string_representer)

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


def get_backfill_dataset_ids() -> list[str]:
    """Dataset ids backfill-kubernetes can run for: those with an update CronJob
    (it provides the worker resource shapes and deployed image)."""
    dataset_ids = []
    for dataset in DYNAMICAL_DATASETS:
        try:
            resources = dataset.operational_kubernetes_resources(
                "placeholder-image-tag"
            )
        except NotImplementedError:
            continue
        if any(isinstance(r, ReformatCronJob) for r in resources):
            dataset_ids.append(dataset.dataset_id)
    return sorted(dataset_ids)


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
                        "uses": "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
                        "with": {"sparse-checkout": "."},
                    },
                    {
                        "name": "Configure AWS Credentials",
                        "uses": "aws-actions/configure-aws-credentials@ec61189d14ec14c8efccab744f656cffd0e33f37",
                        "with": {
                            "role-to-assume": "${{ secrets.AWS_ROLE_TO_ASSUME }}",
                            "aws-region": "${{ secrets.AWS_REGION }}",
                        },
                    },
                    {
                        "name": "Install kubectl",
                        "uses": "azure/setup-kubectl@15650b3ad78fff148532a140b8a4c821796b2d7b",
                        "with": {"version": "latest"},
                    },
                    {
                        "name": "Update kubeconfig",
                        "run": "aws eks update-kubeconfig --name ${{ secrets.EKS_CLUSTER_NAME }} --region ${{ secrets.AWS_REGION }}",
                    },
                    {
                        "name": "Submit job (SEE LOGS)",
                        "run": LiteralString(
                            r"""#!/bin/bash
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

echo ""
echo "## Job Created Successfully"
echo ""
echo "CronJob: ${CRONJOB_NAME}"
echo ""
echo "Job Name: ${JOB_NAME}"
echo ""
echo "### Monitoring"
echo ""
echo "- Sentry cron status: https://dynamical.sentry.io/issues/alerts/rules/crons/reformatters/${CRONJOB_NAME}/details/"
echo "- Sentry job logs: https://dynamical.sentry.io/explore/logs/?logsQuery=job_name%3A${JOB_NAME}"
echo "- Manual Get Jobs: https://github.com/${{ github.repository }}/actions/workflows/manual-get-jobs.yml"
echo "- Manual Get Pods: https://github.com/${{ github.repository }}/actions/workflows/manual-get-pods.yml"

# Write to job summary
{
  echo "## Job Created Successfully"
  echo ""
  echo "**CronJob:** \`${CRONJOB_NAME}\`"
  echo ""
  echo "**Job Name:** \`${JOB_NAME}\`"
  echo ""
  echo "### Monitoring"
  echo ""
  echo "- [Sentry cron status](https://dynamical.sentry.io/issues/alerts/rules/crons/reformatters/${CRONJOB_NAME}/details/)"
  echo "- [Sentry job logs](https://dynamical.sentry.io/explore/logs/?logsQuery=job_name%3A${JOB_NAME})"
  echo "- [Manual Get Jobs](https://github.com/${{ github.repository }}/actions/workflows/manual-get-jobs.yml)"
  echo "- [Manual Get Pods](https://github.com/${{ github.repository }}/actions/workflows/manual-get-pods.yml)"
} >> $GITHUB_STEP_SUMMARY
"""
                        ),
                    },
                ],
            }
        },
    }


def generate_backfill_workflow(dataset_ids: list[str]) -> dict[str, Any]:
    """Generate the workflow_dispatch workflow that kicks off a backfill.

    Exposes only safe operations: creating a new store, overwriting chunk data,
    and overwriting metadata (which never trims and only expands with an explicit
    append_dim_end plus both overwrite flags — the CLI enforces all guards)."""
    return {
        "name": "Manual: Backfill",
        "on": {
            "workflow_dispatch": {
                "inputs": {
                    "dataset_id": {
                        "description": "Dataset to backfill",
                        "required": True,
                        "type": "choice",
                        "options": dataset_ids,
                    },
                    "operation": {
                        "description": "create-new-store fails if the store exists; overwrite-* require it to exist. Backfilling a newly added variable = overwrite-chunks-and-metadata + filter_variable_names.",
                        "required": True,
                        "type": "choice",
                        "options": [
                            "create-new-store",
                            "overwrite-chunks",
                            "overwrite-metadata",
                            "overwrite-chunks-and-metadata",
                        ],
                    },
                    "append_dim_end": {
                        "description": "Exclusive end timestamp (ISO). Leave empty for the default: an existing store's current end (extent unchanged), or the current time for a new store. Setting this past an existing store's end extends it (requires overwrite-chunks-and-metadata); trimming is never supported.",
                        "required": False,
                        "type": "string",
                    },
                    "filter_start": {
                        "description": "Only process regions at or after this timestamp (ISO, optional)",
                        "required": False,
                        "type": "string",
                    },
                    "filter_end": {
                        "description": "Only process regions before this timestamp (ISO, optional)",
                        "required": False,
                        "type": "string",
                    },
                    "filter_variable_names": {
                        "description": "Comma-separated variable names to process (optional, default all)",
                        "required": False,
                        "type": "string",
                    },
                    "jobs_per_pod": {
                        "description": "Region jobs per worker pod",
                        "required": False,
                        "type": "string",
                        "default": "1",
                    },
                    "max_parallelism": {
                        "description": "Maximum concurrent worker pods",
                        "required": False,
                        "type": "string",
                        "default": "100",
                    },
                }
            }
        },
        "concurrency": {
            "group": "k8s-manual-${{ github.actor }}-${{ github.run_id }}",
            "cancel-in-progress": False,
        },
        "permissions": {"id-token": "write", "contents": "read"},
        "jobs": {
            "backfill": {
                "name": "Backfill",
                "runs-on": "ubuntu-24.04",
                "environment": MANUAL_K8S_GITHUB_ENVIRONMENT,
                "steps": [
                    {
                        "uses": "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
                    },
                    {
                        "name": "Install uv",
                        "uses": "astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78",
                        "with": {
                            "enable-cache": True,
                            "cache-dependency-glob": "uv.lock",
                        },
                    },
                    {
                        "name": "Set up Python",
                        "uses": "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
                        "with": {"python-version-file": ".python-version"},
                    },
                    {
                        "name": "Install the project",
                        "run": "uv sync --all-extras --dev --locked",
                    },
                    {
                        "name": "Configure AWS Credentials",
                        "uses": "aws-actions/configure-aws-credentials@ec61189d14ec14c8efccab744f656cffd0e33f37",
                        "with": {
                            "role-to-assume": "${{ secrets.AWS_ROLE_TO_ASSUME }}",
                            "aws-region": "${{ secrets.AWS_REGION }}",
                        },
                    },
                    {
                        "name": "Install kubectl",
                        "uses": "azure/setup-kubectl@15650b3ad78fff148532a140b8a4c821796b2d7b",
                        "with": {"version": "latest"},
                    },
                    {
                        "name": "Update kubeconfig",
                        "run": "aws eks update-kubeconfig --name ${{ secrets.EKS_CLUSTER_NAME }} --region ${{ secrets.AWS_REGION }}",
                    },
                    {
                        "name": "Start backfill (SEE LOGS)",
                        "env": {"DYNAMICAL_ENV": "prod"},
                        "run": LiteralString(
                            r"""#!/bin/bash
set -euo pipefail

DATASET_ID="${{ github.event.inputs.dataset_id }}"
OPERATION="${{ github.event.inputs.operation }}"
APPEND_DIM_END="${{ github.event.inputs.append_dim_end }}"
FILTER_START="${{ github.event.inputs.filter_start }}"
FILTER_END="${{ github.event.inputs.filter_end }}"
FILTER_VARIABLE_NAMES="${{ github.event.inputs.filter_variable_names }}"
JOBS_PER_POD="${{ github.event.inputs.jobs_per_pod }}"
MAX_PARALLELISM="${{ github.event.inputs.max_parallelism }}"

ARGS=(--jobs-per-pod "${JOBS_PER_POD}" --max-parallelism "${MAX_PARALLELISM}")
case "${OPERATION}" in
  create-new-store) ;;
  overwrite-chunks) ARGS+=(--overwrite-chunks) ;;
  overwrite-metadata) ARGS+=(--overwrite-metadata) ;;
  overwrite-chunks-and-metadata) ARGS+=(--overwrite-chunks --overwrite-metadata) ;;
esac
if [ -n "${APPEND_DIM_END}" ]; then
  ARGS+=(--append-dim-end "${APPEND_DIM_END}")
fi
if [ -n "${FILTER_START}" ]; then
  ARGS+=(--filter-start "${FILTER_START}")
fi
if [ -n "${FILTER_END}" ]; then
  ARGS+=(--filter-end "${FILTER_END}")
fi
if [ -n "${FILTER_VARIABLE_NAMES}" ]; then
  IFS=',' read -ra VARIABLE_NAMES <<< "${FILTER_VARIABLE_NAMES}"
  for VARIABLE_NAME in "${VARIABLE_NAMES[@]}"; do
    ARGS+=(--filter-variable-names "${VARIABLE_NAME}")
  done
fi

echo "Running: uv run main ${DATASET_ID} backfill-kubernetes ${ARGS[*]}"
uv run main "${DATASET_ID}" backfill-kubernetes "${ARGS[@]}"

{
  echo "## Backfill Started"
  echo ""
  echo "**Dataset:** \`${DATASET_ID}\`"
  echo ""
  echo "**Operation:** \`${OPERATION}\`"
  echo ""
  echo "### Monitoring"
  echo ""
  echo "- [Manual: Get Jobs](https://github.com/${{ github.repository }}/actions/workflows/manual-get-jobs.yml)"
  echo "- [Manual: Get Pods](https://github.com/${{ github.repository }}/actions/workflows/manual-get-pods.yml)"
  echo "- [Sentry logs](https://dynamical.sentry.io/explore/logs/)"
} >> $GITHUB_STEP_SUMMARY
"""
                        ),
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
                        "uses": "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
                        "with": {"sparse-checkout": "."},
                    },
                    {
                        "name": "Configure AWS Credentials",
                        "uses": "aws-actions/configure-aws-credentials@ec61189d14ec14c8efccab744f656cffd0e33f37",
                        "with": {
                            "role-to-assume": "${{ secrets.AWS_ROLE_TO_ASSUME }}",
                            "aws-region": "${{ secrets.AWS_REGION }}",
                        },
                    },
                    {
                        "name": "Install kubectl",
                        "uses": "azure/setup-kubectl@15650b3ad78fff148532a140b8a4c821796b2d7b",
                        "with": {"version": "latest"},
                    },
                    {
                        "name": "Update kubeconfig",
                        "run": "aws eks update-kubeconfig --name ${{ secrets.EKS_CLUSTER_NAME }} --region ${{ secrets.AWS_REGION }}",
                    },
                    {
                        "name": "Get jobs (SEE LOGS)",
                        "run": LiteralString(
                            """#!/bin/bash
set -euo pipefail

# Get jobs and save output (capture both stdout and stderr for "No resources found" message)
OUTPUT=$(kubectl get jobs --sort-by=.metadata.creationTimestamp 2>&1)

# Print to logs (plaintext version of summary)
echo "## Kubernetes Jobs"
echo ""
echo "$OUTPUT"
echo ""
echo "### Monitoring"
echo ""
echo "- Sentry crons overview: https://dynamical.sentry.io/insights/crons/"
echo "- Sentry logs: https://dynamical.sentry.io/explore/logs/"

# Write to job summary
echo "## Kubernetes Jobs" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY
echo '```' >> $GITHUB_STEP_SUMMARY
echo "$OUTPUT" >> $GITHUB_STEP_SUMMARY
echo '```' >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY
echo "### Monitoring" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY
echo "- [Sentry crons overview](https://dynamical.sentry.io/insights/crons/)" >> $GITHUB_STEP_SUMMARY
echo "- [Sentry logs](https://dynamical.sentry.io/explore/logs/)" >> $GITHUB_STEP_SUMMARY
"""
                        ),
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
                        "uses": "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
                        "with": {"sparse-checkout": "."},
                    },
                    {
                        "name": "Configure AWS Credentials",
                        "uses": "aws-actions/configure-aws-credentials@ec61189d14ec14c8efccab744f656cffd0e33f37",
                        "with": {
                            "role-to-assume": "${{ secrets.AWS_ROLE_TO_ASSUME }}",
                            "aws-region": "${{ secrets.AWS_REGION }}",
                        },
                    },
                    {
                        "name": "Install kubectl",
                        "uses": "azure/setup-kubectl@15650b3ad78fff148532a140b8a4c821796b2d7b",
                        "with": {"version": "latest"},
                    },
                    {
                        "name": "Update kubeconfig",
                        "run": "aws eks update-kubeconfig --name ${{ secrets.EKS_CLUSTER_NAME }} --region ${{ secrets.AWS_REGION }}",
                    },
                    {
                        "name": "Get pods (SEE LOGS)",
                        "run": LiteralString(
                            """#!/bin/bash
set -euo pipefail

# Get pods and save output (capture both stdout and stderr for "No resources found" message)
OUTPUT=$(kubectl get pods --sort-by=.metadata.creationTimestamp 2>&1)

# Print to logs (plaintext version of summary)
echo "## Kubernetes Pods"
echo ""
echo "$OUTPUT"
echo ""
echo "### Monitoring"
echo ""
echo "- Sentry crons overview: https://dynamical.sentry.io/insights/crons/"
echo "- Sentry logs: https://dynamical.sentry.io/explore/logs/"

# Write to job summary
echo "## Kubernetes Pods" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY
echo '```' >> $GITHUB_STEP_SUMMARY
echo "$OUTPUT" >> $GITHUB_STEP_SUMMARY
echo '```' >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY
echo "### Monitoring" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY
echo "- [Sentry crons overview](https://dynamical.sentry.io/insights/crons/)" >> $GITHUB_STEP_SUMMARY
echo "- [Sentry logs](https://dynamical.sentry.io/explore/logs/)" >> $GITHUB_STEP_SUMMARY
"""
                        ),
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
# Do not edit manually - changes will be overwritten by prek hook

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
        (
            generate_backfill_workflow(get_backfill_dataset_ids()),
            "manual-backfill.yml",
        ),
        (generate_get_jobs_workflow(), "manual-get-jobs.yml"),
        (generate_get_pods_workflow(), "manual-get-pods.yml"),
    ]

    for workflow, filename in workflows:
        write_workflow(workflow, filename, workflows_dir)

    log.info("All workflow files generated successfully!")


if __name__ == "__main__":
    main()
