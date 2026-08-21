#!/usr/bin/env python3
"""Update generated choices in manual GitHub Actions workflows."""

import re
from pathlib import Path

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.logging import get_logger

log = get_logger(__name__)


def get_all_cronjob_names() -> list[str]:
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
    dataset_ids = []
    for dataset in DYNAMICAL_DATASETS:
        try:
            resources = dataset.operational_kubernetes_resources(
                "placeholder-image-tag"
            )
        except NotImplementedError:
            continue
        if any(isinstance(resource, ReformatCronJob) for resource in resources):
            dataset_ids.append(dataset.dataset_id)
    return sorted(dataset_ids)


def update_choice_options(
    workflow_path: Path, input_name: str, choices: list[str]
) -> None:
    assert choices
    assert all(re.fullmatch(r"[a-z0-9-]+", choice) for choice in choices)

    lines = workflow_path.read_text().splitlines(keepends=True)
    input_indexes = [
        index for index, line in enumerate(lines) if line.strip() == f"{input_name}:"
    ]
    assert len(input_indexes) == 1, (
        f"Expected one {input_name!r} input in {workflow_path}, found {len(input_indexes)}"
    )

    input_index = input_indexes[0]
    input_indent = len(lines[input_index]) - len(lines[input_index].lstrip())
    options_index = next(
        index
        for index in range(input_index + 1, len(lines))
        if lines[index].strip() == "options:"
        and len(lines[index]) - len(lines[index].lstrip()) > input_indent
    )
    option_indent = lines[options_index][: -len(lines[options_index].lstrip())]
    first_choice_index = options_index + 1
    end_choice_index = first_choice_index
    while end_choice_index < len(lines) and lines[end_choice_index].startswith(
        f"{option_indent}- "
    ):
        end_choice_index += 1
    assert end_choice_index > first_choice_index, (
        f"Expected existing choices for {input_name!r} in {workflow_path}"
    )

    generated_choices = [f"{option_indent}- {choice}\n" for choice in choices]
    workflow_path.write_text(
        "".join(
            lines[:first_choice_index] + generated_choices + lines[end_choice_index:]
        )
    )
    log.info(f"Updated {input_name} choices in {workflow_path}")


def main() -> None:
    workflows_dir = Path(__file__).parents[2] / ".github" / "workflows"
    assert workflows_dir.exists(), (
        f"Workflows directory does not exist: {workflows_dir}"
    )

    update_choice_options(
        workflows_dir / "manual-create-job-from-cronjob.yml",
        "cronjob_name",
        get_all_cronjob_names(),
    )
    update_choice_options(
        workflows_dir / "manual-backfill.yml",
        "dataset_id",
        get_backfill_dataset_ids(),
    )


if __name__ == "__main__":
    main()
