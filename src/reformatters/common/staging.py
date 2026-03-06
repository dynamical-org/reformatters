import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.iterating import item
from reformatters.common.kubernetes import CronJob
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import replace

log = get_logger(__name__)

# 63 is the kubernetes DNS label limit. We use 52 to leave room for the
# CronJob controller's job name suffix (dash + up to 10-digit unix-minutes timestamp).
_MAX_KUBERNETES_NAME_LENGTH = 52


def staging_cronjob_name(dataset_id: str, version: str, suffix: str) -> str:
    """Build a staging cronjob name, trimming dataset_id if needed to fit the kubernetes limit."""
    version_dashes = version.replace(".", "-")
    max_id_len = _MAX_KUBERNETES_NAME_LENGTH - len(f"stage--v{version_dashes}-{suffix}")
    assert max_id_len > 0, (
        "Version and suffix are too long to fit in kubernetes name limit"
    )
    return f"stage-{dataset_id[:max_id_len]}-v{version_dashes}-{suffix}"


def rename_cronjob_for_staging(
    cronjob: CronJob, dataset_id: str, version: str
) -> CronJob:
    """Create a copy of a CronJob with a staging-prefixed name."""
    assert cronjob.name.startswith(f"{dataset_id}-")
    suffix = cronjob.name.removeprefix(f"{dataset_id}-")
    new_name = staging_cronjob_name(dataset_id, version, suffix)
    return replace(cronjob, name=new_name)


def find_dataset(
    datasets: Sequence[DynamicalDataset[Any, Any]], dataset_id: str
) -> DynamicalDataset[Any, Any]:
    return item(d for d in datasets if d.dataset_id == dataset_id)


def validate_version_matches_template(
    dataset: DynamicalDataset[Any, Any], version: str
) -> None:
    template_version = dataset.template_config.version
    assert template_version == version, (
        f"Version {version!r} does not match template config version {template_version!r}"
    )


def get_main_version(dataset: DynamicalDataset[Any, Any]) -> str:
    """Read the dataset version from main branch's template zarr.json via git show."""
    template_path = dataset.template_config.template_path()
    # Convert absolute path to repo-relative path
    repo_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    relative_zarr_json = template_path.relative_to(repo_root) / "zarr.json"

    result = subprocess.run(  # noqa: S603
        ["git", "show", f"origin/main:{relative_zarr_json}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Failed to read {relative_zarr_json} from origin/main: {result.stderr}"
    )
    zarr_metadata = json.loads(result.stdout)
    main_version: str = zarr_metadata["attributes"]["dataset_version"]
    return main_version


def validate_version_differs_from_main(
    dataset: DynamicalDataset[Any, Any], version: str
) -> None:
    main_version = get_main_version(dataset)
    assert version != main_version, (
        f"Staging version {version!r} is the same as the version on main. "
        "Staging would conflict with production cronjobs."
    )


def staging_cronjob_names(dataset_id: str, version: str) -> list[str]:
    return [
        staging_cronjob_name(dataset_id, version, "update"),
        staging_cronjob_name(dataset_id, version, "validate"),
    ]


def staging_branch_name(dataset_id: str, version: str) -> str:
    return f"stage/{dataset_id}/v{version}"


def cleanup_staging_resources(
    dataset_id: str,
    version: str,
) -> None:
    cronjob_names = staging_cronjob_names(dataset_id, version)
    branch = staging_branch_name(dataset_id, version)

    # Delete kubernetes cronjobs
    log.info(f"Deleting kubernetes cronjobs: {cronjob_names}")
    subprocess.run(  # noqa: S603
        ["/usr/bin/kubectl", "delete", "cronjob", *cronjob_names, "--ignore-not-found"],
        check=True,
    )

    # Delete git branch
    log.info(f"Deleting remote branch {branch}")
    result = subprocess.run(  # noqa: S603
        ["git", "push", "origin", "--delete", branch],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        log.info(f"Deleted remote branch {branch}")
    else:
        log.info(f"Could not delete branch {branch}: {result.stderr.strip()}")

    log.info(
        "Cleanup complete. Dataset store and Sentry cron monitors were NOT deleted. "
        f"Manually delete Sentry monitors: {cronjob_names}"
    )
