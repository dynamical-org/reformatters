import json
import subprocess
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import replace

log = get_logger(__name__)

_MAX_KUBERNETES_NAME_LENGTH = 63


def staging_cronjob_name(dataset_id: str, version: str, suffix: str) -> str:
    version_dashes = version.replace(".", "-")
    return f"staging-{dataset_id}-v{version_dashes}-{suffix}"


def rename_cronjob_for_staging(
    cronjob: CronJob, dataset_id: str, version: str
) -> CronJob:
    """Create a copy of a CronJob with a staging-prefixed name."""
    # Cronjob names follow the pattern {dataset_id}-{suffix} (e.g. "noaa-gfs-forecast-update")
    assert cronjob.name.startswith(f"{dataset_id}-"), (
        f"Unexpected cronjob name {cronjob.name!r}, expected prefix {dataset_id!r}-"
    )
    suffix = cronjob.name.removeprefix(f"{dataset_id}-")
    new_name = staging_cronjob_name(dataset_id, version, suffix)
    assert len(new_name) <= _MAX_KUBERNETES_NAME_LENGTH, (
        f"Staging cronjob name {new_name!r} exceeds {_MAX_KUBERNETES_NAME_LENGTH} char kubernetes limit"
    )
    return replace(cronjob, name=new_name)


def find_dataset(
    datasets: Sequence[DynamicalDataset[Any, Any]], dataset_id: str
) -> DynamicalDataset[Any, Any]:
    matches = [d for d in datasets if d.dataset_id == dataset_id]
    assert len(matches) == 1, (
        f"Dataset {dataset_id!r} not found. Available: {[d.dataset_id for d in datasets]}"
    )
    return matches[0]


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
    """Return the expected staging cronjob names for cleanup."""
    return [
        staging_cronjob_name(dataset_id, version, "update"),
        staging_cronjob_name(dataset_id, version, "validate"),
    ]


def staging_branch_name(dataset_id: str, version: str) -> str:
    return f"stage/{dataset_id}/v{version}"


_SENTRY_ORG = "dynamical"
_SENTRY_API_BASE = "https://sentry.io/api/0"


def _delete_sentry_monitor(slug: str, auth_token: str) -> None:
    url = f"{_SENTRY_API_BASE}/organizations/{_SENTRY_ORG}/monitors/{slug}/"
    request = urllib.request.Request(url, method="DELETE")  # noqa: S310
    request.add_header("Authorization", f"Bearer {auth_token}")
    try:
        urllib.request.urlopen(request)  # noqa: S310
        log.info(f"Deleted Sentry monitor {slug}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            log.info(f"Sentry monitor {slug} not found, skipping")
        else:
            raise


def cleanup_staging_resources(
    dataset_id: str,
    version: str,
    sentry_auth_token: str | None,
) -> None:
    cronjob_names = staging_cronjob_names(dataset_id, version)
    branch = staging_branch_name(dataset_id, version)

    # Delete kubernetes cronjobs
    log.info(f"Deleting kubernetes cronjobs: {cronjob_names}")
    subprocess.run(  # noqa: S603
        ["/usr/bin/kubectl", "delete", "cronjob", *cronjob_names, "--ignore-not-found"],
        check=True,
    )

    # Delete Sentry monitors
    if sentry_auth_token:
        for name in cronjob_names:
            _delete_sentry_monitor(name, sentry_auth_token)
    else:
        log.info(
            "No --sentry-auth-token provided, skipping Sentry monitor cleanup. "
            f"Manually delete monitors: {cronjob_names}"
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
        "Cleanup complete. Dataset store was NOT deleted — clean up manually when ready."
    )
