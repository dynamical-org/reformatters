import base64
import itertools
import json
import os
import subprocess
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

import httpx
import typer
from croniter import croniter

from reformatters.common import betterstack, docker, kubernetes, staging
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.logging import get_logger

log = get_logger(__name__)

_UPTIME_API = "https://uptime.betterstack.com/api/v2"

# How late a cron may start before we alert (matches the Sentry checkin_margin we ran with).
_START_GRACE = timedelta(minutes=10)


def deploy_operational_resources(
    datasets: Iterable[DynamicalDataset[Any, Any]],
    docker_image: str | None = None,
    dataset_id_filter: str | None = None,
    cronjob_transform: Callable[[kubernetes.CronJob], kubernetes.CronJob] | None = None,
) -> None:
    image_tag = docker_image or docker.build_and_push_image()

    reformat_jobs: list[kubernetes.Job] = []

    for dataset in datasets:
        if dataset_id_filter is not None and dataset.dataset_id != dataset_id_filter:
            continue

        try:
            dataset_cronjobs = list(dataset.operational_kubernetes_resources(image_tag))
        except NotImplementedError:
            log.info(
                f"Skipping deploy for {dataset.__class__.__name__}, "
                "`operational_kubernetes_resources` not implemented."
            )
            continue

        if cronjob_transform is not None:
            dataset_cronjobs = [cronjob_transform(cj) for cj in dataset_cronjobs]

        reformat_jobs.extend(dataset_cronjobs)

    assert len(reformat_jobs) > 0, "No cronjobs to deploy" + (
        f" for dataset_id_filter={dataset_id_filter!r}" if dataset_id_filter else ""
    )

    _provision_heartbeats(
        cj for cj in reformat_jobs if isinstance(cj, kubernetes.CronJob)
    )

    k8s_resource_list = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            reformat_job.as_kubernetes_object() for reformat_job in reformat_jobs
        ],
    }

    subprocess.run(
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=json.dumps(k8s_resource_list),
        text=True,
        check=True,
    )

    log.info(
        f"Deployed {[item['metadata']['name'] for item in k8s_resource_list['items']]}"  # type: ignore[index]
    )


def _provision_heartbeats(cron_jobs: Iterable[kubernetes.CronJob]) -> None:
    # Only update/validate crons get heartbeats; other crons (e.g. archive) are skipped.
    eligible = [cj for cj in cron_jobs if cj.command[0] in betterstack.HEARTBEAT_STEPS]
    if not eligible:
        log.info(
            "No update/validate cron jobs to provision Better Stack heartbeats for."
        )
        return
    if not os.getenv("BETTERSTACK_API_KEY_RW"):
        raise RuntimeError(
            "BETTERSTACK_API_KEY_RW is required to provision Better Stack "
            "heartbeats before deploying CronJobs."
        )
    url_map = reconcile_heartbeats(eligible)
    write_heartbeat_secret(url_map)


class HeartbeatSpec(NamedTuple):
    key: str
    name: str
    period: timedelta
    grace: timedelta


def schedule_period(schedule: str) -> timedelta:
    """Smallest gap between consecutive cron fire times, i.e. the heartbeat period."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    itr = croniter(schedule, base)
    times = [itr.get_next(datetime) for _ in range(64)]
    return min(b - a for a, b in itertools.pairwise(times))


def heartbeat_specs(cron_job: kubernetes.CronJob) -> list[HeartbeatSpec]:
    """Two heartbeats per cron: a tight-grace start and a run-length-grace complete.

    The start heartbeat detects a no-start within _START_GRACE regardless of run length;
    the complete heartbeat's grace must cover the full run.
    """
    step: betterstack.Step = cron_job.command[0]  # ty: ignore[invalid-assignment]
    assert step in betterstack.HEARTBEAT_STEPS, (
        f"Unexpected cron command: {cron_job.command}"
    )
    prefix = betterstack.cron_name_prefix(cron_job.name, step)
    period = schedule_period(cron_job.schedule)
    return [
        HeartbeatSpec(
            betterstack.heartbeat_key(prefix, step, "start"),
            betterstack.heartbeat_name(prefix, step, "start"),
            period,
            _START_GRACE,
        ),
        HeartbeatSpec(
            betterstack.heartbeat_key(prefix, step, "complete"),
            betterstack.heartbeat_name(prefix, step, "complete"),
            period,
            _START_GRACE + cron_job.pod_active_deadline,
        ),
    ]


def _api_client(token: str) -> httpx.Client:
    return httpx.Client(
        base_url=_UPTIME_API,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )


def _list_heartbeats(client: httpx.Client) -> dict[str, dict[str, Any]]:
    """name -> heartbeat attributes (including id and url), across all pages."""
    by_name: dict[str, dict[str, Any]] = {}
    url: str | None = "/heartbeats"
    while url:
        response = client.get(url)
        response.raise_for_status()
        body = response.json()
        for item in body["data"]:
            attributes = item["attributes"]
            by_name[attributes["name"]] = {"id": item["id"], **attributes}
        url = body.get("pagination", {}).get("next")
    return by_name


def _upsert_heartbeat(
    client: httpx.Client, existing: dict[str, dict[str, Any]], spec: HeartbeatSpec
) -> str:
    payload = {
        "name": spec.name,
        "period": int(spec.period.total_seconds()),
        "grace": int(spec.grace.total_seconds()),
    }
    current = existing.get(spec.name)
    if current is None:
        response = client.post("/heartbeats", json=payload)
        response.raise_for_status()
        log.info(f"Created heartbeat {spec.name}")
        body = response.json()["data"]
        attributes = body["attributes"]
        existing[spec.name] = {"id": body["id"], **attributes}
        return attributes["url"]

    if current["period"] != payload["period"] or current["grace"] != payload["grace"]:
        response = client.patch(f"/heartbeats/{current['id']}", json=payload)
        response.raise_for_status()
        current.update(payload)
        log.info(f"Updated heartbeat {spec.name}")
    return current["url"]


def reconcile_heartbeats(cron_jobs: Iterable[kubernetes.CronJob]) -> dict[str, str]:
    """Idempotently create/update one start + one complete heartbeat per cron job.

    Returns the {dataset_id}_{step}_{role} -> url map. Requires the
    BETTERSTACK_API_KEY_RW env var (operator-supplied at deploy).
    """
    token = os.environ["BETTERSTACK_API_KEY_RW"]

    url_map: dict[str, str] = {}
    with _api_client(token) as client:
        existing = _list_heartbeats(client)
        for cron_job in cron_jobs:
            for spec in heartbeat_specs(cron_job):
                url_map[spec.key] = _upsert_heartbeat(client, existing, spec)
    return url_map


def _load_existing_heartbeat_secret() -> dict[str, str]:
    result = subprocess.run(  # noqa: S603
        [
            "/usr/bin/kubectl",
            "get",
            "secret",
            kubernetes.BETTERSTACK_HEARTBEATS_SECRET_NAME,
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if "not found" in result.stderr.lower():
            return {}
        result.check_returncode()

    body = json.loads(result.stdout)
    encoded_contents = body.get("data", {}).get("contents")
    if encoded_contents is None:
        return {}
    contents = json.loads(base64.b64decode(encoded_contents).decode())
    assert isinstance(contents, dict)
    return contents


def _apply_heartbeat_secret(url_map: dict[str, str]) -> None:
    manifest = subprocess.run(  # noqa: S603
        [
            "/usr/bin/kubectl",
            "create",
            "secret",
            "generic",
            kubernetes.BETTERSTACK_HEARTBEATS_SECRET_NAME,
            f"--from-literal=contents={json.dumps(url_map)}",
            "--dry-run=client",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    subprocess.run(
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=manifest,
        text=True,
        check=True,
    )
    log.info(
        f"Wrote {kubernetes.BETTERSTACK_HEARTBEATS_SECRET_NAME} secret with {len(url_map)} heartbeat urls"
    )


def write_heartbeat_secret(url_map: dict[str, str]) -> None:
    """Persist the heartbeat URL map into the k8s secret the cron pods load at runtime."""
    _apply_heartbeat_secret(_load_existing_heartbeat_secret() | url_map)


def delete_staging_heartbeats(dataset_id: str, version: str) -> None:
    """Delete a staging version's heartbeats from Better Stack and the k8s secret."""
    token = os.environ["BETTERSTACK_API_KEY_RW"]

    remove_keys: list[str] = []
    remove_names: list[str] = []
    for step in betterstack.HEARTBEAT_STEPS:
        cron_name = staging.staging_cronjob_name(dataset_id, version, step)
        prefix = betterstack.cron_name_prefix(cron_name, step)
        for role in betterstack.HEARTBEAT_ROLES:
            remove_keys.append(betterstack.heartbeat_key(prefix, step, role))
            remove_names.append(betterstack.heartbeat_name(prefix, step, role))

    with _api_client(token) as client:
        existing = _list_heartbeats(client)
        for name in remove_names:
            heartbeat = existing.get(name)
            if heartbeat is None:
                continue
            response = client.delete(f"/heartbeats/{heartbeat['id']}")
            response.raise_for_status()
            log.info(f"Deleted heartbeat {name}")

    remove_key_set = set(remove_keys)
    remaining = {
        key: url
        for key, url in _load_existing_heartbeat_secret().items()
        if key not in remove_key_set
    }
    _apply_heartbeat_secret(remaining)


def register_commands(
    app: typer.Typer, datasets: Sequence[DynamicalDataset[Any, Any]]
) -> None:
    @app.command()
    def deploy(
        docker_image: str | None = None,
    ) -> None:
        deploy_operational_resources(datasets, docker_image)

    @app.command()
    def deploy_staging(
        dataset_id: str,
        version: str,
        docker_image: str,
    ) -> None:
        """Deploy staging cronjobs for a single dataset version."""
        dataset = staging.find_dataset(datasets, dataset_id)
        staging.validate_version_matches_template(dataset, version)
        staging.validate_version_differs_from_main(dataset, version)

        def transform(cronjob: kubernetes.CronJob) -> kubernetes.CronJob:
            return staging.rename_cronjob_for_staging(cronjob, dataset_id, version)

        deploy_operational_resources(
            datasets,
            docker_image=docker_image,
            dataset_id_filter=dataset_id,
            cronjob_transform=transform,
        )

    @app.command()
    def cleanup_staging(
        dataset_id: str,
        version: str,
        force: bool = False,
    ) -> None:
        """Clean up staging resources: Better Stack heartbeats, kubernetes cronjobs and git branch."""
        staging.find_dataset(datasets, dataset_id)  # validate dataset_id
        if not force:
            cronjob_names = staging.staging_cronjob_names(dataset_id, version)
            branch = staging.staging_branch_name(dataset_id, version)
            log.info(
                f"Will delete heartbeats, cronjobs {cronjob_names} and branch {branch}. "
                "Run with --force to execute."
            )
            return
        delete_staging_heartbeats(dataset_id, version)
        staging.cleanup_staging_resources(dataset_id, version)
