import itertools
import json
import os
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, NamedTuple

import httpx
from croniter import croniter

from reformatters.common.kubernetes import (
    BETTERSTACK_HEARTBEATS_SECRET_NAME,
    CronJob,
    ValidationCronJob,
    load_secret,
)
from reformatters.common.logging import get_logger

log = get_logger(__name__)

_UPTIME_API = "https://uptime.betterstack.com/api/v2"

# Matches the Sentry checkin_margin we ran with: how late a cron may start before we alert.
_START_GRACE = timedelta(minutes=10)

HeartbeatRole = Literal["start", "complete"]


class HeartbeatSpec(NamedTuple):
    role: HeartbeatRole
    name: str
    period: timedelta
    grace: timedelta


def heartbeat_names(cron_name: str) -> dict[HeartbeatRole, str]:
    return {"start": f"{cron_name}.start", "complete": f"{cron_name}.complete"}


def schedule_period(schedule: str) -> timedelta:
    """Smallest gap between consecutive cron fire times, i.e. the heartbeat period."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    itr = croniter(schedule, base)
    times = [itr.get_next(datetime) for _ in range(64)]
    return min(b - a for a, b in itertools.pairwise(times))


def heartbeat_specs(cron_job: CronJob) -> list[HeartbeatSpec]:
    """Two heartbeats per cron: a tight-grace start and a run-length-grace complete.

    The start heartbeat detects a no-start within _START_GRACE regardless of run length;
    the complete heartbeat's grace must cover the full run (see plans/673_betterstack.md).
    """
    period = schedule_period(cron_job.schedule)
    names = heartbeat_names(cron_job.name)
    return [
        HeartbeatSpec("start", names["start"], period, _START_GRACE),
        HeartbeatSpec(
            "complete",
            names["complete"],
            period,
            _START_GRACE + cron_job.pod_active_deadline,
        ),
    ]


def ping(url: str, *, failed: bool = False) -> None:
    response = httpx.post(f"{url}/fail" if failed else url, timeout=10)
    response.raise_for_status()


def load_heartbeat_urls() -> dict[str, dict[str, str]]:
    """cron_name -> {"start": url, "complete": url}. Empty outside prod."""
    return load_secret(BETTERSTACK_HEARTBEATS_SECRET_NAME)


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
        return response.json()["data"]["attributes"]["url"]

    if current["period"] != payload["period"] or current["grace"] != payload["grace"]:
        response = client.patch(f"/heartbeats/{current['id']}", json=payload)
        response.raise_for_status()
        log.info(f"Updated heartbeat {spec.name}")
    return current["url"]


def reconcile_heartbeats(cron_jobs: Iterable[CronJob]) -> dict[str, dict[str, str]]:
    """Idempotently create/update one start + one complete heartbeat per cron job.

    Returns the cron_name -> {"start": url, "complete": url} map. Requires the
    BETTERSTACK_API_TOKEN_RW env var (operator-supplied at deploy).
    """
    token = os.environ["BETTERSTACK_API_TOKEN_RW"]
    status_page_id = os.getenv("DYNAMICAL_BETTERSTACK_STATUS_PAGE_ID")

    url_map: dict[str, dict[str, str]] = {}
    with _api_client(token) as client:
        existing = _list_heartbeats(client)
        for cron_job in cron_jobs:
            urls: dict[str, str] = {}
            for spec in heartbeat_specs(cron_job):
                urls[spec.role] = _upsert_heartbeat(client, existing, spec)
            url_map[cron_job.name] = urls

            # The validation completing is the public freshness signal for status.dynamical.org.
            if status_page_id and isinstance(cron_job, ValidationCronJob):
                _attach_to_status_page(
                    client,
                    status_page_id,
                    existing,
                    heartbeat_names(cron_job.name)["complete"],
                )

    return url_map


def _attach_to_status_page(
    client: httpx.Client,
    status_page_id: str,
    existing: dict[str, dict[str, Any]],
    heartbeat_name: str,
) -> None:
    heartbeat = existing[heartbeat_name]
    response = client.post(
        f"/status-pages/{status_page_id}/resources",
        json={
            "resource_id": int(heartbeat["id"]),
            "resource_type": "Heartbeat",
            "public_name": heartbeat_name.removesuffix(".complete"),
        },
    )
    # 422 = already attached; the status page resource set is idempotent on our side.
    if response.status_code != httpx.codes.UNPROCESSABLE_ENTITY:
        response.raise_for_status()


def write_heartbeat_secret(url_map: dict[str, dict[str, str]]) -> None:
    """Persist the heartbeat URL map into the k8s secret the cron pods load at runtime."""
    manifest = subprocess.run(  # noqa: S603
        [
            "/usr/bin/kubectl",
            "create",
            "secret",
            "generic",
            BETTERSTACK_HEARTBEATS_SECRET_NAME,
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
        f"Wrote {BETTERSTACK_HEARTBEATS_SECRET_NAME} secret with {len(url_map)} cron jobs"
    )
