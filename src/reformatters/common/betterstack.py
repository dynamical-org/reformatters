import base64
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
    load_secret,
)
from reformatters.common.logging import get_logger

log = get_logger(__name__)

_UPTIME_API = "https://uptime.betterstack.com/api/v2"

# How late a cron may start before we alert (matches the Sentry checkin_margin we ran with).
_START_GRACE = timedelta(minutes=10)

Step = Literal["update", "validate"]
Role = Literal["start", "complete"]


class HeartbeatSpec(NamedTuple):
    key: str
    name: str
    period: timedelta
    grace: timedelta


def heartbeat_key(dataset_id: str, step: Step, role: Role) -> str:
    return f"{dataset_id}_{step}_{role}"


def heartbeat_name(dataset_id: str, step: Step, role: Role) -> str:
    return f"reformatters {dataset_id} {step} {role}"


def schedule_period(schedule: str) -> timedelta:
    """Smallest gap between consecutive cron fire times, i.e. the heartbeat period."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    itr = croniter(schedule, base)
    times = [itr.get_next(datetime) for _ in range(64)]
    return min(b - a for a, b in itertools.pairwise(times))


def heartbeat_specs(cron_job: CronJob) -> list[HeartbeatSpec]:
    """Two heartbeats per cron: a tight-grace start and a run-length-grace complete.

    The start heartbeat detects a no-start within _START_GRACE regardless of run length;
    the complete heartbeat's grace must cover the full run.
    """
    step: Step = cron_job.command[0]  # ty: ignore[invalid-assignment]
    assert step in ("update", "validate"), (
        f"Unexpected cron command: {cron_job.command}"
    )
    period = schedule_period(cron_job.schedule)
    return [
        HeartbeatSpec(
            heartbeat_key(cron_job.dataset_id, step, "start"),
            heartbeat_name(cron_job.dataset_id, step, "start"),
            period,
            _START_GRACE,
        ),
        HeartbeatSpec(
            heartbeat_key(cron_job.dataset_id, step, "complete"),
            heartbeat_name(cron_job.dataset_id, step, "complete"),
            period,
            _START_GRACE + cron_job.pod_active_deadline,
        ),
    ]


def ping(url: str, *, failed: bool = False) -> None:
    response = httpx.post(f"{url}/fail" if failed else url, timeout=10)
    response.raise_for_status()


def load_heartbeat_urls() -> dict[str, str]:
    """{dataset_id}_{step}_{role} -> url. Empty outside prod."""
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


def reconcile_heartbeats(cron_jobs: Iterable[CronJob]) -> dict[str, str]:
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
            BETTERSTACK_HEARTBEATS_SECRET_NAME,
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


def write_heartbeat_secret(url_map: dict[str, str]) -> None:
    """Persist the heartbeat URL map into the k8s secret the cron pods load at runtime."""
    merged_url_map = _load_existing_heartbeat_secret() | url_map
    manifest = subprocess.run(  # noqa: S603
        [
            "/usr/bin/kubectl",
            "create",
            "secret",
            "generic",
            BETTERSTACK_HEARTBEATS_SECRET_NAME,
            f"--from-literal=contents={json.dumps(merged_url_map)}",
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
        f"Wrote {BETTERSTACK_HEARTBEATS_SECRET_NAME} secret with {len(merged_url_map)} heartbeat urls"
    )
