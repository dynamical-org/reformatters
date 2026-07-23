import base64
import itertools
import json
import logging
import os
import subprocess
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, NamedTuple, cast

import httpx
from croniter import croniter
from logtail import LogtailHandler

from reformatters.common.kubernetes import (
    BETTERSTACK_HEARTBEATS_SECRET_NAME,
    CronJob,
    load_secret,
)
from reformatters.common.logging import get_logger

log = get_logger(__name__)

# Every reformatters logger is a child of this one (get_logger returns a child of
# the root named "reformatters.<module>"), so attaching here streams our records
# to Better Stack without the chatter other libraries log to the root.
REFORMATTERS_LOGGER_NAME = "reformatters"

# Kubernetes context injected as env vars by common/kubernetes.py, emitted as
# structured fields so Better Stack surfaces them as queryable tags. cron_job_name
# is not derivable from pod metadata, so it must travel with the log.
_CONTEXT_FIELD_ENV_VARS = {
    "cron_job_name": "CRON_JOB_NAME",
    "job_name": "JOB_NAME",
    "pod_name": "POD_NAME",
    "env": "DYNAMICAL_ENV",
}


class _ContextFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._context = {
            field: value
            for field, env_var in _CONTEXT_FIELD_ENV_VARS.items()
            if (value := os.getenv(env_var)) is not None
        }

    def filter(self, record: logging.LogRecord) -> bool:
        for field, value in self._context.items():
            setattr(record, field, value)
        return True


def attach_logtail() -> None:
    """Stream reformatters logs to the Better Stack source when the source token
    and ingesting host are set. Idempotent; a no-op when they are absent (local
    runs) so the only way logs stop reaching Better Stack is unsetting the env."""
    token = os.getenv("BETTERSTACK_SOURCE_TOKEN")
    host = os.getenv("BETTERSTACK_INGESTING_HOST")
    if not (token and host):
        return

    logger = logging.getLogger(REFORMATTERS_LOGGER_NAME)
    if any(isinstance(handler, LogtailHandler) for handler in logger.handlers):
        return

    handler = LogtailHandler(source_token=token, host=f"https://{host}")
    handler.setLevel(logging.INFO)
    handler.addFilter(_ContextFilter())
    logger.addHandler(handler)


# --- Heartbeats -------------------------------------------------------------
# Each update/validate cron gets two heartbeats: a "start" ping when a run begins
# and a "complete" ping when it finishes. The runtime ping is here; deploy-time
# provisioning (period/grace derived from the cron schedule + deadline) is below.

Step = Literal["update", "validate"]
Role = Literal["start", "complete"]

HEARTBEAT_STEPS: tuple[Step, ...] = ("update", "validate")
HEARTBEAT_ROLES: tuple[Role, ...] = ("start", "complete")


def cron_name_prefix(cron_name: str, step: Step) -> str:
    """The cron name without its `-{step}` suffix; equals dataset_id for prod crons
    and the staging-prefixed name for staging crons, so each gets its own heartbeats."""
    return cron_name.removesuffix(f"-{step}")


def heartbeat_key(name_prefix: str, step: Step, role: Role) -> str:
    return f"{name_prefix}_{step}_{role}"


def heartbeat_name(name_prefix: str, step: Step, role: Role) -> str:
    return f"reformatters {name_prefix} {step} {role}"


def ping(url: str, *, failed: bool = False) -> None:
    response = httpx.post(f"{url}/fail" if failed else url, timeout=10)
    response.raise_for_status()


def load_heartbeat_urls() -> dict[str, str]:
    """{name_prefix}_{step}_{role} -> heartbeat url. Empty outside prod."""
    return {
        key: str(url)
        for key, url in load_secret(BETTERSTACK_HEARTBEATS_SECRET_NAME).items()
    }


def _safe_ping(url: str, *, failed: bool = False) -> None:
    # Monitoring must never break a run; a dropped ping only risks a false alert.
    try:
        ping(url, failed=failed)
    except httpx.HTTPError:
        log.warning("Better Stack heartbeat ping failed", exc_info=True)


@contextmanager
def monitor_heartbeat(
    cron_name: str,
    step: Step,
    *,
    send_start: bool = True,
    send_result: bool = True,
) -> Iterator[None]:
    """Ping this cron's Better Stack heartbeats around the wrapped run: a start ping
    when it begins and a complete ping when it finishes (`/fail` on error). A no-op
    for heartbeats missing from the url map (non-prod, or a not-yet-provisioned cron)."""
    urls = load_heartbeat_urls()
    prefix = cron_name_prefix(cron_name, step)
    start_url = urls.get(heartbeat_key(prefix, step, "start"))
    complete_url = urls.get(heartbeat_key(prefix, step, "complete"))

    if send_start and start_url is not None:
        _safe_ping(start_url)
    try:
        yield
    except Exception:
        if send_result and complete_url is not None:
            _safe_ping(complete_url, failed=True)
        raise
    else:
        if send_result and complete_url is not None:
            _safe_ping(complete_url)


@contextmanager
def monitor_cron_run(
    cron_job: CronJob,
    reformat_job_name: str,  # noqa: ARG001  # RunMonitor interface; heartbeats key off the cron, not the job
    *,
    send_in_progress: bool = True,
    send_result: bool = True,
) -> Iterator[None]:
    """A `DynamicalDataset` RunMonitor that pings this cron's Better Stack heartbeats.

    A no-op for crons without an update/validate step (e.g. archive). The cron name
    comes from the CRON_JOB_NAME env when set (staging-aware), else the cron's own name.
    """
    command = cron_job.command[0]
    if command not in HEARTBEAT_STEPS:
        yield
        return
    step = cast("Step", command)
    cron_name = os.getenv("CRON_JOB_NAME") or cron_job.name
    with monitor_heartbeat(
        cron_name, step, send_start=send_in_progress, send_result=send_result
    ):
        yield


# --- Deploy-time heartbeat provisioning -------------------------------------
# Called from common/deploy.py at deploy time. Reconciles one start + one complete
# uptime heartbeat per update/validate cron via the Better Stack Uptime API, and
# writes the {key -> url} map into the k8s secret the cron pods load at runtime.

_UPTIME_API = "https://uptime.betterstack.com/api/v2"

# How late a cron may start before we alert.
_START_GRACE = timedelta(minutes=10)


class HeartbeatSpec(NamedTuple):
    key: str
    name: str
    period: timedelta
    grace: timedelta


def provision_heartbeats(cron_jobs: Iterable[CronJob]) -> None:
    """Reconcile heartbeats for the update/validate crons and persist their url map.

    Only update/validate crons get heartbeats; other crons (e.g. archive) are skipped.
    Requires the BETTERSTACK_API_KEY_RW env var when there is anything to provision.
    """
    eligible = [cj for cj in cron_jobs if cj.command[0] in HEARTBEAT_STEPS]
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
    write_heartbeat_secret(reconcile_heartbeats(eligible))


def schedule_period(schedule: str) -> timedelta:
    """Largest gap between consecutive cron fire times. Using the largest (not the
    typical) gap as the heartbeat period keeps an irregular schedule's long quiet
    stretches from expiring the heartbeat and firing a false incident."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    itr = croniter(schedule, base)
    times = [itr.get_next(datetime) for _ in range(64)]
    return max(b - a for a, b in itertools.pairwise(times))


def heartbeat_specs(cron_job: CronJob) -> list[HeartbeatSpec]:
    """Two heartbeats per cron: a tight-grace start and a run-length-grace complete.

    The start heartbeat detects a no-start within _START_GRACE regardless of run
    length; the complete heartbeat's grace must cover the full run.
    """
    step = cast("Step", cron_job.command[0])
    assert step in HEARTBEAT_STEPS, f"Unexpected cron command: {cron_job.command}"
    prefix = cron_name_prefix(cron_job.name, step)
    period = schedule_period(cron_job.schedule)
    return [
        HeartbeatSpec(
            heartbeat_key(prefix, step, "start"),
            heartbeat_name(prefix, step, "start"),
            period,
            _START_GRACE,
        ),
        HeartbeatSpec(
            heartbeat_key(prefix, step, "complete"),
            heartbeat_name(prefix, step, "complete"),
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
        body = response.json()["data"]
        existing[spec.name] = {"id": body["id"], **body["attributes"]}
        log.info(f"Created heartbeat {spec.name}")
        return str(body["attributes"]["url"])

    if current["period"] != payload["period"] or current["grace"] != payload["grace"]:
        response = client.patch(f"/heartbeats/{current['id']}", json=payload)
        response.raise_for_status()
        current.update(payload)
        log.info(f"Updated heartbeat {spec.name}")
    return str(current["url"])


def reconcile_heartbeats(cron_jobs: Iterable[CronJob]) -> dict[str, str]:
    """Idempotently create/update one start + one complete heartbeat per cron job.

    Returns the {name_prefix}_{step}_{role} -> url map. Requires the
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

    encoded = json.loads(result.stdout).get("data", {}).get("contents")
    if encoded is None:
        return {}
    contents = json.loads(base64.b64decode(encoded).decode())
    assert isinstance(contents, dict)
    return contents


def _apply_heartbeat_secret(url_map: dict[str, str]) -> None:
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
        f"Wrote {BETTERSTACK_HEARTBEATS_SECRET_NAME} secret "
        f"with {len(url_map)} heartbeat urls"
    )


def write_heartbeat_secret(url_map: dict[str, str]) -> None:
    """Merge into and persist the heartbeat URL map the cron pods load at runtime.

    Merges so a single-dataset (filtered) deploy doesn't drop other datasets' urls.
    """
    _apply_heartbeat_secret(_load_existing_heartbeat_secret() | url_map)


def delete_staging_heartbeats(dataset_id: str, version: str) -> None:
    """Delete a staging version's heartbeats from Better Stack and the k8s secret."""
    from reformatters.common import staging  # noqa: PLC0415

    token = os.environ["BETTERSTACK_API_KEY_RW"]
    remove_keys: set[str] = set()
    remove_names: set[str] = set()
    for step in HEARTBEAT_STEPS:
        prefix = cron_name_prefix(
            staging.staging_cronjob_name(dataset_id, version, step), step
        )
        for role in HEARTBEAT_ROLES:
            remove_keys.add(heartbeat_key(prefix, step, role))
            remove_names.add(heartbeat_name(prefix, step, role))

    with _api_client(token) as client:
        existing = _list_heartbeats(client)
        for name in remove_names:
            heartbeat = existing.get(name)
            if heartbeat is None:
                continue
            client.delete(f"/heartbeats/{heartbeat['id']}").raise_for_status()
            log.info(f"Deleted heartbeat {name}")

    remaining = {
        key: url
        for key, url in _load_existing_heartbeat_secret().items()
        if key not in remove_keys
    }
    _apply_heartbeat_secret(remaining)
