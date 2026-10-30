import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from reformatters.common.kubernetes import CronJob
from reformatters.common.logging import get_logger

from .sources import write_atomic
from .statistics import arrival_stats

log = get_logger(__name__)


@dataclass(frozen=True)
class RuntimeStats:
    startup_estimate: float | None
    processing: float | None
    validation_p95: float | None
    validation_max: float | None
    scheduled_count: int
    validation_count: int
    non_ok_count: int
    off_schedule_count: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    startup_p50: float | None = None
    processing_p50: float | None = None
    validation_p50: float | None = None


def read_checkins(directory: Path | None, monitor: str) -> list[dict[str, Any]]:
    if directory is None or not (directory / f"{monitor}.json").exists():
        return []
    payload = json.loads((directory / f"{monitor}.json").read_text())
    assert isinstance(payload, list), (
        "Check-in exports must be JSON arrays, one per monitor"
    )
    return payload


def measure_runtime(
    directory: Path | None,
    cron: CronJob,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    virtual: bool,
) -> RuntimeStats:
    updates = read_checkins(directory, cron.name)
    validations = read_checkins(
        directory, cron.name.removesuffix("-update") + "-validate"
    )
    lags, processing, validation = [], [], []
    update_ends = []
    validation_errors = [
        pd.Timestamp(checkin["dateUpdated"]).tz_convert(None)
        for checkin in validations
        if checkin["status"] == "error"
    ]
    observed = []
    non_ok = off_schedule = 0
    for checkin in updates:
        timestamp = pd.Timestamp(checkin["dateAdded"]).tz_convert(None)
        if not start <= timestamp < end or checkin.get("environment") != "prod":
            continue
        observed.append(timestamp)
        if checkin["status"] in {"ok", "error"} and checkin.get("duration") is not None:
            update_ends.append(pd.Timestamp(checkin["dateUpdated"]).tz_convert(None))
        if checkin["status"] != "ok":
            non_ok += 1
            continue
        began = checkin.get("dateInProgress")
        if began is None or checkin.get("duration") is None:
            continue
        began = pd.Timestamp(began).tz_convert(None)
        schedule = checkin["monitorConfig"]["schedule"]
        if isinstance(schedule, dict):
            schedule = schedule["value"]
        historical = cron.model_copy(update={"schedule": schedule})
        fire = historical.previous_fire_time(began)
        lag = (began - fire).total_seconds()
        is_retry = lag > 120 and any(
            0 <= (began - failed).total_seconds() <= 300 for failed in validation_errors
        )
        if lag >= 300 or is_retry:
            off_schedule += 1
            continue
        lags.append(lag)
        processing.append(checkin["duration"] / 1000)
    for checkin in validations:
        began = checkin.get("dateInProgress")
        if (
            began is None
            or checkin["status"] not in {"ok", "error"}
            or checkin.get("duration") is None
            or checkin.get("environment") != "prod"
        ):
            continue
        began = pd.Timestamp(began).tz_convert(None)
        if start <= began < end and any(
            0 <= (began - closed).total_seconds() <= 120 for closed in update_ends
        ):
            validation.append(checkin["duration"] / 1000)
    return RuntimeStats(
        startup_estimate=arrival_stats(lags).p95 if lags else None,
        processing=arrival_stats(processing).p95
        if processing and not virtual
        else None,
        validation_p95=arrival_stats(validation).p95 if validation else None,
        validation_max=max(validation) if validation else None,
        scheduled_count=len(lags),
        validation_count=len(validation),
        non_ok_count=non_ok,
        off_schedule_count=off_schedule,
        first=min(observed) if observed else None,
        last=max(observed) if observed else None,
        startup_p50=arrival_stats(lags).p50 if lags else None,
        processing_p50=arrival_stats(processing).p50
        if processing and not virtual
        else None,
        validation_p50=arrival_stats(validation).p50 if validation else None,
    )


def fetch_checkins(
    directory: Path, monitor: str, org: str, start: pd.Timestamp
) -> None:
    path = directory / f"{monitor}.json"
    if path.exists():
        return
    token = os.environ.get("SENTRY_AUTH_TOKEN")
    assert token, (
        "Set SENTRY_AUTH_TOKEN for live Sentry measurements, or use --checkins-dir"
    )
    checkins = []
    url = f"https://sentry.io/api/0/organizations/{org}/monitors/{monitor}/checkins/?per_page=100"
    with httpx.Client(
        headers={"Authorization": f"Bearer {token}"}, timeout=60
    ) as client:
        while url:
            response = checkins_page(client, url)
            if response.status_code == 404:
                break
            response.raise_for_status()
            page = response.json()
            assert isinstance(page, list)
            checkins.extend(page)
            if (
                not page
                or min(
                    pd.Timestamp(item["dateAdded"]).tz_convert(None) for item in page
                )
                < start
            ):
                break
            next_link = response.links.get("next", {})
            url = next_link.get("url", "") if next_link.get("results") == "true" else ""
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(checkins))


def checkins_page(client: httpx.Client, url: str) -> httpx.Response:
    for _ in range(6):
        time.sleep(1)
        response = client.get(url)
        if response.status_code != 429:
            return response
        wait = min(60, float(response.headers.get("Retry-After", "60")))
        log.info("Sentry rate limited; waiting %.0f seconds", wait)
        time.sleep(wait)
    response.raise_for_status()
    raise AssertionError("unreachable")
