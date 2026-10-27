import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from reformatters.common.kubernetes import ReformatCronJob
from scripts.timing_profile.runtime import (
    checkins_page,
    fetch_checkins,
    measure_runtime,
)


@pytest.fixture
def cron() -> ReformatCronJob:
    return ReformatCronJob(
        name="test-update",
        schedule="0 * * * *",
        image="test",
        dataset_id="test",
        cpu="1",
        memory="1G",
    )


def checkin(
    began: str,
    ended: str,
    duration: int,
    *,
    status: str = "ok",
    schedule: str = "10 * * * *",
) -> dict[str, Any]:
    return {
        "dateAdded": began,
        "dateInProgress": began,
        "dateUpdated": ended,
        "duration": duration,
        "status": status,
        "environment": "prod",
        "expectedTime": "1999-01-01T00:00:00Z",
        "monitorConfig": {"schedule": schedule},
    }


def test_runtime_uses_historical_fire_and_maximum_inline_validation_including_errors(
    tmp_path: Path, cron: ReformatCronJob
) -> None:
    updates = [
        checkin("2026-10-26T01:10:20Z", "2026-10-26T01:11:20Z", 60000),
        checkin("2026-10-26T02:10:40Z", "2026-10-26T02:12:40Z", 120000),
        checkin("2026-10-26T03:19:00Z", "2026-10-26T03:20:00Z", 60000),
        checkin(
            "2026-10-26T04:10:20Z", "2026-10-26T04:11:20Z", 60000, status="timeout"
        ),
    ]
    validations = [
        checkin("2026-10-26T01:11:20Z", "2026-10-26T01:11:30Z", 10000),
        checkin("2026-10-26T02:12:40Z", "2026-10-26T02:13:00Z", 20000),
        checkin("2026-10-26T03:20:00Z", "2026-10-26T03:23:00Z", 180000, status="error"),
        checkin("2026-10-26T04:30:00Z", "2026-10-26T04:40:00Z", 600000),
    ]
    (tmp_path / "test-update.json").write_text(json.dumps(updates))
    (tmp_path / "test-validate.json").write_text(json.dumps(validations))
    args = (tmp_path, cron, pd.Timestamp("2026-10-26"), pd.Timestamp("2026-10-27"))
    stats = measure_runtime(*args, virtual=False)
    assert stats.startup_estimate == 39
    assert stats.processing == 117
    assert stats.validation_p95 == 164
    assert stats.validation_max == 180
    assert stats.validation_count == 3
    assert stats.non_ok_count == 1
    assert stats.off_schedule_count == 1
    assert measure_runtime(*args, virtual=True).processing is None


def test_retry_after_failed_validation_is_not_scheduled_runtime(
    tmp_path: Path, cron: ReformatCronJob
) -> None:
    updates = [
        checkin("2026-10-26T01:10:20Z", "2026-10-26T01:11:20Z", 60000),
        checkin("2026-10-26T01:13:00Z", "2026-10-26T01:14:00Z", 60000),
    ]
    validations = [
        checkin("2026-10-26T01:11:20Z", "2026-10-26T01:12:00Z", 40000, status="error")
    ]
    (tmp_path / "test-update.json").write_text(json.dumps(updates))
    (tmp_path / "test-validate.json").write_text(json.dumps(validations))
    stats = measure_runtime(
        tmp_path,
        cron,
        pd.Timestamp("2026-10-26"),
        pd.Timestamp("2026-10-27"),
        virtual=False,
    )
    assert stats.scheduled_count == 1
    assert stats.off_schedule_count == 1


def test_live_fetch_pages_with_environment_token_and_keeps_cache_token_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = []
    client_class = httpx.Client

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-read-only-value"
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(
                200,
                json=[{"dateAdded": "2026-10-26T00:00:00Z"}],
                headers={
                    "Link": '<https://sentry.io/api/0/organizations/test/monitors/test-update/checkins/?cursor=older>; rel="next"; results="true"'
                },
            )
        assert cursor == "older"
        return httpx.Response(200, json=[{"dateAdded": "2026-09-01T00:00:00Z"}])

    def client(*, headers: dict[str, str], timeout: int) -> httpx.Client:
        return client_class(
            transport=httpx.MockTransport(respond), headers=headers, timeout=timeout
        )

    monkeypatch.setenv("SENTRY_AUTH_TOKEN", "test-read-only-value")
    monkeypatch.setattr("scripts.timing_profile.runtime.httpx.Client", client)
    monkeypatch.setattr("scripts.timing_profile.runtime.time.sleep", lambda _: None)
    fetch_checkins(tmp_path, "test-update", "test", pd.Timestamp("2026-10-01"))
    exported = (tmp_path / "test-update.json").read_text()
    assert len(json.loads(exported)) == 2
    assert "test-read-only-value" not in exported
    fetch_checkins(tmp_path, "test-update", "test", pd.Timestamp("2026-10-01"))
    assert len(requests) == 2


def test_empty_exports_leave_all_components_unmeasured(
    tmp_path: Path, cron: ReformatCronJob
) -> None:
    stats = measure_runtime(
        tmp_path,
        cron,
        pd.Timestamp("2026-10-26"),
        pd.Timestamp("2026-10-27"),
        virtual=False,
    )
    assert stats.startup_estimate is None
    assert stats.processing is None
    assert stats.validation_max is None


def test_sentry_rate_limit_respects_bounded_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return (
            httpx.Response(429, headers={"Retry-After": "120"})
            if calls == 1
            else httpx.Response(200, json=[])
        )

    monkeypatch.setattr("scripts.timing_profile.runtime.time.sleep", sleeps.append)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert checkins_page(client, "https://sentry.io/api/0/test").status_code == 200
    assert sleeps == [1, 60, 1]
