from datetime import timedelta
from typing import Self
from unittest.mock import Mock

import pytest

from reformatters.common import betterstack
from reformatters.common.betterstack import HeartbeatSpec
from reformatters.common.kubernetes import ReformatCronJob, ValidationCronJob


def _cron(name: str, schedule: str, deadline: timedelta) -> ReformatCronJob:
    return ReformatCronJob(
        name=name,
        schedule=schedule,
        pod_active_deadline=deadline,
        image="image-tag",
        dataset_id="example",
        cpu="1",
        memory="1G",
    )


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("0 0 * * *", timedelta(days=1)),
        ("22 5,11,17,23 * * *", timedelta(hours=6)),
        ("57 */3 * * *", timedelta(hours=3)),
        ("3 * * * *", timedelta(hours=1)),
    ],
)
def test_schedule_period(schedule: str, expected: timedelta) -> None:
    assert betterstack.schedule_period(schedule) == expected


def test_heartbeat_specs_grace_rules() -> None:
    cron = _cron("example-update", "0 0 * * *", timedelta(minutes=30))
    start, complete = betterstack.heartbeat_specs(cron)

    assert start == HeartbeatSpec(
        "start", "example-update.start", timedelta(days=1), timedelta(minutes=10)
    )
    # complete grace = checkin_margin (10m) + pod_active_deadline (30m)
    assert complete == HeartbeatSpec(
        "complete", "example-update.complete", timedelta(days=1), timedelta(minutes=40)
    )


def test_ping_ok_and_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_post = Mock(return_value=Mock())
    monkeypatch.setattr(betterstack.httpx, "post", mock_post)

    betterstack.ping("https://hb.example/abc")
    assert mock_post.call_args.args[0] == "https://hb.example/abc"

    betterstack.ping("https://hb.example/abc", failed=True)
    assert mock_post.call_args.args[0] == "https://hb.example/abc/fail"


def test_upsert_heartbeat_creates_when_missing() -> None:
    client = Mock()
    client.post.return_value.json.return_value = {
        "data": {"attributes": {"url": "https://hb/new"}}
    }
    spec = HeartbeatSpec("start", "a.start", timedelta(days=1), timedelta(minutes=10))

    url = betterstack._upsert_heartbeat(client, {}, spec)

    assert url == "https://hb/new"
    client.post.assert_called_once()
    assert client.post.call_args.kwargs["json"] == {
        "name": "a.start",
        "period": 86400,
        "grace": 600,
    }


def test_upsert_heartbeat_patches_when_grace_changed() -> None:
    client = Mock()
    existing = {
        "a.start": {
            "id": "7",
            "url": "https://hb/existing",
            "period": 86400,
            "grace": 999,
        }
    }
    spec = HeartbeatSpec("start", "a.start", timedelta(days=1), timedelta(minutes=10))

    url = betterstack._upsert_heartbeat(client, existing, spec)

    assert url == "https://hb/existing"
    client.post.assert_not_called()
    client.patch.assert_called_once_with(
        "/heartbeats/7", json={"name": "a.start", "period": 86400, "grace": 600}
    )


def test_upsert_heartbeat_noop_when_unchanged() -> None:
    client = Mock()
    existing = {
        "a.start": {
            "id": "7",
            "url": "https://hb/existing",
            "period": 86400,
            "grace": 600,
        }
    }
    spec = HeartbeatSpec("start", "a.start", timedelta(days=1), timedelta(minutes=10))

    url = betterstack._upsert_heartbeat(client, existing, spec)

    assert url == "https://hb/existing"
    client.post.assert_not_called()
    client.patch.assert_not_called()


def test_reconcile_heartbeats_builds_url_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BETTERSTACK_API_KEY_RW", "token")
    monkeypatch.delenv("DYNAMICAL_BETTERSTACK_STATUS_PAGE_ID", raising=False)
    monkeypatch.setattr(betterstack, "_api_client", lambda token: _NullClient())
    monkeypatch.setattr(betterstack, "_list_heartbeats", lambda client: {})
    monkeypatch.setattr(
        betterstack,
        "_upsert_heartbeat",
        lambda client, existing, spec: f"https://hb/{spec.name}",
    )

    cron_jobs = [
        _cron("example-update", "0 0 * * *", timedelta(minutes=30)),
        ValidationCronJob(
            name="example-validate",
            schedule="0 1 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image="image-tag",
            dataset_id="example",
            cpu="1",
            memory="1G",
        ),
    ]

    url_map = betterstack.reconcile_heartbeats(cron_jobs)

    assert url_map == {
        "example-update": {
            "start": "https://hb/example-update.start",
            "complete": "https://hb/example-update.complete",
        },
        "example-validate": {
            "start": "https://hb/example-validate.start",
            "complete": "https://hb/example-validate.complete",
        },
    }


class _NullClient:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None
