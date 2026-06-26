import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from kubernetes import client

from reformatters.common.webhook import receiver
from reformatters.common.webhook.triggers import WxOpticonTrigger

SECRET = "test-secret"  # noqa: S105


def _dataset(dataset_id: str, triggers: tuple[WxOpticonTrigger, ...]) -> object:
    return SimpleNamespace(
        dataset_id=dataset_id, source_arrival_triggers=lambda: triggers
    )


def _gfs_dataset() -> object:
    return _dataset(
        "noaa-gfs-forecast",
        (WxOpticonTrigger(product_id="external-noaa-gfs-aws", trigger="complete"),),
    )


def _sign(body: bytes, timestamp: str, secret: str = SECRET) -> str:
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def _payload(kind: str = "complete", **overrides: object) -> dict[str, object]:
    return {
        "event_id": "noaa-gfs/external-noaa-gfs-aws/2026-06-10T06:00Z/" + kind,
        "product_id": "external-noaa-gfs-aws",
        "kind": kind,
        "init_time": "2026-06-10T06:00:00Z",
        **overrides,
    }


def test_build_trigger_map_includes_only_datasets_with_triggers() -> None:
    trigger_map = receiver.build_trigger_map(
        [_gfs_dataset(), _dataset("noaa-gfs-analysis", ())]  # ty: ignore[invalid-argument-type]
    )
    assert trigger_map == {
        ("external-noaa-gfs-aws", "complete"): [
            receiver.TriggerTarget("noaa-gfs-forecast-update", source_complete=True)
        ]
    }


def test_trigger_string_qualifies_progress_with_lead_group() -> None:
    progress = receiver.WebhookPayload.model_validate(
        _payload("progress", lead_group="f384")
    )
    assert receiver.trigger_string(progress) == "progress:f384"
    complete = receiver.WebhookPayload.model_validate(_payload("complete"))
    assert receiver.trigger_string(complete) == "complete"


def test_webhook_job_name_is_deterministic_and_dns_safe() -> None:
    name = receiver.webhook_job_name("noaa-gfs-forecast-update", "some/event/id")
    assert name == receiver.webhook_job_name(
        "noaa-gfs-forecast-update", "some/event/id"
    )
    assert len(name) <= 63
    assert name.replace("-", "").isalnum()


def test_verify_signature() -> None:
    body = b'{"a":1}'
    timestamp = "2026-06-10T09:45:00+00:00"
    assert receiver.verify_signature(SECRET, timestamp, body, _sign(body, timestamp))
    assert not receiver.verify_signature(SECRET, timestamp, body, "sha256=deadbeef")
    assert not receiver.verify_signature(
        SECRET, timestamp, b'{"a":2}', _sign(body, timestamp)
    )
    assert not receiver.verify_signature(SECRET, timestamp, body, None)


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("WXOPTICON_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(receiver.config, "load_incluster_config", lambda: None)
    return TestClient(receiver.create_app([_gfs_dataset()]))  # ty: ignore[invalid-argument-type]


def _post(
    app_client: TestClient, payload: dict[str, object], *, secret: str = SECRET
) -> httpx.Response:
    body = json.dumps(payload).encode()
    timestamp = datetime.now(UTC).isoformat()
    return app_client.post(
        "/webhooks/wxopticon",
        content=body,
        headers={
            receiver.TIMESTAMP_HEADER: timestamp,
            receiver.SIGNATURE_HEADER: _sign(body, timestamp, secret),
        },
    )


def test_healthz(app_client: TestClient) -> None:
    assert app_client.get("/healthz").status_code == 200


def test_bad_signature_rejected(app_client: TestClient) -> None:
    assert _post(app_client, _payload(), secret="wrong").status_code == 401  # noqa: S106


def test_stale_timestamp_rejected(app_client: TestClient) -> None:
    body = json.dumps(_payload()).encode()
    timestamp = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    response = app_client.post(
        "/webhooks/wxopticon",
        content=body,
        headers={
            receiver.TIMESTAMP_HEADER: timestamp,
            receiver.SIGNATURE_HEADER: _sign(body, timestamp),
        },
    )
    assert response.status_code == 401


def test_unmatched_product_ignored(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = MagicMock()
    monkeypatch.setattr(receiver, "create_job_from_cronjob", created)
    response = _post(app_client, _payload(product_id="external-noaa-hrrr-aws"))
    assert response.status_code == 204
    created.assert_not_called()


def test_matched_event_creates_job_with_complete_flag(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_create(batch_v1, cron_job_name, job_name, *, source_complete):  # noqa: ANN001, ANN202
        calls.append(
            {
                "cron_job_name": cron_job_name,
                "job_name": job_name,
                "source_complete": source_complete,
            }
        )
        return True

    monkeypatch.setattr(receiver, "create_job_from_cronjob", fake_create)
    monkeypatch.setattr(receiver.client, "BatchV1Api", MagicMock)
    response = _post(app_client, _payload())
    assert response.status_code == 202
    assert len(calls) == 1
    assert calls[0]["cron_job_name"] == "noaa-gfs-forecast-update"
    assert calls[0]["source_complete"] is True


def _fake_cronjob() -> SimpleNamespace:
    container = SimpleNamespace(env=None)
    return SimpleNamespace(
        spec=SimpleNamespace(
            job_template=SimpleNamespace(
                spec=SimpleNamespace(
                    template=SimpleNamespace(
                        spec=SimpleNamespace(containers=[container])
                    )
                )
            )
        )
    )


def test_create_job_from_cronjob_adds_complete_env() -> None:
    batch = MagicMock()
    cronjob = _fake_cronjob()
    batch.read_namespaced_cron_job.return_value = cronjob
    created = receiver.create_job_from_cronjob(
        batch, "noaa-gfs-forecast-update", "job-1", source_complete=True
    )
    assert created is True
    batch.create_namespaced_job.assert_called_once()
    env = cronjob.spec.job_template.spec.template.spec.containers[0].env
    assert any(e.name == "SOURCE_RUN_COMPLETE" and e.value == "true" for e in env)


def test_create_job_from_cronjob_duplicate_returns_false() -> None:
    batch = MagicMock()
    batch.read_namespaced_cron_job.return_value = _fake_cronjob()
    batch.create_namespaced_job.side_effect = client.ApiException(status=409)
    created = receiver.create_job_from_cronjob(
        batch, "noaa-gfs-forecast-update", "job-1", source_complete=False
    )
    assert created is False
