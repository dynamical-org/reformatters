"""FastAPI receiver that turns wxopticon source-arrival webhooks into reformatter
update jobs by cloning the dataset's existing `<dataset>-update` CronJob into a Job.

The webhook just fires the update early; the cron remains the backup. See docs/webhooks.md.
"""

import hashlib
import hmac
import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

from fastapi import FastAPI, Request, Response
from kubernetes import client, config
from pydantic import BaseModel, ConfigDict

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.iterating import digest
from reformatters.common.logging import get_logger

log = get_logger(__name__)

SIGNATURE_HEADER = "X-Wxopticon-Signature"
TIMESTAMP_HEADER = "X-Wxopticon-Timestamp"
_MAX_SKEW = timedelta(minutes=5)
_SECRET_ENV = "WXOPTICON_WEBHOOK_SECRET"  # noqa: S105
_NAMESPACE = "default"


class WebhookPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str
    product_id: str
    kind: str
    init_time: str
    lead_group: str | None = None


class TriggerTarget(NamedTuple):
    cron_job_name: str
    source_complete: bool


def trigger_string(payload: WebhookPayload) -> str:
    if payload.kind == "progress" and payload.lead_group:
        return f"progress:{payload.lead_group}"
    return payload.kind


def build_trigger_map(
    datasets: Sequence[DynamicalDataset[Any, Any]],
) -> dict[tuple[str, str], list[TriggerTarget]]:
    """Map (product_id, trigger) -> the update CronJobs that want it."""
    trigger_map: dict[tuple[str, str], list[TriggerTarget]] = {}
    for dataset in datasets:
        for trigger in dataset.source_arrival_triggers():
            target = TriggerTarget(
                cron_job_name=f"{dataset.dataset_id}-update",
                source_complete=trigger.is_complete,
            )
            trigger_map.setdefault((trigger.product_id, trigger.trigger), []).append(
                target
            )
    return trigger_map


def verify_signature(
    secret: str, timestamp: str, body: bytes, signature_header: str | None
) -> bool:
    if signature_header is None:
        return False
    expected = hmac.new(
        secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature_header)


def _timestamp_is_fresh(timestamp: str) -> bool:
    return abs(datetime.now(UTC) - datetime.fromisoformat(timestamp)) < _MAX_SKEW


def webhook_job_name(cron_job_name: str, event_id: str) -> str:
    """DNS-safe (≤63 char) Job name derived from event_id so redelivered events
    create the same Job (k8s rejects the duplicate with 409)."""
    dataset_id = cron_job_name.removesuffix("-update")
    return f"{dataset_id[:21]}-wh-{digest([event_id], length=10)}"


def create_job_from_cronjob(
    batch_v1: client.BatchV1Api,
    cron_job_name: str,
    job_name: str,
    *,
    source_complete: bool,
) -> bool:
    """Clone the CronJob's jobTemplate into a one-off Job. Returns False if the Job
    already exists (duplicate event), True if created."""
    cronjob = batch_v1.read_namespaced_cron_job(cron_job_name, _NAMESPACE)
    job_spec = cronjob.spec.job_template.spec
    if source_complete:
        container = job_spec.template.spec.containers[0]
        container.env = [
            *(container.env or []),
            client.V1EnvVar(name="SOURCE_RUN_COMPLETE", value="true"),
        ]
    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=job_name),
        spec=job_spec,
    )
    try:
        batch_v1.create_namespaced_job(_NAMESPACE, job)
        return True
    except client.ApiException as e:
        if e.status == 409:
            return False
        raise


def create_app(datasets: Sequence[DynamicalDataset[Any, Any]]) -> FastAPI:
    app = FastAPI()
    trigger_map = build_trigger_map(datasets)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/webhooks/wxopticon")
    async def wxopticon(request: Request) -> Response:
        secret = os.environ[_SECRET_ENV]
        body = await request.body()
        timestamp = request.headers.get(TIMESTAMP_HEADER, "")
        signature = request.headers.get(SIGNATURE_HEADER)
        if not timestamp or not _timestamp_is_fresh(timestamp):
            return Response(status_code=401)
        if not verify_signature(secret, timestamp, body, signature):
            return Response(status_code=401)

        payload = WebhookPayload.model_validate_json(body)
        targets = trigger_map.get((payload.product_id, trigger_string(payload)))
        if not targets:
            log.info(
                f"No trigger for {payload.product_id} {trigger_string(payload)}; ignoring."
            )
            return Response(status_code=204)

        config.load_incluster_config()
        batch_v1 = client.BatchV1Api()
        for target in targets:
            job_name = webhook_job_name(target.cron_job_name, payload.event_id)
            created = create_job_from_cronjob(
                batch_v1,
                target.cron_job_name,
                job_name,
                source_complete=target.source_complete,
            )
            log.info(
                f"{'Created' if created else 'Duplicate (skipped)'} job {job_name} "
                f"for {payload.event_id}"
            )
        return Response(status_code=202)

    return app
