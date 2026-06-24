# Webhook-triggered operational updates

Operational updates normally run on a fixed Kubernetes CronJob schedule timed for *after*
source data is expected to land. That schedule is a guess with safety margin, so updates are
later than necessary when data arrives early and fail when it is late.

[wxopticon](https://status.dynamical.org) watches the upstream sources and POSTs a signed
webhook the moment a run crosses a readiness boundary. A dataset can opt in to having that
webhook trigger its update immediately. **The existing update cron stays on as a backup** — if a
webhook is missed or the receiver is down, the scheduled run still updates the dataset.

```
wxopticon ──(signed POST on source arrival)──▶ webhook receiver (in-cluster Deployment)
                                                   │ clone <dataset>-update CronJob → Job
                                                   ▼
                                          noaa-gfs-forecast-update Job  ◀── also fired by the cron (backup)
```

## Opting a dataset in

Override `source_arrival_triggers()` on the dataset to return the wxopticon products + triggers
that should fire its update. The update CronJob (`<dataset_id>-update`) is what gets cloned, so no
extra Job config is needed.

```python
def source_arrival_triggers(self) -> Sequence[WxOpticonTrigger]:
    return (WxOpticonTrigger(product_id="external-noaa-gfs-aws", trigger="complete"),)
```

`trigger` is one of `started`, `complete`, or `progress:<lead_group>` (e.g. `progress:f384`).
Use `complete` when the reformatter processes the full run. Datasets with no override stay
cron-only. `noaa-gfs-forecast` is the current pilot.

## Backup cron — skip when already done or in progress

Both the webhook- and cron-triggered runs execute the same `update` command and derive the same
data-driven `run_key` (the newest append-dim value being written). They coordinate through marker
files in the object store (`{dataset}/_internal/operational/runs/<run>/`):

- A run already marked **complete** → skip.
- A run with a fresh **in-progress** lease from a *different* job → skip. A stale lease (crashed
  job, older than the lease timeout) is ignored so the cron can take over — self-healing.

Only a run triggered by a wxopticon `complete` event marks the run complete (the receiver tags the
Job with `SOURCE_RUN_COMPLETE=true`). Cron and `progress:*` runs leave the run re-runnable so the
backup still reprocesses the tail to catch late data when the webhook path is unavailable.

This means the backup cron is normally a no-op (the webhook already did the work) but still fully
updates the dataset whenever the webhook path fails. See
`src/reformatters/common/operational_run_guard.py`.

## The webhook contract (from wxopticon)

- `POST` JSON to the receiver. Headers: `X-Wxopticon-Signature: sha256=<hex>` and
  `X-Wxopticon-Timestamp: <ISO Z>`. Signature = `hmac_sha256(secret, "{timestamp}.{body}")`.
- The receiver rejects requests with a stale timestamp (>5 min skew) or bad signature (401),
  acks unmatched products (204), and creates a Job for matched ones (202).
- Delivery is at-least-once; the receiver derives the Job name from `event_id`, so a redelivered
  event creates the same Job name and Kubernetes rejects the duplicate (409 → ignored).

## Deploying the receiver

`uv run main deploy` applies the operational cronjobs **and** the webhook receiver resources
(ServiceAccount + Role/RoleBinding to create Jobs from CronJobs, Deployment, Service, Ingress) —
see `src/reformatters/common/webhook/deploy.py`. Run the receiver locally with
`uv run main serve-webhooks`.

Deploy-time prerequisites (see constants in `webhook/deploy.py`):

- A public HTTPS host for the receiver (`RECEIVER_HOST`) with an ACM certificate
  (`ACM_CERTIFICATE_ARN`) on the ALB Ingress, and a DNS record pointing at it.
- A Kubernetes secret `wxopticon-webhook` with key `WXOPTICON_WEBHOOK_SECRET` holding the
  subscription secret returned at registration.

## Registering the wxopticon subscription

After the receiver is deployed and reachable over HTTPS:

```
WXOPTICON_ADMIN_TOKEN=… uv run main register-webhook-subscription https://<RECEIVER_HOST>/webhooks/wxopticon
```

This builds the subscription `targets` from every dataset's `source_arrival_triggers()` and calls
the admin-gated wxopticon API. It prints the subscription `id` and the `secret` (shown once) —
store the secret in the `wxopticon-webhook` Kubernetes secret. Pass `--subscription-id <id>` to
update an existing subscription instead of creating a new one.

## Troubleshooting

- **A webhook-triggered job didn't run**: the cron backup still covers the update. Check the
  receiver Deployment logs and that the wxopticon subscription targets the right product/trigger.
- **The cron keeps doing full work**: expected when the webhook path is unavailable. Confirm the
  receiver is reachable and the subscription is active.
- **Duplicate jobs**: deduped by `event_id`-derived Job name and the per-run lease; safe because
  `update` is idempotent.
