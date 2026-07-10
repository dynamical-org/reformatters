# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset there are two workflows: `{dataset-id}-update` runs first, followed by `{dataset-id}-validate`.

## Sentry monitoring
_Requires sentry organization invitation._
- [Crons overview](https://dynamical.sentry.io/insights/crons/) - Start here. If all green, we're good. If red, click in to see the issue and logs.
- [Issues](https://dynamical.sentry.io/issues/?statsPeriod=12h)
- [Logs](https://dynamical.sentry.io/explore/logs/) - You can filter these by job name

## Better Stack heartbeats
Each dataset has four heartbeats — `reformatters {dataset-id} {update|validate} {start|complete}` — pinged at the start and on completion of its update and validate cron jobs (a failed job pings the complete heartbeat's `/fail` URL). They are provisioned automatically by `uv run main deploy` (requires the `BETTERSTACK_API_KEY_RW` env var / GitHub Actions secret), and their URLs are stored in the `betterstack-heartbeats` kubernetes secret that every cron pod mounts.

## Better Stack collector (log shipping)
Job logs (stdout/stderr from every backfill, update, and validate pod) are shipped to Better Stack by a cluster-wide collector DaemonSet, separate from the heartbeats above. It is **not** managed by `uv run main deploy` — it is a one-time cluster install via `deploy/aws/install_betterstack_collector.sh` (re-runnable to upgrade). Because the collector tails container stdout, no application or per-dataset wiring is needed; new datasets are covered automatically. The collector's source token lives only in Better Stack and the `better-stack` namespace, distinct from `BETTERSTACK_API_KEY_RW`.

```
COLLECTOR_SECRET=<source-token> deploy/aws/install_betterstack_collector.sh
```

Verify with `kubectl -n better-stack get daemonset,pods -o wide` — expect one collector pod per node.

## Kubernetes cluster operations
Accessible via manually triggered github actions. Follow link and click "run workflow". _Requires repo write permisisons._
- [Get jobs](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-jobs.yml) - What's running now/recently and its status
- [Get pods](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-pods.yml) - Failed pods will be visible here. Jobs retry pods, but if multiple pods have failed we'll likely need a code change to fix the issue.
- [Create job from cronjob](https://github.com/dynamical-org/reformatters/actions/workflows/manual-create-job-from-cronjob.yml) - Use this to manually re-run any workflow. This is safe to do.

## Troubleshooting
- **Validation fails**: Re-run update, then re-run validation using "Create job from cronjob" in GitHub Actions. This most commonly happens because data was missing at the source at update time.
- **Update times out**: Use "Get jobs" and "Get pods" to check status. If update finishes successfully, but late, re-run validation.
- **Update fails**: Look at issues and logs. Failed jobs usually require a code change to fix (e.g. structural change to data at the source). If it appears a code change is needed, make a PR, merge it, wait for the deploy action to complete, then re-run the update and validation workflows. If it appears transient, re-run the update job followed by validation.