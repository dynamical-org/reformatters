# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset there are two workflows: `{dataset-id}-update` runs first, followed by `{dataset-id}-validate`.

## Better Stack monitoring
_Requires Better Stack (dynamical.org team) invitation._
- **Logs** — the `reformatters` source (Live tail). Filter by the `cron_job_name` / `job_name` / `pod_name` / `env` fields.
- **Errors** — the `reformatters` errors application (grouped exceptions).
- **Heartbeats** — one `reformatters <dataset> <update|validate> <start|complete>` uptime monitor per cron. A missing start ping means a run didn't fire; a missing complete ping (or a `/fail`) means a run didn't finish. Period is derived from the cron schedule and grace from its `pod_active_deadline`, so heartbeats mirror the cron config automatically.
  - Heartbeats are provisioned at deploy time; the deploy requires the `BETTERSTACK_API_KEY_RW` secret. Redeploy to reconcile them after changing a cron's schedule or deadline.

## Kubernetes cluster operations
Accessible via manually triggered github actions. Follow link and click "run workflow". _Requires repo write permisisons._
- [Get jobs](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-jobs.yml) - What's running now/recently and its status
- [Get pods](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-pods.yml) - Failed pods will be visible here. Jobs retry pods, but if multiple pods have failed we'll likely need a code change to fix the issue.
- [Create job from cronjob](https://github.com/dynamical-org/reformatters/actions/workflows/manual-create-job-from-cronjob.yml) - Use this to manually re-run any workflow. This is safe to do.

## Troubleshooting
- **Validation fails**: Re-run update, then re-run validation using "Create job from cronjob" in GitHub Actions. This most commonly happens because data was missing at the source at update time.
- **Update times out**: Use "Get jobs" and "Get pods" to check status. If update finishes successfully, but late, re-run validation.
- **Update fails**: Look at issues and logs. Failed jobs usually require a code change to fix (e.g. structural change to data at the source). If it appears a code change is needed, make a PR, merge it, wait for the deploy action to complete, then re-run the update and validation workflows. If it appears transient, re-run the update job followed by validation.