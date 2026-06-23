# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset there are two workflows: `{dataset-id}-update` runs first, followed by `{dataset-id}-validate`.

## Better Stack monitoring
_Requires Better Stack team invitation._
- [Heartbeats](https://uptime.betterstack.com/) - Start here. Each cron has a `{cronjob-name}.start` (fires within ~10m of the scheduled time) and a `{cronjob-name}.complete` (fires after a successful run) heartbeat. If all green, we're good.
- [status.dynamical.org](https://status.dynamical.org) - Public status page; driven by each dataset's `{dataset-id}-validate.complete` heartbeat.
- [Errors](https://dynamical.sentry.io/issues/?statsPeriod=12h) - Captured via the Sentry SDK pointed at Better Stack's errors application.
- [Logs](https://telemetry.betterstack.com/) - You can filter these by `job_name`.

## Kubernetes cluster operations
Accessible via manually triggered github actions. Follow link and click "run workflow". _Requires repo write permisisons._
- [Get jobs](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-jobs.yml) - What's running now/recently and its status
- [Get pods](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-pods.yml) - Failed pods will be visible here. Jobs retry pods, but if multiple pods have failed we'll likely need a code change to fix the issue.
- [Create job from cronjob](https://github.com/dynamical-org/reformatters/actions/workflows/manual-create-job-from-cronjob.yml) - Use this to manually re-run any workflow. This is safe to do.

## Troubleshooting
- **Validation fails**: Re-run update, then re-run validation using "Create job from cronjob" in GitHub Actions. This most commonly happens because data was missing at the source at update time.
- **Update times out**: Use "Get jobs" and "Get pods" to check status. If update finishes successfully, but late, re-run validation.
- **Update fails**: Look at issues and logs. Failed jobs usually require a code change to fix (e.g. structural change to data at the source). If it appears a code change is needed, make a PR, merge it, wait for the deploy action to complete, then re-run the update and validation workflows. If it appears transient, re-run the update job followed by validation.