# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset an update (`-update`) CronJob runs first, followed by a validation (`-validate`) CronJob. Names may shorten the dataset ID to fit Kubernetes' length limit; use the generated choices in "Create job from cronjob" below.

## Sentry monitoring
_Requires sentry organization invitation._
- [Crons overview](https://dynamical.sentry.io/insights/crons/) - Start here. If all green, we're good. If red, click in to see the issue and logs.
- [Issues](https://dynamical.sentry.io/issues/?statsPeriod=12h)
- [Logs](https://dynamical.sentry.io/explore/logs/) - You can filter these by `cron_job_name` / `job_name` / `pod_name` attributes.

Alerts route to Slack `#ops-reformatters` via each project's alert rule.

## Kubernetes cluster operations
Accessible via manually triggered github actions. Follow link and click "run workflow". _Requires repo write permisisons._
- [Get jobs](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-jobs.yml) - What's running now/recently and its status
- [Get pods](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-pods.yml) - Failed pods will be visible here. Jobs retry pods, but if multiple pods have failed we'll likely need a code change to fix the issue.
- [Create job from cronjob](https://github.com/dynamical-org/reformatters/actions/workflows/manual-create-job-from-cronjob.yml) - Use this to manually re-run any workflow. This is safe to do.

## Troubleshooting
- **Validation fails**: Re-run update, then re-run validation using "Create job from cronjob" in GitHub Actions. This most commonly happens because data was missing at the source at update time.
- **Update times out**: Use "Get jobs" and "Get pods" to check status. If update finishes successfully, but late, re-run validation. A run whose logs end with `Received SIGTERM, exiting` was stopped by kubernetes (eviction, pod active deadline, or a replacing fire); one whose logs stop with no such line was killed outright (e.g. out of memory) or stopped making progress on its own.
- **Update fails**: Look at issues and logs. Failed jobs usually require a code change to fix (e.g. structural change to data at the source). If it appears a code change is needed, make a PR, merge it, wait for the deploy action to complete, then re-run the update and validation workflows. If it appears transient, re-run the update job followed by validation.
- **`noaa-hrrr-forecast-18-hour-virtual-fast-validate` fails on `CheckMirrorWindow`**: the oldest data in the window does not decode. Check `noaa-hrrr-nomads-mirror-gribs`, the R2 bucket's lifecycle rule, and the R2 public domain. Files NOMADS has rotated away are unrecoverable in this product; use `noaa-hrrr-forecast-18-hour-virtual` for those forecasts.
- **`noaa-hrrr-forecast-18-hour-virtual-fast-validate` or `-update` fails with an assertion that the store's `init_time` labels are not the template's**: in validation, the most recent update fire did not drop the window's expired positions (it failed early, never ran, or is suspended), or the store was just created by a backfill whose end was not that fire's time. Fix the update; the next scheduled fire drops the positions and realigns the store. In the update, another writer moved the window while this one ran. Ensure only one update writer is active; the next scheduled fire recovers.
