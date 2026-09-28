# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset an update (`-update`) CronJob runs processing and validation in its final pod. The same pod deadline applies to both. Names may shorten the dataset ID to fit Kubernetes' length limit; use the generated choices in "Create job from cronjob" below.

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
- **Validation fails**: The final update pod exits 20 and Kubernetes marks its Job Failed. In production it may have created one child Job with an `-r1` suffix; inspect that child or wait for the next scheduled update before starting a manual run. There is at most one automatic retry. It is skipped when the next schedule fire is within the update pod's numeric deadline (`next_fire <= now + activeDeadlineSeconds`), including for MRMS. Use "Create job from cronjob" to rerun the update when needed. The `validate` CLI command remains available for a standalone check. Missing source data at update time is a common cause.
- **ECMWF 46-day archive completes but its Job fails**: Update submission failures fail the archive Job too; inspect the submission error. The GRIB archiver submits both daily and 6-hourly updates after its archive loop finishes with the newest initialization complete. A replacement archiver pod uses the same update Job names to avoid duplicate submissions. The 09:00 and 10:00 UTC update schedules remain backstops. Triggered update Jobs have no CronJob owner reference, so scheduled replacement cannot delete them; they can overlap a scheduled update, with Icechunk finalization detecting conflicting publication. Local runs and custom archive destinations do not trigger updates.
- **Update times out**: Use "Get jobs" and "Get pods" to check status. A run whose logs end with `Received SIGTERM, exiting` was stopped by kubernetes (eviction, pod active deadline, or a replacing fire); one whose logs stop with no such line was killed outright (e.g. out of memory) or stopped making progress on its own.
- **Update fails**: Look at issues and logs. Failed jobs usually require a code change to fix (e.g. structural change to data at the source). If it appears a code change is needed, make a PR, merge it, wait for the deploy action to complete, then re-run the update. If it appears transient, re-run the update job.
