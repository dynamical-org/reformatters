# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset there are two workflows: `{dataset-id}-update` runs first, followed by `{dataset-id}-validate`. A few source archives and mirrors that feed a dataset run their own cron (e.g. `ecmwf-ifs-ens-46-day-gribs-archive-grib-files`, `noaa-hrrr-nomads-mirror-gribs`); they have no validate step.

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
- **`noaa-hrrr-forecast-18-hour-virtual-validate` fails on `CheckMirroredFilesReachNodd`**: a file the NOMADS mirror copied 36 hours ago has still not appeared on NODD, so the store's refs for it can only point at the mirror, which expires files after three days. HEAD the key on `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/`; if NODD is simply late the next update fire rewrites the refs once it lands, if NODD never publishes it the refs break at expiry and that init's file is lost as on every NODD-only product. If the mirror cron (`noaa-hrrr-nomads-mirror-gribs`) itself misbehaves, suspend it: no new copies are made, the update takes every file from NODD, and files already in the mirror are still rewritten from NODD by later fires.
