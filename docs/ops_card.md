# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset there are two workflows: `{dataset-id}-update` runs first, followed by `{dataset-id}-validate`. A few source archives and caches that feed a dataset run their own cron (e.g. `ecmwf-ifs-ens-46-day-gribs-archive-grib-files`, `noaa-hrrr-nomads-cache-mirror-gribs`); they have no validate step.

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
- **`noaa-hrrr-forecast-18-hour-virtual-validate` fails on `CheckNomadsCacheRepointed`**: a file the NOMADS mirror cached hours ago has no repoint marker. HEAD the same key on `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/` and its `.idx`: if NODD lacks it, this is upstream lateness and the cached copy is kept (it is never expired until repointed), so wait; if NODD has it, the repoint is failing: read the update's logs for that key (`Skipping ...` lines name a short index or a build error) and re-run the update. If the mirror cron (`noaa-hrrr-nomads-cache-mirror-gribs`) itself misbehaves, suspend it: the update falls back to NODD for new files, and refs already pointing at the cache are still repointed by later fires. Set `NoaaHrrrForecast18HourVirtualRegionJob.cache_first = False` to stop selecting the cache in code without touching the repoint machinery.
