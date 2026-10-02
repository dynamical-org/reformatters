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
- **ECMWF 46-day freshness**: Check the GRIB archiver first; it triggers both updates after the newest initialization is archived. Updates share the archiver's 53-hour expected-publication delay, including the newest eligible 00 UTC initialization at exactly 05 UTC two days later. The 48-hour licence minimum does not establish ECDS availability. Updates retain the latest stored initialization for repair and include every later eligible initialization for catch-up. The 09/10 UTC update schedules remain backstops; the age cutoff does not guarantee a delayed archive is complete.
- **Validation fails**: Re-run update, then re-run validation using "Create job from cronjob" in GitHub Actions. This most commonly happens because data was missing at the source at update time.
- **Update times out**: Use "Get jobs" and "Get pods" to check status. If update finishes successfully, but late, re-run validation. A run whose logs end with `Received SIGTERM, exiting` was stopped by kubernetes (eviction, pod active deadline, or a replacing fire); one whose logs stop with no such line was killed outright (e.g. out of memory) or stopped making progress on its own.
- **Update fails**: Look at issues and logs. Failed jobs usually require a code change to fix (e.g. structural change to data at the source). If it appears a code change is needed, make a PR, merge it, wait for the deploy action to complete, then re-run the update and validation workflows. If it appears transient, re-run the update job followed by validation.
- **Earthdata unauthorized (401)** (IMERG, SMAP): `Earthdata session rejected (401)` logs at info while the download retries with a fresh session. `Download failed <url>` with `401 Client Error` or `Failed to get token from NASA Earthdata` means retries were exhausted; check Earthdata Login availability, the `nasa-earthdata` credentials, and data-access approvals.

_This card gives only the first steps for responding to an operational error. Keep noncritical details elsewhere so on-call operators can find what they need quickly. Agents: ask before adding to this file._
