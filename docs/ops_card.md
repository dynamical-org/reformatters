# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset an update (`-update`) CronJob runs processing in parallel across one or more pods and validation in its final pod. The pod deadline applies to update processing and validation together. Names may shorten the dataset ID to fit Kubernetes' length limit; use the generated choices in "Create job from cronjob" below.

## Sentry monitoring
_Requires sentry organization invitation._

The `-update` monitor covers processing/publication; `-validate` covers the checks afterward. Validation failure leaves the update monitor successful but fails the Kubernetes Job. If processing fails, validation never starts and both monitors should alert.
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
- **Materialized update skips**: Source-gated forecast updates log a skip reason from `_internal/{job_name}/plan/ready.json`. An older unfinished sibling can own the work; otherwise there may be no new source-ready init. Final-worker validation still checks currency and data quality. Products with delayed extensions can opt into frontier reprocessing. The public source summary is required for selecting new inits; request/schema errors fail loudly. See [update coordination](parallel_processing.md#materialized-update-plan).
- **Repeated sibling skips**: Inspect the named older Job, including Pending pods. A triggered Job that never schedules can block subsequent updates until its scheduling problem or the stale Job is resolved; the CronJob's Replace policy does not remove manually triggered Jobs.
- **Source summary fails**: Materialized planning stops until the public wxopticon summary is reachable and valid. A removed or renamed product requires updating the dataset's source mapping. Kubernetes pod retries apply, but this processing failure does not create a validation-retry Job.
- **Validation fails**: The final update pod exits 20 and its Job fails. Check for an automatic retry before using "Create job from cronjob" to rerun the update. Retries are bounded and submitted only if the next scheduled fire is farther away than the pod deadline. Missing source data at update time is a common cause. The `validate` CLI remains available for standalone checks.
- **Update times out**: Use "Get jobs" and "Get pods" to check status. A run whose logs end with `Received SIGTERM, exiting` was stopped by kubernetes (eviction, pod active deadline, or a replacing fire); one whose logs stop with no such line was killed outright (e.g. out of memory) or stopped making progress on its own.
- **Update fails**: Look at issues and logs. Jobs that raise an error usually require a code change to fix (e.g. structural change to data at the source). If it appears a code change is needed, make a PR, merge it, wait for the deploy action to complete, then re-run the update. If it appears transient, re-run the update job.

_This card gives only the first steps for responding to an operational error. Keep noncritical details elsewhere so on-call operators can find what they need quickly. Agents: ask before adding to this file._
