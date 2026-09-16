# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset an update (`-update`) CronJob runs first, followed by a validation (`-validate`) CronJob. Names may shorten the dataset ID to fit Kubernetes' length limit; use the generated choices in "Create job from cronjob" below. A few source archives and mirrors that feed a dataset run their own cron (e.g. `ecmwf-ifs-ens-46-day-gribs-archive-grib-files`, `noaa-hrrr-nomads-mirror-gribs`); they have no validate step.

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
- **`noaa-hrrr-forecast-18-hour-virtual-fast-validate` fails on `CheckMirrorRefsRepointed`**: a file first ingested from the NOMADS mirror 36 hours ago still has its refs pointing there, because NODD has not published it (`https://noaa-hrrr-bdp-pds.s3.amazonaws.com/<key>` and its `.idx`). The mirror expires files three days after copying them, and a read of a chunk whose mirror object is gone raises. If NODD is simply late, the next update fire after it lands rewrites the refs and clears the record; nothing to do. If NODD never publishes it, remove that file's refs from `main` by hand, then its record: open a writable session on the primary repo, and for every variable the file carries (`generate_source_file_coords` for that init, lead and file type lists them) and every level of a vertical group, `store.delete` the chunk key at that init and lead (`RegionJob.chunk_key(out_loc, var)` names it), commit once, and delete `<store>/_internal/pending-repoint/<key with / as __>`. Older snapshots keep the dangling refs and still raise; only `main` is repaired. There is no tool for this yet. If the mirror cron (`noaa-hrrr-nomads-mirror-gribs`) itself misbehaves, set `suspend=True` on its CronJob in code and merge (a `kubectl` suspend is undone by the next deploy): no new copies are made, the update takes every file from NODD, and files already ingested from the mirror are still rewritten from NODD by later fires.
