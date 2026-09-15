# dynamical.org reformatters operations card

_Report issues to feedback@dynamical.org._

For each dataset an update (`-update`) CronJob runs first, followed by a validation (`-validate`) CronJob. Names may shorten the dataset ID to fit Kubernetes' length limit; use the generated choices in "Create job from cronjob" below.

## Sentry monitoring
_Requires sentry organization invitation._
- [Crons overview](https://dynamical.sentry.io/insights/crons/) - Start here. If all green, we're good. If red, click in to see the issue and logs.
- [Issues](https://dynamical.sentry.io/issues/?statsPeriod=12h)
- [Logs](https://dynamical.sentry.io/explore/logs/) - You can filter these by `cron_job_name` / `job_name` / `pod_name` attributes.

Alerts route to Slack `#ops-reformatters` via each project's alert rule.

## Peak memory

After a CLI command starts, it logs `Peak memory: <value> GiB` on success, exception, or a handled SIGTERM/keyboard interrupt. Our Kubernetes containers expose the cgroup v2 counter at `/sys/fs/cgroup/memory.peak`. This counter covers the container's processes, including multiprocessing workers, and includes charged page cache and kernel memory. It measures the cgroup's lifetime peak, so a reused cgroup can include earlier work. Reformatter job pods run one container and use `restartPolicy: Never`; each retry gets a new pod.

Use these logs across representative worker pods to inform container memory requests. The peak includes reclaimable page cache, so it can exceed the memory the job needs to keep resident. If the cgroup counter is unavailable, the log explicitly labels `/proc/self/status` `VmHWM` as **process RSS only**; that fallback excludes children and is not a container sizing measurement. If neither source is readable, the log says `Peak memory: unavailable`. SIGKILL/OOM kills and failures before CLI startup cannot emit this exit log.

## Kubernetes cluster operations
Accessible via manually triggered github actions. Follow link and click "run workflow". _Requires repo write permisisons._
- [Get jobs](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-jobs.yml) - What's running now/recently and its status
- [Get pods](https://github.com/dynamical-org/reformatters/actions/workflows/manual-get-pods.yml) - Failed pods will be visible here. Jobs retry pods, but if multiple pods have failed we'll likely need a code change to fix the issue.
- [Create job from cronjob](https://github.com/dynamical-org/reformatters/actions/workflows/manual-create-job-from-cronjob.yml) - Use this to manually re-run any workflow. This is safe to do.

## Troubleshooting
- **Validation fails**: Re-run update, then re-run validation using "Create job from cronjob" in GitHub Actions. This most commonly happens because data was missing at the source at update time.
- **Update times out**: Use "Get jobs" and "Get pods" to check status. If update finishes successfully, but late, re-run validation. The image requests SIGINT for container shutdown; a handled SIGINT emits the `Peak memory:` line and exits with code 130 without a traceback. A direct SIGTERM logs `Received SIGTERM, exiting` before the peak line. An outright kill (e.g. out of memory or an expired shutdown grace period) cannot log a peak; check the pod termination reason to distinguish it from a process that stopped making progress.
- **Update fails**: Look at issues and logs. Failed jobs usually require a code change to fix (e.g. structural change to data at the source). If it appears a code change is needed, make a PR, merge it, wait for the deploy action to complete, then re-run the update and validation workflows. If it appears transient, re-run the update job followed by validation.
