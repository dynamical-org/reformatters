# Operational update timing

Timing values are hand-set constants in each dataset's code. This document defines the rules for selecting those values; the [adoption plan](plans/operational_update_timing.md) tracks their rollout.

## Latency and readiness units

Latency runs from source availability to readability in the primary store.

- **Publish as soon as the source has the data.**
  - Virtual datasets poll from just before the source's earliest normal arrival until the next scheduled run takes over, so each file lands within seconds, even when the source is late.
  - A materialized dataset runs the moment its virtual sibling (or archive job) sees a complete unit. Datasets with nothing to trigger them run as soon as they are reliably available when the source is publishing normally.

Measure the source inputs the dataset actually consumes:

- A virtual unit is one usable source file, including its index where required.
- A materialized forecast unit is the complete run, or an explicit tier when lead groups arrive far apart.
- An observational analysis unit is one append position. Its anchor is the end of its valid period.
- A forecast-derived analysis unit contains every position one source cycle contributes, including the accumulated boundary input required by its last position. Its anchor is the source cycle's initialization, rather than each position's label.

With the source on schedule, the latency targets are less than ten seconds for virtual files and less than ten minutes from the unit's last file for a materialized dataset with a completion trigger. An untriggered materialized dataset publishes within startup plus processing time of its scheduled fire. That class can miss the ten-minute target at typical source timing. Source lateness, failures, and the named exceptions below can miss targets and alert; report those misses.

## Measure before setting constants

Run the advisory profiler locally:

```bash
uv run python -m scripts.timing_profile --days 35 --checkins-dir /path/to/checkins --output /tmp/values-table.md
```

Omit `--checkins-dir` to fetch fresh check-ins using a read-only `SENTRY_AUTH_TOKEN` supplied in the environment. See `--help` for the export format and measurement limitations. The tool changes no dataset code and has no recurring schedule.

The profiler measures source arrival p50, p95, p99, and earliest normal arrival (`E`) after the anchor. It aggregates timestamps into usable files and complete readiness units before taking percentiles. It reports startup (`L`) as an estimate from scheduled fire to update check-in, materialized processing through publish (`R`), and validation duration (`V`), including both p95 and the observed maximum. Virtual check-in duration includes source waiting and cannot establish processing time. Historical check-in schedules establish the fire time, but check-ins alone do not establish which source initialization a run processed. Missing components remain unmeasured.

Use a trailing window, normally 35 days. Inspect sample count, actual coverage, missing and censored units, source identity, reuploads, and source regime changes. Rewritten or reprocessed timestamps do not establish first availability. A dataset without anonymous arrival evidence keeps its constants. Do not add credentialed source access just to tune its schedule.

`E` is the minimum first-file arrival in the window after excluding verified outliers. Keep recurring fast episodes. Measure cycle classes separately when their median arrivals differ by at least fifteen minutes, using first visibility for virtual datasets and complete-unit arrival for materialized datasets.

The numerical formulas are implemented in the profiler. The late threshold is `T = p95 + max(p95 − p50, 15 min)`. The recommended untriggered schedule offset is `S = min(p95 + max(p95 − p50, 5 min), p99 + 1 min)`. The spread term tolerates rare outliers; the p99 cap avoids overshooting a wide but bounded tail. The profiler shows the three candidate offsets and the chosen minimum against the measured distribution. `T` identifies a late source. `S` selects the cut between normal arrival and an anomalous delay for an untriggered materialized dataset. Review the chosen `S` against the candidate comparison before adopting a cron or deriving values from it. Values remain hand-set, and named exceptions require manual review. The fraction of complete observed units arriving after `S` is arrival evidence, not a prediction of validation alerts: a multi-position unit can pass with a complete prefix.

## Schedules

Use one cron per dataset, with one minute offset across all its cycles:

- Virtual schedules fire at `a + E − L − 1 min`, using raw earliest file arrival across all tiers and cycle classes. Later classes spend the difference polling.
- Untriggered materialized schedules use the latest class's `S` for the first readiness tier. Earlier classes accept that wait as latency.
- Where a materialized dataset has a completion trigger, its cron is a backstop at the first readiness tier's `T`. Change a consumer's cron together with activation of its trigger.

Check every cycle class against the resulting single cron. Configure deadlines using measured startup, processing, and validation, leaving room for validation after polling. See [virtual operational updates](virtual_datasets.md#operational-updates) and [parallel processing](parallel_processing.md) for the mechanisms in use.

## Virtual polling

For CronJobs on `Allow`, fire at `a + E − L − 1 min`, where `a` is the source anchor, `E` the earliest normal usable-file arrival after it, and `L` the p95 startup duration. Choose one minute offset from the earliest cycle class. Poll until the next fire plus `L + 1 min`, so a file arriving at the next fire remains observable while its successor starts. Exit early when nothing in the selected window remains pending.

Set both pod and Job active deadlines to `Δf + L + 1 min + V + 30 s`, with `Δf` the fire interval and `V` the largest observed validation duration. Set `virtual_poll_deadline_grace = V + 30 s`. The cutoff is the selected fire plus the pod deadline minus that grace. Round Sentry's `max_runtime` up to whole minutes so it does not expire before the pod deadline.

Select the fire once at execution start for initial job construction, the polling cutoff and end-of-run manifest/decode validation, including when startup crosses the successor fire. Currency checks still use current time. A replacement starting after the successor fire may select its window and cutoff; the original Job deadline bounds lifetime even though the replacement's pod deadline restarts.

Validate once after polling ends. Validation failure exits 20 and fails the Job; virtual datasets on `Allow` create no validation re-pass Job. Source lateness can therefore alert at poll end under the currency rule below. Datasets still on `Replace` retain their existing retry lifecycle until adoption. See [virtual operational updates](virtual_datasets.md#operational-updates) for writer coordination and recovery.

## Currency and integrity validation

`CheckCurrentData.max_delay` is measured from an append position's label. It must never require a position before the fire intended to publish it:

- For virtual datasets, `max_delay = T_first_file + 4 min + (a − label)`. Here first file means the file that makes a new position visible. Where discovery gates each position on its own-cycle files, measure the maximum of those files per position, then the minimum across positions. Forecast virtuals use the earliest usable file. Polling start `E` always uses the raw earliest usable file.
- For materialized datasets, `max_delay = F + (a − earliest label) + max(0, Δf − anchor spacing)`, where `F` is the intended fire's offset (`S` for an untriggered run, `T` for a backstop), and `Δf` is the fire interval. The last term accounts for a fire covering several anchors.

Use one scalar per dataset. Later positions in a multi-position unit are detected later. Compute it against the final single cron, including minute rounding and every cycle class, rather than against processing completion. Use the profiler's recommendations to review the current cron, deadlines, and `max_delay` together.

Currency does not replace integrity checks. Use `CheckVirtualManifestCompleteness` to require complete older positions and detect interior gaps, and use recent NaNs, decode health, replica, and expected-shard checks as appropriate. Tune leading completeness tiers and NaN thresholds by hand when they false-alert. Integrity failures can page before the currency fire. Validation errors page except for a queued materialized run's first failure when a re-pass is submitted; Sentry cron monitors also alert on missed and timeout check-ins. See [validation](validation.md) for configuring and inspecting checks.

## Materialized deadlines and monitoring

Materialized pod and Job active deadlines equal the interval between scheduled fires, `Δf`. Sentry's `max_runtime` matches the pod deadline, so a hung update alerts at that deadline. Virtual budgets follow [virtual polling](#virtual-polling).

The update check-in covers the start decision, queue wait, processing, and finalization. A skipping Job closes it successfully without validation. Validation has its own check-in after publication. End-of-run backoff is outside both check-ins.

A queued materialized run's first validation sends only a terminal check-in (`ok`, or `error` when no re-pass is submitted); re-passes keep `in_progress`. With no first-pass timeout check-in, an established monitor covers a lost first validation or re-pass, including off-schedule triggers, through a missed scheduled slot at the fire plus the pod deadline plus 10 minutes, unless another run's validation check-in arrives first.

## Materialized queue and re-passes

These rules apply to production materialized updates whose CronJobs adopt `Allow` with the deadlines above. [Job state and ordering](parallel_processing.md#materialized-job-state-and-ordering) define the coordination files and Job identity. A Job is **writing** while Kubernetes reports neither a `Complete` nor a `Failed` condition and its matching-UID state is absent or `run`. Failed indices alone do not make a Job terminal; running siblings may still write. A stuck writer blocks newer Jobs until it finishes or its Job deadline expires.

Pod 0 decides once, before any index computes region jobs. Job annotations determine the kind:

- `dynamical.org/run-kind: trigger` identifies a trigger.
- Other Jobs annotated `cronjob.kubernetes.io/instantiate: manual` are manual, including operator clones and re-passes.
- Remaining Jobs are scheduled. Ownership does not determine the kind.

A scheduled Job takes the backstop check when a trigger for its CronJob was created within `Δf/2` before its own creation. It waits for older writers without recording a state, then checks only `CheckCurrentData` against the published store. It records `skip` if current, otherwise `run`. The comparison uses the scheduled Job's own creation time even when pod 0 is replaced later. Every other Job records `run`.

All indices, including replacement pods, reuse the matching-UID decision. `skip` and `finalized` exit 0 without processing or validation. On `run`, each index waits until no older Job is writing, checking every 30 seconds, before reading the store and computing jobs. Waiting only on older Jobs prevents deadlock; a finalized Job does not block the queue while validating or sleeping. A Job deciding whether to skip never records `run` first, so it cannot suppress a re-pass and then skip.

Each adopted materialized builder accepts the explicit `job_reference_time` argument. Its current Job's own `creationTimestamp`, normalized to UTC, anchors window and planning calculations; a re-pass uses its own Job's creation time. All indices and replacements therefore compute the same window across initialization boundaries. Forwarding a reference to an unsupported builder fails rather than dropping it. Download, polling, and validation clocks keep current time. This reference is distinct from a virtual update's selected `job_fire_time`.

The last index follows the [publication and cleanup order](parallel_processing.md#cleanup), then always validates, including when it had no region jobs. On successful validation, the run is done. A failure backs off as below. A re-pass failure pages before backing off; a first failure pages after its backoff only if no re-pass is submitted, and otherwise logs a warning with the failed checks:

- Sleep 5 minutes before `-r1`, 10 before `-r2`, 20 before `-r3`, and 30 before each later re-pass, through `-r16`.
- From 30 seconds into sleep, check every 30 seconds and immediately before submission for a newer nonterminal Job with matching-UID `run` state. Yield to one without submitting; absent, skipped, finalized, terminal, or stale-UID state does not suppress a re-pass.
- Otherwise, `create_job_from_cronjob` submits the next `-rN` as an ordinary manual update. It recomputes its repair work from store state. Re-passes use `retry_job_name(max_retries=16)` and do not apply the legacy next-fire deadline suppression. Suspending the CronJob prevents submission.
- Exit 20 whether a re-pass was submitted or the pod yielded. A failing `-r16` reports and exits without sleeping or submitting another Job.

An interrupted backoff is lost; a later trigger or scheduled fire covers it. A replacement that sees finalized state exits successfully, including after an interruption in cleanup, validation, or backoff. Every persistent failure can start another chain at each fire, and sleeping pods retain their resources. See the [operations card](ops_card.md#troubleshooting) for first response steps.

Materialized datasets outside this lifecycle retain one immediate `-r1`, submitted only when the next scheduled fire is farther away than the pod deadline. A multi-worker replacement after cleanup waits for deleted setup files until its deadline; a single-worker replacement reruns the update. Inspect termination reasons before rerunning the full update as a new Job.

## Completion triggers

A virtual producer dispatches its configured materialized consumers when its window becomes complete: it started with pending files, all pending files became present before poll end, and no file was skipped as unreadable. Completion counts regardless of which overlapping pod committed the last file. A window complete at startup or incomplete at poll end produces no trigger; the consumer's backstop cron covers those cases.

The write loop returns the completion flag. Dispatch happens after polling and before validation, outside the commit path. Producers list their consumers in `materialized_update_triggers`; each consumer accepts only units it publishes. Only consumers whose scheduled fire publishes one append position are eligible. A multi-producer consumer takes its trigger from the source that usually completes last; the backstop covers the others.

Trigger Job names are deterministic per consumer and unit, fit the indexed Job name budget, and never end in a numeric suffix. `create_job_from_cronjob` deduplicates a 409 only for the same CronJob name and UID and respects consumer suspension. Trigger Jobs carry `dynamical.org/run-kind: trigger` on Job metadata and enter the materialized queue. Admission permits that exact annotation value only on the Job; labels and pod templates remain pinned. Staging producers cannot trigger production consumers.

A triggered consumer's cron is its backstop at the latest cycle class's late threshold T, rounded up to the fire minute. Its `max_delay` equals the cron's actual offset from the position label, so the currency check cannot skip a missing unit at the fire. Move the cron and currency allowance together with trigger activation. Submission failures are logged without preventing producer validation; lost triggers are covered by that cron.

The IFS 46-day archiver emits one trigger per init. GEFS 35-day extension tier 2 receives no trigger: its source window does not normally complete during a producer run, and the consumer's age-gated update window covers it on a later init's trigger or backstop.

## Repair window

`operational_update_jobs` defines the repair window, usually starting inclusively at the latest stored position. A dataset that needs a wider window defines one. Gaps outside it page and require a [backfill](backfill.md). Forecast-derived analyses publish a non-ragged leading edge through result trimming or discovery gating.

## Named exceptions

The dataset implementations define the values and boundaries of these exceptions:

- GEFS 35-day virtual uses one daily window; intermediate extension waves can wait until the next fire.
- GEFS 35-day materialized extension has no completion trigger and waits for an eligible later run after its age gate.
- WeatherNext 2 uses all members and levels of one variable and lead as its unit, measured from the unit's last object.
- MRMS uses one hourly run; pass 1 is an exception.

Arrivals beyond a fire's chosen offset alert when currency or integrity checks detect them. Choose offsets to make this rare under normal source operation.

## Possible follow-up: quiet re-pass window

Not implemented. If an untriggered materialized dataset with a high-variance source alerts more often than is acceptable, give it a per-dataset quiet window: re-pass validation failures stay quiet until the fire plus that window, and page only if the unit is still missing after it. Late units still publish through re-passes as soon as they arrive; only the alert waits.
