# Operational update timing adoption plan

The [timing conventions](../operational_update_timing.md) define latency, readiness units, measurement, schedules, currency validation, repair windows, and exceptions. Numerical recommendations come only from `src/scripts/timing_profile/`; constants remain in dataset code. This plan tracks adoption of those rules and the remaining lifecycle work. Remove it after every cohort has adopted the conventions and the evergreen documentation covers the completed mechanisms.

Prefer simple mechanisms and existing utilities. Accept occasional late alerts or a named exception without compromising reader or writer safety. If implementation reveals unexpected complexity, review the options before adding machinery.

## Changes and status

| Step | Scope | Status | Dataset adoption |
|---|---|---|---|
| B: measurement and documentation | K1: anonymous source timestamps, exported or live read-only check-ins, advisory timing and candidate comparisons; documentation foundation | Prepared | No timing constants changed |
| B′: urgent currency values | K9: fix `max_delay` values that require positions before their existing fires | Pending | None tracked here yet |
| A: overlap primitives | K2: configurable concurrency; K3: fresh-session conflict recovery; K4: metadata refresh using its committing session | Pending | Inert until a cohort enables overlap |
| C: virtual pilot and cohorts | K5/K9: poll windows, deadlines, and currency values; virtual lifecycle documentation | Pending | Pilot: HRRR 18-hour and RRFS sub-hourly; remaining virtual families follow |
| D: materialized pilot and cohorts | K2/K6/K7/K9: queue, deadlines, validation re-passes, untriggered schedules; K10: analysis boundary windows | Pending | Pilot: HRRR 48-hour; remaining materialized families follow |
| G: completion triggers and cohorts | K8: completion dispatch, Job identity, backstop decisions, admission; move consumer schedules with trigger activation | Pending | After both producer and consumer lifecycles are adopted |

Each later change updates this table with its PR link and adopted datasets. The core dependency order is B, A, C pilot, D pilot, G pilot, then family cohorts. Urgent currency corrections can proceed independently against existing crons. An untriggered cron move waits for review of the profiler's `S` comparison. A consumer's move from `S` to `T` lands with its trigger activation.

## Remaining lifecycle contracts

### Virtual polling

Discover about once per second and commit each tick. Exit when nothing remains pending. Poll until the next fire plus startup and an overlap allowance. Use `Allow` concurrency. A conflicted commit opens a fresh session, filters already-present files again, and recommits; a metadata refresh reads the store size from its committing session.

Pod and Job deadlines include the fire interval, startup, overlap, maximum observed validation duration, and shutdown allowance. Validation runs once after polling, and its failure exits with the validation-failure code. Virtual runs create no re-pass Job. Unreadable files are skipped for that pod's lifetime and retried at the next fire. Document these rules in the evergreen timing and virtual-dataset docs with the pilot implementation.

### Materialized admission and queue

Enable this lifecycle only with `Allow` concurrency and pod and Job deadlines equal to the fire interval. The update check-in covers the start decision, queue wait, processing, and finalize. Backoff sleep is outside it; Sentry's maximum runtime matches the pod deadline.

Store the Job UID and `run`, `skip`, or `finalized` in `state.json` in its coordination directory. Another UID's state counts as absent. Order Jobs by creation timestamp, then name. A Job is writing while Kubernetes reports it neither Complete nor Failed and its state is absent or `run`.

Pod zero decides before computing jobs. Every ordinary Job records `run`; every index waits until no older Job is writing before computing its jobs. Replacement pods reuse the recorded decision. A finalized Job exits successfully without writing again. Immediately after the finalize CAS, record `finalized` before cleanup; the final pod then always validates. Waiting only on older writers prevents deadlock and lets a queued Job proceed while an older finalized Job validates or sleeps.

Scheduled, trigger, and manual Jobs are distinguished by annotations. `dynamical.org/run-kind: trigger` identifies a trigger. Without `cronjob.kubernetes.io/instantiate: manual`, an untagged Job is scheduled; all others are manual, including operator clones and re-passes. Manual Jobs always run. Once triggers are active, a scheduled Job with a trigger created within half the fire interval before its creation first waits for older writers, then skips if currency passes. It records no `run` state before that decision.

### Materialized validation re-passes

A validation failure pages. Unless the Job is already `-r16`, wait 5, 10, or 20 minutes, then 30 minutes on subsequent steps. From one minute into sleep, check each minute and immediately before submission for a newer nonterminal Job with `run` state. Yield to such a Job; otherwise clone the next re-pass from the CronJob and exit with the validation-failure code. A re-pass recomputes work from store state. Interrupted sleeps are lost; a later trigger or fire covers recovery. Retain state files until ordinary cleanup.

Persistent integrity failures can cause repeated full passes and alerts while sleeping pods retain their resources. During a known persistent failure, suspend the CronJob or fix the validator. Add that guidance to the operations docs with the materialized lifecycle. Keep the deprecated plain-Zarr primary on its existing lifecycle until removal; reader safety remains mandatory.

### Triggers

Dispatch off the commit path and before validation when a virtual window started pending, becomes complete, skipped no unreadable files, and has not reached poll end. Completion by an overlapping writer still counts. A window complete at startup or incomplete at poll end sends no trigger.

Only consumers whose fire publishes exactly one append position receive completion triggers. Use deterministic consumer-and-unit Job names and the clone helper's conflict identity check. A producer triggers only consumer units it publishes; a consumer with several producers takes a trigger only from the producer that normally completes last. The IFS 46-day archiver dispatches once per initialization. Lost triggers are covered by the backstop cron.

Permit only `dynamical.org/run-kind: trigger` as the added Job annotation in clone admission, never on the pod template. Deploy that permission before the first annotated trigger. Activate each consumer's annotation, backstop schedule, and currency values together.

### Analysis boundaries

Extend forecast-derived analysis windows through the next cycle's boundary input, with an exclusive template end one position beyond it. The existing unconditional trim then removes the boundary rather than a complete position from the target cycle. Verify this using real region-job overrides and template factories before documenting the rule as adopted.

## Verification and rollout gates

- Measurement: aggregate all required indexes, members, tiers, and analysis boundary inputs before percentiles. Show coverage and missing or censored units. Validation sizing uses its observed maximum. Keep rewritten/reprocessed timestamps and virtual processing duration unmeasured where they do not establish the required quantity.
- Virtual overlap: test duplicate and mixed batches, primary/replica conflicts, metadata growth races, unreadable files, and predecessor disruption. Measure commit latency during overlap against the virtual target.
- Materialized queue: test simultaneous Jobs, tie ordering, stale UIDs, replacement and late-starting indexes, finalized-state recovery, terminal failed writers, zero-work workers, and work spanning a fire.
- Re-passes: test healthy and late sources, persistent missing data through the retry limit, and newer Jobs created during sleep that have not yet recorded `run` or have already failed.
- Trigger seams: test complete-at-start, incomplete, unreadable, conflicting final batches, another writer completing the entire batch, duplicate triggers, consumer cadence filtering, dispatch before validation, and every Job-kind classification.
- Admission: check allowed trigger annotations and reject other values, extra keys, and pod-template annotations using the real API before rollout. Verify Job-list permissions and clone label compatibility.
- Values: compare current and proposed fires, deadlines, and `max_delay` by cycle class. Confirm currency never requires a position before its intended fire, including scalar delays for multi-position units and the final rounded single cron. Unmeasured datasets keep their constants. Report GEFS sibling-order flips when evidence permits.

Roll out each cohort in a quiet source window after review. Mechanism reviews trace whole update paths and trigger-to-queue-to-validation seams; later value reviews use fresh measurement output. After all cohorts, audit code against the evergreen rules, remove superseded retry paths and this plan, and check for stale documentation.
