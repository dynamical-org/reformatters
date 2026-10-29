# Operational update timing adoption plan

The [timing conventions](../operational_update_timing.md) define latency, readiness units, measurement, schedules, currency validation, repair windows, and exceptions. Numerical recommendations come only from `src/scripts/timing_profile/`; constants remain in dataset code. This plan tracks adoption of those rules and the remaining lifecycle work. Remove it after every cohort has adopted the conventions and the evergreen documentation covers the completed mechanisms.

Prefer simple mechanisms and existing utilities. Accept occasional late alerts or a named exception without compromising reader or writer safety. If implementation reveals unexpected complexity, review the options before adding machinery.

## Changes and status

| Step | Scope | Status | Dataset adoption |
|---|---|---|---|
| [B: measurement and documentation](https://github.com/dynamical-org/reformatters/pull/1261) | anonymous source timestamps, exported or live read-only check-ins, advisory timing and candidate comparisons; documentation foundation | Draft | No timing constants changed |
| B′: urgent currency values | audit `max_delay` against existing fires | Not needed: an audit found current `max_delay` values already satisfy the currency rule | No correction required |
| [A: overlap primitives](https://github.com/dynamical-org/reformatters/pull/1227) | configurable concurrency; fresh-session conflict recovery; metadata refresh using its committing session | Draft; implemented | Conflict recovery and metadata refresh apply to existing datasets; overlap activation is adopted per cohort |
| [C: virtual pilot](https://github.com/dynamical-org/reformatters/pull/1262) | poll windows, deadlines, and currency values; virtual lifecycle documentation | Draft; pilots implemented | HRRR 18-hour and RRFS sub-hourly virtual use overlapping polling with bounded Jobs and selected-fire validation; family adoption is tracked below |
| [D: materialized pilot and cohorts](https://github.com/dynamical-org/reformatters/pull/1260) | queue, deadlines, validation re-passes, untriggered schedules; analysis boundary windows | Draft; pilot implemented | HRRR 48-hour uses the materialized queue, Job-reference-time window, bounded re-passes, and chosen untriggered timing; remaining materialized families and analysis boundaries follow |
| [G: completion triggers and cohorts](https://github.com/dynamical-org/reformatters/pull/1265) | completion dispatch, Job identity, backstop decisions, admission; move consumer schedules with trigger activation | Draft; pilot implemented | HRRR 48-hour virtual uses overlapping polling and triggers its queued materialized consumer; the consumer uses its triggered backstop timing and currency allowance; remaining cohorts follow |
| [F1: virtual families](https://github.com/dynamical-org/reformatters/pull/1267) | overlapping polling, deadlines, grace and currency values | Draft; implemented | GFS forecast/analysis; GEFS analysis, 10-day and 16-day; HRRR analysis; RTMA; RRFS 84-hour, 18-hour and ensemble; REFS hourly/3-hourly; AIFS single; GEFS 35-day daily-window exception; WN2 operational with source timing retained and 3-second polling ticks |

Suspended WN3 and WN2 historical keep their existing lifecycle. Active RTMA-RU and URMA retain `Replace` concurrency and the [legacy validation retry policy](../operational_update_timing.md#virtual-polling); adopting overlapping `Allow` polling for them remains separate scheduling work. GEFS 35-day virtual retains its short daily polling window, and WN2 operational retains its gated source timing.

Adoption status is tracked here; recommendations do not activate polling, overlapping writers, retries, or triggers. Each later change updates this table with its PR link and adopted datasets. The core dependency order is B, A, C pilot, D pilot, G pilot, then family cohorts. The profiler recommends `S = min(p95 + max(p95 − p50, 5 min), p99 + 1 min)` for untriggered runs and retains all three candidates plus the chosen row for review. Untriggered cron moves and derived currency values use that chosen offset; adoption and named exceptions remain subject to manual review. A consumer's move from `S` to `T` lands with its trigger activation.

## Remaining lifecycle contracts

### Virtual polling

The [polling lead rule](https://github.com/dynamical-org/reformatters/pull/1291) is adopted by active `Allow` virtual crons. Fires use earliest first visibility across the measured history, leading it by 15 minutes unless that fire's cutoff precedes a retained class's complete-arrival p99, in which case they use 3 minutes. GEFS 16-day, HRRR 18-hour, RRFS sub-hourly and REFS 3-hourly use 3 minutes. Each fire respects its selected window. GEFS 35-day retains its daily window, and WeatherNext 2 retains its seven-hour landing gate and existing fire. RRFS sub-hourly requires 6/18 files at the newest position; older positions remain complete. Deadlines and currency values are unchanged.

Discover about once per second and commit each tick. Exit when nothing remains pending. Poll until the next fire plus startup and an overlap allowance. Use `Allow` concurrency. A conflicted commit opens a fresh session, filters already-present files again, and recommits; a metadata refresh reads the store size from its committing session.

Pod and Job deadlines include the fire interval, startup, overlap, maximum observed validation duration, and shutdown allowance. Validation runs once after polling, and its failure exits with the validation-failure code. Virtual runs create no re-pass Job. Unreadable files are skipped for that pod's lifetime and retried at the next fire. Document these rules in the evergreen timing and virtual-dataset docs with the pilot implementation.

### Materialized cohorts

The [materialized deadlines and monitoring](../operational_update_timing.md#materialized-deadlines-and-monitoring) and [queue and re-pass rules](../operational_update_timing.md#materialized-queue-and-re-passes) are implemented and documented with the pilot. Remaining work is cohort adoption, tracked above. The deprecated plain-Zarr primary retains its existing lifecycle until removal.

### Triggers

The [completion-trigger rules](../operational_update_timing.md#completion-triggers) and [dispatch mechanism](../parallel_processing.md#completion-triggers) are implemented with the pilot. Remaining work is cohort adoption, tracked above. Deploy the trigger annotation permission before activating each cohort.

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
