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
| [D: materialized pilot and cohorts](https://github.com/dynamical-org/reformatters/pull/1260) | queue, deadlines, validation re-passes, untriggered schedules; analysis boundary windows | Draft; pilot implemented | HRRR 48-hour uses the materialized queue, Job-reference-time window, bounded re-passes, and chosen untriggered timing; remaining materialized families are tracked below |
| [G: completion triggers and cohorts](https://github.com/dynamical-org/reformatters/pull/1265) | completion dispatch, Job identity, backstop decisions, admission; move consumer schedules with trigger activation | Draft; pilot implemented | HRRR 48-hour virtual uses overlapping polling and triggers its queued materialized consumer; the consumer uses a +2:04 backstop and matching currency allowance, measured from 365 days of its surface-only source unit; remaining cohorts follow |
| [F1: virtual families](https://github.com/dynamical-org/reformatters/pull/1267) | overlapping polling, deadlines, grace and currency values | Draft; implemented | GFS forecast/analysis; GEFS analysis, 10-day and 16-day; HRRR analysis; RTMA; RRFS 84-hour, 18-hour and ensemble; REFS hourly/3-hourly; AIFS single; GEFS 35-day daily-window exception; WN2 operational with source timing retained and 3-second polling ticks |
| [NOAA materialized family](https://github.com/dynamical-org/reformatters/pull/1269) | untriggered timing, materialized lifecycle, reference-time windows and analysis boundaries | Draft; implemented | GFS forecast +5:44 and analysis +3:58 (one cycle class), GEFS analysis +4:03 and 35-day tier 1 +7:15, HRRR analysis +1:10 (currency +3:10), and MRMS; GEFS extension keeps its 28-hour eligibility gate; the next day's tier-1 trigger requests the prior init at about +30:40, or the 07:15 backstop requests it at +31:15 when no trigger came; GFS/GEFS analysis remain schedule-only; MRMS keeps its current timing pending its separate schedule follow-up |
| [F3: other materialized families](https://github.com/dynamical-org/reformatters/pull/1268) | required materialized reference-time contract, lifecycle budgets and reviewed timing | Draft; implemented | IFS ENS 15-day and both 46-day variants, AIFS single and ensemble, ICON-EU, HRDPS, IMERG early and late, SWANN, SMAP, CHIRPS final and preliminary, and NDVI |
| [F4: trigger families](https://github.com/dynamical-org/reformatters/pull/1266) | producer completion mappings, per-init archive trigger identity and annotation, exact T backstops and currency regressions | Draft; implemented on virtual and materialized family lifecycles | GFS forecast (T +5:44) from GFS forecast virtual; GEFS 35-day tier 1 (T +7:15) from GEFS 16-day 0.5° virtual at 00Z only; AIFS single forecast (T +5:39) from AIFS single forecast virtual; IFS ENS 46-day daily and 6-hourly from the archiver. GEFS 35-day extension tier 2 has no completion trigger of its own. |

Suspended WN3 and WN2 historical keep their existing lifecycle. Active RTMA-RU and URMA retain `Replace` concurrency and the [legacy validation retry policy](../operational_update_timing.md#virtual-polling); adopting overlapping `Allow` polling for them remains separate scheduling work. GEFS 35-day virtual retains its short daily polling window, and WN2 operational retains its gated source timing.

Adoption status is tracked here; recommendations do not activate polling, overlapping writers, retries, or triggers. Each later change updates this table with its PR link and adopted datasets. The core dependency order is B, A, C pilot, D pilot, G pilot, then family cohorts. The profiler recommends `S = min(p95 + max(p95 − p50, 5 min), p99 + 1 min)` for untriggered runs and retains all three candidates plus the chosen row for review. Untriggered cron moves and derived currency values use that chosen offset; adoption and named exceptions remain subject to manual review. A consumer's move from `S` to `T` lands with its trigger activation.

## Remaining lifecycle contracts

### Virtual polling

Discover about once per second and commit each tick. Exit when nothing remains pending. Poll until the next fire plus startup and an overlap allowance. Use `Allow` concurrency. A conflicted commit opens a fresh session, filters already-present files again, and recommits; a metadata refresh reads the store size from its committing session.

Pod and Job deadlines include the fire interval, startup, overlap, maximum observed validation duration, and shutdown allowance. Validation runs once after polling, and its failure exits with the validation-failure code. Virtual runs create no re-pass Job. Unreadable files are skipped for that pod's lifetime and retried at the next fire. Document these rules in the evergreen timing and virtual-dataset docs with the pilot implementation.

### Materialized cohorts

The [materialized deadlines and monitoring](../operational_update_timing.md#materialized-deadlines-and-monitoring) and [queue and re-pass rules](../operational_update_timing.md#materialized-queue-and-re-passes) are implemented and documented with the pilot. Remaining work is cohort adoption, tracked above. The deprecated plain-Zarr primary retains its existing lifecycle until removal.

### Triggers

The [completion-trigger rules](../operational_update_timing.md#completion-triggers) and [dispatch mechanism](../parallel_processing.md#completion-triggers) are implemented with the pilot. Remaining work is cohort adoption, tracked above. Deploy the trigger annotation permission before activating each cohort.

### Analysis boundaries

GFS and GEFS materialized analysis adopt the [boundary-window rule](../operational_update_timing.md#repair-window) with real-builder, accumulated-input, trimming, and currency regressions in the NOAA materialized family.

## Verification and rollout gates

- Measurement: aggregate all required indexes, members, tiers, and analysis boundary inputs before percentiles. Show coverage and missing or censored units. Validation sizing uses its observed maximum. Keep rewritten/reprocessed timestamps and virtual processing duration unmeasured where they do not establish the required quantity.
- Virtual overlap: test duplicate and mixed batches, primary/replica conflicts, metadata growth races, unreadable files, and predecessor disruption. Measure commit latency during overlap against the virtual target.
- Materialized queue: test simultaneous Jobs, tie ordering, stale UIDs, replacement and late-starting indexes, finalized-state recovery, terminal failed writers, zero-work workers, and work spanning a fire.
- Re-passes: test healthy and late sources, persistent missing data through the retry limit, and newer Jobs created during sleep that have not yet recorded `run` or have already failed.
- Trigger seams: test complete-at-start, incomplete, unreadable, conflicting final batches, another writer completing the entire batch, duplicate triggers, consumer cadence filtering, dispatch before validation, and every Job-kind classification.
- Admission: check allowed trigger annotations and reject other values, extra keys, and pod-template annotations using the real API before rollout. Verify Job-list permissions and clone label compatibility.
- Values: compare current and proposed fires, deadlines, and `max_delay` by cycle class. Confirm currency never requires a position before its intended fire, including scalar delays for multi-position units and the final rounded single cron. Unmeasured datasets keep their constants. Report GEFS sibling-order flips when evidence permits.

Roll out each cohort in a quiet source window after review. Mechanism reviews trace whole update paths and trigger-to-queue-to-validation seams; later value reviews use fresh measurement output. After all cohorts, audit code against the evergreen rules, remove superseded retry paths and this plan, and check for stale documentation.

## Other materialized family adoption

| Dataset | Timing adoption | Lifecycle adoption |
|---|---|---|
| IFS ENS 15-day | Keep 08:05 UTC and 8h40m currency; separate source/schedule gate | Draft |
| IFS ENS 46-day daily | +58h05m backstop and currency | Draft |
| IFS ENS 46-day 6-hourly | +58h05m backstop and currency | Draft |
| AIFS single | +5h39m, hand value from 2026-10-27 12Z onward | Draft |
| AIFS ensemble | +5h59m, hand value from 2026-10-27 12Z onward | Draft |
| ICON-EU | Keep +3h52m fire; +3h52m currency matches the fire | Draft |
| HRDPS | Keep deployed +4h30m; hand value pending source-tail decision | Draft |
| IMERG early / late | Keep unmeasured timing | Draft |
| SWANN / SMAP | Keep unmeasured timing; rewrites are not arrival evidence | Draft |
| CHIRPS final / preliminary | Keep manual timing | Draft |
| NDVI | Keep manual timing | Draft |

All datasets in this cohort use Allow concurrency and pod/Job deadlines equal to `min(Δf, 6h)`; CHIRPS final uses its minimum calendar fire interval for `Δf`. Every materialized builder takes a required reference time. Existing IFS 46-day archiver submissions stay unannotated until the trigger cohort enables the backstop check.

AIFS single/ensemble use hand values from six complete units beginning 2026-10-27 at 12Z, rounded with `ceil_minute(p99 + 3 min)`. The source provider's 2026-10-20 06Z – 2026-10-27 06Z release slowdown, which it acknowledged as a bug and reverted, is excluded; values use units from 2026-10-27 12Z onward. The sample is small; re-profile after about two weeks. HRDPS keeps its deployed +4h30m schedule; its tail-dominated p99 and short history require a separate timing decision. ICON-EU keeps its +3h52m fire and sets currency to that same offset. IMERG, SWANN and SMAP keep unmeasured source timing; CHIRPS and NDVI keep manual timing. IFS ENS 15-day keeps its source, schedule and currency behind its separate measurement gate while adopting lifecycle budgets.
