# Operational update timing

Publish as soon as the source has the data. Timing values are hand-set constants in each dataset's code. This document defines the rules for selecting those values; the [adoption plan](plans/operational_update_timing.md) tracks their rollout.

## Latency and readiness units

Latency runs from source availability to readability in the primary store. Measure the source inputs the dataset actually consumes:

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

The numerical formulas are implemented in the profiler. The late threshold is `T = p95 + max(p95 − p50, 15 min)`. The draft untriggered schedule offset is `S = p95 + max(p95 − p50, 5 min)`; the profiler compares this with other candidate offsets against the measured distribution. `T` identifies a late source. `S` selects the cut between normal arrival and an anomalous delay for an untriggered materialized dataset. Choose `S` by hand from the candidate comparison before moving a cron or deriving values from it. The fraction of complete observed units arriving after `S` is arrival evidence, not a prediction of validation alerts: a multi-position unit can pass with a complete prefix.

## Schedules

Use one cron per dataset, with one minute offset across all its cycles:

- Virtual schedules fire at `a + E − L − 1 min`, using raw earliest file arrival across all tiers and cycle classes. Later classes spend the difference polling.
- Untriggered materialized schedules use the latest class's `S` for the first readiness tier. Earlier classes accept that wait as latency.
- Where a materialized dataset has a completion trigger, its cron is a backstop at the first readiness tier's `T`. Change a consumer's cron together with activation of its trigger.

Check every cycle class against the resulting single cron. Configure deadlines using measured startup, processing, and validation, leaving room for validation after polling. See [virtual operational updates](virtual_datasets.md#operational-updates-single-writer) and [parallel processing](parallel_processing.md) for the mechanisms in use.

## Currency and integrity validation

`CheckCurrentData.max_delay` is measured from an append position's label. It must never require a position before the fire intended to publish it:

- For virtual datasets, `max_delay = T_first_file + 4 min + (a − label)`. Here first file means the file that makes a new position visible. Where discovery gates each position on its own-cycle files, measure the maximum of those files per position, then the minimum across positions. Forecast virtuals use the earliest usable file. Polling start `E` always uses the raw earliest usable file.
- For materialized datasets, `max_delay = F + (a − earliest label) + max(0, Δf − anchor spacing)`, where `F` is the intended fire's offset (`S` for an untriggered run, `T` for a backstop), and `Δf` is the fire interval. The last term accounts for a fire covering several anchors.

Use one scalar per dataset. Later positions in a multi-position unit are detected later. Compute it against the final single cron, including minute rounding and every cycle class, rather than against processing completion. Use the profiler's recommendations to review the current cron, deadlines, and `max_delay` together.

Currency does not replace integrity checks. Use `CheckVirtualManifestCompleteness` to require complete older positions and detect interior gaps, and use recent NaNs, decode health, replica, and expected-shard checks as appropriate. Tune leading completeness tiers and NaN thresholds by hand when they false-alert. Integrity failures can page before the currency fire. Validation errors page, and Sentry cron monitors alert on missed, timeout, and error check-ins. See [validation](validation.md) for configuring and inspecting checks.

## Repair window

`operational_update_jobs` defines the repair window, usually starting inclusively at the latest stored position. A dataset that needs a wider window defines one. Gaps outside it page and require a [backfill](backfill.md). Forecast-derived analyses publish a non-ragged leading edge through result trimming or discovery gating.

## Named exceptions

The dataset implementations define the values and boundaries of these exceptions:

- GEFS 35-day virtual uses one daily window; intermediate extension waves can wait until the next fire.
- GEFS 35-day materialized extension has no completion trigger and waits for an eligible later run after its age gate.
- WeatherNext 2 uses all members and levels of one variable and lead as its unit, measured from the unit's last object.
- MRMS uses one hourly run; pass 1 is an exception.

Arrivals beyond a fire's chosen offset alert when currency or integrity checks detect them. Choose offsets to make this rare under normal source operation.
