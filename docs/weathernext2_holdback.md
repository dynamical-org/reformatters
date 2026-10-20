# WeatherNext 2 publication policy and withdrawal

The operational virtual product publishes all available forecast steps as Google's source files arrive, including steps whose valid time is in the future. The historical 2022–2024 product is unchanged.

Operational updates revisit 17 days of initializations: the 15-day forecast horizon plus two days for late source files. The next normal update fills missing steps within that window, including steps withheld by a prior publication policy. A source step missing for longer than the retry window needs a whole-archive completeness scan and targeted backfill. `expected_forecast_length` describes the source horizon.

The publication switch is `GoogleWeathernext2ForecastVirtualRegionJob.publication_holdback` in `src/reformatters/google/weathernext2/forecast_virtual/region_job.py`. `None` publishes every available step. Set it to `pd.Timedelta("1h")` to admit only steps with `init_time + lead_time <= scheduled_fire_time - 1h`; a zero holdback still excludes future valid times. Delayed backfills capture the current UTC time minus one hour once per `get_jobs()` call. Historical processing ignores the switch. Delayed mode also restores the template extent and `CheckNoRefsInsideHoldback` validator. That validator probes representative chunks in the update window; it is a regression alert, not an exhaustive remediation audit.

Reinstating the delay also requires redeploying the valid-time gate in the `weathernext-proxy-worker` Worker in `dynamical-org/ops`. Removing Icechunk refs alone cannot block guessable source paths or listing routes. Follow the containment order below.

## Audit

`src/scripts/weathernext2_holdback.py audit` reads Icechunk metadata anonymously. It pins one snapshot and probes every chunk in every held-back `(init, lead, variable)` within the committed array grids, including all ensemble members and pressure levels. It reads no source weather chunks. The report identifies the snapshot and cutoff and lists counts for every affected step and variable. A nonzero count exits with status 1. This checks extra refs; the completeness scan in [validation.md](validation.md) checks missing expected refs.

For withdrawal, first contain access and stop writers as described below. Record the withdrawal cutoff with a UTC offset, equal to the withdrawal time minus one hour. The audit requires an explicit `--cutoff`, independent of the live publication switch. Use that same cutoff through deletion, publication, and the post-publication audit. Substitute the recorded values for the capitalized placeholders below; do not reuse an old incident's snapshot or cutoff.

An aging forecast can disappear from the restricted set without any deletion. This only resolves exposure for a fixed set of refs: an unrestricted writer continually adds later valid times. The last recorded valid time plus one hour is the age-out time only if nothing adds later refs. An audit of `main` does not certify other branches, tags, old snapshots, cached objects, or direct Worker routes.

## Withdrawal

These are operator actions, not part of running tests or merging a draft. Keep the operational update cron suspended in code until remediation is complete. Every deployment applies all cron definitions, so a cluster-only suspension can be undone by an unrelated merge. One operator must exclusively own `holdback-purge` from creation through publication and branch cleanup; the main-pointer CAS does not lock that temporary branch. Suspension prevents new jobs; also wait for already-running update and backfill jobs to finish before recording the starting snapshot.

1. Redeploy the proxy Worker's valid-time gate first, covering chunk access and object/listing routes. Verify the boundary and address cached responses before proceeding.

2. Set `publication_holdback = pd.Timedelta("1h")` on the WN2 region job and `suspend=True` on the operational update cron in code, then deploy. Verify its image and suspension, and wait for running update and backfill jobs to finish. Keep the Worker gate active throughout remediation and after updates resume.

3. Record the `main` snapshot, cutoff, ancestry, and any other branches/tags. The following read-only inventory reports each branch/tag tip and the full main ancestry; repeat it before expiry and after GC.

```sh
uv run python -c 'import icechunk; from reformatters.common.logging import get_logger; log = get_logger("wn2-inventory"); repo = icechunk.Repository.open(icechunk.http_storage("https://google-weathernext2.r2.dynamical.org/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk")); log.info({"branches": {name: repo.lookup_branch(name) for name in repo.list_branches()}, "tags": {name: repo.lookup_tag(name) for name in repo.list_tags()}}); log.info([(s.id, s.written_at.isoformat(), s.message) for s in repo.ancestry(branch="main")])'
```

Run the pinned public audit:

```sh
uv run src/scripts/weathernext2_holdback.py audit https://google-weathernext2.r2.dynamical.org/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET --output /tmp/wn2-before
```

Dry-run the deletion, then explicitly approve the mutation. The tool writes only to `holdback-purge`, committing one variable/manifest window at a time, and exhaustively verifies that branch after deletion. It leaves `main` untouched. Save the printed starting snapshot and branch tip, plus each window's time and process peak memory.

```sh
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py delete google-weathernext2-forecast-operational-virtual --cutoff CUTOFF_WITH_UTC_OFFSET
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py delete google-weathernext2-forecast-operational-virtual --cutoff CUTOFF_WITH_UTC_OFFSET --force
```

4. Publish only after reviewing the branch audit. Publication audits the branch again and atomically moves `main` only if it still matches the recorded starting snapshot. If another writer advanced `main`, investigate and restart the remediation from the new head; do not substitute the new head merely to bypass the check.

```sh
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py publish google-weathernext2-forecast-operational-virtual --from-snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py publish google-weathernext2-forecast-operational-virtual --from-snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET --force
```

5. Older snapshots still expose removed references. Resolve any additional branches/tags retaining them. Record the clean tip's `written_at` as `CLEAN_TIP_TIMESTAMP`. The expiry dry run prints only the URI and threshold, not affected snapshot IDs: review the recorded ancestry against the clean-tip timestamp, then compare ancestry after expiry and confirm the clean tip still opens before GC. Review the GC dry-run deletion counts separately; they destroy snapshot history and can break readers pinned to it. The storage-options secret must be the WeatherNext R2 secret, not the default AWS secret. These commands load credentials internally without printing them.

```sh
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key list --verbose
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key expire --older-than CLEAN_TIP_TIMESTAMP
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key expire --older-than CLEAN_TIP_TIMESTAMP --force
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key garbage-collect --older-than CLEAN_TIP_TIMESTAMP
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key garbage-collect --older-than CLEAN_TIP_TIMESTAMP --force
```

6. Confirm `main` still names the clean tip before expiry and after GC, and record the GC deletion summary. Repeat the public audit on the clean snapshot with the same cutoff. Verify an expired snapshot no longer opens over the public endpoint; check any CDN caches separately. The clean snapshot must remain readable.

7. Run one controlled update, including its final validation, while the schedule remains suspended. Start it just after an update schedule slot so its polling deadline has time remaining. Manual jobs use the most recent scheduled fire, not their start time, to choose the publication cutoff. If missing eligible steps have aged outside the 17-day window, run the [targeted backfill](backfill.md) first, starting at the oldest affected initialization. For scheduled fire `F`, compare the window start `floor_6h(F - 1h) - 17 days` against the oldest purged init; a later window start needs backfill. An add-only backfill cannot remove restricted refs and is not a substitute for the purge. Select the old gap with an exclusive `--filter-end` at the window start. Filters select whole regions, so a boundary region may also rewrite eligible refs just inside the window. Omit `--append-dim-end` when repairing the existing extent; extending it beyond the operational template would make the holdback guard fail until updates catch up.

If backfill is needed, the sizing below matches the operational backfill recorded in [meta #188](https://github.com/dynamical-org/meta/issues/188); measure memory for the repair scope.

```sh
DYNAMICAL_ENV=prod uv run main google-weathernext2-forecast-operational-virtual backfill-kubernetes --overwrite-chunks --filter-start OLDEST_PURGED_INIT --filter-end WINDOW_START --jobs-per-pod 30 --max-parallelism 10
```

Controlled update and validation:

```sh
kubectl create job --from=cronjob/google-wn2-forecast-operational-virtual-update wn2-holdback-first-update
kubectl --kubeconfig ~/.kube/low-permissions.kubeconfig --context low-permissions logs -f job/wn2-holdback-first-update
```

Inspect validation in the final update pod before the next update schedule slot changes its expected set. Record presence-probe count, manifests rewritten, commit time, peak RSS, and validation duration against the update pod deadline. Completeness and decode health each probe the 17-day window; the no-extra-refs guard adds its own representative probes. Use measurements before changing manifest splits or sharing presence results. If log streaming is unavailable, inspect job completion and the monitoring paths in [ops_card.md](ops_card.md). A whole-archive completeness scan uses its own wall-clock cutoff and can report newly eligible steps missing until the next update; it does not share an earlier worker cutoff.

8. Restore the schedule with a follow-up code change setting `suspend=False`, then verify deployment and subsequent scheduled fires. Unsuspension can immediately trigger a missed update schedule. Complete the deployment before the next update fire after the controlled update; if that window is missed, repeat the controlled update for the new window before resuming, so catch-up validation sees the expected data. Update catalog wording and examples to reflect the delayed policy and its valid-time boundary.
