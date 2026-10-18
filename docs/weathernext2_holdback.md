# WeatherNext publication holdback

WeatherNext 2 operational and WeatherNext 3 virtual products are an exception to publishing forecasts as they arrive: only steps with `init_time + lead_time <= reference_time - 1h` are admitted under [Google's terms](https://storage.googleapis.com/weathernext-public/terms-of-use.pdf). Updates use the scheduled fire as `reference_time`; backfill workers and retries share the launch reference time. Newly eligible omissions are filled by the operational sweep.

WeatherNext 2 operational updates revisit 17 days of initializations: the 15-day forecast horizon plus two days for late source files. Recent initializations are intentionally partial. Coordinates retain all 60 leads; unpublished steps read as NaN. `expected_forecast_length` describes the source horizon, not the number of leads currently published. A source step missing for longer than the retry window needs a whole-archive completeness scan and targeted backfill.

The WeatherNext 2 update schedule is `5 1,7,13,19 * * *`; its final pod validates the result. Decode health samples both ends of the update window so a partial newest initialization does not eliminate long-lead coverage. `CheckNoRefsInsideHoldback` checks representative chunks in the update window; it is a regression alert, not an exhaustive remediation audit.

## WeatherNext 3 release validation

Start a controlled update just after a scheduled fire. Before unsuspending updates, require a [completeness scan](validation.md) and a controlled update with successful inline validation using the committed snapshot's `publication_cutoff`. Source completion markers alone are insufficient. If validation crosses a schedule boundary and uses a different cutoff, repeat the run. Audit every committed snapshot against its own cutoff, including snapshots from failed runs.

Measure the complete update, including commits and validation, within the configured memory and deadline. The first pod must succeed without restarts or replacements; stop replacements after a failure before repeating the timing run.

## Audit

`src/scripts/weathernext2_holdback.py audit` reads Icechunk metadata anonymously. It pins one snapshot and probes every chunk in every held-back `(init, lead, variable)` within the committed array grids, including all ensemble members and pressure levels. It reads no source weather chunks. The report identifies the snapshot and cutoff and lists counts for every affected step and variable. A nonzero count exits with status 1. This checks extra refs; the completeness scan in [validation.md](validation.md) checks missing expected refs.

Record a cutoff with a UTC offset equal to the audit time minus one hour. Use that same cutoff through deletion, publication, and the post-publication audit. Substitute the recorded values for the capitalized placeholders below; do not reuse an old incident's snapshot or cutoff.

```sh
uv run src/scripts/weathernext2_holdback.py audit https://google-weathernext2.r2.dynamical.org/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET --output /tmp/wn2-before
```

An aging forecast can disappear from the restricted set without any deletion. This only resolves exposure for a fixed set of refs: an old-rule writer continually adds later valid times. The last recorded valid time plus one hour is the age-out time only if nothing adds later refs. An audit of `main` does not certify other branches, tags, old snapshots, cached objects, or direct Worker routes.

## Recorded-cutoff audits

Use the snapshot's recorded `publication_cutoff` for target-reference and source-location audits, even after restricted forecasts have aged out. Explicit-cutoff mode is for legacy remediation. See [holdback_audit.py](../src/reformatters/google/weathernext_virtual/holdback_audit.py) for the audit entry points.

Independently audit source locations with the product's parser. Measure memory on a pilot before enumerating the full archive's locations.

Require provenance on every non-root snapshot and nondecreasing effective cutoffs along `main`; snapshot timestamps alone do not establish publication safety. Audit retained intermediate snapshots and branch tips as well as the finalized tip.

## Remediate an existing store

These are operator actions, not part of running tests or merging a draft. Keep the operational update cron suspended in code until remediation is complete. Every deployment applies all cron definitions, so a cluster-only suspension can be undone by an unrelated merge. One operator must exclusively own `holdback-purge` from creation through publication and branch cleanup; the main-pointer CAS does not lock that temporary branch. Suspension prevents new jobs; also wait for already-running update and backfill jobs to finish before recording the starting snapshot.

1. Set `suspend=True` on the operational update cron and deploy the holdback implementation. Verify its image, schedule, and suspension in the cluster. An emergency suspension before deploy can use the following commands, but must be followed by the durable code change:

```sh
kubectl patch cronjob google-wn2-forecast-operational-virtual-update --type=merge -p '{"spec":{"suspend":true}}'
kubectl get cronjob google-wn2-forecast-operational-virtual-update -o yaml
```

2. Record the `main` snapshot, cutoff, ancestry, and any other branches/tags. The following read-only inventory reports each branch/tag tip and the full main ancestry; repeat it before expiry and after GC.

```sh
uv run python -c 'import icechunk; from reformatters.common.logging import get_logger; log = get_logger("wn2-inventory"); repo = icechunk.Repository.open(icechunk.http_storage("https://google-weathernext2.r2.dynamical.org/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk")); log.info({"branches": {name: repo.lookup_branch(name) for name in repo.list_branches()}, "tags": {name: repo.lookup_tag(name) for name in repo.list_tags()}}); log.info([(s.id, s.written_at.isoformat(), s.message) for s in repo.ancestry(branch="main")])'
```

 Run the pinned public audit. Dry-run the deletion, then explicitly approve the mutation. The tool writes only to `holdback-purge`, committing one variable/manifest window at a time, and exhaustively verifies that branch after deletion. It leaves `main` untouched. Save the printed starting snapshot and branch tip, plus each window's time and process peak memory.

```sh
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py delete google-weathernext2-forecast-operational-virtual --cutoff CUTOFF_WITH_UTC_OFFSET
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py delete google-weathernext2-forecast-operational-virtual --cutoff CUTOFF_WITH_UTC_OFFSET --force
```

3. Publish only after reviewing the branch audit. Publication audits the branch again and atomically moves `main` only if it still matches the recorded starting snapshot. If another writer advanced `main`, investigate and restart the remediation from the new head; do not substitute the new head merely to bypass the check.

```sh
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py publish google-weathernext2-forecast-operational-virtual --from-snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py publish google-weathernext2-forecast-operational-virtual --from-snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET --force
```

4. Older snapshots still expose removed references. Resolve any additional branches/tags retaining them. Record the clean tip's `written_at` as `CLEAN_TIP_TIMESTAMP`. The expiry dry run prints only the URI and threshold, not affected snapshot IDs: review the recorded ancestry against the clean-tip timestamp, then compare ancestry after expiry and confirm the clean tip still opens before GC. Review the GC dry-run deletion counts separately; they destroy snapshot history and can break readers pinned to it. The storage-options secret must be the WeatherNext R2 secret, not the default AWS secret. These commands load credentials internally without printing them.

```sh
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key list --verbose
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key expire --older-than CLEAN_TIP_TIMESTAMP
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key expire --older-than CLEAN_TIP_TIMESTAMP --force
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key garbage-collect --older-than CLEAN_TIP_TIMESTAMP
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key garbage-collect --older-than CLEAN_TIP_TIMESTAMP --force
```

5. Confirm `main` still names the clean tip before expiry and after GC, and record the GC deletion summary. Repeat the public audit on the clean snapshot with the same cutoff. Verify an expired snapshot no longer opens over the public endpoint; check any CDN caches separately. The clean snapshot must remain readable.

6. Run one controlled update, including its final validation, while the schedule remains suspended. The update refreshes stored description metadata from the merged template. Start it just after an update schedule slot so its polling deadline has time remaining. Manual jobs use the most recent scheduled fire, not their start time, to choose the publication cutoff. If missing eligible steps have aged outside the 17-day window, run the [targeted backfill](backfill.md) first, starting at the oldest affected initialization. For scheduled fire `F`, compare the window start `floor_6h(F - 1h) - 17 days` against the oldest purged init; a later window start needs backfill. An add-only backfill cannot remove restricted refs and is not a substitute for the purge. Select the old gap with an exclusive `--filter-end` at the window start. Filters select whole regions, so a boundary region may also rewrite eligible refs just inside the window. Omit `--append-dim-end` when repairing the existing extent; extending it beyond the operational template would make the holdback guard fail until updates catch up.

If backfill is needed, measure memory for the repair scope and tune parallelism:

```sh
DYNAMICAL_ENV=prod uv run main google-weathernext2-forecast-operational-virtual backfill-kubernetes --overwrite-chunks --filter-start OLDEST_PURGED_INIT --filter-end WINDOW_START --jobs-per-pod 30 --max-parallelism 10
```

Controlled update and validation:

```sh
kubectl create job --from=cronjob/google-wn2-forecast-operational-virtual-update wn2-holdback-first-update
kubectl logs -f job/wn2-holdback-first-update
```

Inspect inline validation in the final update pod before the next update schedule slot changes its expected set. Record presence-probe count, manifests rewritten, commit time, peak RSS, and validation duration against the update pod deadline. Completeness and decode health each probe the 17-day window; the no-extra-refs guard adds its own representative probes. Use measurements before changing manifest splits or sharing presence results. If log streaming is unavailable, inspect job completion and the monitoring paths in [ops_card.md](ops_card.md). A manifest completeness scan uses the pinned snapshot's recorded cutoff and covers the requested range, the whole archive by default. A legacy operational snapshot without provenance uses a fresh cutoff and can report newly eligible steps missing. Replay `launch_scope` explicitly to check only a filtered launch's selection.

7. Restore the schedule with a follow-up code change setting `suspend=False`, then verify deployment and subsequent scheduled fires. Unsuspension can immediately trigger a missed update schedule. Complete the deployment before the next update fire after the controlled update; if that window is missed, repeat the controlled update for the new window before resuming, so catch-up validation sees the expected data. Update staging STAC wording and examples at the same time: use an eligible lead (such as 6h after a successful update), explain partial initializations, and remove claims of a 48-hour valid-time boundary. A latest-init lead-240h example is intentionally unfilled under this rule.

## Release boundary

Removing Icechunk refs does not prevent requests to guessable source paths or listing routes at `wn.dynamical.org`. Release requires an explicit decision on a Worker-side valid-time gate or documented authorization for that access, including cache behavior. Bucket-owner billing evidence or explicit acceptance of the egress risk is a separate gate. Neither is established by a successful manifest audit.
