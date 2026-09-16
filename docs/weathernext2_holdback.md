# WeatherNext 2 publication holdback

The operational virtual product admits a forecast step when `init_time + lead_time <= scheduled_fire_time - 1h`. Backfills capture one clock value for the same rule. The historical 2022–2024 product is independent of this clock. The boundary follows [Google's terms](https://storage.googleapis.com/weathernext-public/terms-of-use.pdf) and the valid-time interpretation recorded in [meta #201](https://github.com/dynamical-org/meta/issues/201).

Operational updates revisit 17 days of initializations: the 15-day forecast horizon plus two days for late source files. Recent initializations are intentionally partial. Coordinates retain all 60 leads; unpublished steps read as NaN. `expected_forecast_length` describes the source horizon, not the number of leads currently published. A source step missing for longer than the retry window needs a whole-archive completeness scan and targeted backfill.

The update schedule is `5 1,7,13,19 * * *`; validation follows at `5 2,8,14,20 * * *`. Decode health samples both ends of the update window so a partial newest initialization does not eliminate long-lead coverage. `CheckNoRefsInsideHoldback` checks representative chunks in the update window; it is a regression alert, not an exhaustive remediation audit.

## Audit

`src/scripts/weathernext2_holdback.py audit` reads Icechunk metadata anonymously. It pins one snapshot and probes every chunk in every held-back `(init, lead, variable)` within the committed array grids, including all ensemble members and pressure levels. It reads no source weather chunks. The report identifies the snapshot and cutoff and lists counts for every affected step and variable. A nonzero count exits with status 1. This checks extra refs; the completeness scan in [validation.md](validation.md) checks missing expected refs.

Record a cutoff with a UTC offset equal to the audit time minus one hour. Use that same cutoff through deletion, publication, and the post-publication audit. Substitute the recorded values for the capitalized placeholders below; do not reuse an old incident's snapshot or cutoff.

```sh
uv run src/scripts/weathernext2_holdback.py audit https://google-weathernext2.r2.dynamical.org/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET --output /tmp/wn2-before
```

An aging forecast can disappear from the restricted set without any deletion. This only resolves exposure for a fixed set of refs: an old-rule writer continually adds later valid times. The last recorded valid time plus one hour is the age-out time only if nothing adds later refs. An audit of `main` does not certify other branches, tags, old snapshots, cached objects, or direct Worker routes.

## Remediate an existing store

These are operator actions, not part of running tests or merging a draft. Keep both operational crons suspended in code until remediation is complete. Every deployment applies all cron definitions, so a cluster-only suspension can be undone by an unrelated merge. Suspension prevents new jobs; also wait for already-running update and backfill jobs to finish before recording the starting snapshot.

1. Deploy the holdback implementation with both operational crons suspended. Verify their images, schedules, and suspension in the cluster. An emergency suspension before deploy can use the following commands, but must be followed by the durable code change:

```sh
kubectl patch cronjob google-wn2-forecast-operational-virtual-update --type=merge -p '{"spec":{"suspend":true}}'
kubectl patch cronjob google-wn2-forecast-operational-virtual-validate --type=merge -p '{"spec":{"suspend":true}}'
kubectl get cronjob google-wn2-forecast-operational-virtual-update google-wn2-forecast-operational-virtual-validate -o yaml
```

2. Record the `main` snapshot, cutoff, ancestry, and any other branches/tags. Run the pinned public audit. Dry-run the deletion, then explicitly approve the mutation. The tool writes only to `holdback-purge`, committing one variable/manifest window at a time, and exhaustively verifies that branch after deletion. It leaves `main` untouched. Save the printed starting snapshot and branch tip, plus each window's time and process peak memory.

```sh
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py delete google-weathernext2-forecast-operational-virtual --cutoff CUTOFF_WITH_UTC_OFFSET
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py delete google-weathernext2-forecast-operational-virtual --cutoff CUTOFF_WITH_UTC_OFFSET --force
```

3. Publish only after reviewing the branch audit. Publication audits the branch again and atomically moves `main` only if it still matches the recorded starting snapshot. If another writer advanced `main`, investigate and restart the remediation from the new head; do not substitute the new head merely to bypass the check.

```sh
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py publish google-weathernext2-forecast-operational-virtual --from-snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET
DYNAMICAL_ENV=prod uv run src/scripts/weathernext2_holdback.py publish google-weathernext2-forecast-operational-virtual --from-snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET --force
```

4. Older snapshots still expose removed references. Resolve any additional branches/tags retaining them. Record the clean tip's `written_at` as `CLEAN_TIP_TIMESTAMP`. Review expiry and GC separately; they destroy snapshot history and can break readers pinned to it. The storage-options secret must be the WeatherNext R2 secret, not the default AWS secret. These commands load credentials internally without printing them.

```sh
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key list --verbose
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key expire --older-than CLEAN_TIP_TIMESTAMP
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key expire --older-than CLEAN_TIP_TIMESTAMP --force
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key garbage-collect --older-than CLEAN_TIP_TIMESTAMP
DYNAMICAL_ENV=prod uv run src/scripts/icechunk_utils.py --repo s3://dynamical-google-weathernext2/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --k8s-secret weathernext2-storage-options-key garbage-collect --older-than CLEAN_TIP_TIMESTAMP --force
```

5. Confirm `main` still names the clean tip before expiry and after GC, and record the GC deletion summary. Repeat the public audit on the clean snapshot with the same cutoff. Verify an expired snapshot no longer opens over the public endpoint; check any CDN caches separately. The clean snapshot must remain readable.

6. Run one controlled update and validation while schedules remain suspended. The update refreshes stored description metadata from the merged template. Start it just after an update schedule slot so its polling deadline has time remaining. Manual jobs use the most recent scheduled fire, not their start time, to choose the publication cutoff. If missing eligible steps have aged outside the 17-day window, run the [targeted backfill](backfill.md) first, starting at the oldest affected initialization. For scheduled fire `F`, compare the window start `floor_6h(F - 1h) - 17 days` against the oldest purged init; a later window start needs backfill. An add-only backfill cannot remove restricted refs and is not a substitute for the purge.

```sh
kubectl create job --from=cronjob/google-wn2-forecast-operational-virtual-update wn2-holdback-first-update
kubectl logs -f job/wn2-holdback-first-update
kubectl create job --from=cronjob/google-wn2-forecast-operational-virtual-validate wn2-holdback-first-validate
kubectl logs -f job/wn2-holdback-first-validate
```

Run validation promptly after the update completes, before the next update schedule slot changes its expected set. Record presence-probe count, manifests rewritten, commit time, peak RSS, and validation duration against the 30-minute deadline. Completeness and decode health each probe the 17-day window; the no-extra-refs guard adds its own representative probes. Use measurements before changing manifest splits or sharing presence results.

7. Restore both schedules with a follow-up code change setting `suspend=False`, then verify deployment and subsequent scheduled fires. Unsuspension can immediately trigger a missed schedule. Update staging STAC wording and examples at the same time: use an eligible lead (such as 6h after a successful update), explain partial initializations, and remove claims of a 48-hour valid-time boundary. A latest-init lead-240h example is intentionally unfilled under this rule.

## Release boundary

The public proxy at `wn.dynamical.org` is maintained in `dynamical-org/ops`. Removing Icechunk refs does not prevent requests to guessable source paths or listing routes. Release requires an explicit decision on a Worker-side valid-time gate or documented authorization for that access, including cache behavior. Bucket-owner billing evidence or explicit acceptance of the egress risk is a separate gate. Neither is established by a successful manifest audit.
