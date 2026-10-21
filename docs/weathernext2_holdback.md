# WeatherNext publication policy and withdrawal

WeatherNext 2 operational and all four WeatherNext 3 virtual products publish every available source forecast step, including future-valid steps. Each family has an independent `publication_holdback` class variable: `GoogleWeathernext2ForecastOperationalVirtualRegionJob` and `GoogleWeathernext3ForecastVirtualRegionJob`. `None` admits all source steps; changing either line to `pd.Timedelta("1h")` restores `init_time + lead_time <= reference_time - 1h` for that family. WeatherNext 2 historical is unchanged. The shared `PUBLICATION_HOLDBACK` constant is the fixed one-hour duration used to interpret legacy provenance and audit delayed publication; it is not the live switch.

Updates use the scheduled fire as `reference_time`; backfill workers and retries share the launch reference time. Operational and backfill commits record `reference_time` and `launch_scope`, plus `publication_policy=unrestricted` or a delayed `publication_cutoff`. Completeness scans replay the recorded policy independently of the live switch. Existing delayed snapshots remain scannable. Snapshots without policy provenance require an explicit cutoff to reconstruct delayed expectations.

WeatherNext 2 operational updates revisit 17 days of initializations: the 15-day forecast horizon plus two days for late source files. All 60 leads are candidates for every initialization in the source-landing window; steps unavailable at the source read as NaN. `expected_forecast_length` describes the source horizon, not the number of leads currently published. A source step missing for longer than the retry window needs a whole-archive completeness scan and targeted backfill.

The WeatherNext 2 update schedule is `5 1,7,13,19 * * *` and remains suspended for the rollout below; its final pod validates the result. The unrestricted update keeps the existing extent, using a scheduling heuristic: `newest_init = floor_6h(reference_time - 7h)`. The seven-hour source-landing delay predicts availability rather than guaranteeing it, and is independent of forecast valid time: all 60 leads of that init are candidates. Source inits sampled in October 2026 landed at 6.1–7.5 hours after initialization (median 6.8 hours, excluding one 46-hour outlier). Decode health samples both ends of the update window. `CheckNoRefsInsideHoldback` runs only in delayed mode and checks representative chunks in that window; it is a regression alert, not an exhaustive remediation audit.

## Terms notices

WeatherNext 2 operational and all four WeatherNext 3 datasets use `license="other"`: Google's Real-Time Terms of Use and CC BY 4.0, as set out in those Terms. Every WeatherNext product describes its transformations in `modifications`; WeatherNext 2 historical retains its CC BY 4.0 attribution and license. WeatherNext 3 historical attribution cites the Earth Engine page for its grid.

Each of these five Icechunk repositories carries `notices/<revision>/TERMS_OF_USE.pdf`, `LEGALLY_BINDING_TERMS_OF_USE.txt`, and `NOTICE.txt`. Every group's `notice_files` attribute links to these files. The shared renderer takes the source product name from the operational template config. Revisions are the first 12 SHA-256 hex characters of the sorted file names and bytes (each followed by a NUL). Bundles are immutable: an existing path must contain exactly the expected bytes. Writers publish and read back the files before committing metadata that links to them. Whole-prefix copies include them; Icechunk session exports do not.

The daily `WeatherNext Terms check` workflow compares Google's live PDF with `src/reformatters/google/weathernext_terms_of_use.pdf` and fails on a fetch failure or changed bytes. On failure, distinguish download problems from changed terms. Review any new terms, replace the vendored PDF and `TERMS_SHA256` in `weathernext_terms.py`, update the notice wording as required, regenerate all five operational templates, and review its new bundle URLs before deploying. Keep prior bundle revisions available for older snapshots. Terms changes require an operator decision; the workflow does not alter publication.

## WeatherNext 3 release validation

WeatherNext 3 crons remain suspended. Updates revisit the source horizon plus two days so previously withheld steps and late files are rediscovered. Unrestricted updates include initializations through `floor(reference_time - 6h, init_frequency)`; source listings determine which steps are available. The six-hour source-landing delay is a scheduling heuristic, not an availability guarantee. It starts polling near the measured earliest arrivals: interim inits typically land at 6.4–6.7 hours, and synoptic inits at a median 7.2 hours (observed maximum 10.3 hours). All leads arrive in one burst. The hourly `10 * * * *` cadence and 25-minute poll window let later fires finish late arrivals without repeatedly listing inits too young to land.

Start a controlled update just after a scheduled fire. Before unsuspending updates, require a [completeness scan](validation.md) and a controlled update with successful inline validation. Scan the pinned snapshot using its recorded policy and reference time. Source completion markers alone are insufficient. If inline validation crosses a schedule boundary, repeat the run for the new launch scope.

Unrestricted WN2 and WN3 15-day completeness allow at least 5% of the newest init and require every older init to be complete. WN3 48-hour products allow at least 5% of each of the newest two inits and require every older init to be complete: a synoptic init can finish landing after the following hourly init starts. WN2 reserves five minutes of its 30-minute pod deadline for validation; WN3 retains its 15-minute grace. Delayed mode retains its original thresholds.

Measure the complete update, including commits and validation, within the configured memory and deadline. The first pod must succeed without restarts or replacements; stop replacements after a failure before repeating the timing run.

## Unrestricted rollout

Keep WN2 and all four WN3 crons suspended in code. Deploy the Worker change before ingesting unrestricted refs, including support for unrestricted whole-variable listings, and drain active writers. Do not use the first operational update to fill the backlog. Unrestricted updates reject oversized pending sets before source discovery; use the fixed-range backfill below.

1. At suspension, record the last successful old-policy snapshot, its scheduled fire, and its `publication_cutoff` (fire minus one hour). In that pinned snapshot, record `WITHHELD_START`: the oldest init whose last lead was not yet past the old cutoff, using `init_time + expected_forecast_length > publication_cutoff`. For WN2 this is the first six-hour init strictly after `publication_cutoff - 360h`. Inventory any older outage gaps separately.
2. Pin the rollout reference time `F`. WN2's exclusive `ROLLOUT_END` is `floor_6h(F - 7h) + 6h`; WN3's is `floor(F - 6h, init_frequency) + init_frequency`. These landing extents are scheduling heuristics, not guarantees. Backfill the fixed range `[WITHHELD_START, ROLLOUT_END)`. Keep that start if rollout is delayed; never replace it with a freshly computed `now - 17 days`. Extend the end for later landing inits as needed, and repair separately inventoried older outages. Use full ISO timestamps with seconds precision.
3. Pilot one dense, fully landed four-init WN2 manifest window within that fixed range and the committed extent at `--jobs-per-pod 4 --max-parallelism 1`. Set `PILOT_START` on a four-init manifest boundary and `PILOT_END` one day later. A single worker processes all four one-init jobs and builds one four-init commit. Record peak RSS and commit duration for that commit before choosing concurrency.

```sh
DYNAMICAL_ENV=prod uv run main google-weathernext2-forecast-operational-virtual backfill-kubernetes --overwrite-chunks --filter-start PILOT_START --filter-end PILOT_END --reference-time F --jobs-per-pod 4 --max-parallelism 1
```

One full WN2 init is `60 × 64 × (8 + 6 × 13) = 330,240` refs. One init job per pod bounds newly built refs to that size; larger workers can build up to four inits per manifest batch (1,320,960 refs). [Backfill sizing](backfill.md#sizing-a-virtual-backfill) distinguishes added refs from manifests rewritten. Use the measured pilot to choose concurrency before running the fixed range:

```sh
DYNAMICAL_ENV=prod uv run main google-weathernext2-forecast-operational-virtual backfill-kubernetes --overwrite-chunks --filter-start WITHHELD_START --filter-end ROLLOUT_END --reference-time F --jobs-per-pod 1 --max-parallelism CHOSEN_PARALLELISM
```

Keep `--overwrite-chunks`: the existing-store CLI has no add-only worker mode. With unchanged source objects it rewrites byte-identical refs, so retrying the fixed range is idempotent and resumable. `--overwrite-metadata` alone launches no workers. Omitting `--append-dim-end` repairs the committed extent; if it ends before `ROLLOUT_END`, add `--overwrite-metadata --append-dim-end ROLLOUT_END`. Preserve `F` on retries.

4. WN3 first enablement also requires an initial backfill through its fixed `ROLLOUT_END`, starting at the chosen initial archive start or recorded `WITHHELD_START` for an existing delayed store. Pilot at one job per pod and one concurrent pod. For a new store omit overwrite flags; for an existing store use the bounded rewrite procedure above. Backfills admit future-valid leads. WN3 enablement also requires the termination procedure below to be ready.
5. Pin the resulting snapshot and run the manifest scan over the fixed coverage range. Inspect residual missing refs and repair older gaps before a controlled update. Only the leading completeness tiers may still be mid-landing: the newest two inits for WN3 48-hour products, or the newest init for WN2 and WN3 15-day products. Every older init must be complete. Run one controlled update with successful inline validation while schedules remain suspended. Record its snapshot, pending count, peak RSS, commit duration, and validation duration. Restore schedules only in follow-up changes after these checks pass. If the end advances, extend coverage without moving `WITHHELD_START`.

### Update size limit

Before discovery, an unrestricted update sums the refs each pending coordinate produces, including ensemble members, pressure levels, and statistics. The bound is `manifest_init_split × refs per full product init`, derived from the same chunk axes used to build refs. WN2's bound is `4 × 60 × 64 × (8 + 6 × 13) = 1,320,960` refs. A new landing init plus one missed-fire init and one straggling lead across all variables total `2 × 330,240 + 5,504 = 665,984` refs; three full inits total 990,720. Seven older inits missing their six pressure-level variables plus one fully missing newest init total 2,426,880 refs and exceed the bound despite having only 3,360 coordinates.

WN3 bounds include all six statistics:

| Product | Refs per init | Manifest inits | Ref bound |
| --- | ---: | ---: | ---: |
| 15-day, 0.1° | `360 × 19 × 6 = 41,040` | 64 | 2,626,560 |
| 15-day, 0.05° | `360 × 2 × 6 = 4,320` | 64 | 276,480 |
| 48-hour, 0.1° | `48 × 19 × 6 = 5,472` | 384 | 2,101,248 |
| 48-hour, 0.05° | `48 × 2 × 6 = 576` | 384 | 221,184 |

A newest init plus one missed-fire init plus one entire straggling init uses 3/64 of the 15-day bound or 3/384 of the 48-hour bound. Source delays can exceed these ordinary cases; an oversized backlog fails with its ref count, bound, and this rollout link instead of building a giant ref batch. The size guard applies only to unrestricted updates; backfill batching is unchanged.

## Withdrawal containment

If delayed publication must be restored, first enable the Worker valid-time gate for chunk and listing routes, including cached responses, and suspend the affected update crons durably in code. Wait for active update and backfill jobs to finish before inventorying snapshots. Then restore the family's one-hour switch. Changing the switch prevents new restricted refs; it does not remove existing refs.

WeatherNext 3 withdrawal containment is the Worker gate plus suspending all four crons. WeatherNext 3 ref-removal tooling is not yet provided. The destructive commands below support only WeatherNext 2 operational.

## Termination

WN3 enablement is gated on readiness to carry out this procedure, distinct from restoring delayed publication:

1. Block the WeatherNext routes in the proxy Worker. Restoring the publication gate is not enough: it rejects only chunks that are currently ineligible, and already eligible chunks stay readable.
2. Cached responses in the Worker's `ChunkCache` have a 24-hour freshness lifetime, which does not guarantee deletion. The Worker has no purge operation, and a Cloudflare zone or hostname purge does not reach this cache, so deleting cached copies needs a restricted Workers Cache purge, which does not exist yet.
3. Suspend the crons and drain active writers.
4. For WN2, purge refs with the existing tool or delete the operational repo prefix.
5. For WN3, delete its repo prefixes; ref-removal tooling is not provided.
6. Post a dated notice in the dataset description and catalog.

## WeatherNext 2 withdrawal audit

`src/scripts/weathernext2_holdback.py audit` reads Icechunk metadata anonymously. It pins one snapshot and probes every chunk in every held-back `(init, lead, variable)` within the committed array grids, including all ensemble members and pressure levels. It reads no source weather chunks. The report identifies the snapshot and cutoff and lists counts for every affected step and variable. A nonzero count exits with status 1. This checks extra refs; the completeness scan in [validation.md](validation.md) checks missing expected refs.

Record a cutoff with a UTC offset equal to the audit time minus one hour. Use that same cutoff through deletion, publication, and the post-publication audit. Substitute the recorded values for the capitalized placeholders below; do not reuse an old incident's snapshot or cutoff.

```sh
uv run src/scripts/weathernext2_holdback.py audit https://google-weathernext2.r2.dynamical.org/google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk --snapshot START_SNAPSHOT --cutoff CUTOFF_WITH_UTC_OFFSET --output /tmp/wn2-before
```

An aging forecast can disappear from the restricted set without any deletion. This only resolves exposure for a fixed set of refs: an unrestricted writer continually adds later valid times. The last recorded valid time plus one hour is the age-out time only if nothing adds later refs. An audit of `main` does not certify other branches, tags, old snapshots, cached objects, or direct Worker routes.

## Recorded-cutoff audits

For delayed snapshots, use the recorded `publication_cutoff` for target-reference and source-location audits, even after restricted forecasts have aged out. Unrestricted snapshots have no delayed cutoff: withdrawal audits require an explicit `--cutoff`, as do all audit/delete/publish CLI commands. These remediation tools do not consult the live policy switches. See [holdback_audit.py](../src/reformatters/google/weathernext_virtual/holdback_audit.py) for the audit entry points.

Independently audit source locations with the product's parser. Measure memory on a pilot before enumerating the full archive's locations.

For an ancestry containing only delayed commits, require provenance on every non-root snapshot and nondecreasing effective cutoffs along `main`; snapshot timestamps alone do not establish publication safety. The recorded-cutoff ancestry and source-location audits reject unrestricted snapshots without a cutoff; use the explicit-cutoff target audit for withdrawal remediation. Audit retained intermediate snapshots and branch tips as well as the finalized tip.

## Remediate WeatherNext 2 after withdrawal

These are operator actions, not part of running tests or merging a draft. Keep the operational update cron suspended in code until remediation is complete. Every deployment applies all cron definitions, so a cluster-only suspension can be undone by an unrelated merge. One operator must exclusively own `holdback-purge` from creation through publication and branch cleanup; the main-pointer CAS does not lock that temporary branch. Suspension prevents new jobs; also wait for already-running update and backfill jobs to finish before recording the starting snapshot.

1. Set `suspend=True` on the operational update cron, restore its one-hour switch, and deploy. Verify its image, schedule, and suspension in the cluster. An emergency suspension before deploy can use the following commands, but must be followed by the durable code change:

```sh
kubectl patch cronjob google-wn2-forecast-operational-virtual-update --type=merge -p '{"spec":{"suspend":true}}'
kubectl --kubeconfig ~/.kube/low-permissions.kubeconfig --context low-permissions get cronjob google-wn2-forecast-operational-virtual-update -o yaml
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
kubectl --kubeconfig ~/.kube/low-permissions.kubeconfig --context low-permissions logs -f job/wn2-holdback-first-update
```

Inspect inline validation in the final update pod before the next update schedule slot changes its expected set. Record presence-probe count, manifests rewritten, commit time, peak RSS, and validation duration against the update pod deadline. Completeness and decode health each probe the 17-day window; the no-extra-refs guard adds its own representative probes. Use measurements before changing manifest splits or sharing presence results. If log streaming is unavailable, inspect job completion and the monitoring paths in [ops_card.md](ops_card.md). A manifest completeness scan uses the pinned snapshot's recorded policy and covers the requested range, the whole archive by default. Use `scan_manifest(..., publication_cutoff=...)` for a legacy delayed snapshot without provenance. Replay `launch_scope` explicitly to check only a filtered launch's selection.

7. Restore the schedule with a follow-up code change setting `suspend=False`, then verify deployment and subsequent scheduled fires. Unsuspension can immediately trigger a missed update schedule. Complete the deployment before the next update fire after the controlled update; if that window is missed, repeat the controlled update for the new window before resuming, so catch-up validation sees the expected data. When restoring delayed publication, update catalog wording and examples to describe partial initializations and use an eligible lead, such as 6h after a successful update.

## Release boundary

Removing Icechunk refs does not prevent requests to guessable source paths or listing routes at `wn.dynamical.org`. Withdrawal containment requires a Worker-side valid-time gate, including cache behavior. Bucket-owner billing evidence or explicit acceptance of the egress risk is a separate gate. Neither is established by a successful manifest audit.
