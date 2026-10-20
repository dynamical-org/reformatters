# WeatherNext 2 publication policy and withdrawal

The operational virtual product publishes all available forecast steps as Google's source files arrive, including steps whose valid time is in the future. The historical 2022–2024 product is unchanged.

Operational updates revisit 17 days of initializations: the 15-day forecast horizon plus two days for late source files. Routine updates retry missing steps within that window. Fill the held-back backlog with a backfill before enabling unrestricted updates; see the rollout below. A source step missing for longer than the retry window needs a whole-archive completeness scan and targeted backfill. `expected_forecast_length` describes the source horizon.

The publication switch is `GoogleWeathernext2ForecastVirtualRegionJob.publication_holdback` in `src/reformatters/google/weathernext2/forecast_virtual/region_job.py`. `None` publishes every available step. Set it to `pd.Timedelta("1h")` to admit only steps with `init_time + lead_time <= scheduled_fire_time - 1h`; a zero holdback still excludes future valid times. Delayed backfills capture the current UTC time minus one hour once per `get_jobs()` call. Historical processing ignores the switch. Delayed mode also restores the template extent and `CheckNoRefsInsideHoldback` validator. That validator probes representative chunks in the update window; it is a regression alert, not an exhaustive remediation audit.

Unrestricted updates include initializations through `floor_6h(scheduled_fire_time - 7h)`. The seven-hour source-landing delay is a scheduling heuristic, not an availability guarantee, and is independent of forecast valid time: all 60 leads of that init are candidates. Source files typically arrive together about 6–7.5 hours after initialization. Manifest completeness requires at least 5% of the newest ingested initialization and all of every older initialization; delayed mode retains full completeness for its eligible steps. WN2 reserves five minutes of its 30-minute pod deadline for validation, ending polling at scheduled fire plus 25 minutes.

Reinstating the delay also requires redeploying the valid-time gate in the `weathernext-proxy-worker` Worker in `dynamical-org/ops`. Removing Icechunk refs alone cannot block guessable source paths or listing routes. Follow the containment order below.

## Terms notices

The operational `license` attribute is a statement covering both “Real-Time Data & Forecasts” and “Historical Data”, using [Google's labels](https://developers.google.com/weathernext/guides/disclaimers) and linking to the applicable Terms and CC BY 4.0. Its attribution includes Google's copyright and required real-time citation. The notice bundle carries the Real-Time Experimental Data Terms; dynamical-stac maps the statement to STAC `license: other` with two titled license links. Values relating to a time less than one hour ago or in the future fall under those Terms; older values remain CC BY 4.0. This metadata does not change the publication holdback. The historical product retains its Earth Engine attribution and `license="CC-BY-4.0"`.

Each operational Icechunk repository carries `notices/<revision>/TERMS_OF_USE.pdf`, `LEGALLY_BINDING_TERMS_OF_USE.txt`, and `NOTICE.txt`. Every group's `notice_files` attribute links to these files. `NOTICE.txt` contains the copyright, binding notice, modifications, attribution, restricted-territory notice, and non-endorsement notice. The shared renderer takes the source product name, modifications, and territory/non-endorsement notices from the operational template config. Revisions are the first 12 SHA-256 hex characters of the sorted file names and bytes (each followed by a NUL). Bundles are immutable: an existing path must contain exactly the expected bytes. Writers publish and read back the files before committing metadata that links to them. Whole-prefix copies include them; Icechunk session exports do not.

The daily `WeatherNext Terms check` workflow compares Google's live PDF with `src/reformatters/google/weathernext_terms_of_use.pdf` and fails on a fetch failure or changed bytes. On failure, distinguish download problems from changed terms. Review any new terms, replace the vendored PDF and `TERMS_SHA256` in `weathernext_terms.py`, update the notice wording as required, regenerate the operational template, and review its new bundle URLs before deploying. Keep prior bundle revisions available for older snapshots. Terms changes require an operator decision; the workflow does not alter publication.

## Unrestricted rollout

Keep the operational update cron suspended in code. Deploy the unrestricted reformatter and the Worker's unrestricted chunk and listing access, and drain active writers. Do not use the first operational update to fill the backlog. Unrestricted updates reject oversized pending sets before source discovery; use the fixed-range backfill below. Backfill mode splits work by manifest window. WN3 is not on main.

1. At suspension, record the last successful old-policy snapshot, its scheduled fire, and its `publication_cutoff` (fire minus one hour). In that pinned snapshot, record `WITHHELD_START`: the oldest init whose last lead was not yet past the old cutoff, using `init_time + expected_forecast_length > publication_cutoff`. For WN2 this is the first six-hour init strictly after `publication_cutoff - 360h`. Inventory any older outage gaps separately.
2. Pin the rollout reference time `F`. The exclusive `ROLLOUT_END` is `floor_6h(F - 7h) + 6h`. This landing extent is a scheduling heuristic, not a guarantee. Backfill the fixed range `[WITHHELD_START, ROLLOUT_END)`. Keep that start if rollout is delayed; never replace it with a freshly computed `now - 17 days`. Extend the end for later landing inits as needed, and repair separately inventoried older outages. Use full ISO timestamps with seconds precision.
3. Pilot one dense, fully landed four-init WN2 manifest window within that fixed range and the committed extent at `--jobs-per-pod 4 --max-parallelism 1`. Set `PILOT_START` on a four-init manifest boundary and `PILOT_END` one day later. A single worker processes all four one-init jobs and builds one four-init commit. Record peak RSS and commit duration for that commit before choosing concurrency.

```sh
DYNAMICAL_ENV=prod uv run main google-weathernext2-forecast-operational-virtual backfill-kubernetes --overwrite-chunks --filter-start PILOT_START --filter-end PILOT_END --jobs-per-pod 4 --max-parallelism 1
```

One full WN2 init is `60 × 64 × (8 + 6 × 13) = 330,240` refs. One init job per pod bounds newly built refs to that size; larger workers can build up to four inits per manifest batch (1,320,960 refs). [Backfill sizing](backfill.md#sizing-a-virtual-backfill) distinguishes added refs from manifests rewritten. Use the measured pilot to choose concurrency before running the fixed range:

```sh
DYNAMICAL_ENV=prod uv run main google-weathernext2-forecast-operational-virtual backfill-kubernetes --overwrite-chunks --filter-start WITHHELD_START --filter-end ROLLOUT_END --jobs-per-pod 1 --max-parallelism CHOSEN_PARALLELISM
```

Keep `--overwrite-chunks`: the existing-store CLI has no add-only worker mode. With unchanged source objects it rewrites byte-identical refs, so retrying the fixed range is idempotent and resumable. `--overwrite-metadata` alone launches no workers. Omitting `--append-dim-end` repairs the committed extent; if it ends before `ROLLOUT_END`, add `--overwrite-metadata --append-dim-end ROLLOUT_END`. Keep other writers stopped until finalization.

4. Pin the resulting snapshot and run the manifest scan over the fixed coverage range, using its read-only snapshot session and the unrestricted publication switch. Use a fresh checkpoint directory for that snapshot. Inspect residual missing refs and repair older gaps before a controlled update; the newest init may still be mid-landing, but every older init must be complete. Run one controlled update with successful inline validation while the cron remains suspended, just after the chosen scheduled fire. Record its snapshot, pending count, peak RSS, commit duration, and validation duration. If the backfill misses that slot, advance the end and finish the corresponding backfill and pinned scan before the controlled update, without moving `WITHHELD_START`.
5. Restore `suspend=False` in a follow-up PR only after these checks pass. Verify the deployment and subsequent scheduled fires. Keep the publication switch at `None`.

### Update size limit

Before discovery, an unrestricted operational update's missing candidates must not produce more refs than one dense manifest batch: `manifest_init_split × refs per complete init`. Count every member and pressure-level chunk produced by each candidate, using the same chunk axes as ref generation. WN2's bound is `4 × 330,240 = 1,320,960` refs. A new landing init adds 330,240 refs; one missed fire adds at most another 330,240, plus a one-lead straggler across all variables adds 5,504: 665,984 refs. Even an entire additional late init totals 990,720, below the bound.

Source delays can exceed these ordinary cases; an oversized backlog fails with its count, bound, and this rollout link instead of building a giant ref batch. The size guard applies only to unrestricted operational updates; delayed publication, historical processing, and backfill manifest batching retain their existing paths.

## Termination

This procedure is distinct from restoring delayed publication:

1. Restore the Worker gate or block the route.
2. Restoring the gate blocks access to cached bytes immediately because it runs before the Worker's `ChunkCache`. Cached copies expire within 24 hours; a restricted purge operation does not exist yet, and a Cloudflare zone or hostname purge does not reach this cache.
3. Suspend the cron and drain active writers.
4. Purge WN2 refs with the existing tool or delete the operational repo prefix.
5. Post a dated notice in the dataset description and catalog.

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
