# RRFS/REFS October 2026 array retirement

NOAA [SCN 26-48, October 15 update](https://www.weather.gov/media/notification/pdf_2026/scn26-048_Updated_RRFS_and_REFS_Implementation.aaf.pdf)
retires freezing-rain and frozen-rain/sleet accumulations and freezing-rain
probabilities. The corresponding messages disappear at the October 15 12Z
initialization in the public parallel feed. REFS hourly composite-reflectivity
spread also disappears from the inspected source files; its removal from this
archive is approved, although the notice does not announce that source change.

This migration removes 32 array paths from six existing v0.1.0 stores. It removes
them from **all initialization times on main**, including historical values.
The pre-migration snapshot remains accessible through a retained Icechunk tag.
The dataset version and URL do not change. Communicate this breaking schema
change and the historical tag naming pattern before execution. Send a completion
notice with the verified tag and snapshot for each store after execution.

| Dataset ID | Removed arrays |
| --- | --- |
| `noaa-rrfs-forecast-18-hour-virtual` | `freezing_rain_run_total_surface`, `freezing_rain_surface`, `frozen_precipitation_run_total_surface`, `frozen_precipitation_surface` |
| `noaa-rrfs-forecast-84-hour-virtual` | The same four arrays |
| `noaa-rrfs-forecast-sub-hourly-virtual` | `freezing_rain_run_total_surface`, `frozen_precipitation_run_total_surface` |
| `noaa-rrfs-ens-forecast-virtual` | `freezing_rain_run_total_surface`, `freezing_rain_surface` |
| `noaa-refs-forecast-hourly-virtual` | Three `neighborhood_probability_of_freezing_rain_1_hour_surface_above_*` arrays and `composite_reflectivity` |
| `noaa-refs-forecast-3-hourly-virtual` | Sixteen freezing-rain neighborhood probability arrays with 3/6/12/24-hour windows |

The migration script contains the exact allowlist. Categorical precipitation
types, percent frozen precipitation, snowfall, surface height, PMMN reflectivity,
and every other variable remain. No source selector is remapped to another
physical quantity.

## Release and migration are one operation

The validation-exception revert and all 32 template removals ship in the **same
PR and image**. Never deploy the revert separately. Retain partial-file ingestion:
one defective source file must not prevent healthy files from committing.

Merge and deploy this retirement before the generalized virtual partial-ingestion
change in #1218. That change must then rebase and pass its combined checks.
Deploying #1218 first would log errors for the retired fields until this retirement
deploys; the old missing-source validation exceptions do not suppress ingestion
errors. Main merges deploy automatically, so two nearby merges are not a reliable
way to obtain one combined deployment.

Normal operational metadata refresh does **not** delete arrays. Both the update
structural guard and metadata-overwrite guard reject a template missing an existing
array. A template-only deployment therefore stops new-image updates until migration.
Conversely, an old-image writer can recreate deleted arrays from its older template.
Neither migrating before deployment nor ignoring active old-image retries is safe.

## Choose a quiet window before merging

All steps here are for the production operator. Preparation and review do not
authorize an agent to merge, write production stores, trigger Jobs, or change
CronJobs. No suspension or manual trigger is needed in the planned procedure.

1. Finish review, tests, dry-run plans, credentials preparation, and the six-store
   command list before merging. Resolve each exact store URI from production STAC;
   do not use a bucket prefix guessed from a dataset name. Check the stored dataset
   ID and record the current snapshot. A plan made while updates are running must
   be refreshed after those writers finish.
2. Inspect actual CronJob schedules, active Jobs, retry Jobs, pod images, and
   deadlines using the read-only kubeconfig. Require **every old-image writer for
   a store** to finish before migrating that store, including application-created
   `-r1` Jobs. Building and deploying the new image may overlap those old jobs;
   deleting arrays may not. Do not infer quiescence from an absent original pod
   or a red cron check-in.
3. Allow enough time for CI, image publication, deployment, six fresh dry runs,
   all six migration commits, and verification before the next relevant fire.
   The observed deployment of the preceding release took up to roughly 25 minutes;
   use current workflow progress and a margin, not a promise based on that sample.
   Sub-hourly runs are the limiting cadence. A run may finish early, but its deadline
   can leave only about a minute before the next fire. Do not assume a full quiet
   hour. Prepare its plan and credentials in advance, refresh the expected head
   after its last old writer exits, then apply in that gap. If there is insufficient
   margin, use the controlled missed-fire fallback below; do not introduce a suspension.

The schedules observed for this incident are below. Re-read live schedules at
execution time; these times are UTC.

| Dataset | Scheduled fires | Pod deadline |
| --- | --- | --- |
| RRFS sub-hourly | Hourly at :15 | 59 minutes |
| RRFS 18-hour | 01:40, 04:40, 07:40, 10:40, 13:40, 16:40, 19:40, 22:40 | 60 minutes |
| RRFS 84-hour | 01:40, 07:40, 13:40, 19:40 | 135 minutes |
| RRFS ensemble | 01:15, 07:15, 13:15, 19:15 | 160 minutes |
| REFS hourly and 3-hourly | 01:55, 07:55, 13:55, 19:55 | 125 minutes |

For the five non-sub-hourly stores, a conservative example is after the 22:40
18-hour run and any retry finish, but before 01:15. At maximum deadlines that is
approximately 23:41–01:14. Start the build/deploy early enough to use this window,
and check actual completions rather than assuming the example applies.

**Expected rollout failures: zero for those five stores, and zero or one affected
scheduled fire for sub-hourly.** Target zero by migrating sub-hourly in its observed
post-run gap. If its old writer runs to the deadline and the gap is too short,
allow the first new-image fire to hit the structural guard, then migrate promptly
after confirming the old writer is gone. Pod retries have roughly five minutes of exponential backoff in total; migration
within that budget may recover the same fire. Otherwise the next natural fire
recovers it. Do not migrate while an old-image
writer is running merely to avoid that alert. One affected fire can generate
multiple pod exception events plus a cron-monitor failure; notification rules
prevent promising an exact email/push count. If more than one fire would be missed,
stop and escalate the rollout plan. No suspension or Job deletion is included.

REFS hourly remains red on its retired reflectivity field until the first
post-migration update. To meet the five-store zero-failure target, choose a window
after its existing fire and retry chain have terminal Job conditions. A validation
retry takes the CronJob image at the time of failure: before deployment it can
create another old-image writer; after deployment but before migration it can
create a new-image structural-guard failure. If deployment overlaps that decision,
budget an additional affected REFS hourly fire. Pod retries within an existing Job
keep that Job's image. A missing pod alone never establishes writer quiescence.

## Execute the approved migration

1. Merge the combined release and watch its normal deployment. Verify **all six**
   live CronJob templates use the combined image. Recheck that no old-image writer
   started during the build/deploy interval. If one did, wait for it to terminate
   before migrating its store; reassess the next-fire margin.
2. Run a fresh read-only plan for each store and review the exact deletion set,
   dataset identity, and expected main snapshot. Keep an operator ledger containing
   each URI, expected snapshot, historical tag, and migration result.
3. Apply each reviewed plan with its expected snapshot. Among stores with no active
   old writer, process the nearest upcoming fire first. Handle sub-hourly in its
   separate post-run gap; do not delay the other five while it is still running.
   Each store gets one deletion commit;
   tag creation is a separate reference write. There is no cross-store transaction.
4. If main moved, **stop and re-plan**. Never force or automatically rebase a schema
   deletion over an unknown concurrent writer. Do not expire snapshots or run
   garbage collection. If any store fails, leave completed stores alone, record
   the partial rollout, and resolve the failure before that store's next fire.
5. Verify each committed schema against the new template and verify retained refs,
   coordinate bytes, and extents against its saved pre-migration snapshot. The
   migration changes no retained array encoding or data values. The two approved
   PMMN comment corrections are applied separately by the normal metadata refresh
   on the next successful update; that later metadata commit is expected.
6. Observe the next natural updates. Strict current-data, manifest-completeness and
   decode-health checks should pass for the retained variables. No missing-source
   validation option remains. Keep the prior snapshots/tags until historical-access
   and retention policy explicitly permit otherwise.

For each ledger entry, substitute its reviewed dataset, S3 URI and snapshot:

```bash
RRFS_DATASET='noaa-rrfs-forecast-sub-hourly-virtual'
RRFS_STORE='s3://<reviewed-bucket>/<reviewed-store-prefix>'
RRFS_SNAPSHOT='<reviewed-main-snapshot>'

# Read-only plan; anonymous public-store access is the default.
uv run python src/scripts/migrate_rrfs_retired_arrays.py \
  --dataset "$RRFS_DATASET" --repo "$RRFS_STORE" \
  --expected-snapshot "$RRFS_SNAPSHOT"

# Operator only, after deployment and writer-quiescence checks.
uv run python src/scripts/migrate_rrfs_retired_arrays.py \
  --dataset "$RRFS_DATASET" --repo "$RRFS_STORE" \
  --expected-snapshot "$RRFS_SNAPSHOT" --auth-from-env --apply
```

The script uses existing environment credentials only when explicitly requested;
it does not load Kubernetes secrets. Local disposable repositories use `file:///`
URIs. It verifies the dataset identity and retired-array dimensions, creates the
historical tag, deletes exactly the approved nodes, and checks Icechunk's pending
diff before committing without rebase. Unconsolidated retained metadata is
untouched; inline consolidated metadata loses only the deleted array entries.

An immediate rerun with the original expected snapshot recognizes the exact tagged
direct migration commit and makes no changes. If any later update has advanced
main, that old invocation fails closed: inspect the ledger and history rather than
trying to delete again. A tag left by an interrupted attempt can be reused only
when it still points to the expected snapshot.

Do not substitute `overwrite-metadata`, raw S3 object deletion, or a full-store
rewrite. Verification should use the snapshot diff and selected reads, not a full
archive scan or dereferencing every virtual chunk.

## Historical access and rollback

The tag is `rrfs-retired-arrays-v1-before-<dataset-id>-<pre-migration-snapshot>`.
Record the emitted tag and snapshot ID for every store in the operator ledger. Open the
tag with `repo.readonly_session(tag=...)`, retaining the same virtual-container
credentials used for the live store. The old snapshot contains the retired array
metadata and references, but reading it still depends on NOAA retaining the source
GRIB objects. A tag does not copy or preserve NOAA's bytes.

Do not reset main to the historical snapshot after new updates have committed:
that would discard newer ingestion. A rollback needs its own reviewed plan to
restore the retired array nodes without losing intervening refs, together with a
matching image. Do not roll back only the image, which could recreate empty arrays.

## Validation report and STAC publication

Prepare local validation-report drafts by copying the currently published reports,
retaining their approved notes, removing the retired variable sections and table
rows, and updating the two PMMN comments. Report the source-method discontinuities
under dated `### Review notes`; do not encode archive-era facts as timeless variable
comments. Rebuild affected availability plots/statistics against the migrated schema
before publication: deleting Markdown sections does not regenerate a plot.

Use dated NOAA-notice notes for the categorical diagnosis and REFS PMMN method
changes; their exact first changed cycles are not established. Put these notes in
the validation reports, including historical PMMN no-echo floors and the fact that
zero can be valid reflectivity. Replace draft retirement language with the verified
per-store tags and snapshots in the completion notice.

After **all six** stores are migrated, their normal metadata refreshes have applied
the PMMN comments, and the updated validation reports are approved for publication:

```bash
# In the dynamical-stac checkout, after reading that repository's instructions:
./scripts/generate
git diff -- stac/
```

Review that the six collections omit exactly the retired variables and retain the
same store assets, dimensions, and unrelated metadata. Commit the regenerated
`stac/` output and hand off a catalog PR; do not publish a catalog-only removal
before the stores change. The generator reads live stores, so generating against
unmigrated production cannot validate the final catalog. Record that post-migration
step as operator work, not as a completed action in the code PR.

After rollout and publication, remove this single-use script, its tests and this
runbook in a cleanup change; retain the execution ledger and historical-access
notice outside the application source.
