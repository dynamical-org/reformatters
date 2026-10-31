# Bounded ENS profiling backfill

Manual Backfill's `profile=true` runs all 19 IFS ENS variables for March 30–31,
2024, with parallelism 2. It uses the deployed main image and the normal MARS
reader, transforms, encodings, worker assignment and commit path. The separate
template starts March 30 and preserves the production coordinate chunk size
5840. The short scratch store measures neither early-MARS/Open Data throughput
nor full-history manifest/commit cost.

The driver generates a fresh UUID after validating the request. Data goes to
`s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens/<run-id>/data/ifsens-profile/v0.0.1-profiling.icechunk`;
coordination stays under that run's `data/` and profiles under its `profiles/`.
The instance is unregistered, has no CronJob, replica or catalog entry, and
cannot select production or an existing store. Worker 0 creates a clean
repository and tags its root; other workers open only that owned repository.
A setup crash requires a fresh workflow dispatch/run id, not worker-0 recovery.

Profiling is off by default. Its wrapper adds per-phase wall time for download,
read/decode, transform, local chunk write, upload, commit, setup and waits;
20 Hz nonblocking py-spy stacks with subprocess discovery enabled; one-second cgroup/process CPU,
memory and peak-memory readings, network counters and filesystem use. Immutable
phase/resource segments upload every 30 seconds. Final evidence includes
checksums, assignment, image commit, lock hash, workload exit receipt and stack
profile. The in-pod sampler self-test must pass before source work. It measures
five plain/sampled pairs each of fixed CPU work and imports, under a shared
240-second probe deadline. Either median paired ratio above 1.10 sets
`sampler_overhead_warning` in the initial and final manifests and logs a pod-level
warning; work continues. The threshold flags measurement bias, not destination
safety or a cost limit. Host contention can inflate the ratio; a warning
preserves the work while marking its timing estimate as potentially biased. The evidence
verifier lists affected attempts, including ones without a final manifest.
Timeouts, a failed sampler executable or a self-test with no samples still fail
before source work. There is no automatic retry or re-dispatch.
Nonblocking sampling avoids pausing the workload but can miss or read inconsistent
stacks. Inspect sampler errors/lag and per-process coverage; use phase/resource
metrics for timing and treat stacks as approximate attribution. Short-lived
children can finish before discovery, so missing child stacks are not evidence
that write-side child work is cheap. Compare phase totals with stack coverage.
Sampler process failure, upload failure, missing exit receipt or source failure fails the
attempt; so does failed profiling, defined below. CPU is requested, not limited;
parent/child CPU and overlapping phase wall times are not additive. Network deltas include index/profile traffic;
filesystem use includes concurrent downloads, chunk output and profile files.
Sampler CPU/RSS is recorded separately within the process-tree counters; cgroup
totals include it. Final manifests record numeric sample/error counts, error
fraction, lag warnings and per-process coverage. Error fractions above 0.25 or
missing totals mark stacks low-confidence without failing phase/resource evidence.
Parent-only coverage is explicit; it does not prove the absence of child work.
Failed profiling means any of: sampler failure, kill, signal, flush failure, or a
profile that is missing, is invalid JSON, lacks valid `profiles`, or has zero samples;
each fails the attempt, and `profile_error` names a profile failure. Non-UTF-8 output is
the one exception to the profile gate, because nonblocking sampling can write it into
frame strings. The profile is decoded strictly first (`profile_decode: strict`). On a
UTF-8 decode error only, a replacement decode in memory counts the samples
(`profile_decode: replacement`); the raw profile is uploaded and hashed unchanged. If
every replacement character is inside `shared.frames` strings, the attempt can
complete, `py_spy_low_confidence` is forced true, and the manifest records
`profile_replacement_chars`, `profile_affected_frames` and `profile_affected_samples`
(samples referencing an affected frame). Replacement characters elsewhere are
`replacement_outside_frames`, a failed attempt. Do not rank frames from a
replacement-decoded profile: its other frames are not proven correct either. The
initial manifest's `selftest_profile_quality` records the same fields for the self-test.
The private sampler log is copied to pod stderr after recording and is not
uploaded. Retain pod logs when interpreting missing or weak stack evidence.

## Stack coverage and parent-only pilot scope

Child sampling is not guaranteed. Inspect every pod separately: parent-only
stacks still support phase wall-time and cgroup CPU/RSS measurements, but cannot
attribute worker-pool CPU. Reads use workload threads; child processes perform
write-side work. A complete evidence export is not proof of useful stack coverage.
The operator accepts this reduced scope before dispatch and includes per-pod
child hits, errors, lag and effective sample rates in the analysis. Cancellation
can lose the stack profile entirely; flushed phase/resource segments remain useful.

## Acceptance and launch

Merge deploys the shared image through the normal main pipeline. Dispatch waits
for that exact commit's deployment and uses its SHA-tagged image. There is no
separate image upload or staging deployment. The operator runs every workflow
dispatch, Kubernetes mutation and cleanup below; agents only observe read-only.
Commands use Bash and GNU date. Start each session with `set -euo pipefail`.

Before dispatch, the operator accepts **code-only isolation using the existing
production-capable storage key**, public scratch data/profiles, and sponsorship
fit for temporary objects in the Open Data bucket. This is not an IAM boundary.
The production storage secret is mounted in namespace `default`; never inspect
its contents. A dedicated scoped credential would require another change.
After merge/deploy, check the ENS operational Job/CronJob status and validation
logs with the read-only context below; stop if production is unhealthy. Check
spare node/EBS capacity and arrange an observer. Up to 38 retained 30G
claims (28 GiB provisioned each, 1,064 GiB total) can remain until pod deletion;
completed workers do not imply early PVC cleanup.

```bash
set -euo pipefail
read -r -p 'Accept code-only isolation with the production-capable key, public scratch/profiles, confirmed sponsorship fit and potentially parent-only stacks? Type ACCEPT: ' PILOT_ACCEPT
test "$PILOT_ACCEPT" = ACCEPT
gh workflow run manual-backfill.yml --repo dynamical-org/reformatters --ref main -f dataset_id=ecmwf-ifs-ens-forecast-15-day-0-25-degree -f operation=create-new-store -f append_dim_end=2024-04-01T00:00:00 -f filter_start=2024-03-30T00:00:00 -f filter_end=2024-04-01T00:00:00 -f filter_contains= -f filter_variable_names= -f jobs_per_pod=1 -f max_parallelism=2 -f profile=true
```

Record the workflow run, main SHA, Job name and generated run id from its summary
and keep that receipt in a durable operator directory backed up through analysis.
The local export path below is an example; arrange its backup before deleting
remote profiles. Keep receipts and reports at least as long as the profiles.
Public evidence and the summary include the source SHA, not the private image URI.
Sampled worker arguments contain public run/job/SHA values only.
The summary prints the identity before submission, so an interrupted driver can
leave an accepted Job: inspect that exact Job before deciding whether to retry.
Workflow success means submission succeeded, not that processing completed.
Workflow cancellation does not cancel the Kubernetes Job. Do not reuse a run id
or retry a failed worker. The Job uses backoff 0, max failed indexes 0, replacement
policy `Failed`, no disruption-ignore rule and a six-hour active deadline. A Spot reclaim is a
terminal failure and counts against the shared caps.
Kubernetes can still duplicate an index in unusual controller failures; worker-0
create-only setup refuses a second creator. Count every observed pod attempt.

Set these from the summary and Job's UTC start time, not from the dispatch time:

```bash
export PILOT_RUN=<32-lowercase-hex-run-id> PILOT_JOB=<ifsens-profile-backfill-job-name> PILOT_START=<UTC-ISO-Job-startTime>
[[ "$PILOT_RUN" =~ ^[a-f0-9]{32}$ ]]
[[ "$PILOT_JOB" == ifsens-profile-backfill-* ]]
export PILOT_ROOT="s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens/$PILOT_RUN"
export PILOT_EVIDENCE="$HOME/ifsens-profile-evidence/$PILOT_RUN"
mkdir -p "$PILOT_EVIDENCE"
export PILOT_DEADLINE=$(date -u -d "$PILOT_START +6 hours" +%s)
```

## Observe and stop

Shared ceilings across all attempts and any optional comparison: **six hours,
12 running pod-hours (36 requested vCPU-hours), 0.75 TB received, 0.25 TB sent,
$25 compute/PVC/storage**. Parallelism and Job deadline bound scheduled work;
bytes, cumulative attempts and dollars are operator-enforced, not hard billing
limits. Include node boot/idle time and PVC lifetime. Source inventory implies
332,371,560,326 bytes per complete leg before protocol/retry overhead, about
0.33 TB. The write allowance includes scratch chunks and profiles.

Observe at least every minute; download fresh evidence every 30 seconds. Sum
per-pod network deltas across completed/failed/active attempts, and record node
instance type/price, pod start/finish, PVC creation/deletion and imageID. Stop on
any cap, 401/403, source error, OOM, wrong destination/image, failed profiling as
defined above (a replacement-decoded profile with samples is not one) or upload,
terminal worker failure, or no advancing evidence for two upload intervals while work continues. If counters/cost cannot be tracked, stop rather
than assume headroom. A partial leg remains useful with verified evidence.

```bash
kubectl --kubeconfig <read-only-kubeconfig> --context <read-only-context> -n default get job "$PILOT_JOB" -o json
kubectl --kubeconfig <read-only-kubeconfig> --context <read-only-context> -n default get pods -l "job-name=$PILOT_JOB" -o json
kubectl --kubeconfig <read-only-kubeconfig> --context <read-only-context> -n default get pvc -o json
aws s3 sync "$PILOT_ROOT/profiles/" "$PILOT_EVIDENCE/profiles/" --no-sign-request --only-show-errors
kubectl --kubeconfig <read-only-kubeconfig> --context <read-only-context> -n default logs -l "job-name=$PILOT_JOB" --all-containers --prefix --tail=-1 > "$PILOT_EVIDENCE/pod-logs.txt"
```

Keep raw pod logs private: errors can contain source URLs. Capture logs before
healthy cleanup; urgent cancellation takes priority. The operator cancels with
normal grace so the wrapper can drain the workload and flush partial evidence:

```bash
kubectl --context <operator-context> -n default delete job "$PILOT_JOB" --cascade=foreground --wait=true --timeout=120s
```

Never force deletion or broaden selectors. The wrapper budgets 25 seconds for
workload drain and 15 for final flush within the 60-second pod grace. An
incomplete final upload is an incomplete attempt even if chunks committed.

## Evidence and optional comparison

From the matching main checkout, verify the downloaded evidence locally:

```bash
export KUBECONFIG=<read-only-kubeconfig>
uv run python -m reformatters.ecmwf.ifs_ens.profiling.verify --output-dir "$PILOT_EVIDENCE/profiles" > "$PILOT_EVIDENCE/verification.json"
find "$PILOT_EVIDENCE/profiles" -name manifest-final.json -exec jq -c '{job: .metadata.job_name, worker: .metadata.worker_index, overhead_warning: .sampler_overhead_warning, child_hits: .profile_child_samples, parent_only: .profile_parent_only, low_confidence: .py_spy_low_confidence, profile_decode: .profile_decode, profile_error: .profile_error, error_fraction: .py_spy_error_fraction, lag_warnings: .py_spy_lag_warnings, samples_per_second: ((.py_spy_samples // 0) / .sampler_elapsed_seconds)}' {} + > "$PILOT_EVIDENCE/stack-quality.jsonl"
jq -e --arg job "$PILOT_JOB" '.jobs[$job].missing_workers == [] and .incomplete == 0' "$PILOT_EVIDENCE/verification.json"
```

A checksum failure is corrupt evidence; a missing final manifest is incomplete.
The sample-rate denominator includes drain time, so the rate is a lower bound.
The numerator sums samples across processes and can exceed 20 per second; compare
aggregate rates only between similarly shaped workloads.
Zero lag warnings means no recognized warning lines. Include stack-quality.jsonl
in the report even when the checksum/worker completeness check passes.
Inspect `overhead_warning_attempts` in verification.json and label their runtime
estimates as potentially biased; a warning does not invalidate phase evidence.
Inspect `invalid_utf8_attempts` there too. The verifier derives it from the raw
profile bytes, also for older manifests without quality fields, and rejects a manifest
that disagrees. Those attempts can count as complete and pass the jq gate, never as
strict: exclude their stacks from frame attribution, keeping phase/resource evidence.
Reconcile coverage with observed pods and the exact run/image/assignment. A
successful primary has 38 covered worker indexes and 38 distinct variable-day
units. Retain partial segments and pod termination records for failed attempts.

Only after primary success, verified export and Job/PVC cleanup, consider a
second dispatch with `jobs_per_pod=10` and every other input unchanged. It creates
a different run id/store and four workers (10/10/9/9 units). Skip it if the
primary exceeded 375 GB received or 125 GB sent, or the remaining time/cost is
insufficient. It must be stopped at the **original** `PILOT_DEADLINE`; its own
six-hour Job deadline does not extend the shared budget. Keep both receipts and
sum their usage. No jobs-per-pod 30 leg. Repeated source reads can warm caches:
compare download separately from setup and commit amortization.

After analysis, report measured per-unit runtime, reserved and actual CPU,
memory/disk peaks, network bytes, node/PVC cost and the uncertainty in extrapolating
two dates to the whole history. Recommend jobs-per-pod from those measurements.
Rank optimizations by profile evidence, estimated savings and confidence; do not
claim 200–300-pod throughput or full-store commit cost from this low-parallelism
scratch pilot.

## Cleanup ownership and retention

The operator records pods/PVC owner references before deleting each completed
Job, and verifies its claims disappear afterward. Claims have pod owners, not
the Job label; do not delete claims by a broad label. Export/check evidence first
for healthy completion; failure cancellation can precede export. Job TTL is a
fallback, not the cleanup schedule.

Export evidence, then delete each run's scratch `data/` within seven days of
launch, including
coordination and orphan repositories. Keep `profiles/` at least 30 days and until
analysis/export acceptance. Record the UTC launch date for each run separately.
The following commands are separate operator sessions; failed gates stop them.
They recheck every available final checksum and require exported evidence. For a
healthy complete run also require the complete-worker jq check above. For a
failed run, accept its documented missing workers/partial evidence explicitly
before cleanup; an attempt with no verifiable final evidence needs a separate
retention decision rather than bypassing the gate.

```bash
set -euo pipefail
export PILOT_RUN=<32-lowercase-hex-run-id> PILOT_LAUNCH_DATE=<YYYY-MM-DD>
export PILOT_EVIDENCE="$HOME/ifsens-profile-evidence/$PILOT_RUN"
[[ "$PILOT_RUN" =~ ^[a-f0-9]{32}$ ]]
uv run python -m reformatters.ecmwf.ifs_ens.profiling.verify --output-dir "$PILOT_EVIDENCE/profiles" > "$PILOT_EVIDENCE/verification.json"
jq -e '.attempts > 0 and .files_verified > 0' "$PILOT_EVIDENCE/verification.json"
read -r -p 'Job stopped, evidence exported and this exact data prefix disposable? Type DELETE-DATA: ' PILOT_CLEANUP
test "$PILOT_CLEANUP" = DELETE-DATA
aws s3 rm "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens/$PILOT_RUN/data/" --recursive --dryrun
aws s3 rm "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens/$PILOT_RUN/data/" --recursive
```

At or after launch plus 30 days, in a separate session:

```bash
set -euo pipefail
export PILOT_RUN=<32-lowercase-hex-run-id> PILOT_LAUNCH_DATE=<YYYY-MM-DD>
export PILOT_EVIDENCE="$HOME/ifsens-profile-evidence/$PILOT_RUN"
[[ "$PILOT_RUN" =~ ^[a-f0-9]{32}$ ]]
test "$(date -u +%s)" -ge "$(date -u -d "$PILOT_LAUNCH_DATE +30 days" +%s)"
uv run python -m reformatters.ecmwf.ifs_ens.profiling.verify --output-dir "$PILOT_EVIDENCE/profiles" > "$PILOT_EVIDENCE/verification.json"
jq -e '.attempts > 0 and .files_verified > 0' "$PILOT_EVIDENCE/verification.json"
read -r -p 'Analysis and verified evidence export accepted? Type DELETE-PROFILES: ' PILOT_CLEANUP
test "$PILOT_CLEANUP" = DELETE-PROFILES
aws s3 rm "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens/$PILOT_RUN/profiles/" --recursive --dryrun
aws s3 rm "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens/$PILOT_RUN/profiles/" --recursive
```
