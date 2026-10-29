# IFS ENS profiling pilot

This one-off entrypoint profiles 38 variable-day units: all 19 variables for
2024-03-30 and 2024-03-31. It uses the materialized ENS pipeline at main commit
`44a8fede0b31e337093d616fcac379b08195afb6`, with a separate template starting
2024-03-30 and version `0.0.0-profiling-20261029`. It does not register a dataset
or create a CronJob. The primary leg has 38 indexed workers, parallelism 2 and
one unit per pod. An optional comparison has four workers with 10, 10, 9 and 9
units, at parallelism 2.

## Destination and ownership

For a fresh token `pYYYYMMDD-<12 lowercase hex digits>`, data and coordination
live beneath `s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens-20261029/<token>/data/<leg>/`.
The repository ends in
`ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.0.0-profiling-20261029.icechunk`.
Profiles are separate, beneath `<token>/profiles/<leg>/<job>/<pod UID>/<attempt>/`.
All are publicly readable, uncatalogued scratch objects.

The pilot uses the existing production-capable storage credential. Exact path,
version and replica checks plus create-only ownership checks provide code-level
isolation; the credential does not provide IAM isolation. Before launch, the
operator must accept this arrangement and confirm that temporary scratch use
fits the bucket's sponsorship, or request a scoped-credential code change, review and new build instead. No
credential is needed for manifest generation or image building.

Worker 0 creates an empty repository and tags its root with the token and leg.
Every repository open verifies ownership. Existing repositories are refused by
worker 0. A crash between creation and tagging requires a fresh token, as does
any failed pilot rerun. There is no automatic application retry. Indexed Job
retries are disabled and a failed index fails the Job. Spot interruption or eviction fails the Job. Kubernetes can replace a pod in
the narrow interval before its terminal state is counted; create-only setup
refuses a replacement worker 0. Each attempt has its own evidence directory.
Other workers allow 30 minutes for worker 0 to schedule, pull the image and
create the repository, within the Job deadline.

## Evidence

The worker records elapsed wall time for source download, read/decode,
transformation, local chunk encoding/writing, upload, commit, setup, waits and
finalization. Template-helper commits are included within setup/finalization
timings; only worker commits have a separate commit timer. These phases overlap; their times must not be summed. Local write
time is the parent's wait for the encoding pool. The per-process CPU stacks
from `py-spy --subprocesses` distinguish work inside those children. Decode and
download hooks execute in the worker's threads.

A supervisor samples cgroup CPU, memory current/peak, IO, process-tree RSS/CPU,
pod network counters, PVC and shared-memory filesystem use once per second.
CPU is requested at 3 vCPU, not capped at 3. Filesystem use includes concurrent
files on that mount; process RSS can count shared pages more than once. Profiles
include the sampler's overhead. The native container smoke recorded sampling lag
during a cold start; retain sampler warnings and compare startup separately.
Sampling at 20 Hz does not establish negligible overhead. Final manifests
record sampler rate and elapsed time alongside sample counts; the speedscope
profiles identify sampled processes and threads. Idle threads are omitted, so
rate multiplied by wall time is not an expected total CPU-sample count. Source-read bytes use network counter deltas,
which also include index and storage traffic; use phase boundaries to attribute
them. Initial/final manifests identify the image, source revision, lock hash,
assignment and pod attempt. Closed segments upload every 30 seconds. Node loss
can lose the last 30 seconds; the final stack file is best effort on cancellation.
A failed source read, profiler self-test or evidence upload fails the pilot.
No environment, credentials, source URLs or exception messages are copied into
structured evidence. The fixed workload command is safe to appear in CPU stacks.

The wrapper uses 25 seconds for process-group drain and 15 seconds for final
upload, inside the pod's 60-second termination grace. Kernel or pod admission
restrictions on sampling must be resolved before a pilot; do not add privileges
to bypass a failed self-test.

## Build and delivery

The PR workflow **IFS ENS profiling image** builds a multi-platform OCI archive
for linux/amd64 and linux/arm64. It publishes a public GitHub artifact for seven
days, with checksums, image index, platform digests and provenance. It has no
registry credentials or deployment step. The ordinary Dockerfile is extended by
`deploy/profiling.Dockerfile.fragment`; the application lock remains unchanged.

The following commands are for the operator. Run them from a Bash shell in the
reviewed checkout. Registry authentication and cluster mutations use the
operator's own access. Agents use only the explicit low-permissions Kubernetes
context for observation and must not execute the mutations below.

Choose the completed workflow run and set `PILOT_SHA` to its reviewed PR head.
Set `PILOT_REPOSITORY` to the existing reformatters ECR repository. These are
identity inputs; do not substitute a mutable image tag in a manifest.

```bash
set -euo pipefail
export PILOT_CI_RUN=<completed-workflow-run-id> PILOT_SHA=<reviewed-40-character-head-sha> PILOT_REPOSITORY=<existing-reformatters-ecr-repository>
mkdir -p /tmp/ifsens-profiling-artifact && gh run download "$PILOT_CI_RUN" --repo dynamical-org/reformatters --name "ifsens-profiling-image-$PILOT_SHA" --dir /tmp/ifsens-profiling-artifact
(cd /tmp/ifsens-profiling-artifact && sha256sum -c SHA256SUMS)
test "$(jq -r .source.head_sha /tmp/ifsens-profiling-artifact/provenance.json)" = "$PILOT_SHA"
export PILOT_INDEX=$(jq -er .index_digest /tmp/ifsens-profiling-artifact/platform-manifests.json)
skopeo copy --all --preserve-digests oci-archive:/tmp/ifsens-profiling-artifact/image.oci.tar "docker://$PILOT_REPOSITORY:ifsens-profile-$PILOT_SHA"
skopeo inspect --raw "docker://$PILOT_REPOSITORY@$PILOT_INDEX" > /tmp/ifsens-profiling-registry-index.json
test "sha256:$(sha256sum /tmp/ifsens-profiling-registry-index.json | cut -d' ' -f1)" = "$PILOT_INDEX"
export PILOT_IMAGE="$PILOT_REPOSITORY@$PILOT_INDEX"
```

The preserved index pins its platform manifests. Check that `platform-manifests.json`
contains exactly linux/amd64 and linux/arm64. No staging branch or deployment is
needed. The ARM sampler still has to pass its in-pod self-test before source work.

## Primary launch

The caps below apply to the whole pilot, including any comparison. The storage-options secret must exist in namespace `default`; the operational
ENS jobs use it there. Do not read its contents. Verify free
EC2/NodePool capacity, EBS quota and storage class before launch. With no early
PVC garbage collection, the primary retains up to 38 claims of 30G (28 GiB
rounded provisioning each): about 1,064 GiB. Claim lifetime lasts until pod
removal, not worker completion. Record node types/prices and PVC creation/deletion
for billing. Arrange an observer before creating the Job.

```bash
export PILOT_RUN="p$(date -u +%Y%m%d)-$(openssl rand -hex 6)" PILOT_LEG=jpp1
export PILOT_JOB="ifsens-profile-$PILOT_RUN-$PILOT_LEG"
docker run --rm --network none "$PILOT_IMAGE" preflight --run-token "$PILOT_RUN" --leg "$PILOT_LEG" > /tmp/ifsens-pilot-preflight.json
docker run --rm --network none "$PILOT_IMAGE" manifest --run-token "$PILOT_RUN" --leg "$PILOT_LEG" --image "$PILOT_IMAGE" --seconds 21600 > /tmp/ifsens-pilot-job.json
jq -e '.spec.completions == 38 and .spec.parallelism == 2 and .spec.backoffLimitPerIndex == 0 and .spec.maxFailedIndexes == 0' /tmp/ifsens-pilot-job.json
jq -e --arg prefix "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens-20261029/$PILOT_RUN/data/$PILOT_LEG/" '.repository | startswith($prefix) and endswith("v0.0.0-profiling-20261029.icechunk")' /tmp/ifsens-pilot-preflight.json
jq -e --arg image "$PILOT_IMAGE" --arg job "$PILOT_JOB" '.metadata.name == $job and .spec.activeDeadlineSeconds == 21600 and .spec.template.spec.terminationGracePeriodSeconds == 60 and .spec.template.spec.containers[0].image == $image and ([.spec.template.spec.containers[0].env[].name] | index("DYNAMICAL_SENTRY_DSN") == null) and ([.spec.podFailurePolicy.rules[].action] | index("Ignore") == null)' /tmp/ifsens-pilot-job.json
read -r -p 'Accept code-only isolation with the production-capable key, public scratch/profiles and confirmed sponsorship fit? Type ACCEPT: ' PILOT_ACCEPT
test "$PILOT_ACCEPT" = ACCEPT
kubectl --context <operator-context> -n default create --dry-run=server -f /tmp/ifsens-pilot-job.json
kubectl --context <operator-context> -n default create -f /tmp/ifsens-pilot-job.json
```

The acceptance prompt is a launch gate; stop unless it passes. The manifest
equivalence test pins all other resources, mounts, security and environment to
the ordinary driver-generated Job. The server dry-run must pass. It is not permission to relax admission or security
settings. The first worker's durable initial manifest and sample segments must
appear before treating the launch as healthy. Record actual pod architecture,
image digest, node, PVC, start/end and worker assignment. The 6-hour Job deadline
bounds elapsed runtime for this leg. The observer and operator enforce the
remaining caps; they are not hard AWS billing limits.

Shared ceilings: **6 processing hours, 12 running pod-hours (36 requested
vCPU-hours), 0.75 TB source reads, 0.25 TB output writes, $25 compute/PVC/storage**.
The Job deadline and parallelism bound scheduled processing time; replacement
and terminating pods can temporarily overlap, so account for actual attempts.
Count completed and failed attempts. Stop on any cap, source failure, OOM,
wrong destination/image, failed profiling/upload, or terminal worker error.
Do not automatically retry or resume. A cancelled partial pilot remains useful
if its durable evidence is complete enough for the question being measured.

```bash
kubectl --context <operator-context> -n default delete job "$PILOT_JOB" --cascade=foreground --wait=true --timeout=120s
```

Observe with the read-only context; Marsh issues any cancellation. Pod start/end
timestamps account for pod-hours; evidence network deltas account for transfer,
with rx/tx treated conservatively as reads/writes. The 0.75 TB/0.25 TB caps
include index/profile traffic and retries. Account for already completed
pods as well as the active ones. Watch at least every minute, and retrieve fresh
profile segments every 30 seconds. Stop if evidence stops advancing for two
upload intervals while work continues. Cost includes node and retained PVC
lifetime, not just container time.

```bash
kubectl --kubeconfig /home/marsh/.kube/low-permissions.kubeconfig --context low-permissions -n default get pods -l "job-name=$PILOT_JOB" -o wide -w
kubectl --kubeconfig /home/marsh/.kube/low-permissions.kubeconfig --context low-permissions -n default get job "$PILOT_JOB" -o json
kubectl --kubeconfig /home/marsh/.kube/low-permissions.kubeconfig --context low-permissions -n default get pods -l "job-name=$PILOT_JOB" -o json
```

Capture logs early and before healthy Job deletion so sampler warnings survive
pod cleanup. Logs from an already deleted/replaced pod may be unavailable. Treat
raw pod logs as private operator evidence; do not publish unredacted source-error
messages or credential-bearing URLs. On an urgent stop, cancellation takes
priority over log capture.

```bash
kubectl --kubeconfig /home/marsh/.kube/low-permissions.kubeconfig --context low-permissions -n default logs -l "job-name=$PILOT_JOB" --all-containers --prefix --tail=-1 > "/tmp/$PILOT_JOB-pod-logs.txt"
```

Use normal grace; do not force deletion. Export closed evidence and verify its
checksums before deleting a healthy completed Job. Deleting a failed Job is an
immediate stop action: remote evidence survives it.

## Optional comparison and cleanup

Run the comparison only after primary success, verified durable evidence and
primary Job/PVC cleanup. Freeze it if the primary exceeds 375 GB read or 125 GB
written, or measured time/cost leaves insufficient shared headroom. Same-source
re-reading may warm caches: compare phase times and setup/commit amortization,
not total runtime alone. There is no jobs-per-pod 30 leg.

```bash
export PILOT_LEG=jpp10 PILOT_REMAINING_SECONDS=<remaining-shared-processing-seconds>
export PILOT_JOB="ifsens-profile-$PILOT_RUN-$PILOT_LEG"
docker run --rm --network none "$PILOT_IMAGE" manifest --run-token "$PILOT_RUN" --leg "$PILOT_LEG" --image "$PILOT_IMAGE" --seconds "$PILOT_REMAINING_SECONDS" > /tmp/ifsens-pilot-job-jpp10.json
jq -e '.spec.completions == 4 and .spec.parallelism == 2' /tmp/ifsens-pilot-job-jpp10.json
kubectl --context <operator-context> -n default create --dry-run=server -f /tmp/ifsens-pilot-job-jpp10.json
kubectl --context <operator-context> -n default create -f /tmp/ifsens-pilot-job-jpp10.json
```

The comparison adds at most four claims (112 GiB). After each leg, export the
profiles with the operator's AWS access. Validate every final manifest's SHA256
map against the downloaded files; a missing final manifest marks an incomplete
attempt. The verification result lists missing worker indexes for each Job;
require none before calling a leg complete. Successful attempts must include
the stack profile, workload receipt and a resource segment. Retain partial
segments and pod termination evidence too.

```bash
export PILOT_ROOT="s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens-20261029/$PILOT_RUN"
aws s3 sync "$PILOT_ROOT/profiles/" "/tmp/ifsens-profile-evidence-$PILOT_RUN/" --only-show-errors
docker run --rm --network none --mount "type=bind,src=/tmp/ifsens-profile-evidence-$PILOT_RUN,dst=/evidence,readonly" "$PILOT_IMAGE" verify-evidence --output-dir /evidence > "/tmp/$PILOT_RUN-evidence-verification.json"
kubectl --context <operator-context> -n default get pods -l job-name="$PILOT_JOB" -o json > "/tmp/$PILOT_JOB-final-resources.json"
kubectl --context <operator-context> -n default get pvc -o json > "/tmp/$PILOT_JOB-final-pvcs.json"
kubectl --context <operator-context> -n default delete job "$PILOT_JOB" --cascade=foreground --wait=true --timeout=120s
```

PVCs have pod owner references rather than the Job label; separately list PVCs
and verify those recorded for this Job disappear after its pods are deleted.
Do not delete claims by a broad label. The operator owns cleanup: delete scratch
data within seven days and profiles after 30 days, only after the evidence export
and analysis have been verified. Data cleanup includes coordination and orphaned
repositories but does not touch the separate profiles prefix.

Set the UTC launch date once and retain it with the run receipt. Marsh deletes
data after verified evidence export and no later than launch plus seven days.
For an October 29 launch, that deadline is November 5, 2026.

```bash
export PILOT_LAUNCH_DATE=<actual-YYYY-MM-DD-launch-date>
test -s "/tmp/$PILOT_RUN-evidence-verification.json"
[[ "$PILOT_RUN" =~ ^p[0-9]{8}-[a-f0-9]{12}$ ]] && aws s3 rm "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens-20261029/$PILOT_RUN/data/" --recursive --dryrun
[[ "$PILOT_RUN" =~ ^p[0-9]{8}-[a-f0-9]{12}$ ]] && aws s3 rm "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens-20261029/$PILOT_RUN/data/" --recursive
```

Start each separate cleanup session with `set -euo pipefail`; a failed gate must
stop the session. Marsh deletes profiles at launch plus 30 days after accepting the analysis and
verified export (November 28, 2026 for an October 29 launch). This is a separate
step, not part of data cleanup. The commands below use GNU date.

```bash
test "$(date -u +%s)" -ge "$(date -u -d "$PILOT_LAUNCH_DATE +30 days" +%s)"
test -s "/tmp/$PILOT_RUN-evidence-verification.json"
read -r -p 'Analysis and evidence export accepted? Type DELETE-PROFILES: ' PILOT_CLEANUP
test "$PILOT_CLEANUP" = DELETE-PROFILES
[[ "$PILOT_RUN" =~ ^p[0-9]{8}-[a-f0-9]{12}$ ]] && aws s3 rm "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens-20261029/$PILOT_RUN/profiles/" --recursive --dryrun
[[ "$PILOT_RUN" =~ ^p[0-9]{8}-[a-f0-9]{12}$ ]] && aws s3 rm "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens-20261029/$PILOT_RUN/profiles/" --recursive
```

Keep the ECR profiling tag and local artifact/evidence copies until the 30-day
analysis handoff, then Marsh removes that single tag through ECR and removes the
named local run directories/JSON. The GitHub artifact expires after seven days;
copy it before then if needed. Keep the final report and checksums in the planning
record. This one-off PR stays draft and is closed after the pilot; it is not a
request to merge permanent profiling or dataset code into main.

## Analysis after execution

Report per-unit and per-worker phase distributions, CPU seconds, peak cgroup
memory, read/write bytes, disk peaks and total pod/node/PVC lifetime. Reconcile
source bytes against the index-derived 332,371,560,326 bytes per leg; that figure
checks metadata coverage, not payload integrity. Invoice cost uses actual nodes,
Spot price/AZ, Auto Mode and provisioned PVC lifetime; resource reservations are
not billed CPU time. Report missing evidence and cancelled work explicitly.

Recommend jobs-per-pod only from observed amortization and the pod deadline,
with a ranked list of measured bottlenecks and savings estimates. Each saving
must state the phase fraction, proposed intervention, uncertainty and validation
needed. This pilot covers one late-MARS era at two concurrent pods; it does not
establish early-MARS/Open Data throughput, 200-pod source capacity or commit cost
in a large existing repository. Repeated-source cache effects, sampling overhead
and small-repository commit costs bound extrapolation to the full backfill.
