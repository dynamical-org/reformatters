# ENS replica handoff and recovery

Ordinary deployment retains direct replica writes and the approved primary origin
2024-04-01. The independent mirror is dormant until a verified operator handoff.
The replica keeps that fixed origin and daily updates; a different primary origin
requires an explicitly reviewed `approved_origins` configuration change. Coordinate
chunks remain 5840. This protocol does not perform a historical prepend or authorize
production activation, migration, retention changes, or consumer communication.

## Handoff

Rehearse against an authorized scratch prefix before production handoff. The operator
commands live in `uv run src/scripts/ens_replica_mirror.py`; `--help` describes their
arguments. They use the configured ENS stores and must only be run against an
explicitly authorized destination.

1. `pending HANDOFF_ID` atomically persists a pending marker. This stops **all ENS
   primary daily publication**, not just replica writes, until successful adoption
   or explicit abort. Updated writers refuse new writes and recheck the marker
   before every replica batch and metadata publication. A marker read failure fails
   closed. Schedule the outage and approve its duration and verification cost
   separately with the operator. Code does not authorize any of them.
2. Drain **every** direct writer, including older deployed images that cannot observe
   the marker. Confirm jobs are terminal and pods/processes absent. Reconcile any
   partial direct writes against the chosen successfully published primary snapshot.
3. `adoption-dry-run SNAPSHOT REPORT` verifies schema, layout, timestamp coordinates,
   exact mapped shard keys and encoded lengths. It reads metadata and coordinates,
   but does not GET data payloads. Review the report's exact inventory digest and
   full residual read budget.
4. `adopt SNAPSHOT INVENTORY_SHA256 REHEARSAL DRAINED_WRITERS --max-read-bytes BYTES`
   recomputes the inventory under exclusive ownership. The entire residual must
   fit the explicit budget **before the first data GET**; the default is zero.
   Inline source bytes are compared from manifests and do not consume GET budget.
   Every residual shard is compared byte for byte. Only successful verification
   creates the active checkpoint and durable proof record.

There is no size-only proof, sampling shortcut, opaque external-proof bypass, or
checksum shortcut. Equal lengths are insufficient. Adoption can require reading
the entire retained archive from both stores; that cost is a separate operator
decision. A dry run does not authorize those reads. Rehearsal and drain attestations
are prerequisites, not evidence of byte equality. Snapshot, destination identity,
schema, coordinates, keys and reference inventory bind the reviewed digest.

To abandon a pending handoff, drain all writers and obtain explicit operator
approval to resume direct publication, reconciling any partial direct writes first.
`abort-handoff DRAINED_WRITERS OPERATOR_EVIDENCE` requires nonempty evidence for both,
holds the same exclusive lock, and persists an audit containing the pending state,
evidence and UTC timestamp before removing the marker. An audit or marker-removal
failure leaves the marker blocking writes. Active handoffs cannot be aborted by
this command. Retain abort audits; removing the marker permits new direct writes
and primary daily publication again. It does not repair partial data or restart jobs.

## Daily mirroring and retry

An active handoff disables direct ENS replica writes. After successful primary
publication, the mirror reads a pinned published snapshot and copies changed
encoded data shards, mapping source timestamps to fixed replica offsets. Time
shards must contain one timestamp. Deleted source shards delete the mapped replica
shards. Coordinates, including progress coordinates and missing values, are read
from the source, sliced by timestamp and re-encoded with the identical replica
layout; their keys cannot simply be shifted by the origin offset.

Data and coordinates precede metadata publication. Replica root coverage and
attribution stay replica-specific. The checkpoint advances only after success.
Plain Zarr is not transactional: readers can observe partially replaced existing
chunks during an interrupted copy, even though expanded metadata is published last.
Pending replay reads `init_time` independently and validates each array's append
length against the durable checkpoint and target. It tolerates interrupted metadata
growth while still rejecting invalid labels, incompatible schemas and out-of-range
lengths. Array metadata precedes the final consolidated root metadata write.

## Shared materialized-dataset behavior and retention

The coordination changes apply to all materialized Icechunk operational updates,
including single-worker jobs: the source pin, ready marker, worker results and
publication receipts remain durable. The published first append coordinate must
equal the configured `append_dim_start`; ENS alone can select an explicitly approved
origin. All materialized writers, including backfills, compare actual writable
stored append labels before writing and again during finalization.

The shared final Icechunk branch commit does not rebase. A stale main CAS fails;
worker commits on the job branch retain their existing conflict handling. Across
datasets, plain-Zarr metadata follows successful primary publication. The finalizer
rechecks the primary tip immediately before copying plain-Zarr metadata and rejects
a detected intervening publication. Another publisher can still advance after that
check: it cannot make plain Zarr transactional or serialize independent publishers.

Keep complete receipt/pin/result sets while a job identity can retry. Nothing
deletes these records automatically. Later operator archival is allowed only after
the job is terminal and retries for that identity are permanently disallowed.

`uv run main ecmwf-ifs-ens-forecast-15-day-0-25-degree mirror-replica` finishes a
durable pending target before catching up to the pinned main snapshot. A later
post-commit hook does the same. A mirror failure after primary publication does not
require rerunning ingestion. Unexpected non-ancestral changes fail closed.

## Origin epochs and rollback

Drain publishers and mirror to `Sfinal` before changing the primary origin. Prepare
`Bfinal` separately and retain both snapshots. `epoch-dry-run Sfinal Bfinal REPORT`
compares retained timestamps, coordinates, schema and shard inventories directly
between snapshots, without rereading the replica. Native chunk ID, offset and length
equality or inline-byte equality prove unchanged bytes. Different references are
residuals requiring an explicit complete byte-comparison budget.

`prepare-epoch Sfinal Bfinal INVENTORY_SHA256 --max-read-bytes BYTES` verifies that
proof and durably records the intent and last mirror checkpoint before an operator
publishes `Bfinal` using a conditional main reset from `Sfinal`. This is not a
transaction across Icechunk and plain Zarr. A prepared intent blocks mirroring
while main still equals `Sfinal`; it is never automatically cancelled. After a
successful reset, retry reconciles actual main with `Bfinal` or its descendant and
advances the baseline. The same protocol applies to rollback.

For an unexecuted or failed reset, stop the publisher first, then use
`cancel-epoch Sfinal Bfinal EVIDENCE`. Cancellation refuses a published `Bfinal`
ancestry and records the observed main and attestation. Unexplained ancestry
changes require operator investigation.

## Ownership and abandoned writers

All mutating mirror controls and copies use an atomic, non-expiring backend lock.
Local ownership uses an atomic link; S3 uses conditional creation. There is no lease
expiry or automatic stealing. A crashed process can leave a lock indefinitely.

Before `recover-lock OWNER ATTESTATION`, independently confirm the owning job is
terminal and all its pods/processes are gone. The JSON attestation contains
`job_name`, `terminal_state` (`Complete`, `Failed` or `Deleted`),
`pods_absent_checked_at` (UTC timestamp) and `confirmed_by`. Recovery matches the
recorded owner and persists the attestation before deleting its lock. Serialize
manual recovery itself; never recover a live writer's lock.

## Validation and rollout gates

Active ENS validation compares the timestamp intersection and independently checks
replica freshness. Dormant ENS and other datasets retain exact replica matching.
Scratch rehearsal, exact adoption proof, explicit verification-cost approval,
older-writer drain, watched daily cycles, approved origin, migration/communication
approval and retention hold remain operator gates before any production cutover.
