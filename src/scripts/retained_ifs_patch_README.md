# Retained IFS field repair

This one-off tool repairs six variable/init units (seven output fields) in a fresh descendant of the retained IFS 15-day snapshot. Use the unmerged `retained-branch-repair-runner` branch at the handoff commit. Its PR is intentionally closed; do not reopen it or merge the tool. Main and the protected retained branch are never writable targets.

## Scope and checks

The version-2 manifest binds the immutable base, candidate name, metadata, code/lock hash, source URLs and exclusive byte ranges, GRIB identity and hashes, seven independently confirmed expected-field hashes, and approved encoded/decoded shard hashes. Sources stream anonymously from Source Coop (MARS) and GCS (Open Data). The local path needs no Docker image, Kubernetes Job, PVC, emulator or input-payload bundle.

The runner fetches eight messages: four instantaneous fields, one precipitation-type field and precipitation cumulative L108/L111/L114. The two corrected precipitation outputs are L111 and L114. Production decoding, transformation and rounding must reproduce the frozen expected float32 hashes. Fetches check HTTP 206 ranges, lengths, SHA256, GRIB identity and grid coordinates.

One spatial shard is processed at a time. All uint32 values outside the approved member/lead mask must remain identical, including NaN payloads and signed zero. A codec round trip is checked over every complete unit. All six units are staged before branch creation, and the encoded and decoded shard hashes must match the independently verified preparation. Only then does the runner create a fresh candidate and commit the approved keys once. It checks metadata, coordinates, repository-wide changed keys, ancestry, full-unit values and protected refs after commit. It never resets, deletes, merges or promotes a branch.

The independent reader does not import the patch or its comparator. It checks every value in all six complete units, the seven expected fields, encoded/decoded hashes and exact changed-key scope. The patch retains Zarr's default omission of fill-only inner chunks; repaired encoding may differ from original encoding, but decoded uint32 identity outside the mask and approved repaired bytes are required.

## Preparation and runtime

Before handoff, anonymous preparation writes a local stage under `/tmp`. Independent verification checks that stage against the immutable base. Its receipt and the implementation review are pinned in the handoff bundle alongside the final manifest and source commit. The manifest pins the final commit's runtime. The earlier independent stage remains valid because only wrapper files changed, as recorded in the reviewed equivalence evidence; its original runtime receipt is preserved separately. The actual launch rechecks its runtime and all staged hashes before any branch creation.

The local workload covers 90 spatial shards and approximately 108 GB of decoded complete-unit data. Shards are bounded in memory; the handoff reports measured peak memory, disk and elapsed time. Licensed source payloads and derived staging remain under `/tmp`. Preserve local receipts and staging until verification is complete.

The operator wrapper archives the pinned unmerged commit for each attempt and uses the locked Python environment. It sets nice 10, limits affinity to at most four available CPUs and numerical libraries to one thread. A machine-local exclusive lock prevents concurrent attempts on this machine; its child inherits the lock so a killed wrapper cannot release a still-running repair. Attempt directories and logs are unique and never overwrite earlier evidence. Before a fresh repair, available memory and free scratch must meet the bundle thresholds. These are startup checks, not reservations; each retained attempt consumes additional disk.

## Operator commands

Use the reviewed handoff's exact bundle path and SHA256 from the pinned checkout. Only the operator runs a real launch. These placeholders are not a substitute for that handoff:

```bash
uv run --frozen python -m scripts.retained_ifs_patch_local launch \
  --bundle "$BUNDLE" --bundle-sha256 "$BUNDLE_SHA256" \
  --work-dir "$TMP_RUN_ROOT" \
  --approve-retention-and-shared-credentials

uv run --frozen python -m scripts.retained_ifs_patch_local verify \
  --bundle "$BUNDLE" --bundle-sha256 "$BUNDLE_SHA256" \
  --work-dir "$TMP_RUN_ROOT"
```

`launch --dry-run` validates the pinned bundle and describes the sequence without opening a store, creating a work directory or executing subprocesses. Agents may run that mode and anonymous preparation/verification only.

The real launch is a foreground process. Do not interrupt once `core.log` shows `status=prepared`, when branch creation and commit follow. Ctrl-C reaches the child too. Keep the terminal open, or use the handoff's `nohup` command to survive terminal closure and redirect output to its specified log. Launch performs the independent verification itself; the separate verify command is useful after interruption or for a fresh check. Progress and final JSON receipts are saved under the run root. Success means independent verification passed, not merely that a process exited or a commit exists.

## Credentials and retention

The write phase uses Icechunk's standard S3 environment credential provider (`--auth env`), with the operator's existing credentials authorized for the target repository. No Kubernetes Secret or Docker login is needed. Supply the existing `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables, plus `AWS_SESSION_TOKEN` for temporary credentials, in the operator's shell; do not paste their values into the command, report or logs. Credentials must remain valid for the expected run plus five minutes. Temporary credentials (`AWS_SESSION_TOKEN` present) require a timezone-aware `AWS_CREDENTIAL_EXPIRATION`; obtain the actual expiry from the credential issuer, not an invented future date. The wrapper checks that expiry before starting. Long-lived keys may omit it.

The launch approval acknowledges retention of the base/candidate evidence and application-level isolation: the credentials may authorize more than this candidate. Reviewed code, immutable scope and hash checks enforce the branch restriction; this is not branch-specific IAM. No agent reads credentials or invokes writable mode. Source reads themselves need no credentials.

## Interruption and rerun

A busy lock means another attempt or its surviving child still holds it. Wait for that child to exit, then verify. Never delete the lock file to bypass exclusion.

A manual rerun uses repository state, not a success log, to decide what is safe:

- Candidate absent: start a fresh attempt and recompute preparation. Partial local staging is not reused. This is restartability, not resumable uploading.
- Candidate exists with a descendant commit: perform anonymous independent verification only. Never make a second commit, even when the earlier process left no success receipt.
- Candidate exists at the base: stop for investigation. Do not reset, delete or resume writing to it. A different candidate would require a newly approved manifest.
- Protected branch moved, code/evidence mismatch, failed comparison or ambiguous state: exit nonzero and preserve evidence. No automatic retry or rollback.

Verification uses anonymous reads and never writes a store. A failure after commit does not undo the candidate; investigate it using the retained receipts and explicit snapshots. Publication of the repaired archive and the unresolved source-index gap remain separate decisions.

The older Job operator and whole-unit runner are historical code on this draft branch. They are not alternate launch instructions for this handoff.
