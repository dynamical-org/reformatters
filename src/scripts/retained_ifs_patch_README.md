# Retained IFS field repair

This one-off tool repairs six variable/init units (seven output fields) in a fresh descendant of the retained IFS 15-day snapshot. It stays on the unmerged `retained-branch-repair-runner` branch. Do not merge the repair tool. Main and the protected retained branch are never writable targets.

The patch entry point is `scripts.retained_ifs_patch`. It replaces the earlier whole-unit regeneration launch path for this repair. The earlier runner remains available as reviewed historical code; its PVC runbook does not apply to this path.

## What is repaired

The compact version-2 manifest binds the immutable base, candidate name, metadata, exact code/lock hash, source URLs and exclusive byte ranges, GRIB identity and raw hashes, seven independent expected-field hashes, and pre-approved changed-shard hashes. Sources stream anonymously from Source Coop (MARS) and GCS (Open Data). No input bundle of GRIB payloads, PVC, shared-memory volume, upload bucket or Kubernetes helper Job is needed.

The runtime reads eight messages: four instantaneous fields, one precipitation-type field and precipitation cumulative L108/L111/L114. The two corrected precipitation outputs are L111 and L114. It applies production decoding/transformation/rounding and requires the frozen expected float32 hashes. Source checks include exact HTTP 206 ranges, lengths, SHA256, GRIB parameter/member/lead/init identity and grid coordinates.

The runner reads one spatial shard at a time, patches only approved member/lead fields, and checks every uint32 value outside that mask, including NaN payloads and signed zero. After a codec round trip it compares each complete unit. It stages all six units before creating a branch, checks staged encoded and decoded hashes against the independently verified local stage, then creates the new candidate and commits only the sealed changed keys once. Metadata/coordinates, repository-wide changed keys, ancestry, source expectations and the protected ref are checked again after commit. No automatic retry, reset, delete, merge or promotion exists.

## Preparation before the operator launch

These steps are performed locally, with anonymous store/source reads and no remote writes:

1. Freeze the scientific confirmation and compact manifest. The source science is independent of the patch output.
2. Run prepare mode into a fresh directory under `/tmp`, then run `scripts.retained_ifs_patch_verify --stage-dir` against it. This second reader does not import the patch or its comparison code. It verifies all six complete units and the seven expected hashes. Bind the resulting changed-key set and encoded/decoded hashes into the final manifest. The bundle contains relative paths and hashes for the independent-stage receipt and implementation review; launch rehashes both and checks that the stage receipt matches the manifest.
3. Measure memory, encoded staging size and elapsed time. Fill the reviewed handoff bundle's resource requests, deadline and expected duration. Freeze the exact unmerged source commit and bundle SHA256 after tests and opposite-vendor review.

Licensed source payloads and derived staging data stay under `/tmp`. Preserve that evidence through verification. Losing local evidence blocks launch until it is reconstructed and revalidated; hashes alone are not a backup. The excluded source-index gap and eventual publication of the retained archive are separate decisions.

## Operator commands

The handoff supplies absolute paths, source commit, bundle SHA256 and measured time estimates. Run from a clean checkout at the pinned unmerged commit, using its locked environment. The placeholders below are deliberately not executable launch instructions without that reviewed handoff.

```bash
uv run --frozen python -m scripts.retained_ifs_patch_operator launch \
  --bundle "$BUNDLE" --bundle-sha256 "$BUNDLE_SHA256" \
  --work-dir "$NEW_TMP_WORK_DIRECTORY" \
  --approve-retention-and-shared-secret

uv run --frozen python -m scripts.retained_ifs_patch_operator verify \
  --bundle "$BUNDLE" --bundle-sha256 "$BUNDLE_SHA256" \
  --wait-seconds "$REVIEWED_WAIT_SECONDS" --output "$NEW_VERIFICATION_JSON"
```

`launch --dry-run` validates the bundle and prints the operation sequence without executing external commands or creating a work directory. Agents use only this launch mode. The real launch is for the operator.

The launch command archives the exact commit, builds an arm64 image locally with the existing Dockerfile, and runs a light source-only bit-identity check in that exact image. This checks eight source messages and seven frozen expected fields; it is not a separate full cross-architecture proof. Run launch on a native arm64 Docker host (for example an arm64 workstation with an arm64 Docker daemon). The script checks the Docker daemon architecture before building. A missing capability stops before push or Job creation. The light check does not prove that arm64 compression reproduces the locally staged amd64 encoded shard bytes. That equality remains an in-Job precommit gate; a mismatch stops before candidate creation, with no fallback. Cluster arm64 spot capacity is assumed; no emulator, builder or nodepool is installed by this script.

Only after the source check passes does launch push the image through the operator's existing Docker login. It binds the immutable registry manifest to Buildx’s explicit `containerimage.config.digest` (not the driver-dependent local image ID), renders one arm64 spot Job with that digest and the inline manifest, then freshly verifies that the retained ref is unchanged and both candidate and Job names are unused. The chosen Kubernetes context must also match the approved API endpoint hash; an empty context selector uses the operator’s current context after that check. It submits once and writes `launch.json` with Job UID, image digest and candidate name. Success prints `LAUNCHED` and the expected repair duration. Build and source checks may take additional time beyond the measured Job estimate.

The Job has one worker, at most four CPUs, one numerical thread, no service-account token and no retry. Only the existing storage Secret is mounted. The operator's approval acknowledges retention/GC requirements and application-level branch isolation: the shared Secret may authorize more than the candidate, so protection is enforced by reviewed code rather than branch-specific IAM. The operator script never reads Secret values or credential files; Docker/kubectl use the operator's existing login.

`verify` waits read-only for the candidate to advance from the base, then independently checks explicit base/candidate snapshots, parentage, the repository-wide changed-key set, byte-identical metadata/coordinates, approved encoded/decoded shard hashes, and every full-unit value. It uses anonymous storage reads and no cluster credentials. Success prints `PASS` and writes a fresh JSON receipt. It does not promote the candidate or rely on pod logs for correctness.

Any failure exits nonzero. Launch failures write `STOP.json` in their newly created work directory; existing work directories are never reused or modified. An absent/base-only candidate at the verification deadline is `INDETERMINATE`, not evidence that no write occurred. Inspect the Job and candidate read-only before considering another approved attempt. A spot interruption may leave an absent candidate, a candidate at the base, or a committed candidate; there is no automated cleanup or resubmission. Deletion and promotion remain separate operator decisions.


The patch retains Zarr’s default omission of fill-only inner chunks. Its encoded bytes need not reproduce the original archive encoding; the codec round trip must preserve every decoded uint32 value, and the committed bytes must match the independently approved staged bytes. No fallback loosens those comparisons.

Build metadata field semantics follow [Docker’s Buildx reference](https://docs.docker.com/reference/cli/docker/buildx/build/#write-build-result-metadata-to-a-file---metadata-file). A single-platform image manifest or an index containing exactly one Linux arm64 image is accepted; the resolved image is pinned by digest in the Job.
