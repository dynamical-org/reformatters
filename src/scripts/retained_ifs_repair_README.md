# Repairing a retained IFS ensemble snapshot

For the bounded streaming patch path, use [retained_ifs_patch_README.md](retained_ifs_patch_README.md). The PVC/regeneration instructions below describe the earlier runner.

`src/scripts/retained_ifs_repair.py` stages and verifies exact variable/init units from an explicitly pinned IFS 15-day snapshot. With an additional flag it commits only changed data shards to a **fresh descendant branch**. It never publishes, resets or deletes a branch, changes metadata, or opens a writable session on main or the protected input branch. Existing candidate names fail, including on retries.

This is an operator-reviewed repair tool. A reviewed manifest, independent source evidence and explicit execution authorization are prerequisites. It is not a substitute for an ordinary backfill. Run the full-size pipeline in a separately approved Job; a single variable/init buffer can exceed a shared workstation's available memory.

## Inputs and approval boundary

Prepare a read-only input directory containing `manifest.json`, all referenced request rosters/indexes, independent expected `.npy` fields and evidence receipts. Approve the **SHA256 of the exact manifest file** outside that directory. Artifacts use relative paths constrained to the input directory and full SHA256 hashes. The manifest is the approved scope, not an automatically generated permission to repair.

The manifest's Pydantic schema is available with:

```sh
uv run python -c 'import json; from scripts.retained_ifs_repair import Manifest; print(json.dumps(Manifest.model_json_schema(), indent=2))'
```

Required top-level fields:

- `version: 1`, the IFS 15-day `dataset_id`, explicit S3 `repository`, `base_snapshot`, `protected_branch`, and never-existing `candidate_branch`.
- `runtime_sha256`: `runtime_digest(checkout_root)` from a clean checkout of the exact image commit, or compute it inside the digest-pinned image using `/app`. It binds all Python source files, `pyproject.toml` and `uv.lock`; runtime also checks installed numerical/decoder/storage package versions against the lock. The separately built unmerged-branch image must be pinned by digest in the reviewed Job.
- `metadata_sha256`: `metadata_digest(metadata_bytes(store, open_tree(store)))`, where `store` is the read-only session at the exact base snapshot. This covers every metadata key and coordinate chunk, including runtime coordinates.
- `prerequisites`: an artifact containing the independently reviewed prerequisite receipt described below.
- `units`: an exact list of `(variable, init)` pairs, each with a hash-pinned `requests` roster and `fields`. Each field specifies `member`, `lead_hours`, a `values` artifact and a `source_evidence` artifact. Times are naive UTC ISO strings. Duplicate units/fields, unmatched times and out-of-grid fields fail. There is no timestamp/variable Cartesian product.

The prerequisite receipt must contain `status: "pass"`, the matching `base_snapshot`, `runtime_sha256`, and `units_sha256` (the `object_digest` of the units' JSON-mode model dumps). Its reviewed gates must each be `true`: `selector_parity`, `config_parity`, `decoder_transform_parity`, `full_field_source_confirmation`, `retention_review`. These are **review attestations**, not tests the runner can infer from a Boolean. The receipt must include an `evidence` mapping from each gate name to a hash-pinned measurement artifact; those artifacts must contain the actual measurements and source identities. Review them before approving the receipt hash.

Before setting those gates:

1. Compare ordered selected ranges for every coordinate in every proposed unit under the historical and proposed locked environments. Compare the historical data-variable configuration and source routing too. Filenames deliberately need not match. The selector-parity evidence must name each unit's `requests` SHA256 and show that its ordered `(lead_hours, member, url, starts, ends)` entries equal the parity ledger's selected ranges, or are a verified derivation from that ledger.
2. Decode representative MARS and Open Data messages under both locked environments, applying historical scale/offset, casts, deaccumulation and rounding in the same order. Any difference blocks whole-unit regeneration; do not silently switch to a subfield patch. Representative decoding does not exercise unit-wide deaccumulation thresholds, the segment beyond 144 hours or most payloads; the stage-only full-unit comparison outside permitted fields is therefore the decisive parity check. Any difference stops the repair; never widen the permitted mask or switch method without a new reviewed manifest.
3. Independently identify and decode every permitted output field over its complete grid. Check GRIB parameter/level, init, member, forecast step and grid orientation. Fetch messages in isolation to avoid the filename-sharing mechanism. For deaccumulation, include preceding raw accumulations and successor outputs; preserve the raw-baseline and skip-mask algorithm.
4. Verify repository retention/expiry/GC policies preserve the protected snapshot and required ancestry.

Each independent expected array must be float32, latitude × longitude in the base grid's order, after the production transformations. Its source-evidence JSON must bind `status: "pass"`, `independent: true`, `init`, `variable`, `member`, `lead_hours`, `values_sha256`, and nonempty `source_message_sha256s`. Include decoded header identities, exact source URL/ranges and transformation evidence in that reviewed receipt. The runner verifies the bindings and every expected output cell; the operator's evidence review establishes independence and header correctness. Do not use the regenerated staging output as its own reference.

Each request roster contains exactly one entry per generated `(lead_hours, member)` coordinate: `url`, `index_path`, `index_sha256`, ordered `starts` and `ends`, plus those two coordinate fields. `index_path` is relative to the roster's directory. The adapter re-runs the real selector against those pinned index bytes before any data download, verifies the full coordinate roster and source URLs, and uses the fixed range digest. MARS uses the source archive; Open Data uses GCS without mirror substitution. Intentional absent lead-zero coordinates are enumerated separately from failures. Every generated source must download, decode and transform successfully.

## Staging and commit gates

The runner clones the base's raw metadata and coordinate chunks into local staging. It resolves absolute init indices from that snapshot, builds one RegionJob per exact pair, and uses its read/transform/shard pipeline with strict failure handling. A deaccumulation error blocks the unit rather than merely logging. It computes result-coordinate updates and requires equality with the pinned input coordinates.

For every unit, all cells outside the permitted full fields must have identical finite float32 bits and identical NaN masks. Every cell inside each permitted field must equal the independently prepared source array. Missing staging shards, extra data keys, metadata changes and no-op repairs fail. NaN payload-bit differences are ignored; finite signed-zero differences are not. Only shards containing scientific differences are eligible for copying.

All units must pass before branch creation. Commit mode then rechecks the protected ref and candidate absence, creates the candidate explicitly at the pinned snapshot, checks the session base, copies only the validated encoded shard keys, checks the transaction diff and commits once without rebase. It reopens the returned snapshot and repeats metadata, scope and full-unit/source comparisons. It records precommit, commit and final evidence separately. A failed or ambiguous commit is not automatically retried; investigate the refs read-only. Never delete/reset a candidate as a retry shortcut.

Stage-only is the default and creates no branch or writable repository session. A full-size stage-only pilot is still a Job requiring separate authorization and the complete prerequisite evidence set, including full-field source confirmations. Confirm anonymous repository reads before selecting this pilot mode. Measure source bytes, compressed output, peak memory, shared memory, scratch/PVC use and elapsed time before selecting resources or quoting costs. The container memory limit must cover the shared-memory buffer plus all worker/process RSS. The output PVC retains every downloaded source range, temporary shard and staging/evidence file across all units without pruning; account for both pilot and commit runs if they share a PVC. Inputs are read-only; stage files and evidence use a distinct persistent output volume. Provisioning or populating either volume is a separate approved operation. Before launch, review the PVC storage class, size and availability-zone compatibility; the input population and hash-verification procedure; the evidence retrieval/archive procedure; and the authorized cluster identity that will create the Job. Do not delete the output PVC before its evidence is archived. Check nodepool selectors, taints and tolerations against the rendered Job.

## Build an image from the unmerged branch

Keep the repair PR open as a draft; do not merge it. The runner is used from a reviewed commit on `retained-branch-repair-runner`, independently of main. A successful PR quality check does not build its runtime image. Do not use Deploy Operational Updates, Deploy Staging, a `stage/**` branch, or Manual Backfill for this operation: those paths deploy workloads or select main's deployed image.

The following is an **operator procedure after separate image-build/push approval**, not an instruction to execute as part of documentation or code review. It does not grant Job, PVC, data-write or deployment approval. Use Docker Buildx with an already approved remote native `linux/amd64` builder, `jq`, and registry authentication supplied through the approved credential mechanism. Choose an approved image repository accessible to the eventual cluster nodes; do not create a registry or change pull permissions implicitly. No storage credentials or repair input bundle belong in the build context. Review image-retention and tag-immutability policies before pushing; do not replace an existing repair tag. Preserve the unmerged branch and approved commit without rebasing, force-pushing or deleting them while their evidence is in use. This procedure replaces any assumption that a normal main deployment supplies the repair image.

Set `REPAIR_COMMIT` to the externally approved full commit SHA, `REPAIR_IMAGE_REPOSITORY` to the approved registry/repository, `REPAIR_BUILDER` to the approved builder, and `REPAIR_EVIDENCE_DIR` to a fresh local evidence directory. Run from a clone containing the repair branch. Fetching the branch does not approve its new tip: a later branch update requires review and a new explicit commit pin.

```bash
set -euo pipefail
: "${REPAIR_COMMIT:?Set the reviewed full commit SHA}"
: "${REPAIR_IMAGE_REPOSITORY:?Set the approved image repository}"
: "${REPAIR_BUILDER:?Set the approved native amd64 builder}"
: "${REPAIR_EVIDENCE_DIR:?Set a fresh evidence directory}"
git fetch origin retained-branch-repair-runner
test "$(git rev-parse "${REPAIR_COMMIT}^{commit}")" = "$REPAIR_COMMIT"
git merge-base --is-ancestor "$REPAIR_COMMIT" origin/retained-branch-repair-runner
mkdir "$REPAIR_EVIDENCE_DIR"
REPAIR_EVIDENCE_DIR="$(realpath "$REPAIR_EVIDENCE_DIR")"
REPAIR_BUILD_ROOT="$REPAIR_EVIDENCE_DIR/source"
mkdir "$REPAIR_BUILD_ROOT"
git show -s --format='%H %T' "$REPAIR_COMMIT" > "$REPAIR_EVIDENCE_DIR/source-identity.txt"
git archive --format=tar "$REPAIR_COMMIT" > "$REPAIR_EVIDENCE_DIR/source.tar"
sha256sum "$REPAIR_EVIDENCE_DIR/source.tar" > "$REPAIR_EVIDENCE_DIR/source.sha256"
tar -xf "$REPAIR_EVIDENCE_DIR/source.tar" -C "$REPAIR_BUILD_ROOT"
REPAIR_IMAGE_TAG="${REPAIR_IMAGE_REPOSITORY}:retained-repair-${REPAIR_COMMIT}"
docker buildx build --builder "$REPAIR_BUILDER" \
  --platform linux/amd64 --provenance=false --sbom=false \
  --file "$REPAIR_BUILD_ROOT/deploy/Dockerfile" \
  --label "org.opencontainers.image.revision=$REPAIR_COMMIT" \
  --tag "$REPAIR_IMAGE_TAG" \
  --metadata-file "$REPAIR_EVIDENCE_DIR/build-metadata.json" \
  --push "$REPAIR_BUILD_ROOT" 2>&1 | tee "$REPAIR_EVIDENCE_DIR/build.log"
REPAIR_DIGEST="$(jq -er '."containerimage.digest" | select(test("^sha256:[a-f0-9]{64}$"))' "$REPAIR_EVIDENCE_DIR/build-metadata.json")"
REPAIR_IMAGE_DIGEST="${REPAIR_IMAGE_REPOSITORY}@${REPAIR_DIGEST}"
printf '%s\n' "$REPAIR_IMAGE_DIGEST" > "$REPAIR_EVIDENCE_DIR/image-reference.txt"
docker buildx imagetools inspect "$REPAIR_IMAGE_DIGEST" > "$REPAIR_EVIDENCE_DIR/registry-inspect.txt"
docker buildx imagetools inspect "$REPAIR_IMAGE_DIGEST" --raw > "$REPAIR_EVIDENCE_DIR/registry-manifest.json"
docker buildx imagetools inspect "$REPAIR_IMAGE_DIGEST" --format '{{json .Image}}' > "$REPAIR_EVIDENCE_DIR/image-configs.json"
jq -e --arg commit "$REPAIR_COMMIT" \
  '.os == "linux" and .architecture == "amd64" and .config.Labels["org.opencontainers.image.revision"] == $commit' "$REPAIR_EVIDENCE_DIR/image-configs.json"
jq -e '.mediaType | test("image.manifest|manifest.v2")' "$REPAIR_EVIDENCE_DIR/registry-manifest.json"
```

The source archive excludes local untracked files and credentials. The image uses the archived `deploy/Dockerfile` and `uv.lock`, not a moving main checkout. [Buildx's metadata file](https://docs.docker.com/reference/cli/docker/buildx/build/#write-build-result-metadata-to-a-file---metadata-file) supplies `containerimage.digest`; do not substitute `containerimage.config.digest`, a local image ID, or the mutable tag. Review the [registry inspection](https://docs.docker.com/reference/cli/docker/buildx/imagetools/inspect/) against that digest, confirm the image config is `linux/amd64` and its revision label matches the approved commit. This single-platform build explicitly disables attestations, so record the image manifest digest rather than expecting a multi-platform index. Preserve the image and evidence under the approved retention policy until repair verification is complete.

Keep the evidence directory and extracted source on persistent local storage. Later blocks require the same shell variables. After a shell restart, set `REPAIR_EVIDENCE_DIR` to that preserved absolute path, then restore and check the recorded values below against the externally approved source/image pins. If the extracted source is missing, restore it into a fresh `source` directory from the verified archive before continuing; never substitute the current checkout.

```bash
set -euo pipefail
: "${REPAIR_EVIDENCE_DIR:?Set the preserved absolute evidence directory}"
sha256sum -c "$REPAIR_EVIDENCE_DIR/source.sha256"
REPAIR_BUILD_ROOT="$REPAIR_EVIDENCE_DIR/source"
REPAIR_COMMIT="$(awk '{print $1}' "$REPAIR_EVIDENCE_DIR/source-identity.txt")"
REPAIR_IMAGE_DIGEST="$(cat "$REPAIR_EVIDENCE_DIR/image-reference.txt")"
test -f "$REPAIR_BUILD_ROOT/src/scripts/retained_ifs_repair.py"
```

Before using that digest, run credential-free smoke checks on a native amd64 host. Override the Dockerfile's ordinary reformatter entrypoint; do not start a dataset update. The build, smoke tests, pilot and commit Job all use amd64, matching the existing parity measurements:

```bash
set -euo pipefail
: "${REPAIR_BUILD_ROOT:?}" "${REPAIR_IMAGE_DIGEST:?}" "${REPAIR_EVIDENCE_DIR:?}"
test -f "$REPAIR_BUILD_ROOT/src/scripts/retained_ifs_repair.py"
docker run --rm --platform linux/amd64 --network none --read-only \
  --tmpfs /tmp -e NUMBA_CACHE_DIR=/tmp/numba -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app/src \
  -e OMP_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  --entrypoint python "$REPAIR_IMAGE_DIGEST" \
  -c 'import json, platform, sys, rasterio, reformatters; from pathlib import Path; import scripts.retained_ifs_repair as repair; from scripts.retained_ifs_repair import runtime_digest; assert Path(reformatters.__file__).resolve().is_relative_to(Path("/app/src")); assert Path(repair.__file__).resolve().is_relative_to(Path("/app/src")); print(json.dumps({"runtime_sha256": runtime_digest(Path("/app")), "runner_file": repair.__file__, "reformatters_file": reformatters.__file__, "python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__}))' > "$REPAIR_EVIDENCE_DIR/smoke-runtime.json"
docker run --rm --platform linux/amd64 --network none --read-only \
  --tmpfs /tmp -e NUMBA_CACHE_DIR=/tmp/numba -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app/src \
  --entrypoint python "$REPAIR_IMAGE_DIGEST" -m scripts.retained_ifs_repair --help
docker run --rm --platform linux/amd64 --network none --read-only \
  --tmpfs /tmp -e NUMBA_CACHE_DIR=/tmp/numba -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app/src \
  --entrypoint python "$REPAIR_IMAGE_DIGEST" -m scripts.retained_ifs_repair_job --help
```

Compute the independent source-tree hash with the same archived runner:

```bash
set -euo pipefail
: "${REPAIR_BUILD_ROOT:?}" "${REPAIR_IMAGE_DIGEST:?}" "${REPAIR_EVIDENCE_DIR:?}"
test -f "$REPAIR_BUILD_ROOT/src/scripts/retained_ifs_repair.py"
PYTHONPATH="$REPAIR_BUILD_ROOT/src" uv run --project "$REPAIR_BUILD_ROOT" --frozen python \
  -c 'import sys; from pathlib import Path; from scripts.retained_ifs_repair import runtime_digest; print(runtime_digest(Path(sys.argv[1])))' \
  "$REPAIR_BUILD_ROOT" > "$REPAIR_EVIDENCE_DIR/source-runtime-sha256.txt"
test "$(jq -er .runtime_sha256 "$REPAIR_EVIDENCE_DIR/smoke-runtime.json")" = "$(cat "$REPAIR_EVIDENCE_DIR/source-runtime-sha256.txt")"
```

Capture the smoke exit statuses and output. The container runtime hash must equal this source-tree hash. Bind this value into the manifest and prerequisite receipt; retain the commit/tree, archive hash, build log/metadata, registry digest/architecture, revision label and smoke results together. Confirm imports resolve beneath `/app/src` and installed numerical/storage versions equal `uv.lock`, as `verify_inputs` requires. Record `docker buildx inspect "$REPAIR_BUILDER"`, `docker info --format '{{.Architecture}}'` and `uname -m` with operator confirmation that the builder and smoke host are native amd64. Build caches may be reused; do not describe this as a cold or hermetic build. Run the renderer against a synthetic manifest without credentials or real repair inputs and inspect its command override. The Dockerfile uses unpinned apt package versions, so record Python, OS and rasterio/GDAL versions and compare them with the parity environment; a difference requires explicit evidence review rather than an assumption of a reproducible image. Re-attest parity against this exact runtime before approving the manifest. A failed build, inspection, smoke check or hash comparison stops preparation; do not fall back to a main image. Rebuilds can yield another image digest even for the same commit, so each digest needs its own evidence and approval.

## Render and launch

Use `REPAIR_IMAGE_DIGEST` from the separately approved branch build above. Run the renderer from the same archived commit, with `PYTHONPATH="$REPAIR_BUILD_ROOT/src" uv run --project "$REPAIR_BUILD_ROOT" --frozen python -m scripts.retained_ifs_repair_job` (followed by the options below), rather than from main or a later branch tip. Supply an already prepared input PVC containing `manifest.json` at its root and a separate output PVC large enough for staging and retained evidence. Resource values below must come from the reviewed sizing gate; they are not automatic defaults.

```bash
set -euo pipefail
: "${REPAIR_BUILD_ROOT:?}" "${REPAIR_IMAGE_DIGEST:?}" "${REPAIR_EVIDENCE_DIR:?}"
test -f "$REPAIR_BUILD_ROOT/src/scripts/retained_ifs_repair.py"
: "${REPAIR_INPUTS:?Set the approved local manifest directory}"
set -o noclobber
PYTHONPATH="$REPAIR_BUILD_ROOT/src" uv run --project "$REPAIR_BUILD_ROOT" --frozen python -m scripts.retained_ifs_repair_job \
  --manifest "$REPAIR_INPUTS/manifest.json" \
  --name "$REPAIR_JOB" --namespace "$REPAIR_NAMESPACE" \
  --image "$REPAIR_IMAGE_DIGEST" --capacity "$REPAIR_CAPACITY" \
  --input-claim "$REPAIR_INPUT_PVC" --output-claim "$REPAIR_OUTPUT_PVC" \
  --memory "$REPAIR_MEMORY" --shared-memory "$REPAIR_SHM" \
  --scratch "$REPAIR_SCRATCH" --output repair-job.unpinned.json
jq '.spec.template.spec.nodeSelector["kubernetes.io/arch"] = "amd64"' \
  repair-job.unpinned.json > repair-job.json
jq -e --arg image "$REPAIR_IMAGE_DIGEST" \
  '.spec.template.spec.nodeSelector["kubernetes.io/arch"] == "amd64" and .spec.template.spec.containers[0].image == $image' repair-job.json
diff -u <(jq -S . repair-job.unpinned.json) \
  <(jq -S 'del(.spec.template.spec.nodeSelector["kubernetes.io/arch"])' repair-job.json)
sha256sum repair-job.unpinned.json repair-job.json
```

The renderer does not currently select an architecture; the local `jq` step pins amd64. Review the final `repair-job.json` bytes and hash, including this selector and the exact `REPAIR_IMAGE_DIGEST`, before approval. Use this same architecture for both pilot and commit; a different architecture requires new parity evidence. This only writes a local Job definition. It has one worker, four CPUs, one numerical thread, read-only inputs, no service-account token and no automatic retries. It does not create volumes or submit anything. For a separately authorized commit run, render a **new** definition, repeat the same `jq` architecture pin and verification/hash steps for it, and include `--commit-to-new-branch --storage-secret "$REPAIR_STORAGE_SECRET"`; the secret must expose the standard `contents` key. Only that run mounts write-capable storage credentials, and it requires `--capacity on-demand`. A stage-only pilot may use spot if approved. Commit mode regenerates from source in a new work directory; it does not reuse earlier pilot output, so account for repeated source I/O. Use a new Job name and an unused work-directory name for every attempt.

After explicit launch approval, the operator's launch command is:

```bash
set -euo pipefail
test -f repair-job.json
kubectl create -f repair-job.json
```

The container runs `python -m scripts.retained_ifs_repair` with the pinned manifest file/hash and a fresh work directory. Do not run the whole-unit worker on a memory-constrained workstation. Preserve the persistent evidence even if the Job fails. A nonzero exit or `STOP.json` means stop and investigate; Job success alone is insufficient without `result.json` and the returned snapshot's postchecks.

Repair does not authorize publication. Historical extension, overlap reconciliation, unavailable-source gaps, reader compatibility, catalog/version identity and operational writer coordination require a separate decision. A corrected retained snapshot must not replace a newer main snapshot without that review.
