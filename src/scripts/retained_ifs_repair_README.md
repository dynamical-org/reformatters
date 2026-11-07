# Repairing a retained IFS ensemble snapshot

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
- `runtime_sha256`: `runtime_digest(checkout_root)` from a clean checkout of the exact image commit, or compute it inside the digest-pinned image using `/app`. It binds all Python source files, `pyproject.toml` and `uv.lock`; runtime also checks installed numerical/decoder/storage package versions against the lock. The normally built container image must independently be pinned by digest in the reviewed Job.
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

## Render and launch

Use the normally built reviewed image by digest. Supply an already prepared input PVC containing `manifest.json` at its root and a separate output PVC large enough for staging and retained evidence. Resource values below must come from the reviewed sizing gate; they are not automatic defaults.

```sh
uv run python -m scripts.retained_ifs_repair_job \
  --manifest /approved-inputs/manifest.json \
  --name "$REPAIR_JOB" --namespace "$REPAIR_NAMESPACE" \
  --image "$REPAIR_IMAGE_DIGEST" --capacity "$REPAIR_CAPACITY" \
  --input-claim "$REPAIR_INPUT_PVC" --output-claim "$REPAIR_OUTPUT_PVC" \
  --memory "$REPAIR_MEMORY" --shared-memory "$REPAIR_SHM" \
  --scratch "$REPAIR_SCRATCH" --output repair-job.json
```

This only writes a local Job definition. Review its exact bytes and hash. It has one worker, four CPUs, one numerical thread, read-only inputs, no service-account token and no automatic retries. It does not create volumes or submit anything. For a separately authorized commit run, render a **new** definition with `--commit-to-new-branch --storage-secret "$REPAIR_STORAGE_SECRET"`; the secret must expose the standard `contents` key. Only that run mounts write-capable storage credentials, and it requires `--capacity on-demand`. A stage-only pilot may use spot if approved. Commit mode regenerates from source in a new work directory; it does not reuse earlier pilot output, so account for repeated source I/O. Use a new Job name and an unused work-directory name for every attempt.

After explicit launch approval, the operator's launch command is:

```sh
kubectl create -f repair-job.json
```

The container runs `python -m scripts.retained_ifs_repair` with the pinned manifest file/hash and a fresh work directory. Do not run the whole-unit worker on a memory-constrained workstation. Preserve the persistent evidence even if the Job fails. A nonzero exit or `STOP.json` means stop and investigate; Job success alone is insufficient without `result.json` and the returned snapshot's postchecks.

Repair does not authorize publication. Historical extension, overlap reconciliation, unavailable-source gaps, reader compatibility, catalog/version identity and operational writer coordination require a separate decision. A corrected retained snapshot must not replace a newer main snapshot without that review.
