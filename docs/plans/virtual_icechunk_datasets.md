# Virtual Icechunk Datasets

> Design plan. The architecture below is settled. Every icechunk-2.x behavior it
> leans on has been **derisked up front** against icechunk 2.0.5 / zarr 3.1.3 /
> gribberish 0.30.3 on Python 3.14 — the former "spike-to-confirm" items are now
> measured results, folded into each section and collected in
> [Spike results](#spike-results). The retry/commit design did not depend on the
> answers; the one finding that changed code is that **empty commits raise** (see
> [the write loop](#the-virtual-write-loop)).

## Goals

Create virtual Icechunk datasets that complement our existing materialized
(rechunked) time-series datasets. For datasets whose source data lives in a
publicly readable GRIB archive (NOAA NODD, ECMWF and DWD archives on Source
Coop), a virtual dataset stores only Icechunk metadata — virtual chunk
references pointing at byte ranges inside the original GRIB files. No copy of
the data lives in our store.

**Why:**

- **Spatial/map-optimized chunking, complementary to time-series.** Chunks
  follow the native GRIB message shape (one time step, full spatial grid),
  ideal for spatial queries and map rendering. Users pick the materialized
  time-series dataset for time-series extraction and the `-spatial` virtual
  dataset for map/spatial queries over the same underlying data.
- **Every curated source variable.** The marginal storage cost of a variable
  is a handful of byte-range refs, not a rechunked copy, so we can include
  every variable we have metadata for — not just the curated materialized
  subset. (Expansion to "all source variables" is gated on per-variable
  metadata; see [Open questions](#open-questions).)
- **Very low latency updates.** Target ≤ 5 s end-to-end from a new source
  file appearing to its refs being visible. We hit this by keeping a pod up
  and polling *before* files start dropping (publication times are
  predictable), and committing each newly-observed file immediately. Writing
  refs is near-instant — we record byte offsets, we don't move data.

The PR #511 prototype proved the core read/write path end-to-end against real
NOAA GFS S3 GRIB files (see [Appendix A](#appendix-a-prototype-pr-511)). The
prototype is a 63-ref toy; the [Scale](#scale) section below sizes the real
thing.

## Scale

Virtual datasets are *metadata-heavy*, not storage-heavy. One GRIB message —
one field for a single `(var, init, member, lead)` (forecast) or `(var, time)`
(analysis) — becomes one virtual chunk, so ref counts track the full Cartesian
product of the dataset's dimensions, *including the variable axis*.

Rough order-of-magnitude — **GEFS 16-day, 31 members, ~200 var/level combos,
4 inits/day, 6-year archive**. These are back-of-envelope (lead count is
approximate, etc.), meant only to fix the order of magnitude:

| quantity | approx |
|---|---|
| refs per init (`~200 vars × 31 members × ~100 leads`) | ~6 × 10⁵ |
| inits (`6 yr × 365 × 4`) | ~9 × 10³ |
| **total refs** | **~10⁹ (billions)** |
| source index files (one `.idx` per `(init, member, lead)`) | **~10⁷ (tens of millions)** |

Two consequences drive the rest of this design:

1. **A single unsplit manifest is impractical at this ref count.** Measured
   manifest overhead is **~25 bytes/ref** for realistic distinct GRIB URLs with
   icechunk's default location compression on (~19 with sequential offsets, ~9
   when every ref shares one URL; see [Spike results](#spike-results)). At order
   10⁹ refs that is **~25 GB** of manifest across the dataset — tiny as raw
   bytes, but far past the point where rewriting one per-array manifest per
   commit is viable, so [splitting](#manifest-splitting) is necessary, not a
   nice-to-have, and the old "stores are KBs–MBs, co-location is free" framing is
   wrong.
2. **Backfill is bound by index-file count, not ref count.** Order 10⁷ index
   downloads is hours of work even at high concurrency, so **backfills must run
   parallel across pods** (see [Backfill](#backfill)). Operational updates touch
   a handful of files per fire and run single-writer.

## The set_virtual_refs loop

This is the entire core operation. Everything else is plumbing around it.

```python
for source_file in source_files:                       # see "what a file covers" below
    index_path = download_index(source_file)           # ~30 KB
    messages = parse_index(index_path, data_vars)       # one entry per GRIB message in the file

    refs = [
        VirtualRef(
            data_var=msg.var,                           # which array this message fills
            out_loc=msg.out_loc,                        # the cell this message fills
            location=source_file.get_url(),             # s3://noaa-gefs-pds/...
            offset=msg.start,
            length=msg.end - msg.start,
        )
        for msg in messages
    ]
    store.set_virtual_refs(refs)                        # all of this file's refs, one batch

session.commit("...")                                   # readers see new data
```

At read time `GribberishCodec` (a zarr v3 `ArrayBytesCodec` serializer) decodes
the raw GRIB message the ref points at. One file's refs are produced and
committed together — that atomicity is a stated invariant (see
[Reader safety](#reader-safety)).

**What a single file+index covers is dataset-specific — don't assume.** A source
file maps to an *opaque set* of `(var, init, member, lead)` (or `(var, time)`)
cells, and `parse_index` yields one entry per GRIB message it contains. The
packing varies by provider:

- **GEFS** — one file per `(init, member, lead)`, all variables inside.
- **ECMWF open-data** — all ensemble members packed into one file.
- **GEFS v12 reforecast** — all lead times for a *single* variable in one file.

Everything downstream — the per-file commit unit, `chunk_key`, `sync_dims_to`,
the filter — treats a file as that opaque cell set. **No part of the design bakes
in "one `.idx` per `X`";** where this document uses GEFS for concrete numbers, it
says so.

**Existing code reused directly:**

| Need | Existing code |
|---|---|
| GRIB URLs | `SourceFileCoord.get_url()` per dataset |
| NOAA `.idx` → `(starts, ends)` | `noaa/noaa_grib_index.py::grib_message_byte_ranges_from_index` |
| ECMWF `.index` → `(starts, ends)` | `ecmwf/ecmwf_grib_index.py` parser (renamed for consistency in PR #2) |
| Per-variable GRIB element / level | `<dataset>InternalAttrs.grib_element`, `grib_index_level`, … |
| Index download | `common/download.py::http_download_to_disk` |
| Candidate source files for a region | `RegionJob.generate_source_file_coords()` (shared with materialized) |

## Class architecture

One `DynamicalDataset` hierarchy; `MaterializedRegionJob` and `VirtualRegionJob`
as siblings under a shared `RegionJob` base.

The current `DynamicalDataset` base already abstracts the right things for both
variants. CLI, store factory, kubernetes resources, validation, Sentry — none
fork. `RegionJob` is where the variant-specific work lives (materialized has the
download/read/shared-memory/write/upload pipeline; virtual has the
poll-and-emit loop and ref helpers), alongside a meaningful shared surface
(`get_jobs` partitioning, `source_groups`, `generate_source_file_coords`, the
`operational_update_jobs` contract). A three-class hierarchy expresses that
cleanly.

```
DynamicalDataset                          # unchanged single base
├── NoaaGefsForecast35DayDataset            (materialized; existing)
├── NoaaGefsForecast35DaySpatialDataset     (virtual; new)
└── …

RegionJob                                 # shared base: fields, get_jobs,
│                                         #   source_groups, generate_source_file_coords,
│                                         #   operational_update_jobs (abstract)
├── MaterializedRegionJob                 # process() pipeline + hooks + tunables
│   ├── NoaaGfsCommonRegionJob → NoaaGfsForecastRegionJob   (base swap only)
│   ├── NoaaHrrrRegionJob → …                                (base swap only)
│   └── …
└── VirtualRegionJob                      # the one virtual write loop, ref helpers,
    │                                     #   process_virtual_refs (abstract generator),
    │                                     #   filter_already_present (default impl),
    │                                     #   sync_dims_to, chunk_key
    ├── NoaaGfsForecastSpatialRegionJob
    └── …
```

`process()` is defined once today and called from exactly one place (the
driver); no dataset overrides it or its materialized helpers
(`_download_processing_group`, `_read_into_data_array`, `_write_shards`,
`_cleanup_local_files`), and nothing calls `super().process()`. So extracting
`MaterializedRegionJob` is a mechanical base-class swap on the ~13 region jobs
that inherit `RegionJob` directly; the four intermediate HRRR/GFS subclasses
ride their parent's swap. Tests pass unchanged. This is PR #1 on its own.

### What lives where

**`DynamicalDataset` (near-unchanged surface):**

- Fields: `template_config`, `region_job_class`, storage configs, plus the new
  `icechunk_virtual_config` (see [Storage configuration](#storage-configuration)).
- CLI, Sentry, `validators()`, `operational_kubernetes_resources()` — unchanged.
- `_process_region_jobs` driver — gains **one** fork: a virtual operational
  update (region job class is a `VirtualRegionJob` subclass *and* this is an
  operational update) takes the single-writer-to-`main` path; everything else
  (materialized backfill/update, **and virtual backfill**) keeps the existing
  temp-branch path.
- A model validator: if `region_job_class` is virtual then every storage
  config must be `ICECHUNK` and `icechunk_virtual_config` must be present with a
  non-empty container set.

**`RegionJob` (shared base):**

- Fields: `tmp_store`, `template_ds`, `data_vars`, `append_dim`, `region`,
  `reformat_job_name`.
- `get_jobs()` — region × variable-group fan-out (see
  [Partitioning](#partitioning)).
- `source_groups()`, `get_processing_region()`, `generate_source_file_coords()`
  (abstract), `operational_update_jobs()` (abstract). The per-region write entry
  point lives on each subclass, not the base — `process()` on
  `MaterializedRegionJob`, `process_virtual()` on `VirtualRegionJob` — so neither
  carries a stub for the other variant's method.

**`MaterializedRegionJob` (subclass):** today's `process(primary_store,
replica_stores)` body, the `download_file` / `read_data` /
`apply_data_transformations` hooks, the parallelism tunables, and the
shared-memory helpers. The existing datasets' region jobs change base class here
and nothing else.

**`VirtualRegionJob` (subclass):**

- `process_virtual(primary_repo, replica_repos, branch)` — runs
  [the one virtual write loop](#the-virtual-write-loop).
- `process_virtual_refs(remaining)` — abstract generator: discover available
  source files and yield whole files' refs a batch at a time. The generator owns
  its own stopping (backfill exhausts `remaining`; an operational generator polls
  and the pod's k8s active-deadline bounds the run).
- `filter_already_present(candidates, store)` — default impl
  (see [Filtering](#filtering-already-present-coordinates)); overridable.
- `sync_dims_to(store, extent)` — idempotent dim/coord/derived-coord expansion
  (see [The virtual write loop](#the-virtual-write-loop)).
- `chunk_key(coord, var, ds)` — coord labels → chunk key, derived from array
  metadata so it can't drift from zarr (see [Chunk keys](#chunk-keys)). Shared
  by filter and emit.

## Storage configuration

Virtual datasets carry one new declarative, per-dataset config object on the
`DynamicalDataset`, holding the **real icechunk objects** directly:

```python
class IcechunkVirtualConfig(FrozenBaseModel):
    # Hold icechunk's own objects. `InstanceOf` relaxes ONLY these fields to an
    # isinstance check — no model-level `arbitrary_types_allowed`, and the
    # DynamicalDataset that holds this config is untouched.
    containers: tuple[InstanceOf[icechunk.VirtualChunkContainer], ...]
    # Manifest splitting policy (see "Manifest splitting").
    manifest_split: InstanceOf[icechunk.ManifestSplittingConfig]
    # No credentials field: every source we point at is public/anonymous-read, so
    # StoreFactory derives anonymous credentials per container. Add an optional
    # per-container credentials field here if a private source ever appears.
```

**Why hold the icechunk objects, not plain fields.** Plain
`prefix`/`region`/`anonymous` + `split_dim`/`split_size` fields are a *lossy
subset*, and the lost pieces are ones we target: a Source Coop container needs
`endpoint_url` + `s3_compatible` (an `s3_store(...)` that resolves to an
S3-compatible store, not plain region); an en-masse repoint to a GCP mirror needs
`gcs_store`; and `ManifestSplittingConfig` supports per-array rules
(`name_matches("temperature.*"): 7` vs `AnyArray(): 30`) and multi-dim splits a
single `split_dim`/`split_size` can't express. Holding the real objects avoids
re-extending a shim the day a non-AWS source appears. `InstanceOf` keeps this
clean under `FrozenBaseModel` (verified: builds with no model-wide
`arbitrary_types_allowed`, still isinstance-validated, survives
`revalidate_instances="always"`). The split verbosity has a terse escape hatch
that still returns the full object — e.g. a helper
`append_dim_split(size, dim="init_time") -> ManifestSplittingConfig` for the
common case, dropping to `from_dict(...)` only when a dataset needs more.

**Nothing serializes this config; `StoreFactory` consumes it directly.** It is
in-code config like `StorageConfig` — k8s workers rebuild the whole dataset
(this included) from the in-code registry by `dataset_id`, never from a
serialized copy. `StoreFactory` holds the whole `IcechunkVirtualConfig`, so at
repo open/create it just uses what's there: register each container with
`set_virtual_chunk_container(c)`, assemble the `RepositoryConfig` (containers +
`manifest_split`), and authorize each container anonymously —
`authorize_virtual_chunk_access=icechunk.containers_credentials({c.url_prefix:
icechunk.s3_anonymous_credentials() for c in cfg.containers})`. No round-trip,
and no introspecting the (opaque) container `.store`.

**Credentials are derived, not configured.** Every source we target is public,
anonymous-read (NOAA NODD, ECMWF, Source Coop), so there is no credentials field —
`StoreFactory` builds an anonymous authorize map straight from the container set.
If a private / requester-pays source ever appears, add an optional per-container
credentials field (held the same way, via `InstanceOf`) and have `StoreFactory`
prefer it; nothing else changes.

**Reader-facing exposure is a separate concern.** External public readers still
need to know which containers to register and authorize (anonymously); that is
surfaced through our STAC catalog + example dataset-open code, not by serializing
`IcechunkVirtualConfig`. icechunk helps: it persists the container set into the
repo (`save_config` → recovered by `Repository.open` / `fetch_config`), so the
published surface can be minimal. Credentials are *not* persisted — a read
without `authorize_virtual_chunk_access` raises, and
`authorized_virtual_container_prefixes` is empty — so the example open code always
supplies the anonymous-credentials map.

This config is held on `DynamicalDataset` (not on the region job, not on the
template) for three reasons:

1. **Cardinality.** Containers and split policy are *per-dataset* and identical
   across the primary store and every replica. A per-store home (`StorageConfig`)
   would force duplicating them on each store and risk drift; a per-region-job
   home is invisible to the reader/STAC path.
2. **One declarative home, writer and reader.** Opening a virtual store needs the
   same containers + authorize map whether writing or reading. Keeping them in one
   per-dataset object gives the writer a single source and lets the STAC /
   example-open path mirror it (it can also recover the container set from the repo
   via `fetch_config`).
3. **It is the unit of repointing.** See below.

`StoreFactory` takes `icechunk_virtual_config` as a single field and applies it
uniformly to every repo it opens (primary + replicas), building the
`icechunk.RepositoryConfig` (registered containers + manifest-split policy) and an
anonymous `authorize_virtual_chunk_access` map (one `s3_anonymous_credentials()`
per container). `__main__.py` registration is identical to materialized.

### Containers as indirection, not just auth

Virtual chunk containers do more than hold anonymous credentials — they are an
indirection layer:

- **Dedup.** Refs are stored relative to a registered container prefix, so the
  repeated `s3://noaa-gefs-pds/...` bucket+prefix isn't copied into every ref.
  On an all-virtual store that dedup is most of the on-disk footprint.
- **En-masse repoint.** icechunk persists refs resolvably-relative to the
  container prefix, so swapping a container registration (e.g. NODD-on-AWS → a
  GCP mirror, or a Source Coop move) repoints every ref at once, with no manifest
  rewrite.

### Manifest splitting

With order 10⁹ refs (see [Scale](#scale)) — ~25 GB of manifest at the measured
~25 bytes/ref — a single per-array manifest would be large enough that every
commit rewriting the whole thing is untenable. icechunk manifests are per-array,
immutable, and content-addressed; a commit rewrites only the split file(s) whose
chunk-index range changed (measured — see [Spike results](#spike-results)). We
therefore **split each array's manifest along the append dimension**:

- An operational append lands only in the current (highest-index) split; every
  historical split is referenced unchanged. **Commit cost = size of the active
  split, not the whole array.** This is what makes the ≤ 5 s budget real — see
  [Operational updates](#operational-updates).
- New appends only ever add higher indices, so splitting stays clean under
  expansion — older splits never shift.

Size the split by the operational commit-cost ceiling (small enough that
rewriting the active split is well under a second), then confirm it's
comfortably under the read-memory ceiling (it will be — the commit-cost
constraint binds first). Measured active-split rewrite cost is ~11 bytes × the
split's ref count (a 200-ref split rewrote ~2.2 KB), so even a multi-week split
holding ~10⁵ refs rewrites ~1 MB per commit — trivially sub-second. For
GEFS-scale arrays the commit-cost ceiling lands around 1–3 weeks of inits per
split.

> **Confirmed (icechunk 2.0.5).** Splitting is per-array along a named dimension:
> `ManifestSplittingConfig.from_dict({ManifestSplitCondition.AnyArray():
> {ManifestSplitDimCondition.DimensionName("init_time"): SPLIT}})`. A commit
> rewrites **only** the split(s) whose chunk-index range changed: appending a new
> init wrote one new ~440 B split manifest while all five historical split
> manifests were carried over unchanged by id; touching an existing split rewrote
> only that one split (~2.2 KB), leaving the other five untouched. The policy is
> persisted in repo config and recovered on reopen.

## Partitioning

`get_jobs()` partitions the append dimension into region jobs. Today it always
reads `encoding["shards"]`. Virtual arrays have **no shards** (one ref per GRIB
message — sharding a virtual ref is meaningless, and the PR #511 prototype uses
`shards=None`, `compressors=()`, `filters=()`). So the partition unit becomes:

> **Use `encoding["shards"]` along the append dim if set, else
> `encoding["chunks"]`.**

This is a two-line policy at the `get_jobs` call site:

```python
kind = "shards" if "shards" in encoding else "chunks"
regions = dimension_slices(template_ds, append_dim, kind=kind)
```

`dimension_slices` stays strict (it is told exactly which key to read), so
`check_for_expected_shards` and every other physical-shard reader keep meaning
*physical shards*. No new config attribute is introduced, and materialized
behavior is byte-for-byte unchanged.

Consequences by flow:

- **Materialized backfill + materialized/virtual** — shards set → shards
  (unchanged).
- **Virtual backfill** — `shards=None` → chunks → one init (or one timestep)
  per job, distributed round-robin across pods. Granularity is fine: backfill
  commits on a time cadence, not per job, so tiny jobs cost nothing, and
  `workers_total` already tunes throughput.
- **Virtual operational** — does not use `get_jobs` partitioning at all; it runs
  a single active-window job (see [Operational updates](#operational-updates)).

## The virtual write loop

Both virtual flows — backfill and operational — run the **same** loop. The only
differences are *where the session is opened* and *how big a batch the generator
yields*:

```python
candidates = self.generate_source_file_coords(scope)          # shared w/ materialized
remaining  = self.filter_already_present(candidates, store)

for refs in self.process_virtual_refs(remaining):            # refs == 1+ whole files' refs
    assert refs                                              # contract: never an empty yield
    self.sync_dims_to(stores, needed_size(refs))            # idempotent; no-op if already covered
    self.set_virtual_refs(stores, refs)                     # per array path, all stores
    commit(stores)                                           # one commit per yield (never empty)
```

`sync_dims_to(stores, extent)` is the single primitive that unifies pre-sizing
and lazy expansion. It grows the append-dim coord, the data-var arrays, and the
derived coords (via the template config's `derive_coordinates`) to cover
`extent`, **writing only the new positions** and leaving existing positions
untouched. It is a **no-op for any store already covering the extent**, and it
computes each store's missing slice from **that store's own committed size** —
not the primary's. That per-store sizing is what makes the
[replica replay](#replica-writes) safe: replicas commit before the primary, so a
partial commit can leave a replica ahead, and appending the primary's slice to an
already-grown replica would duplicate positions.

That single primitive is why the two flows don't need separate write paths:

- **Backfill** pre-sizes the full template up front (worker 0, existing
  `parallel_setup`), so every `sync_dims_to` is a no-op — the resize never
  fires, which is exactly why parallel workers writing disjoint chunks is safe.
- **Operational** is single-writer, so `sync_dims_to` actually grows `main` as
  new inits appear — and being single-writer, that resize is **never
  concurrent**.

**The yield is the commit unit.** There is no timer, no pending buffer between
yields, no `max_seconds_between_commits` knob. The driver commits on every yield
and the generator owns the batching policy — *it* is the thing that knows what's
available, what cadence files arrive on, and whether to wait-and-batch or
yield-immediately. The atomicity invariant (see
[Reader safety](#reader-safety)) says each yield is **whole files** — never
splitting a file across yields — but a yield can contain one file or many.

> **Empty commits raise — so the contract forbids empty yields.** icechunk's
> `session.commit` raises `IcechunkError` ("no changes made to the session")
> rather than no-opping. Rather than guard each commit on
> `session.has_uncommitted_changes`, the loop relies on the generator contract:
> each yield is one commit's worth of **whole files**, so a batch always has refs,
> and **setting refs always dirties the session — even re-emitting byte-identical
> refs** — so the commit is never empty. The loop **asserts** `refs` (a
> loud, early failure if a generator ever violates the contract) and leaves
> `commit_if_icechunk` committing unconditionally. The empty-poll and
> everything-already-present cases never reach the commit at all — they yield no
> batch, so the loop body simply doesn't run (a filter false-negative still
> re-emits and commits an idempotent, content-addressed no-content snapshot).

The two flows fall out naturally:

- **Operational** yields one file (or a small burst that landed in one poll
  iteration) → one commit per yield → per-file visibility, ≤5s budget intact.
- **Backfill** yields a chunk of N whole files sized for one commit's overhead
  (e.g. ~50) → millions of files collapse to ~thousands of commits without any
  outer batcher.

This dodges the only-obvious-but-actually-nasty race the timer model implies
(commit becoming "due" while the generator is blocked on the next file's
availability — needs a committer thread or asyncio `next(gen, timeout=…)`
machinery). There's nothing to flush between yields, so there's nothing to race.

## Operational updates

The whole point is ≤ 5 s visibility, so the operational lifecycle is tuned for
it and runs **single-writer**.

### Single writer, single active-window job

`operational_update_jobs()` for a virtual dataset returns **one** region job
spanning the recent, still-incomplete portion of the append dimension (the
"active window" — the dataset knows its publication cadence). One pod, one
icechunk session on `main`, one commit loop.

What multiple pods bought materialized updates was not compute — a fire ingests
a handful of files, and the prototype set 63 refs in ~1 s. It was N concurrent
long-lived poll loops. We preserve that with a **single** loop whose generator
polls the *union* of all still-missing files across the active window each
iteration:

- Filling out a still-incomplete earlier init's late leads and ingesting a brand
  new init's first leads interleave naturally — they're just entries in the same
  missing set the one loop checks each second.
- A brand new init expands the append dim with **zero possibility of a
  concurrent resize**, because there is exactly one writer. The entire
  concurrent-resize hazard — and the per-batch session-rebuild retry loop the
  earlier draft needed — simply does not exist.

The only parallelism worth having is the *read-only* fan-out (existence checks +
index downloads), which a plain threadpool inside one poll iteration provides.
All `set_virtual_refs` + commits stay single-threaded.

> If an operational fire ever falls far behind (extended downtime), do **not**
> scale it to N writers. Run a [backfill](#backfill) to catch up, then resume
> operational. Operational stays permanently single-writer.

### Latency budget

```
latency ≈ poll_interval + index_download + commit
        ≈ 1 s          + 100–500 ms     + active-split rewrite
        = 2–4 s typical, ≤ 5 s worst case
```

Each new file lands in its own commit (the generator yields per file; the driver
commits per yield — see [The virtual write loop](#the-virtual-write-loop)), so
there is no "wait for batch" latency. `commit` cost is the active manifest split
rewrite (see [Manifest splitting](#manifest-splitting)), **not** the whole array
— this is why splitting is load-bearing for the budget. Several files appearing
in one poll iteration can ride one yield if the generator chooses.

### Step-by-step

1. **CronJob fires** ~5 min before a publication window opens (per dataset).
   `concurrencyPolicy: Forbid` prevents overlapping fires.
2. **A single worker runs `update()`**, which calls `operational_update_jobs()`
   (→ one active-window job).
3. **`update()` detects a virtual dataset** and routes to
   `_run_virtual_operational_update` — the single-writer-to-`main` path: no temp
   branch, no shared setup, no coordination files, no `_process_region_jobs`. It
   opens one writable session on `main` per batch and runs
   [the virtual write loop](#the-virtual-write-loop).
4. **The loop** filters to what's still missing, drives
   `process_virtual_refs(remaining)`, lazily expands via `sync_dims_to`,
   batches refs, and commits on the ~1 s cadence.
5. **The generator exits** when the window is complete; the pod's k8s
   active-deadline bounds the run.
6. **The pod exits.** The next fire starts fresh and resumes via the filter.

### No NaN-padded future

Lazy expansion (never pre-sizing `main` toward "now") is what protects consumers:

- **Analysis** — `time` grows exactly to the data ingested. Querying past the
  end is "out of range," never NaN.
- **Forecast** — a new `init_time` slot is created in the same commit that lands
  its first refs. Querying a new init shows the leads published so far (and NaN
  for unpublished ones — same as materialized forecasts mid-update).

### Crash recovery

`filter_already_present` makes recovery trivial — a re-run picks up from the
last commit.

- **Crash mid-poll (before commit):** pending refs are dropped; the next fire
  re-discovers and re-emits them. Idempotent.
- **Crash after some commits:** committed refs and expansions are durable; the
  filter skips them and processes the rest.
- **Crash mid-expansion (after `sync_dims_to`, before commit):** the expansion
  is part of the uncommitted session and vanishes; the next fire re-expands.

## Backfill

Backfills run the **same** virtual write loop but on the **existing parallel
temp-branch flow** — they are required to be parallel (see [Scale](#scale):
order 10⁷ index files), and parallel writers are safe precisely because the
template is pre-sized.

- **Worker 0 setup** = existing `parallel_setup`: pre-size the full template on
  the `_job_<name>` branch and commit. Because the dims are written at full size
  up front, every worker's `sync_dims_to` is a no-op — no worker ever resizes,
  so workers write **disjoint** chunk refs with no expansion conflict (the
  proven materialized-backfill case, `ConflictDetector` rebase over disjoint
  paths).
- **All workers** write their round-robin slice of refs to the branch with a
  loose commit cadence (e.g. 60 s).
- **Last worker** resets `main` to the branch tip — a cheap pointer move
  regardless of store size. Readers see "empty" or "full," never partial.

The **only** thing that differs from materialized backfill is `process()`'s body
(set refs vs. materialize). No new orchestration, no `is_backfill` flag — the
existing flow is variant-general. This answers the earlier "is backfill
materialized/virtual-specific?" → **general**.

If single-pod-per-range throughput ever pinches, the scaling path stays
branch-based: more workers, each owning a disjoint append-dim range on the
pre-sized branch. Pre-sizing is what makes any of this safe; that's the dividing
line from operational (which can't pre-size `main` without showing NaN futures,
hence single-writer).

## Filtering already-present coordinates

Filtering is what makes operational updates fast in steady state and is the
crash-recovery / idempotency backbone. **There is no auxiliary progress
coordinate** — the icechunk manifest is the source of truth for what's ingested.
(A high-water-mark coord like `ingested_forecast_length` is a denormalized cache
that drifts from truth on out-of-order/gappy lead arrival; deriving "remaining"
straight from the store removes the only thing that could disagree.
`ingested_forecast_length` may still be *published* as a reader-facing output
computed from the manifest, but it is never read back to make decisions.)

```python
def filter_already_present(self, candidates, store, ds):
    # store.exists is async; the real implementation fans out across candidates
    # concurrently. Shown serially for clarity.
    return [
        c for c in candidates
        if (key := self.chunk_key(c.out_loc, self.representative_var(c), ds)) is None
        or not store.exists(key)                            # ref absent from manifest
    ]
```

One overridable hook, one mechanism for both variants:

- The author expresses intent in **coordinate labels** (timestamps /
  timedeltas); `chunk_key` translates label → chunk key. The check is **ref
  existence in the manifest**, never a value read.
- **The representative cell must be one the candidate file actually covers.**
  `representative_var(c)` picks a variable *present in that file* (for
  all-vars-per-file packings, any fixed sentinel; for one-var-per-file packings
  like GEFS v12 reforecast, the candidate's own variable). Probing a global
  indicator var that a given file doesn't contain would never see that file land
  — so the choice is packing-dependent, which is exactly why it's an overridable
  method rather than a fixed `indicator_var` field.
- **Never `.isnull()` on a virtual data var.** A present chunk's value read
  triggers an S3 fetch + GRIB decode; in steady state the recent region is
  mostly populated, so `.isnull()` would decode a near-full region every poll.
  This is the same "manifest check, not decode" the plan applies to NaN
  validation.
- `chunk_key` and the emitter share the same implementation, so filter and emit
  can never disagree about where a coord maps.

**Correctness rests on no false-positives, not on filter precision:**

- A filter **false-negative** (re-emits done work) → `set_virtual_refs` rewrites
  an identical entry. Idempotent. A pure performance papercut.
- A filter **false-positive** (skips undone work) → permanent miss.

The manifest probe has no false-positives (a key is present **iff** we emitted
it; keys are unique per `(var, init, member, lead)` / `(var, time)`). A
high-water-mark coord *does* false-positive on gappy leads — which is why we
don't use one. The `chunk_key` contract **"label not in coords ⟹ remaining"**
guarantees a brand-new position (not yet expanded) is never mistaken for done.

The "is there a ref at key K, without decoding" primitive is
[`IcechunkStore.exists(key)`](https://icechunk.io/en/stable/reference/#icechunk.store.IcechunkStore.exists)
— a per-key probe that hits the manifest (specifically, the relevant split; see
[Manifest splitting](#manifest-splitting)) without fetching or decoding any
chunk bytes. It's `async`, so the filter fans out across the candidate keys
concurrently. The candidate set per fire is small in steady state, so per-key
point lookups beat a bulk `list_prefix` and avoid the "what window to list"
question that scoping would otherwise force. (`list_prefix` remains the right
primitive for bulk scans like the manifest-check validator, which wants the
whole window's present-set at once.)

## Chunk keys

`chunk_key(coord, var, ds)` maps a source file's coordinate labels to the zarr
chunk index its ref is written under — and, for the filter, the string key that
index encodes to. It is the one place coord labels become chunk indices, shared
by both the emitter (which passes the index to `set_virtual_refs`) and the filter
(which encodes the index to a key for `exists()`) — so the only thing we must
guarantee is that **our index math matches zarr's**, exactly, with no parallel
reimplementation of the chunk-key format that could silently drift.

Two design rules enforce that:

1. **Derive geometry from array metadata, not from constants.** `chunk_key`
   reads the variable's `chunks` (and dim order) off `ds[var]` rather than
   assuming `(1, 1, lat, lon)`. The append-dim index is
   `position // chunk_size_along_append_dim` (which is 1 for virtual forecast and
   analysis arrays, but we compute it, not assume it), and it **asserts** the
   coord label resolves to an exact, in-range index — catching any drift between
   our lookup and the array's actual shape immediately rather than writing a
   mis-keyed ref.
2. **Encode the key with zarr's own machinery, not an f-string.** The
   `f"{var}/c/{i}/{j}/…"` form is zarr v3's default-separator chunk-key encoding;
   rather than hardcode it, build the key via the array's
   `chunk_key_encoding` (the same encoder zarr uses at read time). If a future
   array ever used a different separator or encoding, the key would still match
   what readers resolve — by construction, not by our remembering to keep two
   formatters in sync.

The contract **"label not in `ds`'s coords ⟹ remaining"** (a brand-new position
not yet expanded by `sync_dims_to`) is part of this method: rule 1's exact-index
assertion would otherwise fire on an unknown label, so `chunk_key` returns
"absent" for labels not present in the current coord arrays, and the filter
treats that as not-yet-ingested.

> **Confirmed.** The encoder is `array.metadata.chunk_key_encoding` (a
> `DefaultChunkKeyEncoding`); `.encode_chunk_key((i, j, k, l))` returns
> `"c/i/j/k/l"`, and the full store key is `f"{array.path}/{encode_chunk_key(...)}"`.
> A probe with the derived key — `await store.exists(key)` — returns True for a
> set ref and False otherwise. **Emit never needs the string:**
> `set_virtual_refs` / `set_virtual_refs_arr` take chunk *indices*, not key
> strings (`VirtualChunkSpec(index=[i, j, k, l], …)`), so the only hand-rollable
> string lives in the filter's `exists()` probe — and it is built from the
> array's own encoder, not an f-string.

## Derived coordinates

A virtual commit modifies more than data-var chunks; `sync_dims_to` keeps the
following consistent and atomic with the refs:

1. **Append-dim coord values** (`init_time` / `time`) for new positions.
2. **Derived coords** that depend on the append dim (`valid_time`,
   `expected_forecast_length`, etc.), recomputed by calling the template
   config's `derive_coordinates(ds)` on the expanded dataset and writing only
   the new positions. `derive_coordinates` is pure with respect to the passed
   `ds` (it reads only dataset contents plus static config), so calling it
   incrementally is sound — today it is only called on the full template at
   generation, so the incremental call path is new and covered by tests.
3. **Static derived coords** (`latitude`, `longitude`, `spatial_ref`) are
   written once at creation and don't change.

## Per-variable serializer

The one non-trivial template-layer change: each virtual data variable declares
its own `GribberishCodec(var=<grib_element>)` as its zarr **serializer** (the
single `ArrayBytesCodec` slot in `filters → serializer → compressors`).
Materialized arrays use zarr's default `BytesCodec` and never name a serializer.

`Encoding` gains one optional field (PR #2):

```python
class Encoding(pydantic.BaseModel):
    ...
    serializer: dict[str, Any] | None = None   # None ⇒ zarr's default BytesCodec
```

It threads through with no extra plumbing: `assign_var_metadata` already does
`var.encoding = var_config.encoding.model_dump(exclude_none=True)`, so the field
flows into xarray's `.encoding` and on to `to_zarr`.

> **Confirmed (zarr 3.1.3 + xarray 2026.1).** `to_zarr` threads the `serializer`
> encoding dict straight into the array's `zarr.json` codecs —
> `{"name": "gribberish", "configuration": {"var": "TMP"}}` persisted and reopened
> intact. One caveat: `to_zarr` would push data-var *chunk values* through the
> codec, and `GribberishCodec` is decode-only (`_encode_single` raises). The
> existing template path is already safe — `write_metadata` calls
> `to_zarr(..., compute=False)` over **dask-backed** `make_empty_variable` data
> vars, so no chunk is ever materialized through the read-only codec.

### Encoding factory

Each virtual variable needs its *own* `Encoding` (different `var=` per
variable), which the single-`Encoding`-shared-across-all-vars pattern can't
express. Extend the existing `get_data_vars(encoding)` to take a per-var
factory:

```python
def get_data_vars(
    self, make_encoding: Callable[[DataVar], Encoding]
) -> Sequence[DataVar]: ...
```

- **Materialized** call sites pass `make_encoding=lambda _var: encoding` — a
  one-line change at ~4 sites.
- **Virtual** builds per-var:

  ```python
  def make_encoding(var: GefsDataVar) -> Encoding:
      return Encoding(
          dtype="float64",                 # GribberishCodec decodes to float64; see below
          chunks=(1, 1, 721, 1440),        # match the GRIB message shape exactly
          shards=None,                     # virtual refs are not sharded
          serializer={"name": "gribberish",
                      "configuration": {"var": var.internal_attrs.grib_element}},
          compressors=None, filters=None, fill_value=np.nan, ...,
      )
  ```

We keep the factory even though, technically, gribberish ignores `var=` at
decode time (it decodes whatever bytes the ref points at, so a shared
placeholder would still read correct *data*). The factory is still the right
choice because:

1. **Honest metadata now.** `var=` is written into each array's `zarr.json`. A
   shared placeholder would label every array with the same wrong element —
   data decodes fine, metadata lies to anyone inspecting the store.
2. **Robustness.** "`var=` is decode-irrelevant" is a current gribberish
   property, not a contract; writing the honest label means a future gribberish
   that *does* consult `var=` can't break us.
3. **Flexibility.** Future per-var differences in dtype/fill/serializer are free.

### Encoding rules

- **Serializer:** `GribberishCodec(var=<grib_element>)` per variable.
- **Chunk shape:** matches the GRIB message shape exactly — one chunk per
  message. Common examples: `(1, 1, lat, lon)` for a deterministic forecast,
  `(1, 1, 1, y, x)` for an ensemble forecast on a non-geographic CRS, or
  `(1, lat, lon)` for analysis.
- **`shards = None`** — virtual refs are standalone messages; partitioning falls
  back to chunks (see [Partitioning](#partitioning)).
- **No compressors or filters** — GribberishCodec does the full decode.
- **dtype = float64 (native — no cast).** Confirmed: `parse_grib_array` returns
  float64 natively, so declaring float64 keeps the decoded values verbatim —
  `GribberishCodec`'s dtype-cast (`data.astype(declared)`) is skipped because
  declared == native. We declare float64 *to avoid* any astype: a non-float64
  declaration like float32 (Appendix A's prototype) is the only thing that would
  force a lossy downcast. End-to-end read-back (virtual ref → icechunk 2.0.5 →
  xarray `.values`) returned float64 values byte-equal to a direct gribberish
  decode (see [Spike results](#spike-results)).
- **Exception (DWD bz2 GRIBs):** chain zarr's `Bz2Codec` as a filter before
  GribberishCodec. Deferred until verified end-to-end.

## Reader safety

Virtual operational updates commit straight to `main` (no temp branch — that's
how we get per-file visibility in seconds). The reader guarantee rests on one
invariant:

> **One source file → one atomic commit.** All refs originating from a single
> source file (every variable, every ensemble member in that file), together
> with any append-dim expansion their position requires, are written and
> committed in a single icechunk commit. A reader on `main` sees either none of
> a source file's data or all of it.

This is **structural**, not a discipline: `process_virtual_refs` yields one
whole file's refs as the indivisible unit (one `.idx` → one `Sequence[ref]`),
and the commit loop only ever bundles *whole* yields — it can't split a file
because it never holds less than one. `sync_dims_to` + `set_virtual_refs` +
`commit` for an extent are one session, never "expand, commit, then fill."

This is the invariant that lets [filtering](#filtering-already-present-coordinates)
probe a *single representative cell* per file: because a file's refs land
atomically, one cell the file covers being present implies the whole file is
present. (Which cell is representative depends on what the file covers — see the
filter's `representative_var`; that's a packing question, separate from this
atomicity guarantee.)

**Per-file is the intended granularity.** Atomicity is **per file**, not per
init — and that's a feature, not a caveat. The whole low-latency value
proposition is that consumers see new data *as each source file becomes
available*, not at init-complete boundaries: when a forecast's leads trickle in
over the publication window, a reader sees lead 3 the moment its file is
ingested, then lead 6, and so on. Same partial-init shape as today's
materialized forecasts mid-update, but visible in seconds rather than minutes,
and exactly the granularity the filter keys on. GribberishCodec refs are
metadata-only writes — a chunk either has a complete
`(location, offset, length)` or it doesn't, with no torn-bytes failure mode — so
per-file atomicity reduces cleanly to "commit the file's refs together," which
icechunk's transactional commit provides.

**On containment.** The materialized temp branch is *not* a correctness gate we
inspect — finalization resets `main` unconditionally, trusting tested code. So
committing virtual operational updates straight to `main` doesn't remove a check
we relied on. We hold the bar with **tests, not runtime asserts**: PR #3's
integration suite covers yield-count/grouping (one file → one yield holding
exactly the cells that file covers, per the dataset's packing) and an
emit → commit → reopen → read-back round-trip that catches bad chunk keys or
byte ranges in CI.

## Replica writes

With a single writer, replica handling collapses: the operational loop opens a
session on every store (primary + replicas), `set_virtual_refs` on each, and
commits replicas-then-primary per the existing
`storage.commit_if_icechunk(message, primary_store, replica_stores)` ordering.
On any commit failure the batch retries with fresh sessions on all stores,
keeping them aligned; if the retry budget exhausts, the pod fails and the next
fire re-runs the filtered (skip-what-committed) work. "What's committed" is
defined by the **primary** — the filter probes the primary's manifest via
`exists(key)`, because the primary is what readers consume — and committing
replicas-then-primary is what makes that definition safe: if a pod dies in the
window after some replicas have committed but before the primary commits, the
primary is unchanged, so the next fire's filter re-derives the same batch and
replays it on every store. The replica work is identical refs rewritten to
identical entries (idempotent — see [Filtering](#filtering-already-present-coordinates)),
and the primary finally lands its commit. The replicas can briefly be a commit
ahead of the primary across pod restarts; they catch back up via this idempotent
replay, never the other way around.

For datasets without replicas (the expected common case early), this is one
session, one commit per batch. Backfill replicas follow the existing temp-branch
reset ordering.

## Dataset identity

### Naming

Virtual variants get a `-spatial` suffix (tentative — captures the
access-pattern optimization without leaking "virtual"):

- `noaa-gefs-forecast-35-day` (materialized, time-series optimized)
- `noaa-gefs-forecast-35-day-spatial` (virtual, spatial/map optimized)

Class names follow:
`NoaaGefsForecast35DayDataset` → `NoaaGefsForecast35DaySpatialDataset`, plus
matching region job and template config classes.

> Open question: confirm `-spatial` survives first-user feedback; rename
> before the first public dataset if a better name emerges.

### Storage location

Virtual Icechunk stores live in the same S3 buckets as the materialized
datasets, with new dataset IDs and paths. The stores are metadata-only but
**not tiny** — order 10⁹ refs at multi-year scale (see [Scale](#scale)), so
manifest size (once measured) is worth monitoring. Co-location is still fine.

### Reader experience

All targeted source archives have anonymous read access. A reader needs the
container registration + anonymous credentials map. It does **not** reconstruct
our in-code `IcechunkVirtualConfig`: the container *definitions* are recoverable
straight from the store via `Repository.fetch_config` (confirmed), so the reader
path (today the `dynamical_catalog` library wrapping our STAC catalog; bare-path
code is in [Appendix A](#appendix-a-prototype-pr-511)) only has to supply the
anonymous credentials map passed to `authorize_virtual_chunk_access` at open — a
virtual read without it raises with a "you need to authorize the virtual chunk
container" error.

## Provider-specific considerations

File packing (what a single file+index covers) varies by provider and even by
dataset within a provider; it's captured per dataset by
`generate_source_file_coords` + `parse_index`, and nothing downstream assumes a
particular shape (see [the loop](#the-set_virtual_refs-loop)).

**NOAA (GFS, GEFS, HRRR):**

- Plain-text `.idx` files, parsed by
  `noaa/noaa_grib_index.py::grib_message_byte_ranges_from_index`.
- Buckets `s3://noaa-gfs-bdp-pds/`, `s3://noaa-gefs-pds/`,
  `s3://noaa-hrrr-bdp-pds/` — all `us-east-1`, anonymous.
- Packing: GEFS forecast is one file per `(init, member, lead)`, all variables
  inside. GEFS v12 reforecast is one variable per file, all lead times inside —
  so its filter keys on the candidate's own variable, not a global sentinel.

**ECMWF (IFS-ENS, AIFS):**

- JSON Lines `.index` files; ECMWF parser renamed for naming consistency in
  PR #2.
- `s3://ecmwf-forecasts/` (`eu-central-1`) plus Source Coop archives.
- Packing: open-data files pack all ensemble members together (one file covers
  the whole member axis at a given `(init, lead)`).
- IFS-ENS has separate MARS (pre-2024-04) and open-data (post-2024-04) URL
  schemes; the virtual dataset likely covers only the open-data era. **TBD.**

**DWD (ICON-EU):**

- No index files: one variable per `.bz2`-compressed GRIB.
- Chain `Bz2Codec` (filter) before GribberishCodec (serializer). **Blocks DWD
  virtual datasets until verified end-to-end.**

## Implementation plan

### PR #1 — Extract `MaterializedRegionJob`

Move today's `process()` body, the `download_file` / `read_data` /
`apply_data_transformations` hooks, the parallelism tunables, and the
shared-memory helpers into `MaterializedRegionJob(RegionJob)`. Swap each
existing dataset's region job base from `RegionJob` to `MaterializedRegionJob`.
Mechanical; tests pass unchanged.

### PR #2 — Prep

- Rename the ECMWF index parser to `grib_message_byte_ranges_from_index` to
  match NOAA; update callers.
- Add `serializer: dict[str, Any] | None = None` to
  `common/config_models.py::Encoding` (threads through `assign_var_metadata`
  automatically).
- Add the `gribberish` dependency (`~=0.30` pin). No Python-version blocker:
  gribberish 0.30.3 ships cp311–cp314 wheels (incl. `cp314` manylinux/musl/macos/win),
  so it installs cleanly on the project's Python 3.14; the decode path is already
  smoke-tested against `icechunk 2.0.5` / `zarr 3.1.3` (see
  [Spike results](#spike-results)).
- Add `IcechunkVirtualConfig` to `common/storage.py` holding the **real icechunk
  objects** directly via `InstanceOf` (containers + split policy) — plain fields
  would be a lossy subset (no Source Coop `s3_compatible`, no GCS mirror, no
  per-array/multi-dim splits; see [Storage configuration](#storage-configuration)).
  No credentials field (all sources public/anonymous). Nothing serializes it:
  `StoreFactory` consumes the config directly to register containers and build an
  anonymous `authorize_virtual_chunk_access` map (one `s3_anonymous_credentials()`
  per container). Thread `icechunk_virtual_config` through `StoreFactory` and repo
  open/create. Add the `DynamicalDataset` validator (virtual ⇒ all stores ICECHUNK
  ∧ non-empty containers).

### PR #3 — `VirtualRegionJob` + driver

The spike is already done (run ahead of this PR — see
[Spike results](#spike-results)), so PR #3 is just the implementation it derisked.

- Add `VirtualRegionJob` (the `process_virtual` write loop, `process_virtual_refs`
  abstract generator, `filter_already_present` default, `sync_dims_to`,
  `chunk_key`). `process_virtual_refs` yields `Sequence[VirtualRef]` — one commit's
  worth of refs; a `VirtualRef` is `(data_var, out_loc, location, offset, length)`,
  carrying its `DataVar` so `chunk_key` reads the array geometry straight off it.
  The whole-file atomicity invariant (all of a file's refs in one yield) is a
  generator discipline checked by tests, not encoded in a batch type. Emit with
  `set_virtual_refs` (explicit `VirtualChunkSpec` per ref) — it handles arbitrary
  sparse cell sets directly, which the common one-message-per-`(var, cell)`
  packing produces; `set_virtual_refs_arr`'s dense columnar block is a later
  optimization for contiguous backfill ranges. The filter builds keys from
  `array.metadata.chunk_key_encoding.encode_chunk_key` + `store.exists`.
- The per-region write entry point lives on each subclass: `process()` on
  `MaterializedRegionJob`, `process_virtual()` on `VirtualRegionJob` — the
  `RegionJob` base carries neither, so neither subclass needs a raising stub for
  the other's method.
- `process_virtual` is driven from the `icechunk.Repository` objects (not a
  pre-opened store): a committed icechunk session is read-only, so each yielded
  batch opens a **fresh writable session** on the branch, emits, and commits. The
  driver gets `(primary_repo, replica_repos)` from `StoreFactory.icechunk_repos()`;
  the role-tagged, sort-ordered list (for `parallel_setup`/`finalize`) is
  `all_icechunk_repos(sort=…)`. `process_virtual` returns nothing — virtual refs
  live in the manifest, so no per-file result tracking is threaded back.
- Two coordination lifecycles, forked at the entry points (not deep in one
  method): **single-writer streaming** (virtual operational) forks in `update()`
  → `_run_virtual_operational_update`, committing whole files straight to `main`
  with no temp branch / `parallel_setup` / `finalize`; the **parallel temp-branch
  coordinator** (`_process_region_jobs`, shared by materialized backfill,
  materialized operational, and virtual backfill) routes its per-job middle
  through `process_virtual` (fresh sessions) for virtual, else the materialized
  per-job-store loop. **No commit guard** — the generator contract forbids empty
  yields, so the loop `assert`s `refs` and `commit_if_icechunk` stays
  unconditional (see [the write loop](#the-virtual-write-loop)).
- Partitioning: `get_jobs` partitions by `encoding["shards"]` if present, else by
  `encoding["chunks"]` (virtual arrays have no shards). Materialized behavior is
  unchanged.
- `sync_dims_to` appends the new slice of the (already-derived) template via
  `to_zarr(append_dim=…, compute=False)`, keeping data vars dask-lazy so the
  decode-only serializer is never invoked; `chunk_key` derives indices from the
  full-size template, so it never re-reads the growing store. It sizes each store
  from that store's own committed size (not the primary's) so an ahead-of-primary
  replica after a partial commit is never double-appended (see
  [Replica writes](#replica-writes)).
- The `DynamicalDataset` validator gains the virtual-region-job direction of the
  invariant: a `VirtualRegionJob` `region_job_class` requires a non-null
  `icechunk_virtual_config` (the container set is already `min_length=1`), so a
  misconfigured virtual dataset fails at construction rather than at first emit.
- `VirtualRegionJob._processing_region_ds` asserts `get_processing_region() ==
  self.region`: buffering is meaningless for virtual refs (they point at raw
  bytes) and would make adjacent backfill workers emit into each other's regions.
- Storage: `_virtual_repository_config_and_credentials` also accepts
  local-filesystem containers (no credentials) so dev/test virtual datasets can
  point at local files.
- Document the `process_virtual_refs` generator contract: each yield is one
  commit's worth of whole files; operational yields ~1 file at a time for
  per-file visibility, backfill yields batches of ~N files to amortize commit
  overhead. No timer, no `max_seconds_between_commits` knob — the driver commits
  on every yield.
- Tests: a synthetic virtual dataset (local-filesystem container, raw-float64
  "messages" file, default `BytesCodec`) drives unit tests (`chunk_key`,
  sizing) and integration tests — yield-count/grouping, two-worker disjoint
  backfill on a branch, operational single-writer expansion + idempotent second
  fire, and an emit → commit → reopen → **value** read-back round-trip (catches
  bad chunk keys *and* byte ranges in CI). A separate test pins that a
  `GribberishCodec` serializer threads through expansion without being invoked.
  Edge-case coverage: `sync_dims_to` sizing each store from its own committed
  size (a second independent repo stands in for an ahead replica, since the test
  harness maps every store of one dataset to one path), the
  virtual-region-job-needs-config validator, and the buffered-processing-region
  assertion.

### PR #4 — First concrete virtual dataset (end to end)

- Pick from the [candidates](#candidate-first-datasets).
- Refactor its `TemplateConfig` to the `make_encoding` factory; add
  `<dataset>SpatialTemplateConfig` (virtual encoding rules).
- Add `<dataset>SpatialRegionJob` (`process_virtual_refs`, dataset-specific
  `filter_already_present` if needed) and `<dataset>SpatialDataset`
  (`IcechunkVirtualConfig`, `operational_kubernetes_resources`).
- **Resource sizing differs by flow** (see below). Ship only **minimal**
  validators — reuse the existing 2-random-point NaN validators as-is (bounded;
  just slower per point because gribberish decodes a whole message).
- Integration tests: backfill a small slice + read-back; two consecutive
  operational fires (first expands, second sees no new work).

### PR #5 — Second concrete virtual dataset

Different provider (NOAA ↔ ECMWF) to prove the abstractions generalize; refine
`VirtualRegionJob` defaults from what it needs.

### PR #6 — Validation phase

Two sub-phases, sequenced after the first datasets are operationally stable:

1. **Operational, spatial-chunk-optimized validators** — a manifest-presence
   check (shares [filtering](#filtering-already-present-coordinates)'s
   `chunk_key` + ref-existence primitive) plus a bounded sample-decode, designed
   for the one-message-per-chunk access pattern rather than the materialized
   shard layout.
2. **Offline tooling** — update `docs/validation.md` and its plotting/scan
   tooling to be feasible on a virtual store (millions of refs, decode-on-read),
   rather than assuming materialized read economics.

### Later (separate PR per dataset)

- Expand to all curated source variables per dataset, once operationally stable.
- DWD virtual datasets, once the bz2 + GribberishCodec chain is verified.

### Resource sizing (operational vs backfill)

The two flows have different profiles and the dataset's k8s resources should
reflect both:

- **Operational** — single pod, a handful of `.idx`/sec: ~1 CPU, ~2 G memory, no
  shared memory, a few MB ephemeral.
- **Backfill** — index-download-bound at high concurrency: more CPU for parse +
  fd/network headroom, and **stream-and-discard** index files (tens of KB ×
  millions) — never accumulate them.

### Candidate first datasets

| Dataset | Pros | Cons |
|---|---|---|
| GFS forecast | PR #511 prototype exists; simplest structure | Lower spatial demand than HRRR/IFS-ENS? |
| GEFS forecast 35-day | Ensemble dim; high demand | More complex URL/file-type logic |
| IFS-ENS forecast | High demand; exercises ECMWF index | MARS / open-data split |
| HRRR 18-hour forecast | 24 inits/day stresses high-frequency updates | Projected grid (y/x) |
| MRMS 2-minute analysis | 2-min cadence is the toughest test of the ≤5s budget; analysis structure is simplest (one time axis); existing `NoaaMrmsRegionJob` to subclass | High commit cadence (~720/day) stresses the active manifest split |

Decision deferred to PR #4 — pick by demand and convenience at the time.

## Spike results

Run **before** PR #3 (derisked up front rather than during implementation),
against local-filesystem icechunk repos. Environment: **icechunk 2.0.5, zarr
3.1.3, xarray 2026.1.0, numpy 2.4.6, gribberish 0.30.3, Python 3.14.** Every item
landed; none forced a design change except #3 (empty commits raise).

1. **Bytes per ref → ~25 bytes/ref.** Realistic distinct GRIB URLs (~79 chars)
   with random offsets/lengths and icechunk's default location compression:
   **25.4 B/ref** (50 k refs → 1.27 MB manifest). Sequential offsets: 18.6;
   single shared URL: 9.2. At 10⁹ refs → ~25 GB of manifest dataset-wide. Fixes
   the [Scale](#scale) and split-size numbers.
2. **Manifest split semantics → confirmed, as designed.** Per-array along a named
   dim via `ManifestSplittingConfig.from_dict({AnyArray(): {DimensionName("init_time"): N}})`.
   A commit rewrites **only** the split(s) whose index range changed: a new init
   wrote one ~440 B split and carried all 5 historical splits over unchanged by
   id; touching one existing split rewrote ~2.2 KB and left the other 5 untouched.
   Commit cost = active-split size (~11 B × split ref count). Policy persists in
   repo config. (See [Manifest splitting](#manifest-splitting).)
3. **Empty commit → RAISES (design change).** `session.commit` on an unchanged
   session raises `IcechunkError` ("no changes made to the session"), *not* a
   no-op. The implementation avoids this not by guarding the commit but by the
   generator contract: every yield is a non-empty batch and re-emitting
   byte-identical refs *does* dirty the session, so the commit is never empty
   (idempotent replay still commits a harmless, content-addressed snapshot). The
   loop `assert`s `refs`; the empty-poll / all-present cases yield no batch
   and never reach the commit. (See [the write loop](#the-virtual-write-loop).)
4. **Chunk-key encoding API → `array.metadata.chunk_key_encoding`.**
   `.encode_chunk_key((i,j,k,l)) → "c/i/j/k/l"`; full key `f"{array.path}/{…}"`;
   `await store.exists(key)` True/False against the manifest, no decode. Emit uses
   chunk *indices* directly (`VirtualChunkSpec(index=…)` / `set_virtual_refs_arr`),
   so only the filter touches the string key. (See [Chunk keys](#chunk-keys).)
5. **float64 round-trip → confirmed, native (no cast).** `parse_grib_array`
   returns float64 natively, so declaring float64 keeps the values verbatim — the
   codec's `astype` is skipped (declared == native); only a non-float64
   declaration like Appendix A's float32 would downcast. Full path (virtual ref →
   icechunk 2.0.5 → xarray `.values`) returned float64 byte-equal to a direct
   decode of an HRRR TMP message (260.4–317.1 K).
6. **`serializer` through `to_zarr` → confirmed.** xarray writes
   `{"name":"gribberish","configuration":{"var":…}}` straight into `zarr.json`
   codecs and reopens it intact — so each array's grib element (`var`) is
   persisted honestly in its metadata. Caveat: keep data vars **lazy** (dask)
   under `compute=False` so the read-only codec isn't invoked — which
   `write_metadata` already does via `make_empty_variable`. (See
   [Per-variable serializer](#per-variable-serializer).)
7. **`set_virtual_refs` (plural) → confirmed, three forms.**
   `set_virtual_refs(path, list[VirtualChunkSpec], validate_containers)` and the
   columnar `set_virtual_refs_arr(path, chunk_grid_shape, locations, offsets_u64,
   lengths_u64, *, validate_containers, arr_offset, checksum)` (zero-copy;
   empty-string locations silently skipped; `arr_offset` writes a sub-block for
   appends). Both return `None` on success or a list of validation-failed indices.
8. **`IcechunkVirtualConfig` → hold the real icechunk objects in-code; nothing
   serializes it.** The icechunk types don't round-trip declaratively (opaque
   `.store` on `VirtualChunkContainer`; object-keyed `ManifestSplittingConfig.to_dict()`)
   — but that's moot, because nothing needs to: k8s workers rebuild the config
   from the in-code registry by `dataset_id`, and `StoreFactory` uses the config
   directly. So hold the real objects via pydantic `InstanceOf` (verified: works
   under `FrozenBaseModel` with no model-wide `arbitrary_types_allowed`, scoped to
   the field, isinstance-validated) — plain fields would be a lossy subset.
   Separately, `save_config` persists containers + split into the repo (recovered
   by `Repository.open` / `fetch_config`), so the reader/STAC exposure can be
   minimal; credentials are never persisted (a read without
   `authorize_virtual_chunk_access` raises). (See [Storage configuration](#storage-configuration).)

Spike scripts live under `/tmp/spike/` for the run that produced these numbers;
they are throwaway (local-FS repos, synthetic refs, gribberish via `uv run --with`)
and are not checked in — PR #3's integration tests are the durable versions.

## Open questions

1. **DWD bz2 + GribberishCodec chain** — verify end-to-end before DWD datasets.
2. **`-spatial` suffix** — tentative; revisit after first-user feedback.
3. **Variable expansion** — extending to all source variables needs per-variable
   internal attrs (`grib_element`, etc.). Manual curation early; auto-discovery
   from the GRIB index is later tooling.
4. **Polling and backfill batch-size defaults** — per-dataset `poll_interval`
   and the backfill generator's files-per-yield size; defaults emerge from PR #4.
5. **Reforecast / historical archives** — some datasets (e.g. GEFS v12
   reforecast) use different URL schemes for historical data. Virtual backfills
   must handle them; operational updates don't. Scope per dataset.

---

## Appendix A: Prototype (PR #511)

PR #511 (closed, prototype only) demonstrated the full set-virtual-ref loop
end-to-end against real NOAA GFS S3 GRIB files: backfill 3 inits × 7 leads, add
a partial new init via `resize()` + refs, then fill the remaining leads. NaN
fill covered unfilled chunks automatically. It used icechunk 1.1.15; the repo is
now on `icechunk~=2.0`, and the [spike results](#spike-results) re-verified the
read/write path (including the float64 read-back) on icechunk 2.0.5.

**Repository creation with a virtual chunk container:**

```python
storage = icechunk.local_filesystem_storage(str(output_dir))  # or s3_storage(...)
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer("s3://noaa-gfs-bdp-pds/", icechunk.s3_store(region="us-east-1"))
)
repo = icechunk.Repository.create(
    storage, config=config,
    authorize_virtual_chunk_access=icechunk.containers_credentials(
        {"s3://noaa-gfs-bdp-pds/": icechunk.s3_anonymous_credentials()}
    ),
)
```

**Virtual data variable metadata** (chunk shape = native GRIB message; one ref
per message):

```python
arr = root.create_array(
    var.name,
    shape=(len(init_times), len(lead_times), N_LAT, N_LON),
    chunks=(1, 1, N_LAT, N_LON),
    dtype="float64",                                   # see Encoding rules
    fill_value=np.nan,
    serializer=GribberishCodec(var=var.internal_attrs.grib_element),
    compressors=(), filters=(),
    dimension_names=("init_time", "lead_time", "latitude", "longitude"),
)
```

**Batch ref setting + reader** (the production loop uses `set_virtual_refs`):

```python
# write
refs = [(f"{var.name}/c/{i}/{j}/0/0", url, offset, length) for var, (offset, length) in ...]
store.set_virtual_refs(refs)
session.commit("...")

# read
repo = icechunk.Repository.open(storage, config=config,
    authorize_virtual_chunk_access=icechunk.containers_credentials(
        {"s3://noaa-gfs-bdp-pds/": icechunk.s3_anonymous_credentials()}))
ds = xr.open_zarr(repo.readonly_session(branch="main").store, consolidated=False)
# .values on a chunk triggers S3 fetch + gribberish decode
```

**Empirical (PR #511):** setting refs is fast; the bottleneck is index download,
not ref-setting (the plural batch API widens that gap further). Each read fetches
one GRIB message (~1 MB); decode adds small CPU, remote latency dominates.
`resize()` + refs is a valid incremental-growth pattern; missing chunks return
the fill value.

## Appendix B: Earlier prototype (PR #510)

PR #510 (closed) explored a VirtualiZarr-based path (`ManifestArray` /
`ChunkManifest`, `vds.virtualize.to_icechunk`) and a NetCDF4/HDFParser path. We
adopted PR #511's simpler approach instead: `store.set_virtual_refs` directly,
leaning on our existing index parsers, skipping VirtualiZarr entirely.

Key details carried forward:

- **`GribberishCodec` is read/decode-only**, a zarr v3 `ArrayBytesCodec`. It
  decodes to **float64**. `var=` only labels the codec at array-creation; decode
  reads whatever bytes the ref points at (see
  [the factory rationale](#encoding-factory)).
- **Mixed storage in one store:** coordinate arrays are real (compressed) zarr
  data; data variables are virtual GRIB refs. Both live in the same icechunk
  store. This is the model we adopt.

## Appendix C: Existing infrastructure reused

| Concern | Existing module / class | Notes |
|---|---|---|
| CLI commands | `DynamicalDataset.get_cli()` | No new commands; variant differences live in `process()` |
| Region × variable-group fan-out | `RegionJob.get_jobs()` | Partition unit: shards-else-chunks (see [Partitioning](#partitioning)) |
| Worker round-robin | `iterating.get_worker_jobs()` | Backfill only (operational is single-writer) |
| Candidate source files | `RegionJob.generate_source_file_coords()` | Shared with materialized |
| Icechunk store creation | `storage` (StoreFactory, repo open/create) | Extended with `IcechunkVirtualConfig` |
| Temp-branch backfill flow | `parallel_coordination.parallel_setup`, `finalize` | Used unchanged for virtual backfills |
| Coordination files | `StoreFactory.*_coordination_files` | Backfill only; virtual operational uses none |
| Icechunk commit w/ rebase | `storage.commit_if_icechunk` | Replica-then-primary ordering carries over |
| GRIB index parsing | `noaa/noaa_grib_index.py`, `ecmwf/ecmwf_grib_index.py` | ECMWF fn renamed in PR #2 |
| Per-dataset internal attrs | `noaa/models.py::NoaaInternalAttrs`, etc. | Already carry `grib_element`, `grib_index_level` |
| Kubernetes resources | `common/kubernetes.py::ReformatCronJob`, `ValidationCronJob`, `Job` | Reused; no new types |
