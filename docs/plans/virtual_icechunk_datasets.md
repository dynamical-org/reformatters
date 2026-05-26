# Virtual Icechunk Datasets

> Design plan. Some details are deliberately deferred to PR #3 (first concrete
> dataset) where they will be settled by experience rather than speculation;
> those are marked **TBD-by-impl**.

## Goals

Create virtual Icechunk datasets that complement our existing materialized
(rechunked) time-series datasets. For datasets where source data lives in a
publicly accessible GRIB archive (NOAA NODD, ECMWF and DWD archives on
Source Coop), a virtual dataset stores only Icechunk metadata pointing at
byte ranges within the original GRIB files.

**Why:**

- **Spatial/map-optimized chunking, complementary to time-series**: chunks
  follow the native GRIB message shape (1 time step, full spatial grid),
  ideal for spatial queries and map rendering. Users choose the
  materialized `-timeseries` dataset for time-series extraction and the
  `-spatial` dataset for map/spatial queries over the same underlying data.
- **All source variables**: ~zero storage cost per variable means we can
  include every variable in the source archive, not just a curated subset.
- **Very low latency updates**: target ≤ 5s end-to-end from a new
  source file appearing to its virtual refs being visible. We hit
  this by scheduling the pod to be up and polling *before* new files
  start dropping (init publication times are predictable), polling
  with `sleep(1)` between checks, and committing each newly-observed
  file immediately. Writing virtual references is near-instant since
  we're only recording byte offsets, not transferring data.

The PR #511 prototype proved the approach: 15.5 KB on disk for what would
be 350 MB materialized, with real reads via S3 + GribberishCodec working
end-to-end. See [Appendix A](#appendix-a-prototype-reference-pr-511).

## The set_virtual_ref loop

This is the entire core operation. Everything else in this plan is
plumbing around it.

```python
for source_file_coord in source_file_coords:
    # 1. Download the lightweight GRIB index file (~10KB)
    index_path = download_index(source_file_coord)

    # 2. Parse byte ranges for each variable (existing parsers reused)
    starts, ends = parse_index(index_path, data_vars)

    # 3. Record a virtual reference per (variable, chunk) pair
    grib_url = source_file_coord.get_url()  # e.g. s3://noaa-gfs-bdp-pds/...
    for var, start, end in zip(data_vars, starts, ends, strict=True):
        chunk_key = make_chunk_key(var, source_file_coord, template_ds)
        store.set_virtual_ref(
            chunk_key, location=grib_url, offset=start, length=end - start
        )

# 4. Commit — readers immediately see the new data
session.commit("...", rebase_with=icechunk.ConflictDetector())
```

At read time, `GribberishCodec` (a zarr v3 `ArrayBytesCodec` serializer)
decodes the raw GRIB messages. No copy of the data lives in our store —
just the byte offsets.

**Existing code we reuse directly:**

| Need | Existing code |
|---|---|
| GRIB URLs | `SourceFileCoord.get_url()` per dataset |
| NOAA `.idx` parsing → `(starts, ends)` | `noaa/noaa_grib_index.py::grib_message_byte_ranges_from_index` |
| ECMWF JSON Lines `.index` parsing | `ecmwf/ecmwf_grib_index.py::get_message_byte_ranges_from_index` (renamed to `grib_message_byte_ranges_from_index` in PR #1) |
| Per-variable GRIB element / level metadata | `<dataset>InternalAttrs.grib_element`, `grib_index_level`, etc. |
| Index download helper | `common/download.py::http_download_to_disk` |
| Icechunk commit with auto-rebase | `common/storage.py::commit_if_icechunk` |

## Class architecture

### Single `DynamicalDataset`; `MaterializedRegionJob` and `VirtualRegionJob` as siblings under `RegionJob`

The current `DynamicalDataset` base already abstracts the right things
for both variants — we keep one `DynamicalDataset` hierarchy. CLI,
store factory, kubernetes resources, validation, Sentry — none fork.

`RegionJob`, on the other hand, has a substantial set of variant-specific
concerns (materialized has the download/read/shared-memory/write/upload
pipeline and its hooks; virtual has the watcher loop, set-virtual-ref
helpers, and source container declarations). It also has a meaningful
set of shared concerns (get_jobs partitioning, source_groups, the
operational_update_jobs contract). The clearest expression of this is a
three-class hierarchy with `RegionJob` as the shared base and
`MaterializedRegionJob` / `VirtualRegionJob` as siblings. The
materialized/virtual pairing in the names makes the distinction
explicit at every reference site.

```
DynamicalDataset                     # unchanged single base
├── NoaaGfsForecastDataset             (materialized; existing)
├── NoaaGfsForecastSpatialDataset      (virtual; new)
├── NoaaGefsForecast35DayDataset       (materialized; existing)
├── NoaaGefsForecast35DaySpatialDataset (virtual; new)
└── …

RegionJob                            # shared base: fields, get_jobs,
                                     #   source_groups, update_template_with_results,
                                     #   abstract process(), abstract operational_update_jobs(),
                                     #   abstract generate_source_file_coords()
├── MaterializedRegionJob            # the existing process() pipeline,
│                                    #   download_file/read_data/transform hooks,
│                                    #   download/read parallelism tunables
│   ├── NoaaGfsCommonRegionJob       # provider-specific shared bits
│   │   └── NoaaGfsForecastRegionJob (existing, base swap RegionJob → MaterializedRegionJob)
│   ├── NoaaHrrrRegionJob            (existing, base swap)
│   │   ├── NoaaHrrrForecast48HourRegionJob
│   │   └── NoaaHrrrAnalysisRegionJob
│   └── …
└── VirtualRegionJob                 # the watcher loop, set_virtual_ref helpers,
                                     #   process_virtual_refs (abstract),
                                     #   _filter_already_ingested (default impl),
                                     #   _expand_dimensions, _update_ingested_forecast_length,
                                     #   source_virtual_chunk_containers ClassVar
    ├── NoaaGfsForecastSpatialRegionJob       (overrides process_virtual_refs)
    ├── NoaaGefsForecast35DaySpatialRegionJob (overrides process_virtual_refs)
    └── …
```

Moving the materialized `process()` body and helpers down one level
(from today's `RegionJob` to the new `MaterializedRegionJob`) is a
mechanical refactor — base class swap in each existing dataset's
region job file, no semantic change.

#### What lives where

**`DynamicalDataset` (unchanged surface):**

- Fields: `template_config`, `region_job_class`, `store_factory`,
  `dataset_id`, `primary_storage_config`, `replica_storage_configs`.
- CLI: `update_template`, `update`, `backfill_kubernetes`,
  `backfill_local`, `backfill`, `validate_dataset`, `dataset_urls`.
- `_process_region_jobs` driver — extended by checking
  `issubclass(region_job_class, VirtualRegionJob)` to take the
  no-temp-branch path for virtual operational updates.
- Sentry monitoring wrapper.

**`RegionJob` (shared base):**

- Fields: `tmp_store`, `template_ds`, `data_vars`, `append_dim`,
  `region`, `reformat_job_name`.
- `get_jobs()` — region × variable-group fan-out with filters and
  worker round-robin. Shared verbatim today and continues to apply
  to both variants.
- `source_groups()`, `get_processing_region()`,
  `update_template_with_results()`.
- Abstract: `operational_update_jobs()`, `generate_source_file_coords()`,
  `process()`.

**`MaterializedRegionJob` (subclass of `RegionJob`):**

- Hooks: `download_file()`, `read_data()`, `apply_data_transformations()`.
- Tunables: `download_parallelism`, `read_parallelism`,
  `max_vars_per_download_group`.
- `process()` implementation: the current shared-memory pipeline
  plus helpers (`_download_processing_group`, `_read_into_data_array`,
  `_write_shards`, `_cleanup_local_files`).

All existing dataset region jobs change their base class from
`RegionJob` to `MaterializedRegionJob`. Tests pass unchanged.

**`VirtualRegionJob` (subclass of `RegionJob`):**

- Extra fields: `is_backfill: bool`,
  `max_files_per_commit: int`, `max_seconds_between_commits: float`.
- `process()` implementation: opens icechunk session, runs filter
  once, drives generator, lazily expands as needed, sets refs,
  commits with per-batch retry (see [The update process](#the-update-process)).
- `process_virtual_refs()` (abstract — the generator subclasses
  implement).
- `_filter_already_ingested()` default implementation (see
  [Filtering](#filtering-already-ingested-coordinates)).
- `_expand_dimensions(new_append_values)` helper: resizes coord and
  data var arrays, writes new coord values, **also recomputes
  derived coords** (valid_time, ingested_forecast_length placeholder,
  expected_forecast_length) by calling the template config's
  `derive_coordinates` with the expanded ds.
- `_update_ingested_forecast_length(init_time, new_max_lead)`
  helper: bumps the per-init progress coord during a batch commit.
- `download_index()` default implementation calling
  `http_download_to_disk` against a coord-supplied URL.
- `_chunk_key(ref, store)` helper: computes
  `f"{var}/c/{idx0}/{idx1}/..."` from the ref's coord values and the
  current store's coord arrays.
- `source_virtual_chunk_containers` ClassVar for store-factory
  wiring (see [Storage configuration](#storage-configuration)).
- `operational_update_jobs()` calls inherited `get_jobs(...)` with
  `is_backfill=False`; backfill callers go through `get_jobs(...)`
  with `is_backfill=True` (the default).
- `update_template_with_results()`: no-op for virtual. Lazy
  expansion means the committed dim length already matches reality;
  no trim needed.

## The update process

The whole point of this is ≤ 5s updates, so the lifecycle deserves a
clean walkthrough. The same `ReformatCronJob` → indexed Job → N
worker pods scheduling primitive as materialized, but with two key
twists tuned for low latency:

1. The CronJob is **scheduled to fire just before a publication
   window starts** (e.g. 5 minutes before GFS init T begins
   publishing). Each pod is long-lived within its fire — runs for
   the duration of the publication window (hours, not seconds),
   polling with `sleep(1)` for new files. This means the pod is
   already up and watching when the source agency drops a file.
2. Commits go straight to `main` (no temp branch) and dim expansion
   happens lazily inside each region job's `process()`, atomic with
   the corresponding refs. Same model as before.

Net effect on latency:

```
latency ≈ poll_interval + index_download + commit + (occasional rebase retry)
        ≈ 1s          + 100-500ms     + 0.5-1s + (1-2s on conflict)
        = 2-3s typical, ≤5s worst case
```

`concurrencyPolicy: Forbid` on the CronJob prevents a new fire from
launching while the previous fire's pod is still polling.

### No NaN-padded future

Lazy expansion is the deliberate design choice that protects consumers
from seeing empty future time steps:

- **Analysis datasets**: the `time` array grows exactly to match the
  data actually ingested. We never pre-expand to "now" or any other
  forward boundary. Consumers querying past the end see "out of range,"
  not NaN.
- **Forecast datasets**: each new `init_time` slot is created in the
  same commit that lands its first virtual refs. Consumers querying a
  newly-published init see the lead times that have been published so
  far (and NaN for ones that haven't yet — same as today's materialized
  forecasts mid-update).

The materialized flow protects readers from "expanded but empty" via a
temp branch. We can't do that for virtual operational updates without
losing the latency advantage (the whole point is per-file visibility
in seconds). Lazy expansion is the alternative: data and structure
always commit together.

### Step-by-step

1. **CronJob fires** on a schedule timed to publication windows
   (per dataset; e.g. for GFS, fire ~5 min before each 6h init's
   publication starts). `concurrencyPolicy: Forbid` prevents
   overlapping fires.
2. **Indexed Job spawns N workers**. Same env vars as today
   (`JOB_NAME`, `WORKER_INDEX`, `WORKERS_TOTAL`).
3. **Each worker runs `update()`** on the `DynamicalDataset`, which
   calls `operational_update_jobs()` and then `_process_region_jobs()`.
4. **The driver detects this is a virtual operational update** —
   `region_job_class` is a `VirtualRegionJob` subclass and
   `update_template_with_results=True`. It takes the virtual
   operational path: no temp branch, no shared setup step. Each pod
   handles its round-robin slice of region jobs directly.
5. **Each worker processes its assigned region jobs in parallel**. For
   each region job, `VirtualRegionJob.process()`:
   1. Calls `_filter_already_ingested(source_coords)` once at the
      top — narrows the job to files this pod still needs to ingest.
      (Other pods never do this pod's work, so we don't need to
      re-check mid-job.)
   2. Drives the `process_virtual_refs()` generator. The generator
      polls source-file availability with `sleep(1)` between
      checks (HEAD requests on the source URLs are fast and cheap)
      and yields each newly-observed file as a single-file batch
      immediately. Operational region jobs default to
      `max_files_per_commit=1` so each file commits on its own for
      minimum latency.
   3. Each batch commits via the retry loop: open a fresh session
      on `main`, expand the dim if still needed, compute chunk
      keys against current state, `store.set_virtual_ref(...)` for
      each ref, commit with
      `rebase_with=icechunk.ConflictDetector()`. On conflict, throw
      away the session and try again.
   4. The generator exits when all expected files for this job have
      been ingested (e.g.
      `ingested_forecast_length[init] >= expected_forecast_length[init]`)
      or the pod's `pod_active_deadline` approaches.
6. **No finalize step**. There's no temp branch to merge. The driver
   may still wait for all workers (via the existing coordination-file
   mechanism) before exiting, but there's no `reset_branch`.
7. **Pod exits** when its work is done. The next scheduled cron fire
   (for the next publication window) starts a fresh pod that picks
   up any unfinished work via `_filter_already_ingested`.

This flow is intentionally close to the existing materialized flow.
What changes is small: skip the temp branch, expand `main` lazily
inside each region job's process() as data arrives, let each worker
manage its own session and commit cadence.

### Why lazy expansion works with parallel pods

Region jobs partition the append dim by construction — different pods
get different `init_time` (or `time`) ranges. So **two pods never write
the same chunks**, only ever the metadata files involved in dim
expansion (`init_time/zarr.json`, the chunk holding `init_time` values,
each data var's `zarr.json`).

`_filter_already_ingested` runs once at the start of `process()` to
decide what files this job will ingest. We don't re-run it per batch —
no other pod is going to process our work.

The only conflict that can happen is on dim expansion. The scenario:

- Pod A (init T-6h, new): opens session at shape N, expands to N+1
  with `T-6h` at index N, sets refs for T-6h's chunks, tries to commit.
- Pod B (init T, new): in parallel, opens session at shape N, expands
  to N+1 (or further) with both `T-6h` and `T` in sorted order
  (computed from the template — Pod B knows T-6h exists in the
  expected coords even though Pod B isn't ingesting it), sets refs
  for T's chunks, commits first.
- Pod A's commit now conflicts on `init_time/c/0` and the resized
  `<var>/zarr.json` files.

Pod A's response is just: throw away the session, open a fresh one,
re-check whether expansion is still needed (it isn't — Pod B's commit
already added T-6h), re-set the same virtual refs against the current
chunk indices, commit.

```python
# Inside VirtualRegionJob.process(), per commit cycle:
for attempt in range(MAX_RETRIES):
    session = repo.writable_session("main")
    store = session.store

    new_coords = self._compute_dim_expansion_needed(store, refs)
    if new_coords:
        self._expand_dimensions(store, new_coords)
    for ref in refs:
        # Recompute the chunk key from the current store state in case
        # another pod's prior expansion shifted our index.
        key = self._chunk_key(ref, store)
        store.set_virtual_ref(key, ref.url, offset=ref.offset, length=ref.length)

    try:
        session.commit("...", rebase_with=icechunk.ConflictDetector())
        break
    except icechunk.ConflictError:
        time.sleep(0.1 * (2 ** attempt))
```

A few important properties of this model:

- **Expansion is idempotent across pods**. Each pod expands the dim to
  cover everything in the template that it depends on — including
  values it isn't ingesting but that must exist to keep sorted order.
  After any pod commits its expansion, the dim is in the same target
  state. Losers' retries see "expansion already done" and skip it.
- **Filter is once-per-job**, not per-batch. Other pods aren't doing
  our work, so we don't need to re-check what's done mid-job.
- **Retries are cheap**. Byte ranges (from parsed index files) are
  already in hand. The retry recomputes chunk-key indices against the
  current dim shape and re-issues `set_virtual_ref` calls; that's
  microseconds per ref.
- **Conflicts only happen during multi-new-init scenarios**. Single
  new init per fire = single expanding pod = no conflict. Multiple
  new inits = conflicts that converge via the retry loop above. No
  special-case code path.

### Concurrency and ConflictDetector (summary)

`ConflictDetector` rejects a commit when two sessions wrote to the
same path. The per-batch retry loop is the response — it always
converges because:

1. Filter ran once at job start; this pod's work is fixed.
2. Other pods never write this pod's chunks.
3. The only path collisions are on dim-expansion metadata files,
   which are idempotent across pods (everyone targets the same final
   dim state derived from the template).

The existing `storage.commit_if_icechunk` re-runs `session.commit()`
without rebuilding the session's writes, which doesn't help us
here — for virtual operational updates we need to rebuild the
session (fresh write set against the new base) on each attempt.
That logic lives inside `VirtualRegionJob.process()`.

> **TBD-by-impl**: confirm icechunk 2.x's `ConflictDetector` accepts
> rebases when one session resized an array (wrote `<var>/zarr.json`)
> and another session wrote chunks under that array. The retry loop
> above handles either answer; we just want to know how often
> conflicts fire operationally.

### Crash recovery

`_filter_already_ingested` makes crash recovery trivial in steady
state — a re-run picks up where the last commit left off. Specific
crash scenarios:

- **Worker crashes mid-batch (before commit)**: pending refs are
  lost; the next cron fire re-discovers them via filter and re-sets
  them. Idempotent.
- **Worker crashes after some commits but before others**: committed
  refs and expansions are durable; uncommitted ones are not.
  `_filter_already_ingested` skips what's done, processes the rest.
- **Worker crashes mid-expansion (after `_expand_dimensions` but
  before commit)**: expansion is part of the session, not yet
  visible. Next fire sees the unchanged dataset and re-expands.

## The backfill process

Backfills use the same `VirtualRegionJob.process()` and
`process_virtual_refs()` code paths — with two differences:

1. **Temp branch flow** (same as materialized backfills): worker 0
   creates `_job_<job_name>`, expands metadata fully on the branch,
   commits. Workers write refs to the branch. Last worker resets
   `main` to the branch. Readers see "empty" or "full," never
   partial state during a backfill.
2. **Looser commit thresholds**: backfill region jobs are constructed
   with higher `max_files_per_commit` (e.g. 50) and
   `max_seconds_between_commits` (e.g. 60s). Operational region jobs
   default to `max_files_per_commit=1` (commit each file immediately
   for ≤5s visibility). Backfills don't care about per-file latency
   so larger batches reduce commit overhead for the much larger
   volume.

The plumbing for these differences:

- `VirtualRegionJob.get_jobs(...)` accepts an `is_backfill: bool = True`
  parameter (backfill is the more common caller).
  `VirtualRegionJob.operational_update_jobs(...)` calls
  `cls.get_jobs(..., is_backfill=False)`. Each constructed region
  job carries the flag and the appropriate threshold values as
  model fields.
- `_process_region_jobs` checks the region jobs' `is_backfill` flag
  (or, equivalently, `update_template_with_results=False`) to take
  the temp-branch path for backfill and the direct-to-main path
  for operational updates.

Parallelism comes from Kubernetes indexed jobs and
`get_worker_jobs(...)`, identical to materialized backfills.

## Filtering already-ingested coordinates

Filtering is what makes operational updates *fast* in the steady state.
It's not just a crash-recovery fallback.

Region jobs cover whole shards along the append dim. For analysis
datasets and for forecasts issued over multiple days (e.g. GEFS 35-day,
where a shard's worth of init times spans many days), the region job
responsible for the current shard is usually already mostly populated.
Filtering identifies the small subset of source files that are
genuinely new, so the job's effective work is "ingest what's missing"
— typically a handful of files per fire, not a whole shard.

`_filter_already_ingested(source_coords) -> source_coords_remaining`
is a method on `VirtualRegionJob` that subclasses can override. The
default implementation reads the icechunk store and detects which
append-dim positions are already populated. Two interchangeable
strategies:

1. **Coordinate-array introspection** (default, no extra metadata
   needed): open the icechunk store readonly, read the relevant
   coordinate(s) and any "progress" coord (e.g.
   `ingested_forecast_length`), compute which (append_dim_value, …)
   tuples are present, filter out matching `source_coords`. For
   forecasts, "present" means
   `ingested_forecast_length[init_time] >= source_coord.lead_time`.
   For analyses, "present" means `time` contains
   `source_coord.init_time + source_coord.lead_time`.
2. **Manifest probe** (fallback for datasets without a progress
   coord): for each candidate chunk key, ask the icechunk session
   whether a virtual ref exists. Slower (one probe per candidate)
   but completely generic.

The default goes in `VirtualRegionJob` and uses strategy 1 if the
template has `ingested_forecast_length`, otherwise strategy 2. Per
dataset, subclasses can override.

Three independent things this gives us:

- **Fast steady-state updates** — only a few files are actually new.
- **Crash recovery** — restart sees post-crash state, resumes.
- **Idempotency across overlapping cron fires** — even with
  `concurrencyPolicy: Forbid`, if a previous fire partially
  completed, the next fire skips its committed work.

## Derived coordinates and progress tracking

A virtual region job's commits modify three things in the icechunk
store, not just the data variable chunks:

1. **Dim coord values** (`init_time` or `time`) when `_expand_dimensions`
   adds new positions. Atomic with the same commit as the
   corresponding refs.
2. **Derived coords** that depend on the append dim. For HRRR-style
   forecasts these are `valid_time = init_time + lead_time`,
   `expected_forecast_length` (per init hour), and the
   `ingested_forecast_length` placeholder (all-NaT until filled).
   `_expand_dimensions` calls the template config's
   `derive_coordinates(ds)` with the expanded ds and writes the
   computed values for the new positions only (leaving existing
   positions untouched).
3. **Progress signal** (`ingested_forecast_length` for forecasts;
   nothing extra for analyses). Each batch commit calls
   `_update_ingested_forecast_length(init_time_idx, new_max_lead)`
   which bumps the per-init lead-time progress coord. This is what
   `_filter_already_ingested` reads on the next fire.

For analyses, lazy expansion adds new `time` positions; no
per-position derived coord needs updating beyond what
`derive_coordinates` returns. The progress signal is implicit in
the current `time` array length — `_filter_already_ingested`
checks `time` directly.

Static derived coords (`latitude`, `longitude`, `spatial_ref`) are
written once at dataset creation and don't change with dim
expansion.

## Per-variable serializer

The single non-trivial template-layer change for virtual datasets:
each virtual data variable has its own `GribberishCodec(var=grib_element)`
serializer. Materialized datasets share one encoding across vars;
virtual datasets cannot.

A mechanical prerequisite (lands in PR #1): `Encoding` gets one new
optional field, since `GribberishCodec` is a zarr v3 `ArrayBytesCodec`
(the serializer slot in `filters → serializer → compressors`):

```python
class Encoding(pydantic.BaseModel):
    ...
    # None means zarr's default BytesCodec.
    serializer: dict[str, Any] | None = None
```

`template_utils.assign_var_metadata` (wherever encoding flows into
xarray's `encoding` dict) passes `serializer` through to zarr.
Materialized datasets leave it `None` and behavior is unchanged.

### Recommended pattern: encoding factory

Extend HRRR's existing `get_data_vars(encoding)` to take a per-var
encoding factory:

```python
class NoaaGefsCommonTemplateConfig(TemplateConfig[GefsDataVar]):
    def get_data_vars(
        self, make_encoding: Callable[[GefsDataVar], Encoding]
    ) -> Sequence[GefsDataVar]: ...

# Materialized variant: same encoding for every var.
class NoaaGefsForecast35DayTemplateConfig(NoaaGefsCommonTemplateConfig):
    @computed_field
    @cached_property
    def data_vars(self) -> Sequence[GefsDataVar]:
        enc = Encoding(dtype="float32", chunks=(1, 49, 721, 1440), ...)
        return self.get_data_vars(make_encoding=lambda _var: enc)

# Virtual variant: per-var serializer.
class NoaaGefsForecast35DaySpatialTemplateConfig(NoaaGefsCommonTemplateConfig):
    @computed_field
    @cached_property
    def data_vars(self) -> Sequence[GefsDataVar]:
        def make_encoding(var: GefsDataVar) -> Encoding:
            return Encoding(
                dtype="float32",
                chunks=(1, 1, 721, 1440),
                # shards drives region-job partitioning, not storage layout.
                # See "Encoding rules for virtual datasets" below.
                shards=(1, lead_time_count, 721, 1440),
                serializer={"name": "gribberish", "configuration": {"var": var.internal_attrs.grib_element}},
                compressors=None, filters=None, ...,
            )
        return self.get_data_vars(make_encoding=make_encoding)
```

Touches only datasets that grow a virtual variant; HRRR's existing
call sites change `lambda _var: encoding` and that's it.

### Alternatives considered

- **Common config exposes `_data_vars_metadata` (no encoding); variants
  build `DataVar`s separately.** Equivalent power, but doesn't fit
  HRRR's existing `get_data_vars(encoding)` cleanly — we'd reshape it.
- **Virtual variant inherits from materialized and `replace()`s
  encoding per var.** Smallest diff but couples the two configs and
  forces awkward `dataset_attributes` overrides for the differing
  id/version.

We do not retrofit every template config eagerly. Only datasets that
grow a virtual variant adopt the factory pattern.

## Encoding rules for virtual datasets

All virtual variants follow the same encoding rules, regardless of
provider:

- **Serializer**: `GribberishCodec(var=<grib_element>)` per variable.
  See `gribberish.zarr.codec.GribberishCodec`.
- **Chunk shape**: `(1, 1, lat, lon)` for forecast (one chunk per
  (init_time, lead_time) GRIB message) or `(1, lat, lon)` for analysis
  (one chunk per time step).
- **`shards` is set equal to the desired *region partition unit***,
  not to the chunk shape. Icechunk doesn't shard virtual refs (each
  ref is a standalone GRIB message), but `RegionJob.get_jobs()` uses
  `encoding["shards"]` to partition the append dim into region jobs.
  For a forecast with `chunks=(1, 1, lat, lon)`, set
  `shards=(1, full_lead_time, lat, lon)` so one region job covers one
  init time's worth of work — a natural unit for filtering and
  parallelism. The dimension_slices logic in `iterating.py` treats
  this as the partition size; the icechunk store ignores the field.
- **No compressors or filters**: GribberishCodec handles the full
  decode.
- **Exception (DWD, bz2-compressed GRIBs)**: chain zarr's built-in
  `Bz2Codec` as a filter before `GribberishCodec` as the serializer.
  Needs end-to-end verification (see [Open questions](#open-questions)).

## Storage configuration

Source-bucket virtual chunk containers are declared once as a
`ClassVar` on the `VirtualRegionJob` subclass (where URL construction
also lives) and picked up automatically by `DynamicalDataset.store_factory`.
`__main__.py` gets no new surface — virtual dataset registration looks
identical to materialized.

```python
class NoaaGefsForecast35DaySpatialRegionJob(VirtualRegionJob[...]):
    source_virtual_chunk_containers: ClassVar[Sequence[VirtualChunkContainerConfig]] = (
        VirtualChunkContainerConfig(prefix="s3://noaa-gefs-pds/", region="us-east-1"),
    )
```

`VirtualChunkContainerConfig` is intentionally minimal — `prefix`,
`region`, and `anonymous: bool = True`. All current targets
(NOAA/ECMWF/DWD) are publicly readable; adding credentialed sources
is a small additive change later. `StoreFactory` opens the icechunk
repo with an `icechunk.RepositoryConfig` that registers the containers
and wires `authorize_virtual_chunk_access` to
`s3_anonymous_credentials()`.

Hard constraints enforced at construction:

- Virtual variants must use `DatasetFormat.ICECHUNK` for the primary
  store and any replica stores. Mixing zarr v3 + virtual refs isn't
  supported; we assert and fail loudly.
- A virtual `DynamicalDataset` whose `region_job_class` is a
  `VirtualRegionJob` subclass with empty
  `source_virtual_chunk_containers` is a configuration error.

Not yet supported (additive when needed): credentialed source buckets,
HTTP-only source containers, runtime container switching.

> **TBD-by-impl**: confirm icechunk persists the `VirtualChunkContainer`
> list inside the repo config so readers don't need to re-register
> containers. If not, readers construct the same `RepositoryConfig`
> we use here (see Appendix A's reader code).

## Validation

Materialized datasets run validators from `common/validation.py`
(`check_for_expected_shards`, `check_analysis_current_data`,
`check_analysis_recent_nans`, etc.) via the `validate` CLI on a
schedule, with the validator suite returned by the dataset's
`validators()` method.

Virtual datasets keep the same shape — `validators()` returns a
sequence of `DataValidator`s — but the set differs:

- `check_for_expected_shards` — **N/A**. Virtual datasets set
  `shards` only for region-job partitioning (see Encoding rules);
  there are no physical shard objects to check.
- `check_analysis_current_data` / `check_analysis_recent_data` —
  **applicable as-is**. They check that the `time` array contains
  recent timestamps. Lazy expansion guarantees `time` reflects
  ingested data, so this is exactly the right check.
- `check_analysis_recent_nans` / NaN-fraction checks —
  **need adaptation**. Reading "is this chunk NaN?" requires
  fetching the GRIB bytes and decoding, which is much slower than
  reading a real materialized chunk. Two reasonable replacements:
  1. **Manifest check** (cheap): for each chunk position the
     validator wants to inspect, ask the icechunk session whether
     a virtual ref exists. If yes, assume non-NaN (the ref points
     at a real GRIB message). If no, the chunk reads as fill
     value, which IS the NaN condition.
  2. **Sample read** (medium): pick N random chunks from the
     recent window, fetch + decode via GribberishCodec, check
     min/max for sanity. Catches "ref points at garbage" bugs
     that the manifest check would miss.
- Progress coord validation (new): for forecasts, check
  `ingested_forecast_length[recent_init_times]` against
  `expected_forecast_length`. Catches "fire ran but didn't make
  progress."

PR #3 adds the manifest-check and progress-coord validators to
`common/validation.py` and uses them in the first virtual dataset's
`validators()`. The sample-read validator is a follow-up.

## Replica writes

For datasets with replica stores (both primary and replicas are
icechunk), each batch commit writes refs to every store. The existing
`storage.commit_if_icechunk(message, primary_store, replica_stores)`
already commits replicas first, then primary, with `ConflictDetector`
retry — that pattern carries over to virtual updates.

The per-batch retry loop in `VirtualRegionJob.process()` opens
sessions on **all** stores, sets refs on all of them, then commits
replicas-then-primary. On any commit failure (`ConflictError`,
network error, etc.), the whole batch retries with fresh sessions
on all stores — keeping replicas and primary aligned. If the retry
budget exhausts on one store, the pod fails; the next cron fire
re-runs the work (filtered to skip what already committed).

For virtual datasets without replicas (the expected common case
for our first virtual datasets), this collapses to one session,
one commit per batch.

## Dataset identity

### Naming

Virtual variants get a `-spatial` suffix on the dataset ID. This is
**tentative** — it captures the access-pattern optimization without
exposing "virtual" as an implementation detail. Examples:

- `noaa-gefs-forecast-35-day` (materialized, time-series optimized)
- `noaa-gefs-forecast-35-day-spatial` (virtual, spatial/map optimized)

Class names follow the same pattern:
`NoaaGefsForecast35DayDataset` → `NoaaGefsForecast35DaySpatialDataset`,
plus the matching region job and template config classes.

> Open question: confirm `-spatial` survives first-customer feedback.
> If a better name emerges (`-grib`, `-native`), rename before the
> first dataset goes public.

### Storage location

Virtual Icechunk stores live in the same S3 buckets as the
materialized datasets, with new dataset IDs and paths. The stores are
tiny (KBs–MBs of metadata) so co-location has no cost concern.

### Reader experience

All source GRIB archives targeted have anonymous read access. When
opening a virtual dataset, the reader needs the
`VirtualChunkContainer` configuration and the
`containers_credentials` mapping. We hide this behind whatever client
library we point users at (today, the `dynamical_catalog` Python
library wraps our STAC catalog and configures this at open time). For
the bare path, the reader code looks like the example in
[Appendix A](#appendix-a-prototype-reference-pr-511).

## Provider-specific considerations

**NOAA (GFS, GEFS, HRRR):**

- Index format: plain text `.idx` files, parsed by
  `noaa/noaa_grib_index.py::grib_message_byte_ranges_from_index`.
- Source buckets: `s3://noaa-gfs-bdp-pds/`, `s3://noaa-gefs-pds/`,
  `s3://noaa-hrrr-bdp-pds/`. All `us-east-1`, anonymous reads.
- Straightforward — one container per bucket.

**ECMWF (IFS-ENS, AIFS):**

- Index format: JSON Lines `.index` files, parsed by the ECMWF-side
  index parser (renamed in PR #1 for naming consistency with NOAA).
- Source: `s3://ecmwf-forecasts/` in `eu-central-1`, plus Source Coop
  archives.
- Archive transition: IFS-ENS has separate MARS archive (pre-2024-04)
  and open-data (post-2024-04) URL schemes. The virtual dataset
  likely only covers the open-data era. **TBD-by-impl**.

**DWD (ICON-EU):**

- No index files: one variable per `.bz2`-compressed GRIB file.
- Codec pipeline: chain zarr's `Bz2Codec` as a filter before
  `GribberishCodec` as the serializer.
- **Blocks DWD virtual datasets until the chained codec is verified
  end-to-end.**

## Implementation plan

Smaller and tighter than the previous draft.

### PR #1 — Prep work (small, mechanical)

- Rename `get_message_byte_ranges_from_index()` to
  `grib_message_byte_ranges_from_index()` in
  `ecmwf/ecmwf_grib_index.py` and update callers (NOAA and ECMWF
  parsers share a name).
- Add `serializer: dict[str, Any] | None = None` to
  `common/config_models.py::Encoding`. Plumb it through
  `template_utils.assign_var_metadata`.
- Add `gribberish` dependency to `pyproject.toml` using a `~=`
  version specifier (same style as the existing `icechunk~=2.0`,
  `dask~=2026.0`, `kubernetes~=33.1` pins). PR #511 used
  `gribberish>=0.29.0` against icechunk 1.1.15; smoke-test the
  current `~=0.X` release on icechunk 2.0.3 and pin to that.
- Add `VirtualChunkContainerConfig` to `common/storage.py`, plus the
  `virtual_chunk_containers` field on `StoreFactory` and the
  open-time wiring through `_get_icechunk_storage`. Assert
  icechunk-only when set.
- **Extract `MaterializedRegionJob`** from today's `RegionJob`: move
  the materialized `process()` body, the `download_file` /
  `read_data` / `apply_data_transformations` hooks, the parallelism
  tunables, and the `_download_processing_group` /
  `_read_into_data_array` / `_write_shards` / `_cleanup_local_files`
  helpers into a new `MaterializedRegionJob(RegionJob)` class. Swap
  every existing dataset's region job base from `RegionJob` to
  `MaterializedRegionJob`. Mechanical refactor; tests pass unchanged.

### PR #2 — VirtualRegionJob and driver hooks

- Add `VirtualRegionJob` as a sibling of `MaterializedRegionJob`
  under `RegionJob`, with the operational flow described in
  [The update process](#the-update-process).
- Extend `_process_region_jobs` to branch on `region_job_class`
  (virtual operational updates skip the temp branch; backfills and
  materialized updates keep the existing flow).
- Add `is_backfill: bool = True` parameter on
  `VirtualRegionJob.get_jobs`; `operational_update_jobs` passes
  `False`. Each region job carries the flag plus
  `max_files_per_commit` and `max_seconds_between_commits` model
  fields with operational vs backfill values.
- Add the default `_filter_already_ingested` (coord-introspection
  strategy with manifest-probe fallback).
- Add `_expand_dimensions` helper that resizes coord and data var
  arrays AND recomputes derived coords via the template config's
  `derive_coordinates`.
- Add `_update_ingested_forecast_length` helper called per batch
  commit.
- Add the per-batch commit retry loop (open fresh session, recompute,
  set refs, commit, retry on conflict).
- Integration tests against a local icechunk 2.x repo:
  - **Concurrent disjoint-chunk writes** — two pods writing chunks
    at different `init_time` indices, one of them also expanding
    the dim. Verifies `ConflictDetector` accepts a rebase when one
    session resized an array and another wrote chunks under it.
  - **Concurrent expansion conflict** — two pods both expanding for
    new inits simultaneously. Verifies the per-batch retry loop
    converges to a correct final state (both inits present, all
    refs landed at correct indices, derived coords consistent).
  - **Backfill-to-update transition** — backfill leaves a partial
    state; update fills the rest. Verifies the temp-branch and
    direct-to-main paths interoperate cleanly.

### PR #3 — First concrete virtual dataset (end to end)

- Pick from the candidates below.
- Refactor that dataset's `TemplateConfig` to use the encoding-factory
  pattern recommended above.
- Implement `<dataset>SpatialTemplateConfig` with virtual encoding
  rules.
- Implement `<dataset>SpatialRegionJob` with
  `process_virtual_refs()` and the dataset-specific
  `_filter_already_ingested` if needed. Declare
  `source_virtual_chunk_containers`.
- Implement `<dataset>SpatialDataset` with
  `operational_kubernetes_resources()` returning a
  `ReformatCronJob` + `ValidationCronJob`. **Resource sizing differs
  from materialized**: virtual pods need ~1 CPU, ~2G memory, no
  shared memory, minimal ephemeral storage (a few MB for downloaded
  index files). Use sane minimums and tune from there.
- Implement `validators()` using the new manifest-check and
  progress-coord validators (see [Validation](#validation)).
- Integration tests: backfill a small slice, verify read-back via
  GribberishCodec; simulate two consecutive cron fires (first does
  expansion, second sees no new work).

### PR #4 — Second concrete virtual dataset

- Pick a different provider (NOAA vs ECMWF) to validate that the
  abstractions generalize. Refine `VirtualRegionJob` defaults based
  on what the second dataset needs.

### Later (separate PR per dataset)

- Expand variable coverage to "all source variables" per dataset,
  after operational stability is established.
- DWD virtual datasets, once the bz2 + GribberishCodec chain is
  verified.

### Candidate first datasets

| Dataset | Pros | Cons |
|---|---|---|
| GFS forecast | PR #511 prototype exists; simplest structure | Lower spatial-access demand than HRRR or IFS-ENS? |
| GEFS forecast 35-day | Tests ensemble dimension; high demand | More complex URL/file-type logic |
| IFS-ENS forecast | High demand; exercises ECMWF index format | MARS / open-data archive split |
| HRRR 18-hour forecast | 24 inits/day stresses high-frequency updates | Projected grid (y/x not lat/lon) |

Decision deferred to PR #3 — pick by user demand and implementation
convenience at the time.

## Open questions

1. **DWD bz2 + GribberishCodec chained codec**: needs end-to-end
   verification before we start DWD virtual datasets.
2. **`-spatial` suffix**: tentative naming; revisit after first
   customer feedback.
3. **Variable expansion**: when expanding to all source variables, how
   do we handle variables we don't yet have internal attrs for?
   Auto-discovery from the GRIB index, or manual curation? Likely
   manual curation early, with tooling later.
4. **Polling configuration defaults**: per-dataset
   `poll_interval_seconds`, `max_files_per_commit`,
   `max_seconds_between_commits`. Defaults will emerge from PR #3.
5. **Reforecast / historical archives**: some datasets (GEFS v12
   reforecast) have different URL schemes for historical data.
   Virtual backfills must handle them; operational updates don't.
   Scope this per-dataset.
6. **ConflictDetector behavior** (TBD-by-impl, called out in the
   update process section): does icechunk 2.x's `ConflictDetector`
   reject a rebase when one session resized an array and another
   wrote chunks under that array? The integration test in PR #2
   answers this. Either answer is fine — the per-batch retry loop in
   `process()` handles both cleanly; we just want to know what to
   expect operationally.

---

## Appendix A: Prototype reference (PR #511)

PR #511 (closed, prototype only) demonstrated the full set-virtual-ref
loop end-to-end against real NOAA GFS S3 GRIB files. The single-file
script `prototypes/gfs_icechunk_virtual.py` covers three phases:

1. **Backfill**: initialize zarr metadata for 3 init times × 7 lead
   times, fill all virtual refs, commit.
2. **Add new init time (partial)**: `resize()` the `init_time` coord
   and every data variable array to grow `init_time` by 1, set
   virtual refs for 3 of 7 new lead times, commit. (NaN fill values
   automatically cover the unfilled chunks.)
3. **Fill missing lead times**: set the remaining lead times for the
   new init time, commit.

Repository on-disk size after all three phases: **15.5 KB** (vs
~350 MB if materialized).

### Concrete patterns to reuse

**Repository creation with virtual chunk container:**

```python
storage = icechunk.local_filesystem_storage(str(output_dir))

config = icechunk.RepositoryConfig.default()
s3_store = icechunk.s3_store(region="us-east-1")
container = icechunk.VirtualChunkContainer("s3://noaa-gfs-bdp-pds/", s3_store)
config.set_virtual_chunk_container(container)

repo = icechunk.Repository.create(
    storage,
    config=config,
    authorize_virtual_chunk_access=icechunk.containers_credentials(
        {"s3://noaa-gfs-bdp-pds/": icechunk.s3_anonymous_credentials()}
    ),
)
```

**Zarr metadata for a virtual data variable** (chunk shape = native
GRIB message shape):

```python
codec = GribberishCodec(var=var.internal_attrs.grib_element)
arr = root.create_array(
    var.name,
    shape=(len(init_times), len(lead_times), N_LAT, N_LON),
    chunks=(1, 1, N_LAT, N_LON),
    dtype="float32",
    fill_value=np.nan,
    serializer=codec,
    compressors=(),
    filters=(),
    dimension_names=("init_time", "lead_time", "latitude", "longitude"),
)
```

**Setting virtual refs for a single GRIB file (one chunk per variable):**

```python
s3_url = f"s3://noaa-gfs-bdp-pds/{gfs_s3_path(init_time, lead_time)}"
for var in data_vars:
    offset, length = byte_ranges[var.name]
    chunk_key = f"{var.name}/c/{init_time_idx}/{lead_time_idx}/0/0"
    store.set_virtual_ref(chunk_key, location=s3_url, offset=offset, length=length)
```

**Resizing for dimension expansion** (this is what
`_expand_dimensions()` does in the production code):

```python
root = zarr.open_group(store, mode="r+")

# Resize and write to the append-dim coordinate array
init_time_arr = root["init_time"]
init_time_arr.resize((new_init_time_count,))
init_time_arr[new_init_time_count - 1] = new_value_seconds_since_epoch

# Resize all data variable arrays
for var in data_vars:
    arr = root[var.name]
    arr.resize((new_init_time_count, lead_time_count, N_LAT, N_LON))
```

**Reader code (the entire bare-path read flow):**

```python
storage = icechunk.local_filesystem_storage(str(repo_path))  # or s3_storage(...)
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        "s3://noaa-gfs-bdp-pds/", icechunk.s3_store(region="us-east-1")
    )
)
repo = icechunk.Repository.open(
    storage,
    config=config,
    authorize_virtual_chunk_access=icechunk.containers_credentials(
        {"s3://noaa-gfs-bdp-pds/": icechunk.s3_anonymous_credentials()}
    ),
)
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store, consolidated=False)
# Accessing .values on a chunk triggers S3 fetch + gribberish decode.
```

### Empirical observations from PR #511

- Setting virtual refs is fast (~1s per phase of 3 inits × 7 lead times
  × 3 vars = 63 refs). Bottleneck is the index download, not the
  set_virtual_ref calls.
- Each read fetches one GRIB message (~1 MB compressed). Decoding via
  GribberishCodec adds a small CPU cost; remote latency dominates.
- `resize()` + `set_virtual_ref()` is a valid pattern for incremental
  growth; missing chunks return the fill value automatically.
- Existing `grib_message_byte_ranges_from_index()` (NOAA) works
  without modification — `.idx` byte offsets map 1:1 to virtual chunk
  refs.
- Note: the prototype used icechunk 1.1.15; the repo is now on 2.0.3.
  Most calls should still work, but the integration test in PR #2
  verifies the rebase/conflict semantics under 2.x.

---

## Appendix B: Earlier prototype (PR #510)

PR #510 (closed) was an earlier, more exploratory prototype with two
scripts and a longer report:

1. `prototypes/virtual_grib_gribberish.py` — direct GRIB approach using
   gribberish's `_split_file()` to scan messages and VirtualiZarr's
   `ManifestArray` / `ChunkManifest` API to build a virtual dataset,
   then `vds.virtualize.to_icechunk(session.store)` to write.
2. `prototypes/virtual_zarr_icechunk.py` — NetCDF4 approach using
   `virtualizarr.open_virtual_dataset()` with `HDFParser()` to
   virtualize each file, then `xr.concat` along `step` dim, write to
   a local icechunk repo, and append new init times with
   `to_icechunk(append_dim="lead_time")`.

PR #511 (Appendix A) is the simpler, lower-dependency approach we
adopted: it skips VirtualiZarr entirely and uses
`store.set_virtual_ref()` directly, leaning on our existing
`grib_message_byte_ranges_from_index()` parser instead of gribberish's
scanner.

### Key technical details from PR #510 worth carrying forward

- **`GribberishCodec` is read-only**: a zarr v3 `ArrayBytesCodec`
  (serializer). It always outputs `float64`. The `var=` parameter is
  only used at array-metadata write time to label the codec; decode
  reads whatever GRIB bytes the chunk reference points to.
- **Dependency set added by the prototypes**: `icechunk==1.1.15` (now
  superseded by `~=2.0` in this repo), `zarr 3.1.3`, `gribberish`
  (PR #511 used `>=0.29.0`; PR #1 in this plan will use `~=` style).
  VirtualiZarr was used in PR #510 but is **not** needed by our
  adopted PR #511 approach — that's a deliberate simplification.
- **Mixed storage in one Icechunk store**: coordinate arrays are real
  zarr data (with zstd compression); data variables are virtual GRIB
  refs. Both live in the same icechunk store. This is the model we
  adopt.
- **`set_virtual_ref` vs `to_icechunk(region=...)`**: PR #510 explored
  both. Our adopted approach uses `set_virtual_ref` because it's a
  more explicit, direct mapping from "one GRIB message in S3" to
  "one chunk ref in icechunk," and avoids the round-trip through
  VirtualiZarr's in-memory `ManifestArray`.

---

## Appendix C: Existing infrastructure we reuse

The plan leans heavily on infrastructure that already exists. This
section enumerates it so an implementer can find each piece quickly.

| Concern | Existing module / class | Notes |
|---|---|---|
| CLI commands | `DynamicalDataset.get_cli()` | Add no new commands; variant differences live in `process()` |
| Region × variable-group fan-out and filtering | `RegionJob.get_jobs()` | Used unchanged |
| Worker round-robin | `iterating.get_worker_jobs()` | Used unchanged |
| Icechunk store creation, anonymous credentials | `storage._get_icechunk_store`, `_get_icechunk_storage` | Extended with `virtual_chunk_containers` |
| Temp-branch backfill flow | `parallel_coordination.parallel_setup`, `finalize` | Used unchanged for virtual backfills |
| Coordination files (`setup/ready.json`, `results/*`) | `StoreFactory.write/read/count_coordination_files` | Used by virtual backfills (same as materialized); not used by virtual operational updates (no shared setup step) |
| Icechunk commit with rebase | `storage.commit_if_icechunk` | Used inside `VirtualRegionJob.process()` |
| GRIB index parsing | `noaa/noaa_grib_index.py`, `ecmwf/ecmwf_grib_index.py` | Rename ECMWF function for naming consistency in PR #1 |
| Operational job factory | `RegionJob.operational_update_jobs` | Implemented per dataset, virtual or materialized |
| `update_template_with_results` | `RegionJob.update_template_with_results` | Default behavior trims to last processed coord; virtual variants typically don't need to override |
| Existing per-dataset internal attrs | `noaa/models.py::NoaaInternalAttrs`, etc. | Already contain `grib_element`, `grib_index_level`, etc. — exactly what GribberishCodec and the index parsers need |
| Kubernetes resource definitions | `common/kubernetes.py::ReformatCronJob`, `ValidationCronJob`, `Job` | Reused; no new resource types needed |
