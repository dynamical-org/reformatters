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
- **Very low latency updates**: target < 30s (60s acceptable). Writing
  virtual references is near-instant since we're only recording byte
  offsets, not transferring data.

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

### Single `DynamicalDataset`, sibling `MaterializedRegionJob` / `VirtualRegionJob`

The current `DynamicalDataset` base already abstracts the right things for
both variants — we keep one `DynamicalDataset` hierarchy. The CLI commands,
store factory, kubernetes resource declarations, validation, and Sentry
monitoring don't fork.

`RegionJob`, on the other hand, has substantial materialized-specific code
(download/read/shared-memory/write/upload pipeline; `download_file` /
`read_data` / `apply_data_transformations` hooks; parallelism tunables).
The virtual variant has nothing in common with most of it. So we split
`RegionJob` into a minimal base plus two siblings:

```
DynamicalDataset                     # unchanged single base
├── NoaaGfsForecastDataset             (materialized; existing)
├── NoaaGfsForecastSpatialDataset      (virtual; new)
├── NoaaGefsForecast35DayDataset       (materialized; existing)
├── NoaaGefsForecast35DaySpatialDataset (virtual; new)
└── …

RegionJob                            # thin base: get_jobs, source_groups,
                                     #   operational_update_jobs (abstract),
                                     #   update_template_with_results,
                                     #   process (abstract)
├── MaterializedRegionJob            # the existing process() pipeline,
│                                    #   download_file/read_data/transform hooks,
│                                    #   download/read parallelism tunables
│   ├── NoaaGfsCommonRegionJob       # provider-specific shared bits
│   │   └── NoaaGfsForecastRegionJob (existing, no semantic changes)
│   ├── NoaaHrrrRegionJob
│   │   ├── NoaaHrrrForecast48HourRegionJob
│   │   └── NoaaHrrrAnalysisRegionJob
│   └── …
└── VirtualRegionJob                 # the watcher loop, set_virtual_ref helpers,
                                     #   process_virtual_refs (abstract),
                                     #   _filter_already_ingested (default impl)
    ├── NoaaGfsForecastSpatialRegionJob       (overrides process_virtual_refs)
    ├── NoaaGefsForecast35DaySpatialRegionJob (overrides process_virtual_refs)
    └── …
```

We deliberately don't fork `DynamicalDataset` — the variant differences
are confined to the region job side, where they actually live. Review
feedback steered us here ("look at the methods we have that are similar
on the materialized and virtual variants and see if they are/can be just
different implementations of the same abc methods").

### What lives where

**`DynamicalDataset` (single base, unchanged surface):**

- Fields: `template_config`, `region_job_class`, `store_factory`,
  `dataset_id`, `primary_storage_config`, `replica_storage_configs`.
- CLI: `update_template`, `update`, `backfill_kubernetes`, `backfill_local`,
  `backfill`, `validate_dataset`, `dataset_urls`.
- `_process_region_jobs` driver — extended by branching on
  `region_job_class` to skip the temp-branch path for virtual operational
  updates. See [The update process](#the-update-process).
- Sentry monitoring wrapper.

**`RegionJob` (minimal base, mostly unchanged):**

- Fields: `tmp_store`, `template_ds`, `data_vars`, `append_dim`, `region`,
  `reformat_job_name`.
- `get_jobs()`, `source_groups()`, `get_processing_region()`,
  `update_template_with_results()`.
- `operational_update_jobs()` (abstract).
- `process()` (abstract — currently has a default materialized
  implementation; PR #2 moves that body down to `MaterializedRegionJob`).

**`MaterializedRegionJob` (new — current `RegionJob.process()` and friends
move here):**

- Hooks: `download_file()`, `read_data()`, `apply_data_transformations()`.
- Tunables: `download_parallelism`, `read_parallelism`,
  `max_vars_per_download_group`.
- `process()` implementation: the current shared-memory pipeline plus
  helpers (`_download_processing_group`, `_read_into_data_array`,
  `_write_shards`, `_cleanup_local_files`).

This is a **mechanical move** of existing code — no semantic changes.
Every existing dataset's `RegionJob` subclass changes its base from
`RegionJob` to `MaterializedRegionJob` and existing tests pass unchanged.

**`VirtualRegionJob` (new):**

- `process()` implementation: opens icechunk session, runs filter, drives
  generator, lazily expands as needed, sets refs, commits (see
  [The update process](#the-update-process)).
- `process_virtual_refs()` (abstract — the generator subclasses implement).
- `_filter_already_ingested()` default implementation (see
  [Filtering](#filtering-already-ingested-coordinates)).
- `_expand_dimensions(new_append_values)` helper called from `process()`
  when a batch introduces new append-dim values — resizes coord and data
  var arrays and writes the new coord values into the current session.
- `_should_commit()` helper using ClassVar commit thresholds.
- ClassVar tuples: `max_files_per_commit`, `max_seconds_between_commits`
  (operational vs backfill values).
- `download_index()` default implementation calling
  `http_download_to_disk` against a coord-supplied URL.
- ClassVar `source_virtual_chunk_containers` for store-factory wiring
  (see [Storage configuration](#storage-configuration)).

## The update process

The whole point of this is sub-60s updates, so the lifecycle deserves a
clean walkthrough. Virtual operational updates use the **same Kubernetes
scheduling pattern as materialized updates** — `ReformatCronJob` →
indexed Job → N parallel worker pods. The differences from materialized
are localized: no temp branch, commits go straight to `main`, and dim
expansion happens **lazily inside each region job's `process()`** — only
when that job needs to write to an append-dim position that doesn't yet
exist, atomic with the corresponding refs.

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
losing the latency advantage (the whole point is sub-minute visibility).
Lazy expansion is the alternative: data and structure always commit
together.

### Step-by-step

1. **CronJob fires** on its schedule (typically once per minute; see
   [Latency tradeoff](#latency-tradeoff) below). `concurrencyPolicy: Forbid`
   prevents overlapping fires.
2. **Indexed Job spawns N workers**. Same env vars as today
   (`JOB_NAME`, `WORKER_INDEX`, `WORKERS_TOTAL`).
3. **Each worker runs `update()`** on the `DynamicalDataset`, which
   calls `operational_update_jobs()` and then `_process_region_jobs()`.
4. **The driver detects this is a virtual operational update** —
   `region_job_class` is a `VirtualRegionJob` subclass and
   `update_template_with_results=True`. It takes the virtual operational
   path: no temp branch, no shared setup step. Each pod handles its
   round-robin slice of region jobs directly.
5. **Each worker processes its assigned region jobs in parallel**. For
   each region job, `VirtualRegionJob.process()`:
   1. Opens its own writable session on `main`.
   2. Calls `_filter_already_ingested(source_coords)` — typically
      reduces the work to a handful of newly-available files (or zero
      files if this region job has nothing new).
   3. Drives the `process_virtual_refs()` generator. For each batch:
      - If the batch's coords include append-dim values not yet in
        the store, call `_expand_dimensions(new_values)` to resize
        coord and data var arrays and write the new coord values.
        This and the batch's `set_virtual_ref` calls go into the
        same session.
      - Issue `store.set_virtual_ref(...)` for each (var, chunk) in
        the batch.
      - When the pending-refs counter hits `max_files_per_commit`
        or `max_seconds_between_commits`, commit with
        `rebase_with=icechunk.ConflictDetector()` and open a fresh
        session on `main`.
   4. After the generator exhausts (or the pod's
      `pod_active_deadline` approaches), commits any pending refs.
6. **No finalize step**. There's no temp branch to merge. The driver
   may still wait for all workers (via the existing coordination-file
   mechanism) before exiting, but there's no `reset_branch`.
7. **Pods exit**. The next cron fire picks up any files that weren't
   yet published (or were skipped because the pod hit its deadline).

This flow is intentionally close to the existing materialized flow.
What changes is small: skip the temp branch, expand `main` lazily
inside each region job's process() as data arrives, let each worker
manage its own session and commit cadence.

### Why lazy expansion works with parallel pods

Each batch's commit is its own self-contained transaction: open a
fresh session on `main`, decide what refs to set (recomputing chunk
keys against current state), expand the dim if needed, set the refs,
commit. On conflict, throw away the session and run the transaction
again on a fresh session.

Concretely:

```python
# Pseudocode inside VirtualRegionJob.process()
for batch in self.process_virtual_refs():
    for attempt in range(MAX_RETRIES):  # ~5-10 attempts with backoff
        session = repo.writable_session("main")
        store = session.store

        # Recompute against current state — handles the "another pod
        # already added an init at this index" case naturally.
        relevant_refs = self._refs_against_current_state(batch, store)
        if not relevant_refs:
            break  # already ingested by someone else; skip

        new_coords = self._needs_dim_expansion(store, relevant_refs)
        if new_coords:
            self._expand_dimensions(store, new_coords)

        for ref in relevant_refs:
            store.set_virtual_ref(ref.key, ref.url, offset=ref.offset, length=ref.length)

        try:
            session.commit("...", rebase_with=icechunk.ConflictDetector())
            break
        except icechunk.ConflictError:
            # Discard session, retry with a fresh one. The retry's
            # _refs_against_current_state() will see the other pod's
            # commit and recompute target indices accordingly.
            time.sleep(0.1 * (2 ** attempt))
```

This makes the conflict story uniform — there is no "steady state vs
catchup" distinction. A conflict just means "another pod committed
between when I opened my session and when I tried to commit," and
the response is always the same: throw away, recompute, retry. In
steady state this almost never fires; in catchup it fires more
often but is still cheap (recomputing keys and re-issuing
`set_virtual_ref` calls is microseconds; the byte ranges we already
have from the parsed index files).

#### Why this isn't expensive

Each retry redoes:

- One icechunk session open (cheap — metadata only).
- One pass through `_refs_against_current_state` (a small array read).
- A handful of `set_virtual_ref` calls (microseconds each).
- One commit attempt.

No re-downloading of GRIB index files, no re-parsing — we already
have the parsed `(starts, ends)` from when the generator yielded the
batch. The only thing that changes between attempts is which
append-dim indices the refs land at, and that's a small recomputation
from the current `init_time` (or `time`) coord values.

In practice the steady-state path has at most one pod expanding
per cron fire (the one with the newest init / newest time chunk).
Other pods write to existing append-dim positions only and never
collide. Conflicts only occur during catchup-like scenarios
(multiple new inits arriving in one fire after downtime), and the
retry handles them transparently.

### Concurrency and ConflictDetector (summary)

`ConflictDetector` rejects a commit when two sessions wrote to the
same path. The retry loop above is the response — it always works
because lazy expansion + the filter-against-current-state step
naturally recomputes target indices on each attempt.

The existing `storage.commit_if_icechunk` already retries up to 10×
on rebase failure, but that retry just re-runs `session.commit()`
without rebuilding the session's writes. For virtual operational
updates we need a different retry shape (open new session, redo the
batch's writes, commit) which lives inside `VirtualRegionJob.process()`.

> **TBD-by-impl**: confirm icechunk 2.x's `ConflictDetector` accepts
> rebases when one session resized an array (wrote `<var>/zarr.json`)
> and another session wrote chunks under that array. Assumed
> behavior: writes-only conflict tracking, so the rebase passes if
> the write sets don't intersect. If `ConflictDetector` is stricter
> than that, the retry loop above still handles it — we just see
> more conflicts than expected and pay slightly more retries.

### Latency tradeoff

Kubernetes CronJobs fire at most once per minute. End-to-end latency
for a single new file is roughly:

```
latency ≈ cron_interval + pod_startup + source_file_published_to_observed + processing
        ≈ 60s + 5-15s + 0-30s + <5s
        = 70-110s in the typical case
```

For 60s targets this is tight; we accept it. For < 30s targets it
doesn't fit — that would require a different mechanism (long-running
watcher pod, event-driven trigger) which we're explicitly not
building yet. Document the achievable latency per dataset as part of
PR #3.

Within a single fire, `process_virtual_refs()` MAY briefly poll
(check, sleep ~5-10s, check again) up to its pod budget so files
arriving mid-fire still get processed. This trades pod CPU for
lower average latency; appropriate for high-update-rate datasets.

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
`process_virtual_refs()` code paths — but with two important differences:

1. **Temp branch flow** (same as materialized backfills): worker 0
   creates `_job_<job_name>`, expands metadata fully on the branch,
   commits. Workers write refs to the branch. Last worker resets
   `main` to the branch. Readers see "empty" or "full," never partial.
2. **Looser commit thresholds**: `max_files_per_commit[1]` (e.g. 50+)
   and `max_seconds_between_commits[1]` (e.g. 60s+) reduce commit
   overhead for the much larger volume.

The differences are entirely confined to the driver and the
`is_backfill` flag on the region job (set to `True` by `get_jobs()`,
`False` by `operational_update_jobs()`). The generator
`process_virtual_refs()` checks `self.is_backfill` to skip polling
and yield immediately. The driver checks the flag to choose
temp-branch vs direct-to-main.

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

## Per-variable serializer: choose at PR #3

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

The existing HRRR pattern `get_data_vars(encoding)` shares a single
`Encoding` instance across all vars. Three realistic ways to extend
it to support per-var serializers; PR #3 picks one:

### Option A — Encoding factory passed to `get_data_vars`

```python
class NoaaGefsCommonTemplateConfig(TemplateConfig[GefsDataVar]):
    def get_data_vars(
        self, make_encoding: Callable[[GefsDataVar], Encoding]
    ) -> Sequence[GefsDataVar]: ...

class NoaaGefsForecast35DayTemplateConfig(NoaaGefsCommonTemplateConfig):
    @computed_field
    @cached_property
    def data_vars(self) -> Sequence[GefsDataVar]:
        enc = Encoding(dtype="float32", chunks=(1, 49, 721, 1440), ...)
        return self.get_data_vars(make_encoding=lambda _var: enc)

class NoaaGefsForecast35DaySpatialTemplateConfig(NoaaGefsCommonTemplateConfig):
    @computed_field
    @cached_property
    def data_vars(self) -> Sequence[GefsDataVar]:
        def make_encoding(var: GefsDataVar) -> Encoding:
            return Encoding(
                dtype="float32", chunks=(1, 1, 721, 1440), shards=None,
                serializer={"name": "gribberish", "configuration": {"var": var.internal_attrs.grib_element}},
                compressors=None, filters=None, ...,
            )
        return self.get_data_vars(make_encoding=make_encoding)
```

- **Pros**: Single source of truth for variable identity; factory cleanly
  encapsulates per-variant encoding logic.
- **Cons**: Existing HRRR `get_data_vars(encoding: Encoding)` signature
  changes; one-line refactor at every call site (today: HRRR variants
  only). The factory has to receive enough of the variable to
  parameterize the codec, which means a proto-DataVar or the actual
  DataVar pre-encoding — slightly awkward.

### Option B — Common config exposes variable metadata; each variant builds DataVars

The common template config exposes `_data_vars_metadata` — name,
internal_attrs, attrs tuples, no encoding. Each variant's `data_vars`
constructs the full DataVars by attaching its own encoding.

```python
class NoaaGefsCommonTemplateConfig(TemplateConfig[GefsDataVar]):
    @computed_field
    @cached_property
    def _data_vars_metadata(self) -> Sequence[tuple[str, GefsInternalAttrs, DataVarAttrs]]: ...

class NoaaGefsForecast35DayTemplateConfig(NoaaGefsCommonTemplateConfig):
    @computed_field
    @cached_property
    def data_vars(self) -> Sequence[GefsDataVar]:
        enc = Encoding(dtype="float32", chunks=(1, 49, 721, 1440), ...)
        return [GefsDataVar(name=n, encoding=enc, attrs=a, internal_attrs=ia)
                for n, ia, a in self._data_vars_metadata]

class NoaaGefsForecast35DaySpatialTemplateConfig(NoaaGefsCommonTemplateConfig):
    @computed_field
    @cached_property
    def data_vars(self) -> Sequence[GefsDataVar]:
        return [
            GefsDataVar(
                name=n, attrs=a, internal_attrs=ia,
                encoding=Encoding(
                    dtype="float32", chunks=(1, 1, 721, 1440), shards=None,
                    serializer={"name": "gribberish", "configuration": {"var": ia.grib_element}},
                    compressors=None, filters=None, ...,
                ),
            )
            for n, ia, a in self._data_vars_metadata
        ]
```

- **Pros**: No factory callback gymnastics. Clear separation: identity
  in the common config, encoding in the variant.
- **Cons**: HRRR's existing `get_data_vars(encoding)` doesn't fit this
  shape directly — we'd either rename and reshape it (touching the same
  call sites as Option A), or keep both methods.

### Option C — Virtual variant inherits from materialized and replaces encodings

```python
class NoaaGefsForecast35DaySpatialTemplateConfig(NoaaGefsForecast35DayTemplateConfig):
    @computed_field
    @cached_property
    def data_vars(self) -> Sequence[GefsDataVar]:
        return [
            replace(var, encoding=Encoding(
                dtype="float32", chunks=(1, 1, 721, 1440), shards=None,
                serializer={"name": "gribberish", "configuration": {"var": var.internal_attrs.grib_element}},
                compressors=None, filters=None, ...,
            ))
            for var in super().data_vars
        ]
    # Also overrides dataset_attributes (different dataset_id, version).
```

- **Pros**: Zero changes to existing template configs. Smallest diff.
- **Cons**: The virtual variant inherits from the materialized one,
  which couples them — restructuring materialized risks breaking
  virtual. Also requires explicit `dataset_attributes` override for
  the differing id/version, which is awkward boilerplate.

**Recommendation**: PR #3's implementer picks. If they're growing a
virtual variant of a dataset that already uses
`get_data_vars(encoding)` (HRRR), Option A is the lightest touch. If
they're growing it on a dataset that doesn't yet use that pattern
(most), Option B is slightly cleaner. Option C is the fallback.

We do **not** retrofit every template config eagerly. Only datasets
that grow a virtual variant adopt whichever pattern is chosen.

## Encoding rules for virtual datasets

All virtual variants follow the same encoding rules, regardless of
provider:

- **Serializer**: `GribberishCodec(var=<grib_element>)` per variable.
  See `gribberish.zarr.codec.GribberishCodec`.
- **Chunk shape**: `(1, 1, lat, lon)` for forecast (one chunk per
  (init_time, lead_time) GRIB message) or `(1, lat, lon)` for analysis
  (one chunk per time step).
- **No shards**: virtual references map 1:1 to GRIB messages.
- **No compressors or filters**: GribberishCodec handles the full
  decode.
- **Exception (DWD, bz2-compressed GRIBs)**: chain zarr's built-in
  `Bz2Codec` as a filter before `GribberishCodec` as the serializer.
  Needs end-to-end verification (see [Open questions](#open-questions)).

## Storage configuration

A virtual dataset opens its Icechunk repo with one or more
`VirtualChunkContainer`s registered (one per source S3 prefix it
reads from) and `authorize_virtual_chunk_access` credentials wired up.
The bare-minimum `__main__.py` surface for this is **nothing new** —
the container set is a property of the source data, which the
`VirtualRegionJob` subclass already knows about. We declare it once
on the region job class and let the store-factory pick it up:

```python
# In the region job subclass — single source of truth.
class NoaaGefsForecast35DaySpatialRegionJob(VirtualRegionJob[...]):
    source_virtual_chunk_containers: ClassVar[Sequence[VirtualChunkContainerConfig]] = (
        VirtualChunkContainerConfig(prefix="s3://noaa-gefs-pds/", region="us-east-1"),
    )

# In storage.py — opens icechunk with the right containers if needed.
class StoreFactory(FrozenBaseModel):
    ...
    virtual_chunk_containers: Sequence[VirtualChunkContainerConfig] = ()

# In DynamicalDataset.store_factory (one-line change) — pass containers
# through from the region job class.
@computed_field
@property
def store_factory(self) -> StoreFactory:
    containers: Sequence[VirtualChunkContainerConfig] = ()
    if issubclass(self.region_job_class, VirtualRegionJob):
        containers = self.region_job_class.source_virtual_chunk_containers
    return StoreFactory(..., virtual_chunk_containers=containers)

# In __main__.py — virtual dataset registration looks identical to materialized.
NoaaGefsForecast35DaySpatialDataset(
    primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
),
```

`VirtualChunkContainerConfig` is intentionally minimal:

```python
class VirtualChunkContainerConfig(FrozenBaseModel):
    prefix: str       # e.g. "s3://noaa-gefs-pds/"
    region: str       # e.g. "us-east-1"
    # Defaults to anonymous because every target source today is publicly
    # readable. If a future source needs credentials, we add a
    # `k8s_secret_name: str | None` field that mirrors StorageConfig's
    # existing credential loading.
    anonymous: bool = True
```

**Bare-minimum supported functionality (PR #2):**

1. One `VirtualChunkContainerConfig` per source S3 prefix.
2. Anonymous S3 reads (good enough for every NOAA/ECMWF/DWD target).
3. Region declared per container (no inference; explicit beats implicit
   for stable regions).
4. Storage layer asserts virtual variants use icechunk
   (`DatasetFormat.ICECHUNK`) for both primary and any replica stores.
   Mixing zarr v3 + virtual refs isn't supported; fail loudly at
   construction.

**Explicitly not supported yet** (additive when needed):

- Credentialed source buckets.
- HTTP-only source containers.
- Per-job dynamic container registration (multiple prefixes are fine;
  switching at runtime is not).

> **TBD-by-impl**: confirm icechunk persists the `VirtualChunkContainer`
> list inside the repo config so readers don't need to re-register
> containers. If not, readers must construct the same `RepositoryConfig`
> we use here (see Appendix A's reader code).

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
- Add `gribberish>=0.29.0` to `pyproject.toml`.
- Add `VirtualChunkContainerConfig` to `common/storage.py`, plus the
  `virtual_chunk_containers` field on `StoreFactory` and the
  open-time wiring through `_get_icechunk_storage`. Assert
  icechunk-only when set.
- Move the current `RegionJob.process()` body and its materialized
  helpers into a new `MaterializedRegionJob` class; existing region
  jobs inherit from `MaterializedRegionJob`. **No semantic change**;
  all existing tests pass unchanged.

### PR #2 — VirtualRegionJob and driver hooks

- Add `VirtualRegionJob` as a sibling of `MaterializedRegionJob`
  under `RegionJob`, with the operational flow described in
  [The update process](#the-update-process).
- Extend `_process_region_jobs` to branch on `region_job_class`
  (virtual operational updates skip the temp branch; backfills and
  materialized updates keep the existing flow).
- Add the `is_backfill` field on `VirtualRegionJob`, set by
  `get_jobs(...)` (True) and `operational_update_jobs(...)` (False).
- Add the default `_filter_already_ingested` with both strategies.
- Add `_expand_dimensions` helper, called lazily from `process()`
  when a batch's append-dim values exceed the current store shape.
- Add the per-batch commit retry loop (open fresh session, recompute,
  set refs, commit, retry on conflict).
- Add commit-batching ClassVar tuples.
- Integration tests against a local icechunk 2.x repo:
  - **Concurrent disjoint-chunk writes** — two pods writing chunks
    at different `init_time` indices, one of them also expanding
    the dim. Verifies `ConflictDetector` accepts a rebase when one
    session resized an array and another wrote chunks under it.
  - **Concurrent expansion conflict** — two pods both expanding for
    new inits simultaneously. Verifies the per-batch retry loop
    converges to a correct final state (both inits present, all
    refs landed at correct indices).
  - **Backfill-to-update transition** — backfill leaves a partial
    state; update fills the rest. Verifies the temp-branch and
    direct-to-main paths interoperate cleanly.

### PR #3 — First concrete virtual dataset (end to end)

- Pick from the candidates below.
- Refactor that dataset's `TemplateConfig` to expose per-var
  encodings via one of Options A/B/C above.
- Implement `<dataset>SpatialTemplateConfig` with virtual encoding
  rules.
- Implement `<dataset>SpatialRegionJob` with
  `process_virtual_refs()` and the
  dataset-specific `_filter_already_ingested` if needed.
- Implement `<dataset>SpatialDataset` with
  `operational_kubernetes_resources()` returning a
  `ReformatCronJob` + `ValidationCronJob`.
- Integration tests: backfill a small slice, verify read-back via
  GribberishCodec; simulate operational poll.

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
  superseded by `~=2.0` in this repo), `zarr 3.1.3`,
  `gribberish>=0.29.0`. VirtualiZarr was used in PR #510 but is **not**
  needed by our adopted PR #511 approach — that's a deliberate
  simplification.
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
