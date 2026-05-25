# Virtual Icechunk Datasets

## Goals

Create virtual Icechunk datasets that complement our existing materialized (rechunked) time-series datasets. For datasets where source data lives in a publicly accessible GRIB archive (NOAA NODD, ECMWF and DWD archives on Source Coop), a virtual dataset stores only Icechunk metadata pointing at byte ranges within the original GRIB files.

**Why:**
- **Spatial/map-optimized chunking**: Chunks follow the native GRIB message shape (1 time step, full spatial grid), ideal for spatial queries and map rendering.
- **All source variables**: No storage cost means we can include every variable in the source archive, not just a curated subset.
- **Very low latency updates**: Target < 30s (60s acceptable). Writing virtual references is near-instant since we're only recording byte offsets, not transferring data.
- **Complementary access patterns**: Users choose the materialized `-timeseries` dataset for time-series extraction or the `-spatial` dataset for map/spatial queries over the same underlying data.

**How it works** (proven in PR #511 prototype):
1. Download lightweight GRIB index file (~10KB)
2. Parse byte ranges for each variable using existing index parsers
3. Call `store.set_virtual_ref(chunk_key, grib_url, offset=offset, length=length)`
4. Commit to Icechunk — readers immediately see the new data
5. `GribberishCodec` (zarr v3 serializer) decodes raw GRIB messages at read time

The prototype achieved 15.5KB on disk for what would be 350MB materialized.

## Architecture

### Class hierarchy

Introduce base classes that capture what's common between materialized and virtual datasets, then specialize:

```
DynamicalDataset (abstract base)
├── MaterializedDynamicalDataset (current DynamicalDataset, renamed)
└── VirtualDynamicalDataset (new)

RegionJob (abstract base)
├── MaterializedRegionJob (current RegionJob, renamed)
└── VirtualRegionJob (new)
```

#### What lives in the base classes

**DynamicalDataset base:**
- `template_config`, `primary_storage_config`, `replica_storage_configs`, `store_factory`
- `dataset_id`, version management
- CLI scaffolding: `update_template`, `validate_dataset`
- Abstract: `validators()`, `operational_kubernetes_resources()`
- Sentry monitoring helpers

**RegionJob base:**
- `template_ds`, `data_vars`, `append_dim`, `region`, `reformat_job_name`
- Abstract: `generate_source_file_coords()` — reusable across both variants
- `get_jobs()` class method — job creation and round-robin worker distribution
- Abstract: `operational_update_jobs()`, `process()`
- `source_groups()`, `get_processing_region()`

#### What's specialized

**MaterializedDynamicalDataset** adds:
- CLI: `backfill_kubernetes`, `backfill_local`, `process_backfill_region_jobs`, `update`
- The full materialized update flow: write metadata → copy metadata → process → update template → commit

**MaterializedRegionJob** adds:
- `download_file()`, `read_data()`, `apply_data_transformations()`
- `process()`: shared memory buffer → download → read → transform → write shards → upload
- `tmp_store` for local shard staging

**VirtualDynamicalDataset** adds:
- CLI: `virtual_update`, `virtual_backfill_kubernetes`, `virtual_backfill_local`
- Virtual chunk container configuration (per-source S3/HTTP store config)
- `authorize_virtual_chunk_access` credential setup

**VirtualRegionJob** adds:
- Abstract: `poll_virtual_refs()` generator (see [Polling Watcher](#operational-model-polling-watcher))
- `process()`: drives the generator, handles lazy expansion, batches commits

### TemplateConfig sharing

Use the existing pattern from `NoaaHrrrCommonTemplateConfig` where a shared base class defines variables via `get_data_vars(encoding)` and subclasses call it with the appropriate encoding:

```python
class NoaaGefsCommonTemplateConfig(TemplateConfig[GefsDataVar]):
    def get_data_vars(self, encoding: Encoding) -> Sequence[GefsDataVar]:
        return [
            GefsDataVar(name="temperature_2m", encoding=encoding, ...),
            ...
        ]

class NoaaGefsForecast35DayTemplateConfig(NoaaGefsCommonTemplateConfig):
    @computed_field
    @cached_property
    def data_vars(self) -> Sequence[GefsDataVar]:
        return self.get_data_vars(Encoding(
            dtype="float32",
            chunks=(1, 49, 721, 1440),
            shards=(1, 49, 721, 1440),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
            ...
        ))

class NoaaGefsSpatialTemplateConfig(NoaaGefsCommonTemplateConfig):
    @computed_field
    @cached_property
    def data_vars(self) -> Sequence[GefsDataVar]:
        return self.get_data_vars(Encoding(
            dtype="float32",
            chunks=(1, 1, 721, 1440),  # one GRIB message per chunk
            # no shards, no compressors — GribberishCodec is the serializer
            ...
        ))
```

Not all datasets currently use the `get_data_vars(encoding)` pattern. Those that don't will need to adopt it (or a similar approach) when adding a virtual variant. This is not a hard requirement — it's just the cleanest way to share variable definitions.

### Virtual encoding rules

All virtual datasets follow the same encoding rules, regardless of provider:
- **Serializer**: `GribberishCodec(var=grib_element)` — decodes raw GRIB at read time
- **Chunk shape**: `(1, 1, lat, lon)` for forecast or `(1, lat, lon)` for analysis — one GRIB message per chunk
- **No shards**: Virtual references map 1:1 to GRIB messages
- **No compressors or filters**: GribberishCodec handles the full decode
- **Exception: bz2-compressed GRIBs** (e.g., DWD): Use zarr-python's built-in bz2 codec chained before GribberishCodec. _Needs verification._

## Core workflow: virtual reference writing

The fundamental loop, proven in PR #511:

```python
for source_file_coord in source_file_coords:
    index_path = download_index(source_file_coord)  # ~10KB
    byte_ranges = parse_index(index_path, data_vars)  # existing parsers

    grib_url = source_file_coord.get_url()  # s3://bucket/path
    for var, (offset, length) in zip(data_vars, byte_ranges):
        init_i, lead_i = compute_chunk_indices(source_file_coord, template_ds)
        store.set_virtual_ref(
            f"{var.name}/c/{init_i}/{lead_i}/0/0",
            location=grib_url,
            offset=offset,
            length=length,
        )

session.commit("update message")
```

**Existing code reused directly:**
- `SourceFileCoord.get_url()` — S3/HTTP URLs for GRIB files
- `SourceFileCoord.get_idx_url()` / `get_index_url()` — index file URLs
- `grib_message_byte_ranges_from_index()` (NOAA) — parses `.idx` files to `(starts, ends)`
- `get_message_byte_ranges_from_index()` (ECMWF) — parses `.index` JSON Lines to `(starts, ends)`
- Variable internal attrs: `grib_element` (for GribberishCodec), `grib_index_level`, etc.

**Virtual chunk container setup:**

Each source archive needs a `VirtualChunkContainer` registered on the Icechunk repository:

```python
config = icechunk.RepositoryConfig.default()
container = icechunk.VirtualChunkContainer(
    "s3://noaa-gefs-pds/",
    icechunk.s3_store(region="us-east-1"),
)
config.set_virtual_chunk_container(container)

repo = icechunk.Repository.open_or_create(
    storage, config=config,
    authorize_virtual_chunk_access=icechunk.containers_credentials(
        {"s3://noaa-gefs-pds/": icechunk.s3_anonymous_credentials()}
    ),
)
```

All source archives we target have anonymous read access, keeping reader setup simple.

## Operational model: polling watcher

### VirtualRegionJob.process() flow

The common `process()` method in `VirtualRegionJob` drives a generator that subclasses implement. The generator controls what files to process and when to stop:

```python
# VirtualRegionJob base class
def process(self, store: IcechunkStore, session: Session) -> None:
    pending_refs = 0
    last_commit = time.monotonic()
    dimension_expanded = False

    for batch in self.poll_virtual_refs():
        if not dimension_expanded:
            self.expand_dimensions(store)
            dimension_expanded = True

        for var_name, chunk_key, url, offset, length in batch:
            store.set_virtual_ref(chunk_key, url, offset=offset, length=length)
            pending_refs += 1

        if self.should_commit(pending_refs, last_commit):
            session.commit(...)
            session = repo.writable_session("main")  # new session
            store = session.store
            pending_refs = 0
            last_commit = time.monotonic()

    # Final commit for remaining refs
    if pending_refs > 0:
        session.commit(...)
```

### poll_virtual_refs() generator

Subclasses implement this generator. It `yield`s batches of virtual refs and controls the job lifecycle:

```python
# Subclass (e.g., GefsSpatialRegionJob)
def poll_virtual_refs(self) -> Iterator[Sequence[VirtualRef]]:
    source_coords = self.generate_source_file_coords(...)
    remaining = filter_already_ingested(source_coords, self.existing_dataset)

    while remaining:
        newly_available = check_availability(remaining)  # HEAD requests
        if not newly_available:
            time.sleep(poll_interval)
            continue

        for coord in newly_available:
            index_path = download_index(coord)
            byte_ranges = parse_index(index_path, coord.data_vars)
            yield make_virtual_refs(coord, byte_ranges, self.template_ds)
            remaining.remove(coord)

    # Generator exhaustion = job complete
```

For **forecast datasets**: the generator exhausts when all expected lead times for that init time are available (or timeout).

For **analysis datasets**: the generator processes a configured number of time steps. Scheduling determines how often the job runs:
- Bursty sources (e.g., GFS analysis with 6h of files): start before burst, exit when all expected files arrive
- Low-cadence sources (e.g., daily): start at expected time, process one file, exit
- High-cadence sources (e.g., MRMS every 2min): process a configured batch (e.g., one hour's worth), schedule hourly

### Lazy dimension expansion

Dimension expansion (resizing arrays, appending coordinate values) happens on the first file arrival, not upfront. This ensures:
- Readers don't see empty holes before any data is available
- Analysis datasets don't need to pre-allocate unknown lengths
- The first commit always includes both the expansion and the initial data

### Commit batching

Two thresholds, whichever triggers first:
- `max_seconds_between_commits` (e.g., 10s for operational, 60s+ for backfill)
- `max_files_per_commit` (e.g., 5 for operational, 50+ for backfill)

Each commit is atomic — readers always see a consistent state.

### Parallelism

One `VirtualRegionJob` per init time (forecast) or per time window (analysis). Multiple jobs can run concurrently (e.g., catching up on a previous init while processing the latest). Icechunk's `ConflictDetector` handles automatic rebase when concurrent sessions commit.

Kubernetes parallelism follows the existing pattern: `parallelism = N * num_variable_groups()`, where N depends on dataset characteristics. For virtual datasets, variable groups may be less relevant (no download bottleneck), so parallelism may be simpler.

### Crash recovery

If a watcher crashes mid-forecast:
- Committed refs are durable and visible to readers
- On restart, the `filter_already_ingested` step detects what's already done
- The watcher resumes polling for remaining files
- Setting a ref that's already set is safe (idempotent)

## Backfill

Backfills use the same `VirtualRegionJob` code path with:
- Much looser commit batch limits (more files and longer intervals between commits)
- Kubernetes indexed jobs for parallelism (same pattern as materialized backfills)
- No polling — all files already exist, so the generator yields immediately
- Worker distribution via the existing round-robin `get_worker_jobs()` mechanism

## Filtering already-ingested coordinates

> _Design detail to be refined during implementation._

Before the generator starts polling, we filter out source file coords whose data is already in the dataset. Likely approaches:
- For forecast datasets: use `ingested_forecast_length` coordinate (already exists on all forecast datasets) to determine the last fully-ingested lead time per init time
- For analysis datasets: check the last value in the append dimension coordinate
- General fallback: attempt to read a chunk and check if it's NaN vs has data (slower, requires S3 fetch)

The exact interface — whether this is a standalone function that takes the source coord sequence and the existing dataset, or a method on VirtualRegionJob — is TBD. The plan notes this as an open question.

## Dataset identity and storage

### Naming

Virtual datasets get a `-spatial` suffix on the dataset ID (tentative — captures the access pattern optimization without exposing the implementation detail "virtual"):
- `noaa-gefs-forecast-35-day` (materialized, time-series optimized)
- `noaa-gefs-forecast-35-day-spatial` (virtual, spatial/map optimized)

### Storage location

Virtual Icechunk stores live in the same S3 buckets as the materialized datasets, with new dataset IDs and paths. The stores are tiny (KBs of metadata) so co-location has no cost concern.

### Reader experience

All source GRIB archives targeted for virtual datasets have anonymous read access. The `dynamical_catalog` Python library (thin wrapper on our STAC catalog) handles virtual chunk container setup when opening datasets, so readers don't need to manually configure containers.

## Provider-specific considerations

### NOAA (GFS, GEFS, HRRR)
- **Index format**: Plain text `.idx` files, parsed with `grib_message_byte_ranges_from_index()`
- **Source**: `s3://noaa-gfs-bdp-pds/`, `s3://noaa-gefs-pds/`, `s3://noaa-hrrr-bdp-pds/`
- **Straightforward**: one virtual chunk container per bucket, anonymous S3 access

### ECMWF (IFS-ENS, AIFS)
- **Index format**: JSON Lines `.index` files, parsed with `get_message_byte_ranges_from_index()`
- **Source**: `s3://ecmwf-forecasts/` (eu-central-1) and Source Coop archives
- **Archive transition**: IFS-ENS has MARS archive (pre-2024-04) and open data (post-2024-04) with different URL schemes. Virtual dataset likely only covers open data era.

### DWD (ICON-EU)
- **No index files**: One variable per bz2-compressed GRIB file
- **Codec pipeline**: Needs zarr-python bz2 codec chained with GribberishCodec: `codecs=[Bz2Codec(), GribberishCodec(var=element)]`
- **Verification needed**: This chained codec approach needs testing to confirm it works end-to-end before implementing DWD virtual datasets

## Implementation plan

### PR 1: Rename existing classes
Pure rename, no behavior change:
- `DynamicalDataset` → `MaterializedDynamicalDataset`
- `RegionJob` → `MaterializedRegionJob`
- Update all subclasses and imports across the codebase

### PR 2: Extract common base classes
- Create abstract `DynamicalDataset` base with shared code (storage, CLI scaffold, validation, template management)
- Create abstract `RegionJob` base with shared code (generate_source_file_coords, get_jobs, worker distribution)
- `MaterializedDynamicalDataset` and `MaterializedRegionJob` extend these bases
- All existing tests pass unchanged

### PR 3: VirtualDynamicalDataset + VirtualRegionJob base classes
- Implement `VirtualDynamicalDataset` with virtual-specific CLI commands and virtual chunk container management
- Implement `VirtualRegionJob` with `process()` loop (drives generator, lazy expansion, commit batching)
- Define the `poll_virtual_refs()` generator interface
- Add `GribberishCodec` dependency (already in prototype: `gribberish>=0.29.0`)
- Unit tests for the base classes

### PR 4: First concrete virtual dataset
- Implement one dataset end-to-end (candidate list below)
- Adopt `get_data_vars(encoding)` pattern in that dataset's template config if not already present
- Virtual template config subclass with GribberishCodec encoding
- Virtual region job subclass with `poll_virtual_refs()` implementation
- Virtual dynamical dataset subclass with operational Kubernetes resources
- Integration test: backfill a small slice, verify read-back via GribberishCodec
- Operational test: simulate polling for new files

### PR 5: Second virtual dataset
- Test that the pattern generalizes to a different provider/structure
- Refine base class abstractions based on what the second dataset needs

### Later: Expand variable coverage
- After the pipeline is proven, expand datasets to include all source variables
- This is the last step, after operational stability is established

### Candidate first datasets

| Dataset | Pros | Cons |
|---|---|---|
| GFS forecast | Prototype exists (PR #511), simplest structure | Lower demand? |
| GEFS forecast 35-day | Tests ensemble dimension, high demand | More complex URL/file-type logic |
| IFS-ENS forecast | High demand, tests ECMWF index format | MARS/open-data archive split |
| HRRR 18-hour forecast | 24 inits/day tests high-frequency updates | Projected grid (y/x not lat/lon) |

Decision deferred — pick based on user demand and implementation convenience at the time.

## Open questions

1. **Filtering already-ingested coordinates**: Exact mechanism and interface TBD (see [section above](#filtering-already-ingested-coordinates)).
2. **DWD bz2 + GribberishCodec chained codec**: Needs end-to-end verification.
3. **`-spatial` suffix**: Tentative naming for virtual dataset IDs.
4. **Variable expansion**: When expanding to all source variables, how do we handle variables we don't yet have internal attrs for? Auto-discovery from GRIB index, or manual curation?
5. **Polling interval and timeouts**: Per-dataset configuration of `poll_interval`, `max_wait_time`, `max_files_per_commit`, `max_seconds_between_commits`. Sensible defaults TBD.
6. **Reforecast/historical archives**: Some datasets (GEFS v12 reforecast) have different URL schemes for historical data. Virtual backfills need to handle these, but operational updates don't. Scope this per-dataset.
