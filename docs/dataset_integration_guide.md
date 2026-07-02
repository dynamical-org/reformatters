# Dataset Integration Guide

Integrate a dataset to reformat into Zarr.

## Overview

Integrating a dataset in dynamical.org `reformatters` is done by subclassing a trio of base classes, customizing their behavior based on the unique characteristics of your dataset.

There are three core base classes to subclass.

1. `TemplateConfig` defines the dataset **structure**.
1. `RegionJob` defines the **process** by which a region of that dataset is reformatted: **downloading, reading, rewriting.**
1. `DynamicalDataset` brings together a `TemplateConfig` and `RegionJob` and defines the compute resources to operationally update and validate a dataset.

### Words

- **Provider** - the agency or organization that publishes the source data. e.g. ECMWF
- **Model** - the model or system that produced the data. e.g. GFS
- **Variant** - the specific subset and structure of data from the model. e.g. forecast, analysis, climatology. Variant may include any other information needed to distinguish datasets from the same model.
- **Dataset** - a specific provider-model-variant. e.g. noaa-gfs-forecast

### Materialized vs. virtual datasets

Before you start, decide which of two kinds of dataset you are building. The choice picks which `RegionJob` base class you subclass and changes a handful of steps below; everything else is shared.

- **Materialized** (the common case) — download source files, rechunk, and rewrite the bytes into a new Zarr/Icechunk store. Chunks are tuned for time-series extraction. The `RegionJob` subclass is a `MaterializedRegionJob` (download → read → transform → write). Most datasets are materialized.
- **Virtual** — an Icechunk store whose chunks are *references* `(location, offset, length)` pointing into the provider's source files in place, decoded at read time by a per-variable serializer. Chunks follow the native source-message shape, so a virtual dataset serves map/spatial queries, includes every curated variable cheaply, and updates within seconds of publication. The `RegionJob` subclass is a `VirtualRegionJob`. See [docs/virtual_datasets.md](virtual_datasets.md) for the full design.

A virtual dataset typically *complements* a materialized one over the same source data — for example the virtual `noaa-gefs-forecast-10-day-spatial` alongside the materialized GEFS time-series. When in doubt, build materialized.

Each kind has a heavily-commented teaching template you will copy from and a real worked reference to study:

| | Teaching template (copied by `initialize-new-integration`) | Real worked reference |
|---|---|---|
| Materialized | `src/reformatters/example/` + `tests/example/` | any registered dataset, e.g. `src/reformatters/noaa/gfs/forecast/` |
| Virtual | `src/reformatters/example_virtual/` + `tests/example_virtual/` | `src/reformatters/noaa/gefs/forecast_10_day_spatial/` |

This guide points you at those files at each step rather than duplicating their code.

## Integration steps

Before getting started, follow the brief setup steps in README.md > Local development > Setup.

### 0. Explore the source dataset

Explore the source dataset to understand the nuances of what's available and how to access it. See [docs/source_data_exploration_guide.md](source_data_exploration_guide.md).

### 1. Initialize a new integration

```bash
uv run main initialize-new-integration <provider> <model> <variant> --kind <materialized|virtual>
```

Provider, model and variant can contain letters, numbers and dashes (e.g. ICON-EU or analysis-hourly). Capitalization will be normalized for you.

`--kind` is **required** and selects which teaching template is scaffolded — `materialized` copies from `src/reformatters/example/`, `virtual` from `src/reformatters/example_virtual/`. Omitting it errors:

```
Missing option '--kind'. Choose from: materialized, virtual
```

This adds a number of files within `src/reformatters/<provider>/<model>/<variant>` and `tests/<provider>/<model>/<variant>`.

These files contain placeholder implementations of the subclasses referenced above, commented with guidance specific to the kind you chose. Follow the rest of this doc to complete the implementations and integrate your new dataset.

### 2. Register your dataset

Add an instance of your `DynamicalDataset` subclass to the `DYNAMICAL_DATASETS` constant in `src/reformatters/__main__.py`:

```python
from reformatters.provider.model.variant import ProviderModelVariantDataset

DYNAMICAL_DATASETS = [
    ...,
    ProviderModelVariantDataset(
        primary_storage_config=ProviderModelIcechunkAwsOpenDataDatasetStorageConfig(),
]
```

If you plan to write this dataset to a location not maintained by dynamical.org, you can instantiate and pass your own `StorageConfig`, contact feedback@dynamical.org for support. A virtual dataset's storage config has extra requirements covered in step 5.

### 3. Implement `TemplateConfig` subclass

Work through `src/reformatters/$DATASET_PATH/template_config.py`, setting the attributes and method definitions to describe the structure of your dataset. The report generated by following the [source_data_exploration_guide.md](source_data_exploration_guide.md) will be helpful here.

Most of the template is shared between materialized and virtual datasets: declare `dims`, `append_dim`, `append_dim_start`, `append_dim_frequency`, then implement `dataset_attributes`, `coords`, `data_vars`, `dimension_coordinates()`, and optionally `derive_coordinates()`. Coordinates are small materialized arrays in both kinds — their encoding is identical.

Read the [chunk/shard layout tool](./chunk_shard_layout_tool.md) docs and use the tool to find chunk and shard sizes for your data variables.

Using the information in the `TemplateConfig`, `reformatters` writes the Zarr metadata for your dataset to `src/reformatters/$DATASET_PATH/templates/latest.zarr`. Run this command in your terminal to create or update the template based on your `TemplateConfig` subclass:

```bash
uv run main $DATASET_ID update-template
git add src/reformatters/$DATASET_PATH/templates/latest.zarr
```

Tracking the template in git lets us review diffs of any changes to the structure of our dataset. Regenerate it after any metadata change.

Run the tests, making any changes necessary.

```bash
uv run pytest tests/$DATASET_PATH/template_config_test.py
```

#### Virtual: the divergence is entirely in `data_vars` encoding

For a virtual dataset, the only part of the template that differs is the **data variable encoding** (`src/reformatters/example_virtual/template_config.py` isolates this; coords are copied verbatim from a materialized dataset). Two rules:

1. **One chunk per source message.** Each virtual chunk is exactly one source message, so chunk size is `1` along every per-message dim (`init_time`, `lead_time`, and `ensemble_member` if present) and full-width along the message's spatial dims, with `shards=None`. The geometry must match the source file layout exactly — the write loop uses it to place each reference.
2. **Never re-encode the bytes.** No compressors or filters of your own (`compressors=()`, `filters=()`); instead a per-variable `serializer` (a Zarr v3 `ArrayBytesCodec`, e.g. `GribberishCodec`) decodes the raw message at read time. Set `dtype` to whatever that codec produces to avoid a cast.

Two consequences for metadata:

- **The grid must be the source's native grid** — one chunk per message means no regridding is possible (a materialized dataset is free to choose any output grid). The grid must also match the *serializer's* decode orientation, which is not necessarily the materialized/GDAL dataset's: `GribberishCodec` decodes some projected grids (e.g. HRRR) bottom-up (row 0 = south) where GDAL is top-down, so a virtual template that reuses a materialized dataset's `y`/`latitude`/`spatial_ref` verbatim would store the data upside-down relative to its coordinates. Decode one real message and compare to the coordinates before trusting a reused grid; flip the `y` ordering (and the `spatial_ref` GeoTransform) to the serializer's native order if they differ.
- **Served values are the raw source values, unless a *pointwise* codec can reproduce the materialized transform.** A transform that is per-chunk pointwise belongs in the encoding codec pipeline so the virtual dataset stays a drop-in for the materialized one: chain a `zarr.codecs.ScaleOffset` array→array filter for K → °C (it runs *after* `GribberishCodec` decodes the raw Kelvin), and pass `GribberishCodec(var=..., adjust_longitude_range=True)` for the 0–360 → −180–180 longitude convention. For these vars match the materialized name/units exactly. A transform that spans chunks (deaccumulating precip to a rate, temporal differencing) *cannot* be done on read: drop the derived variable and carry only the raw source field under the name/units of the *raw* quantity (e.g. `total_precipitation_surface`, a window accumulation, not `precipitation_surface`, a rate). See `_RAW_VALUE_NAME_OVERRIDES` / `_RAW_VALUE_ATTRS_OVERRIDES` in `src/reformatters/noaa/gefs/forecast_10_day_spatial/template_config.py` and the `ScaleOffset` filter in `src/reformatters/noaa/hrrr/forecast_48_hour_spatial/template_config.py`.

#### Vertical groups (root + `pressure_level` / `model_level`)

A dataset that mixes single-level variables with variables on a dense vertical dimension is a Zarr group hierarchy: single-level/surface variables live at the root (level in the name, e.g. `temperature_2m`), and a variable on a dense set of levels lives in a group named after its vertical dimension (`pressure_level/temperature`). In the template this means:

- `dims` is keyed by group: `dims = {ROOT: (...)}` plus one entry per vertical group whose dims are the root dims with that dimension appended (`dims["pressure_level"] = (..., "pressure_level")`). The group name equals its dimension name.
- Each group's `DataVar`s set `group=...`; a variable's `path` (its Zarr identity) becomes `"<group>/<name>"`. Shared dimension coordinates are duplicated into each group so a group opens standalone.

A variable's group is a per-variable property — it sets that variable's path and dims, not a job boundary, so a single source file can span the root and a vertical group. Lean on the references rather than restating the design: see "Vertical levels" in [AGENTS.md](../AGENTS.md#common-dataset-structures) for the user-facing rules, [docs/virtual_datasets.md](virtual_datasets.md#multi-group-vertical-datasets) for the full design (type scheme, group-keyed `dims`, structural validators, DataTree representation), and `tests/common/virtual_multi_group_test.py` for a worked multi-group virtual dataset. Multi-group is currently exercised only by virtual datasets.

### 4. Implement `RegionJob` subclass

This is where materialized and virtual diverge most. Work through `src/reformatters/$DATASET_PATH/region_job.py`. Both kinds share one method: `generate_source_file_coords` lists every source file required to process the data covered by a region (`processing_region_ds`). It is reused by operational validation, so it must list exactly the files the dataset expects (drop files the source genuinely lacks, e.g. accumulated variables at hour 0). Both kinds also implement `operational_update_jobs` — the factory that returns the `RegionJob`s needed to bring the dataset up to date. You can skip `operational_update_jobs` until you're ready to implement dataset updates; a backfill runs without it.

Write tests for any custom logic you create.

```bash
uv run pytest tests/$DATASET_PATH/region_job_test.py
```

#### Materialized: download → read → write

A `MaterializedRegionJob` rewrites the source bytes. Beyond `generate_source_file_coords` there are two more required methods (see `src/reformatters/example/region_job.py`):

- `download_file` retrieves a specific source file to local disk.
- `read_data` loads data for one variable from a local path and returns a numpy array.

The base class drives the download → read → transform → write-shards pipeline from these. Optional overrides — `source_file_var_groups` (variables that share a source file), `get_processing_region` (buffer the region for deaccumulation/interpolation), `apply_data_transformations` (deaccumulation, etc.), `update_template_with_results` (trim to actually-processed data) — are described in the example; implement only what your dataset needs and delete the rest to use the base implementations.

Once `generate_source_file_coords`, `download_file`, and `read_data` are in place you can run the reformatter locally:

```bash
uv run main $DATASET_ID backfill-local <append_dim_end> --filter-variable-names <data var name>
```

Reformatting locally can be slow. Choosing an `<append_dim_end>` not long after your template's `append_dim_start` and selecting a single variable with `--filter-variable-names` limits the work.

#### Virtual: the core three (`generate_source_file_coords` / `discover_available` / `file_refs`)

A `VirtualRegionJob` is source-agnostic: the base owns the write loop, atomic commits, lazy append-dim growth, chunk-index resolution, and unreadable-file skipping. You fill in a small seam — most of a virtual dataset is three methods, run in order to drive an update (see `src/reformatters/example_virtual/region_job.py` and the real `src/reformatters/noaa/gefs/forecast_10_day_spatial/region_job.py`):

1. `generate_source_file_coords(processing_region_ds, data_var_group) -> [coord]` — list every source file this job covers (shared role, above).
2. `discover_available(pending) -> [(coord, file_size)]` — of the files not yet ingested, the subset fetchable right now, each with its data-file size. For an object store obstore can list (S3/GCS/Azure/local), this is one line via `discover_available_by_obstore_listing(...)`; a source obstore can't list (an HTML directory index, a frontier to probe, "assume every coord is available") implements it directly.
3. `file_refs(coord, file_size) -> [VirtualRef]` — given one available file, every pointer it contributes (a `VirtualRef` is a source byte range → one output chunk), or `[]` to skip it. Resolve byte ranges however the source allows: parse a sidecar `.idx` index, scan the data file, or point at the whole file for one-message files. Refs are in coordinate-label space; the chunk index is resolved centrally.

Between steps 1 and 2 the base runs `filter_already_present` to drop files already in the manifest; steps 2–3 then repeat each tick until everything expected is ingested.

The accompanying `SourceFileCoord` subclass identifies one source file and provides `get_url()` (the canonical data-file URL refs point at — it must start with the registered virtual chunk container prefix, see step 5), `out_loc()` (the `{dim: label}` output cell the file fills — override the default to exclude non-dim fields like a `data_vars` list, or to fold init+lead into a single `time` for an analysis dataset), and optionally `get_index_url()` (the `.idx` sidecar, only if the source publishes one).

**The yield is the commit unit.** `file_refs`'s output flows through the base write loop, which commits each yielded batch of `(source file, its refs)` atomically — a reader on `main` sees either none or all of a file's data, within seconds. Never split a file across yields and never yield nothing (an empty Icechunk commit raises). The manifest, not a coordinate value, is the source of truth for what's already ingested. See [The write loop](virtual_datasets.md#the-write-loop) and [What a `VirtualRegionJob` implementer writes](virtual_datasets.md#what-a-virtualregionjob-implementer-writes).

`operational_update_jobs` for a virtual dataset returns a single job over a recent append-dim window, constructed with `processing_mode="update"` so the write loop *polls* (keeps sweeping `discover_available` until everything publishes) rather than sweeping once. It does not read existing coordinates — the manifest already records what's present (`primary_store` is unused).

A virtual backfill runs once the core three are in place; it uses the standard parallel temp-branch flow on a pre-sized template. See [Backfill](virtual_datasets.md#backfill-parallel-on-a-pre-sized-temp-branch).

### 5. Implement `DynamicalDataset` subclass

`DynamicalDataset` brings the `TemplateConfig` and `RegionJob` together with storage and operational config. Declare `template_config` and `region_job_class`, configure `primary_storage_config` (and any `replica_storage_configs`), and implement `operational_kubernetes_resources()` and `validators()`. Work through `src/reformatters/$DATASET_PATH/dynamical_dataset.py`.

To operationalize your dataset and have the `update` and `validate` Kubernetes cron jobs deployed automatically by GitHub CI, implement `operational_kubernetes_resources()`.

#### Materialized: parallel update, storage of any format

Materialized updates and backfills fan out across Kubernetes indexed jobs. Kubernetes resource values:
  - shared memory: Round the value calculated in the chunk/shard size tool output up to the nearest half GB.
  - memory: 1.5x shared memory.
  - cpu: the number of spatial dimension shards minus 1 to account for kubernetes headroom. e.g. if 2 latitude shards * 4 longitude shards = 8, choose 7 cpu to schedule on an 8 cpu node.
  - ephemeral_storage: 20GB is a good starting point.

Parallelism: Set `workers_total` and `parallelism` on the `ReformatCronJob` using `self.num_variable_groups()`. Multiply by 2 if `operational_update_jobs` reprocesses the most recent time slice (see GEFS datasets for examples).

The update cron schedule should run shortly after the source data is expected to be available and the validate cron should run at `update cron start + update pod_active_deadline`.

#### Virtual: ICECHUNK storage + `icechunk_virtual_config`, single-writer update

A virtual dataset requires every storage config to use the `ICECHUNK` format and a `DynamicalDataset.icechunk_virtual_config` (validated at construction). It declares one `icechunk.VirtualChunkContainer` per source bucket — whose prefix must match `SourceFileCoord.get_url()` — and a `manifest_append_dim_split` that splits each array's chunk-reference manifest along the append dimension so an operational commit only rewrites the active split. Pick the split size by the operational commit-cost ceiling (small enough to rewrite well under a second); see [Manifest splitting](virtual_datasets.md#manifest-splitting). The real example is `src/reformatters/noaa/gefs/forecast_10_day_spatial/dynamical_dataset.py`; the commented teaching version is `src/reformatters/example_virtual/dynamical_dataset.py`.

Virtual operational updates are **single-writer**: one pod commits whole files straight to the Icechunk branch as they arrive — no `workers_total`/`parallelism` fan-out. `operational_kubernetes_resources` defines one update pod (e.g. `cpu="1.7"`, `memory="7G"`) that polls through the source's publication window and exits when the manifest is complete, plus the validate pod. Bound the update pod's `pod_active_deadline` well under the gap between fires so runs never overlap. If an update falls far behind, catch up with a backfill rather than scaling to multiple writers. See [Operational updates: single writer](virtual_datasets.md#operational-updates-single-writer) and [parallel_processing.md](parallel_processing.md#virtual-icechunk-operational-updates-single-writer-exception).

#### Validators

Both kinds return validation functions from `validators()`. Materialized datasets list generic `XarrayDataValidator`s that read the opened dataset (e.g. `check_forecast_current_data`, `check_analysis_recent_nans`).

A virtual dataset mixes those generic validators with two virtual-specific ones (see `validation.py` and the GEFS example):

- `CheckVirtualManifestCompleteness` — re-runs the operational filter over the recent window to assert references *exist* (the virtual analog of materialized `check_for_expected_shards`).
- `CheckVirtualDecodeHealth` — decodes a bounded sample of the references that exist to confirm the serializer and virtual-container authorization work end to end.

Tuning for both is set where they're listed in `validators()`. See [Operational validation](virtual_datasets.md#operational-validation).

#### Integration test with snapshot values

In `dynamical_dataset_test.py` create a test that runs `backfill_local` followed by `update` for a couple data variables and a minimal number of time steps, lead times and ensemble members. Include snapshot value assertions for every data variable that the test processes — check specific known values at specific coordinates (e.g. `assert_allclose(point["temperature_2m"].values, [28.75, 29.23])`). Snapshot values catch silent regressions in data reading, unit conversion, or coordinate alignment that other tests miss. (For a virtual dataset the snapshot is the decoded *raw* source value — see step 3 — and the end-to-end virtual flow is exercised by `tests/common/virtual_region_job_test.py` and `tests/common/virtual_multi_group_test.py`.)

For a **materialized** dataset, wrap the trimmed template in `tests.chunk_utils.shrink_chunks_and_shards` in your test's `get_template` monkeypatch (see existing dataset tests for examples). (A **virtual** dataset skips this: its chunks are already one-per-message — chunk size 1 on the per-message dims and full-width spatial — and shrinking the spatial chunks would break the one-message-per-chunk invariant. Trim with `.isel`/`.sel` only, as `noaa/gefs/forecast_10_day_spatial` and `noaa/hrrr/forecast_48_hour_spatial` do.) At test scale, production chunk geometry means writing thousands of nearly empty chunks, which dominates test runtime without adding coverage. The helper reads each var's existing chunks/shards and the trimmed sizes and shrinks them automatically, keeping each dimension's production chunk and shard counts (capped at two) so multi-chunk and multi-shard dims stay multi and the corresponding write paths stay exercised. Call it *after* trimming (after any `.sel(...)`) so it sees the sizes the test actually writes; pass `dims=[...]` to shrink only some dimensions (e.g. leave the append dim at production size for a shard-boundary test).

```bash
uv run pytest tests/$DATASET_PATH/dynamical_dataset_test.py
```

### 6. Deploy

The details here depend on the computing resources and the Zarr storage location you'll be using. Get in touch with feedback@dynamical.org for support at this point if you haven't already. Operational monitoring and troubleshooting live in [docs/ops_card.md](ops_card.md).

1. Run a backfill on your local computer: `DYNAMICAL_ENV=prod uv run main $DATASET_ID backfill-local <append-dim-end>`. If this is fast enough and you have the disk space, it is a nice and simple approach.
1. If you're working to create a public dynamical.org dataset, run `./deploy/aws/create_new_aws_open_data_bucket.sh <provider>-<model>`
1. Run a backfill on a kubernetes cluster:
   - This supports parallelism across servers to process much larger datasets.
   - Complete the steps in README.md > Deploying to the cloud > Setup.
   - `DYNAMICAL_ENV=prod uv run main $DATASET_ID backfill-kubernetes <append-dim-end> <jobs-per-pod> <max-parallelism>`, then track the job with `kubectl get jobs`.
   - Virtual backfill only: suspend the dataset's `-update` CronJob for the duration and choose `<append-dim-end>` as the last *fully published* position (see [Backfill](virtual_datasets.md#backfill-parallel-on-a-pre-sized-temp-branch)).
1. See operational cronjobs in your kubernetes cluster and check their schedule: `kubectl get cronjobs`.
1. To enable issue reporting and cron monitoring with the error reporting service Sentry, create a secret in your kubernetes cluster with your Sentry account's DSN: `kubectl create secret generic sentry --from-literal='DYNAMICAL_SENTRY_DSN=xxx'`.

## 7. Validate

Follow [docs/validation.md](validation.md) — it walks through running `run-all`, reading `validation_summary.md`, inspecting every plot, and the full data quality checklist.

## 8. Update dataset catalog documentation

Update the dataset catalog docs on `dynamical.org` by adding entries into the `catalog.js`, rebuilding (`npm run build`), and merging updates to main in `https://github.com/dynamical-org/dynamical.org`.
