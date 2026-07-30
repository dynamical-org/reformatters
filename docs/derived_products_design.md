# Derived products: design

Status: proposal, not yet implemented.

A derived product is a dataset we compute from one of our own published archives rather
than from provider source files. The two motivating examples:

- **Ensemble statistics**: read an ensemble forecast archive (e.g. ECMWF IFS ENS, GEFS),
  reduce along `ensemble_member`, write a dataset with a `statistic` dimension
  (mean, standard deviation, percentiles, ...) in its place.
- **Climatologies**: read an analysis (or forecast) archive, reduce over years, write a
  dataset with `day_of_year` (or hour-of-year) in place of `time`/`init_time`.

Out of scope: statistics or climatologies the *provider* publishes (e.g. GEFS `geavg`/
`gespr` files). Those are ordinary materialized datasets read from source files — the
existing machinery already supports them (`GefsStatisticSourceFileCoord`, the
`ensemble_statistic` DataVar attr, the `statistic` entry in `Dim`) and nothing here
applies.

## Design principles

1. **A derived product is a full `DynamicalDataset`.** Own variant directory under
   `<provider>/<model>/` (e.g. `ecmwf/ifs_ens/forecast_15_day_statistics/`), own
   `dataset_id`/`name` following the standard `<provider>-<model>-<variant>` convention,
   own checked-in template, registered in `__main__.py`, operated by the standard
   backfill / update / validate machinery. No parallel pipeline, no special CLI.

2. **The source archive is just another source.** `RegionJob` already models
   "enumerate source units → fetch → read → transform → write". A derived product swaps
   "file at a URL" for "slab of an upstream zarr store"; everything downstream of
   `read_data` (shared memory, rounding, shard writes, parallel coordination, icechunk
   branch commits) is inherited from `MaterializedRegionJob` unchanged. Derived products
   always compute new bytes, so they are always materialized datasets.

3. **Structure sharing is explicit config transformation, not inheritance.** The derived
   `TemplateConfig` instantiates the source's `TemplateConfig` and transforms its
   `coords`/`data_vars` with `reformatters.common.pydantic.replace`. The checked-in
   template diff remains the review surface, exactly as for every other dataset.

## New common machinery

### 1. `DerivedDynamicalDataset`: binding to the source dataset

The one thing a derived product needs that no dataset has today is a handle on its
source's store. Storage wiring lives in `__main__.py`, where the source instance already
exists — so the derived dataset takes it as a field:

```python
class DerivedDynamicalDataset(DynamicalDataset[DATA_VAR, SOURCE_FILE_COORD]):
    source_dataset: DynamicalDataset[Any, Any]
```

```python
# __main__.py
ifs_ens_forecast = EcmwfIfsEnsForecast15Day025DegreeDataset(...)
DYNAMICAL_DATASETS = [
    ifs_ens_forecast,
    EcmwfIfsEnsForecast15DayStatisticsDataset(
        primary_storage_config=...,
        source_dataset=ifs_ens_forecast,
    ),
    ...
]
```

Holding the whole `DynamicalDataset` (not just a `StoreFactory`) gives access to the
source's `template_config` (append dim, structure) and `store_factory.primary_store()`,
which resolves the correct path and credentials per environment (prod/dev/test) for free.

To get the source store into region jobs, add one small, general seam to the base
`DynamicalDataset` rather than special-casing derived products at each call site:

```python
class DynamicalDataset:
    def region_job_kwargs(self) -> dict[str, Any]:
        """Extra constructor kwargs this dataset supplies to its region jobs."""
        return {}
```

`backfill`, `backfill_kubernetes` (job counting), and `update` pass
`**self.region_job_kwargs()` through `RegionJob.get_jobs(...)` /
`operational_update_jobs(...)` into the `cls(...)` constructor call. For every existing
dataset this is a no-op. `DerivedDynamicalDataset` implements it as:

```python
def region_job_kwargs(self) -> dict[str, Any]:
    return {"source_store": self.source_dataset.store_factory.primary_store()}
```

Alternatives considered and rejected:
- A classvar on the derived `RegionJob` subclass binding a source dataset instance at
  import time — duplicates the storage-config choice that `__main__.py` owns.
- Adding `source_store` to the base `RegionJob`/`operational_update_jobs` signatures —
  pushes a derived-only concern onto every dataset.

`DerivedDynamicalDataset` also overrides nothing else. One operational detail: the
kubernetes `secret_names` for its update/validate cron jobs must include the source
store's read secrets (`self.source_dataset.store_factory.k8s_secret_names()`) in
addition to its own.

### 2. `DerivedRegionJob` + `SourceRegionCoord` (`common/derived_region_job.py`)

```python
class SourceRegionCoord(SourceFileCoord):
    """Identifies one slab of the source store to read and reduce.

    Subclasses add selector fields (init_time, lead_time block, spatial tile, ...)
    and implement source_loc() (indexer into the source dataset) alongside the
    usual out_loc() (indexer into the output dataset).
    """
```

`get_url()` returns the source store URL plus the selector — it flows into
`SourceFileResult.url` so update results and logs record exactly what was read.

```python
class DerivedRegionJob(MaterializedRegionJob[DATA_VAR, SOURCE_FILE_COORD]):
    source_store: Store  # injected via region_job_kwargs

    @cached_property
    def source_ds(self) -> xr.Dataset:
        return xr.open_zarr(self.source_store, chunks=None, decode_timedelta=True)
```

What it overrides, and what subclasses implement:

- **No download step.** `download_file(coord)` becomes an availability check: assert
  the coord's source region is present *and complete* in `source_ds`, returning no path.
  Raising `FileNotFoundError` for not-yet-ingested regions reuses the existing
  `DownloadFailed` handling verbatim, including quiet logging for recent append coords
  and reprocessing on the next operational update.
- **Completeness hook.** `source_region_complete(coord) -> bool`, called by the default
  `download_file`. A source update may be mid-flight; deriving from a partially written
  region would bake NaNs into the output. Default for forecast sources: the coord's
  `ingested_forecast_length == expected_forecast_length` at its init times. Analysis
  sources: the source's append coordinate covers the coord (optionally minus a lag
  buffer). Per-product overrides as needed.
- **`read_data(coord, data_var)`** (subclass, as today): select the coord's slab from
  `self.source_ds` and reduce it — along `ensemble_member` for statistics, over years
  grouped by day-of-year for climatologies. The returned array includes the new
  dimension (`statistic`, ...) and is written into the shared output buffer by the
  inherited pipeline. Reductions over NaN-padded regions must respect the source's fill
  semantics (e.g. `nan*` reducers vs. propagating missing members as missing statistics —
  a per-product decision that belongs in `read_data`).
- **`generate_source_file_coords`** (subclass, as today): choose slab granularity. This
  is the memory/IO knob: each coord's input slab (e.g. all 51 members for one
  `init_time` × one `lead_time` chunk) must fit in memory, and slabs should align with
  the source's chunk/shard boundaries to avoid read amplification.
- **`operational_update_jobs`** (base provides a default for same-append-dim products):
  today every implementation ends the template at "what the provider has published";
  for a derived product that becomes "what the source store has completely ingested".
  Default: start from the derived store's max append coordinate (reprocess it, it may
  have been incomplete), end at the source store's append end, `get_jobs(...,
  filter_start=...)`. This state-based diff makes update scheduling self-correcting —
  no event coupling to the source's update job is needed.

Known limitation, unchanged from the rest of the codebase: the materialized write path
is not yet group-aware, so derived products of `pressure_level`/`model_level` variables
wait on materialized multi-group support.

### 3. Template config pattern: transform the source's config

No new `TemplateConfig` base class. The derived config imports and instantiates the
source's config (pure config, no storage required) and builds its own declarations from
it:

```python
_source = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()

class IfsEnsStatisticsTemplateConfig(TemplateConfig[EcmwfIfsDataVar]):
    dims = {ROOT: ("init_time", "lead_time", "statistic", "latitude", "longitude")}
    append_dim = _source.append_dim
    append_dim_start = _source.append_dim_start        # or a later start
    append_dim_frequency = _source.append_dim_frequency

    def dimension_coordinates(self) -> dict[str, Any]:
        source_coords = _source.dimension_coordinates()
        return {
            **{d: source_coords[d] for d in self.dims[ROOT] if d in source_coords},
            "statistic": STATISTICS,  # ("mean", "standard_deviation", ...)
        }

    @property
    def coords(self) -> Sequence[Coordinate]:
        keep = {c.name for c in self.all_dims_and_derived_coords}
        return (
            *(c for c in _source.coords if c.name in keep),
            statistic_coordinate(STATISTICS),
        )

    @property
    def data_vars(self) -> Sequence[EcmwfIfsDataVar]:
        return [
            replace(var, encoding=replace(var.encoding, chunks=..., shards=...))
            for var in _source.data_vars
        ]
```

Points of standardization, so every derived product labels things identically:

- **Shared coordinate factories** in a new `common/derived_coordinates.py`:
  `statistic_coordinate(values)`, `day_of_year_coordinate()`. One blessed set of
  statistic names (spelled out: `"mean"`, `"standard_deviation"`, ..., matching the
  repo's no-abbreviations naming convention) and one blessed day-of-year convention
  (1–366, Feb 29 handling documented on the coordinate's attrs).
- **Variable identity is preserved**: derived vars keep the source vars' names, `attrs`,
  and `keep_mantissa_bits` conventions. What changes is dims/encoding, plus CF
  aggregation metadata: `cell_methods="realization: mean"` etc. cannot be expressed
  per-variable when statistics live on a dimension, so the `statistic` coordinate values
  carry the meaning; climatology vars get `cell_methods="time: mean within days
  time: mean over years"` style attrs which *are* per-variable.
- **Provenance**: add optional `DatasetAttributes` fields set by all derived products,
  e.g. `derived_from_dataset_id` and `derived_from_dataset_version`, so readers (and the
  validation report) can trace the input archive.

## Enablers (small gaps in current machinery)

1. **String coordinates.** `Encoding.dtype` has no string dtype, and the `statistic`
   dimension coordinate wants string values (`"mean"`, ...). Add zarr v3 variable-length
   string support to `Encoding`/template machinery. (Integer codes + `flag_meanings`
   would avoid this but is strictly worse for readers.)

2. **Non-datetime append dim (climatology only).** `day_of_year` as the job-partition
   dim breaks two assumptions: `Dim`/`AppendDim` literals, and
   `TemplateConfig.append_dim_coordinates` returning `pd.date_range`. The underlying
   mechanics are already agnostic: `get_jobs` partitioning, sorted-coordinate binary
   search, and shard math work on any monotonic coordinate. Proposal: recognize that
   `append_dim` conflates *the dim jobs partition along* (every dataset needs one) with
   *the dim the dataset grows along operationally* (only appendable datasets have one).
   A climatology is a **fixed-domain dataset**: `day_of_year` has all 366 values from
   day one, `append_dim_start`/`append_dim_frequency` don't apply, `get_template`
   ignores `end_time`, and the datetime-typed `filter_start/end/contains` options simply
   don't apply. Concretely: add `"day_of_year"` to the dim literals, let
   `append_dim_coordinates` be overridden to return a fixed integer index, and assert
   fixed-domain datasets are never operationally appended.

3. **Climatology "updates" are rewrites, not appends.** As the reference period rolls
   forward (e.g. trailing 30 years, refreshed annually), every value in the archive
   changes. That operation already exists: a re-backfill with `--overwrite-chunks`
   (docs/backfill.md), which is atomic-to-readers via the icechunk temp branch. v1
   climatologies are backfill-only with scheduled/manual refresh; wiring a cron that
   runs a periodic overwrite backfill can come later if refreshes prove frequent enough
   to automate.

## Operational model

- **Scheduling**: the derived update cron runs on the same cadence as the source's,
  offset to after the source's update + validation typically complete. Because
  `operational_update_jobs` diffs derived-store state against source-store state, a
  race just means the work happens on the next run.
- **Validation**: standard validators apply as-is. Add one shared
  `common/validation.py` validator for derived products: recompute a small random
  sample of output values from the source store and compare — cheap, and it guards the
  whole read-reduce-write path end to end.
- **Backfill economics**: an ensemble-statistics backfill reads the source archive
  roughly once per variable-group pass; a climatology backfill reads every time step of
  the source for each variable. Reads go through the source's primary store; run
  backfills in-cluster/in-region as usual.

## What a concrete product looks like

```
src/reformatters/ecmwf/ifs_ens/forecast_15_day_statistics/
├── __init__.py
├── dynamical_dataset.py   # DerivedDynamicalDataset subclass: template_config,
│                          #   region_job_class, validators, cron resources
├── template_config.py     # transform of the source template config (above)
├── region_job.py          # DerivedRegionJob subclass: generate_source_file_coords,
│                          #   read_data (reduce along ensemble_member)
└── templates/latest.zarr/
```

`dataset_id="ecmwf-ifs-ens-forecast-15-day-statistics"`, registered in `__main__.py`
with `source_dataset=` the IFS ENS forecast instance.

## Phasing

1. **Enablers**: string coordinate dtype; `region_job_kwargs` seam;
   `DerivedDynamicalDataset`, `DerivedRegionJob`, `SourceRegionCoord`;
   `statistic_coordinate` helper; provenance attrs.
2. **First product**: ensemble statistics for one ensemble forecast dataset — same
   append dim as its source, so the full operational update/validate machinery applies
   without touching the append-dim machinery.
3. **Climatology enablers**: fixed-domain/non-datetime append dim support,
   `day_of_year_coordinate`; first climatology as backfill-only.
