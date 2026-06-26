# Vertical dimension structure — implementation plan

Status: design agreed, not yet implemented. This plan is the spec for the
implementing agent.

## Goal

Support datasets that mix single-level variables with variables on a dense,
comparable vertical dimension, following the structure documented in `AGENTS.md`
("Vertical levels" under *Common dataset structures*):

- Single-level / surface variables live at the **root**, suffix-named
  (`temperature_2m`, `pressure_surface`).
- A variable on a dense, comparable set of vertical levels has **no suffix** and
  lives in a zarr **group named after its vertical dimension** — the group name and
  the dimension name are the same (`pressure_level`, `model_level`; others may be
  added). E.g. `pressure_level/temperature` has dims `(…, latitude, longitude,
  pressure_level)`.
- Dimension coordinates shared with the root (time/lead time, lat/lon,
  ensemble_member, spatial_ref) are **duplicated into each group** so a group can be
  opened on its own.
- A group with **no variables is omitted**.

First implementation targets are the two datasets that actually need it (both
virtual): **NOAA HRRR** (`wrfprs` pressure + `wrfnat` native model levels) and
**DWD ICON-EU** (pressure + model levels). All other catalog datasets are
single-level-only and must keep working unchanged.

This is design decision **E** plus its sub-decisions from the prior exploration
(closed PR #674); the rules above are the settled outcome.

## Design principles that shaped this plan

- **Declarative, group-keyed.** There is one declarative place to find a group's
  dims, keyed by group. Core never hard-codes vertical dim names; each config
  declares them.
- **Group is a per-variable property, not a job boundary.** A source file can span
  groups (HRRR `wrfprs` → root single-level *and* `pressure_level`), or be finer
  than a group (ICON publishes one file per `(var, level)`). So jobs are scoped as
  today (region × source-file var group); the group only determines, per variable,
  its zarr path, dims, and that its refs carry the vertical level.
- **Coordination stays dataset-wide.** Reader-safety (zarr-v3 deferred metadata;
  icechunk temp-branch atomic merge) is whole-dataset; group structure does not
  change worker setup/finalize counting.

---

## 1. Type scheme for groups

In `common/config_models.py`:

```python
from enum import Enum, auto
from typing import Literal

class RootGroup(Enum):
    ROOT = auto()          # pure sentinel; no usable string value
ROOT = RootGroup.ROOT

# Expand as new dense vertical coordinate types are supported.
type VerticalGroup = Literal["pressure_level", "model_level"]
type Group = VerticalGroup | RootGroup
```

Rationale / verified properties (prototyped against pydantic + `ty`):

- **Typo-safe:** a bad group key is rejected by pydantic validation, and `ty` flags
  it at type-check time (closed `VerticalGroup` literal).
- **Self-describing default:** `DataVar.group: Group = ROOT`; `repr(ROOT)` is
  `<RootGroup.ROOT: 1>`, not an opaque `None`.
- **Root cannot leak into a path.** `ROOT` is not a `str`. The path helper narrows:

  ```python
  def var_path(group: Group, name: str) -> str:
      if group is ROOT:
          return name
      return f"{group}/{name}"   # `group` narrows to VerticalGroup (str) here
  ```

  `ty` confirms that after `if group is ROOT`, the value narrows to `VerticalGroup`,
  and that passing an un-narrowed `Group` where a `str` vertical group is required is
  an error. So it is impossible to accidentally use `ROOT` as a zarr-path segment.

- **Common case reads cleanly:** `dims = {ROOT: ("init_time", "latitude", "longitude")}`.

The vertical group name string **equals** its dimension name (the `group == dim`
invariant), so `dims[g]` for a vertical group contains the dim `g`, and there is a
`Coordinate(name=g, …)` for it.

---

## 2. `TemplateConfig` encoding

### `dims` becomes group-keyed

Replace the single `dims: tuple[Dim, ...]` with:

```python
dims: dict[Group, tuple[Dim, ...]]
```

- `dims[ROOT]` is the root/single-level dims (today's `dims` value verbatim — so an
  existing single-level config migrates by wrapping: `dims = {ROOT: (...)}`).
- `dims["pressure_level"]` is the full dims of the pressure group, e.g.
  `("init_time", "lead_time", "ensemble_member", "latitude", "longitude", "pressure_level")`.
- A `all_dims` property returns the de-duplicated union (for the one core spot that
  needs every dim name — see `derive_coordinates`).

Every subclass reader of the old `self.dims` becomes group-explicit, which is the
point — e.g. a pressure var's encoding:
`chunks = tuple(var_chunks[d] for d in self.dims["pressure_level"])`.

> Migration note: this changes `dims` codebase-wide. Every existing template config
> changes `dims = (...)` → `dims = {ROOT: (...)}`, and its
> `tuple(var_chunks[d] for d in self.dims)` → `... for d in self.dims[ROOT]`. There
> are ~15 such configs (see file list). This is mechanical.

### `DataVar` gains `group`

In `config_models.py`:

```python
class DataVar(FrozenBaseModel, Generic[INTERNAL_ATTRS_co]):
    name: str
    group: Group = ROOT
    ...
    @property
    def path(self) -> str:        # zarr path / identity key
        return var_path(self.group, self.name)
```

A variable's dims are `template_config.dims[var.group]` — **not** stored on the
DataVar (no per-var dims field; nothing to disagree). The var's encoding
`chunks`/`shards` tuples must have length `len(dims[var.group])`; validate this in
`update_template`.

### `dimension_coordinates()` and `coords`

- `dimension_coordinates()` stays a flat `{dim_name: values}` but now covers
  `all_dims` (adds `pressure_level`, `model_level` values). Dim names are unique, so
  one flat dict is unambiguous.
- `coords` stays a flat `Sequence[Coordinate]`, now including a `Coordinate` for each
  vertical dim (`pressure_level`: `standard_name="air_pressure"`, `units="hPa"`,
  `axis="Z"`, `positive="down"`, descending; `model_level`: hybrid level number,
  `axis="Z"`).
- `derive_coordinates` default (line 61) uses `all_dims` instead of `self.dims`.

### Validators (all derived from the config — no core constant)

Add as model validators / asserts in `update_template`:

1. **group == added dim:** for each vertical group `g`, `tuple(d for d in dims[g] if
   d not in dims[ROOT]) == (g,)`.
2. **groups are non-empty:** every key in `dims` other than `ROOT` is used by ≥1 var
   (no orphan dim / empty group), and every `var.group` is a key in `dims`.
3. **`(group, name)` uniqueness** across `data_vars` (parameterized test below).
4. **root var name ≠ any group name** (a zarr node can't be both an array and a
   group at the same path).
5. **coords cover `all_dims`** (the existing `set(coords) == set(self.dims)` becomes
   `== set(all_dims)`).
6. **append-dim chunk size uniform across groups** (so region partitioning is shared
   — see §4).
7. **encoding tuple length** equals `len(dims[var.group])` per var.

---

## 3. Template represented as a DataTree

The multi-group template *is* a `xarray.DataTree`; use it at the template layer.

### `update_template`

Build a DataTree (root dataset + one child dataset per non-empty group), then write
the whole tree in one call:

```python
dt = xr.DataTree.from_dict({"/": root_ds, **{f"/{g}": group_ds for g in groups}})
dt.to_zarr(self.template_path(), mode="w",
           consolidated=False, write_inherited_coords=True)
```

`write_inherited_coords=True` is **required**: by default `DataTree.to_zarr` does
not duplicate shared coords into child groups (it relies on inheritance), which
breaks standalone `open_zarr(group=…)`. With the flag, each group writes its own
copy of the shared coords (verified on icechunk) — satisfying the "groups open
independently" rule while keeping the concise single-call write.

Build details:
- Partition `data_vars` by `var.group`. `make_empty_variable(self.dims[g], …)` per
  var (already takes a dims tuple).
- Each group dataset's coords = shared coords (those whose dim is in `dims[ROOT]`) +
  the group's own vertical coord. Derived coords (`valid_time`, etc.) are computed
  per group too (they depend only on shared dims).
- Assign per-var/per-coord metadata per group (today's `assign_var_metadata` loop,
  run per node).

### `get_template`

Return a `DataTree`. Reindex the append dim and re-derive coordinates in **every**
node, e.g. `dt.map_over_datasets(...)` or a per-node loop reusing
`empty_copy_with_reindex`. Signature change: `get_template(end) -> xr.DataTree`
(and the `get_template_fn` type in `operational_update_jobs`).

### What stays a Dataset

An individual `RegionJob.template_ds` stays a per-relevant-slice `Dataset`. Region
jobs read a var's geometry via `var.path` against the tree/the relevant group; they
do not need the whole DataTree threaded through the materialized Dataset operations.

---

## 4. Region jobs and source file coords

**Group is per-variable; jobs are not per-group.** A job is scoped to a region
(append-dim slice) × a source-file variable group, exactly as today. Within a job,
each variable knows its `group`, hence its `path` and `dims[group]`.

### Shared (`RegionJob`)

- `get_jobs` / `dimension_slices`: append-dim partitioning is unaffected by vertical
  dims (they are intra-variable). `dimension_slices` already indexes per var
  (`var.dims.index(dim)`); ensure it reads the var's own dims (from `dims[group]`).
  Keep the "uniform append-dim chunk size across vars" check (now validated in §2.6).
- Variable identity becomes `(group, name)` / `var.path`. Replace
  `{v.name for v in data_vars}` style lookups and the
  `set(template_ds.data_vars)` assertion with path-based equivalents over the tree.
- Rename `source_groups()` (vars sharing a source file) to avoid colliding with the
  new vertical-group concept (e.g. `source_file_var_groups()`), or document the
  distinction prominently.
- `update_template_with_results` trims per group (each group to its own max processed
  append-dim position) via the DataTree.

### Source file coords

- `out_loc()` for a variable on a vertical dim must include the **level** (exactly
  how `ensemble_member` already rides `out_loc` → `chunk_key`). A source file with one
  message per `(var, level)` expands to one ref per `(var, level)`, each ref's
  `out_loc` carrying its own `pressure_level`/`model_level`.
- A source file may span groups (HRRR `wrfprs` → root single-level + `pressure_level`):
  ref generation routes each message to its var's group path. All refs from one file
  still commit together (per-file atomicity preserved across groups).

### Virtual region job (`VirtualRegionJob`) — the main work

Make these group-aware via `var.path` + `dims[var.group]`:

- `chunk_key(out_loc, var)`: read geometry from the var's group dataset (its dims +
  chunks), not `self.template_ds[var.name]` at root.
- `_emit_refs`: `store.set_virtual_refs(var.path, specs, …)` — group-qualified array
  name (verified to work, e.g. `set_virtual_refs("pressure_level/temperature", …)`).
- `filter_already_present`: open the var's group (`zarr.open_group(store)[var.path]`)
  for the probe; `representative_var` + the probe chunk include a level.
- `_assert_probe_chunk_covered`: group-aware chunk key.
- `sync_dims_to`: grow the append dim in **every** group's duplicated coord (write the
  appended slice to root and each group node). Groups may sit at different append-dim
  lengths transiently under operational lazy growth — acceptable and eventually
  consistent.
- `_append_dim_size`: read per group (they can differ transiently).

### Operational updates (virtual) — single writer across all groups

`operational_update_jobs` still returns **one** job (single writer to `main`, no temp
branch). That one pod ingests files across **all** groups, committing per file. Do
**not** split into per-group jobs (multiple `main` writers would race). Per-group
parallelism is backfill-only (pre-sized temp branch; worker 0 pre-sizes the full
tree so every worker's `sync_dims_to` is a no-op).

### Materialized region job (follow-up, after virtual lands)

`zarr.py` (`copy_data_var`, `copy_zarr_metadata`) and `shared_memory_utils.write_shards`
build chunk paths as `f"{data_var_name}/c/…"` and write via `to_zarr(region=…)` at
root. These need group-prefixed paths (`f"{var.path}/c/…"`) and `to_zarr(group=…)`.
HRRR/ICON-EU are virtual, so this can be a second phase.

---

## 5. File-by-file change list

Core (phase 1, required for virtual):
- `common/config_models.py` — add `RootGroup`/`ROOT`/`VerticalGroup`/`Group`,
  `var_path`; `DataVar.group` + `.path`; encoding-length note.
- `common/template_config.py` — `dims: dict[Group, tuple]` + `all_dims`;
  `update_template` builds + writes a DataTree (`write_inherited_coords=True`),
  partitions vars by group, routes coords, per-group metadata + assertions;
  `get_template` returns a DataTree; `derive_coordinates` uses `all_dims`; validators.
- `common/template_utils.py` — `write_metadata` accepts/writes a DataTree
  (`dt.to_zarr(..., write_inherited_coords=True)`); `empty_copy_with_reindex` runs
  per node.
- `common/region_job.py` — path-based identity; per-var dims via `dims[group]`;
  rename `source_groups`; `update_template_with_results` per group.
- `common/virtual_region_job.py` — `chunk_key`, `_emit_refs`,
  `filter_already_present`, `_assert_probe_chunk_covered`, `sync_dims_to`,
  `_append_dim_size`, representative/probe all group-aware via `var.path`.
- `common/dynamical_dataset.py` — template typed as DataTree; `validators()` walk
  groups; operational path unchanged (single writer already).
- All ~15 existing template configs — `dims = {ROOT: (...)}` and
  `... for d in self.dims[ROOT]`; no behavior change (single group).

Materialized (phase 2):
- `common/zarr.py`, `common/shared_memory_utils.py` — group-prefixed chunk paths and
  `to_zarr(group=…)`.

New datasets (phase 3): NOAA HRRR and DWD ICON-EU virtual configs with
`pressure_level` + `model_level` groups.

---

## 6. Tests

- **New, parameterized across all template configs:** no duplicate `(group, name)`;
  no root var/coord name equal to a group name; each group's added dim == group name;
  `dims` keys ⊆ valid `Group`; coords cover `all_dims`.
- Update `tests/common/common_template_config_subclasses_test.py` and
  `tests/common/datasets_cf_compliance_test.py` to walk the DataTree (root + each
  group) rather than a single flat dataset.
- Round-trip test: a multi-group template writes and reads back with each group
  independently openable via `open_zarr(group=…)` (shared coords present) and via
  `open_datatree`.
- A virtual backfill + operational test on a small multi-group fixture (a file that
  spans root + a vertical group, and a one-file-per-`(var, level)` source) asserting
  refs land at the right `group/name` chunk and per-file atomicity holds across
  groups.

---

## 7. Out of scope / deferred

- **Names for other vertical types** (height for MRMS 3D reflectivity; soil/depth) —
  add to `VerticalGroup` when those datasets are built. The boundary rule for what
  becomes a Z dimension vs. a root single-level var is *dense + comparable* (a single
  selected level like `geopotential_height_500hpa` stays a root single-level var).
- **Non-vertical heterogeneity groups** (e.g. GEFS 0.25°/0.5° across lead time):
  these groups share dims but differ in coordinate *extent*, which `dims[group]`
  (dims-per-group) does not express — they would need per-group coordinate
  definitions, a future extension. Materialized datasets sidestep this by resampling
  to a common grid; only virtual datasets that can't resample need it.
- **Static / time-invariant fields** with fewer dims than their group (e.g.
  orography): none exist today; if needed, add an optional per-var dims override
  defaulting to `dims[group]`.

---

## 8. Verified feasibility (prototypes)

The following were confirmed against a real icechunk store (zarr 3.2, icechunk 2.0,
xarray 2026.4) during design:

- Mixed-dimensionality variables coexist; root + `pressure_level` + `model_level`
  groups write and read back, each independently openable.
- `set_virtual_refs("pressure_level/temperature", …)` works on a group-qualified
  array; the same source byte range can back two array paths.
- `DataTree.from_dict` → path-index `dt["pressure_level/temperature"]` →
  single-call `dt.to_zarr()` → `open_datatree` round-trips; `map_over_datasets`
  reindexes the append dim across nodes.
- `dt.to_zarr(write_inherited_coords=True)` is what makes each group independently
  openable (default omits child coords).
- The `Group` type scheme: pydantic accepts `dict[Group, tuple]`, rejects typo'd
  keys; `ty` narrows `is ROOT` so `ROOT` cannot be used as a path string.
