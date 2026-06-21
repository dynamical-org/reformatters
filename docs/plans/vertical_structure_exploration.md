# Vertical-dimension structure for virtual Icechunk datasets — exploration

Status: exploration / decision input. Not a committed design.

## Why this is on the table now

Our catalogued datasets today are **materialized** (rechunked, copied bytes) and
each one stores a hand-picked **selection of single-level variables**, with the
level folded into the variable name (`temperature_2m`, `wind_u_100m`,
`geopotential_height_500hpa`). Copying bytes made every extra variable and every
extra level cost real storage, so the selection stayed small and flat.

Virtual datasets (see [docs/virtual_datasets.md](../virtual_datasets.md)) change
the economics. A virtual store is just `(location, offset, length)` pointers into
the GRIB messages already sitting in NOAA's open-data buckets. The store is tiny
regardless of how many variables or levels it references. So there is suddenly no
storage reason to *not* include every variable, every pressure level, and every
model level a source publishes. That forces a structural question we never had to
answer while everything was a flat single-level selection:

**How do we lay out a dataset that now has genuine vertical dimensions?**

This doc breaks the dictated brain-dump into discrete questions, lays out the
trade-offs, and reports results from runnable code spikes
(`docs/plans/vertical_structure_spikes/`, reproduced inline below).

## What the code today assumes

A precondition for everything below: the current machinery assumes **one flat
zarr group per dataset**. Concretely:

- `TemplateConfig` exposes a single `dims: tuple[Dim, ...]`, a single flat
  `data_vars` sequence, and a single `coords` sequence
  (`src/reformatters/common/template_config.py`).
- `get_template()` does `xr.open_zarr(self.template_path())` — root group only.
- `VirtualRegionJob.chunk_key()` resolves geometry from `self.template_ds[var.name]`
  and `filter_already_present` opens `zarr.open_group(store)[var.name]` — both
  assume the var lives at the root (`src/reformatters/common/virtual_region_job.py`).
- No dataset anywhere passes `group=` to `to_zarr`/`open_zarr`, and nothing uses
  `DataTree` (confirmed by full-repo survey).

So any "groups" option is a real change to these base classes, not just config.
The spikes below test whether the *storage layer* (icechunk + zarr + xarray)
supports what we'd need before we invest in that.

---

## The open questions

### Q1. Suffix-naming vs. an explicit vertical dimension

- **Option A (status quo):** keep flattening — every (variable, level) pair is its
  own 1-D-over-append-dim array named `temperature_500hpa`, `temperature_2m`, …
- **Option B:** introduce real `pressure_level` / `model_level` dimensions; the
  variable is named plainly (`temperature`) and indexed by level.

Trade-offs:

| | Suffix-name (A) | Real dimension (B) |
|---|---|---|
| Reader ergonomics | `ds.temperature_500hpa` | `ds.temperature.sel(level=500)`, vectorized across levels |
| Cross-level ops (lapse rate, integrals) | manual, name-parsing | natural array math |
| Sparsity tolerance | perfect — only existing pairs exist | a level a var lacks becomes a NaN hole |
| Back-compat with current catalog | identical | breaking |
| CF-conventions friendliness | poor (level in name) | good (level coord with `Z` axis) |

The deciding factor is **density of the (variable × level) grid**, measured in Q3.

### Q2. Should level-type live in zarr groups?

Different vertical coordinate types are genuinely different dimensions: a pressure
variable is indexed by `[time, pressure_level, y, x]`, a model-level variable by
`[time, model_level, y, x]`, a single-level variable by `[time, y, x]`. You cannot
put all three in one flat group without either three separate `level` dims or
forcing everything onto one axis.

Proposed grouping: `single_level`, `pressure_level`, `model_level` (and possibly
`surface`, `ensemble`…), each present only if it has ≥1 variable.

Trade-offs: groups give each level-type its own clean dimension and let a reader
open exactly the slice they want; the cost is more machinery (Q's precondition)
and that a reader must know which group a variable is in.

### Q3. How sparse is the (variable × level) cross product?

This is the empirical crux. Measured against a real GFS `pgrb2.0p25.f000` index
(696 GRIB messages) — see spike results below.

### Q4. "Virtual alias": back-compat names at root pointing to a canonical location

Because refs are just pointers, the *same* GRIB message can be referenced from two
array paths: a back-compat `temperature_2m` at the root **and** a canonical
`single_level/temperature_2m` (or `pressure_level` `temperature.sel(level=…)`).
This is like a replica, but within one store's group tree. Trade-off: you keep a
drop-in-compatible root surface and a clean organized surface simultaneously, at
the cost of doubling the manifest ref count for aliased variables (no byte
duplication). Tested in spike 2.

### Q5. Swap-compatibility with the existing materialized catalog

If a new virtual dataset keeps the current root + suffix names, it is a drop-in
swap for the materialized version — existing readers keep working. The
counter-argument: *now*, while these products are still young, is the cheapest
time to make breaking structural changes.

### Q6. Migration of the existing ~10–11 products

Maintain the old-structure materialized products **and** publish new-structure
virtual versions in parallel? The append-dim/operational machinery makes running
both cheap-ish (virtual updates are a single small pod). Or cut over hard.

### Q7. A consistent rule vs. a known-ahead-of-time layout

Either commit to a rule ("anything on pressure/model levels lives in the
`pressure_level`/`model_level` group with a real level dim; single levels stay
suffix-named") so structure is predictable across datasets, or hard-code each
dataset's known set of (ensemble × model_level × pressure_level) combinations.

### Q8. Ever-any-variables-at-root, or everything in named groups?

Philosophical consistency: is the root always empty (every variable lives in a
named group), or do we keep root variables for the common single-level case?

### Q9. Ensemble interaction

`ensemble_member` is an independent dimension that multiplies into all of the
above. It is orthogonal to level-type and does not change the grouping decision,
but it does matter for chunk/manifest sizing.

---

## Code spikes (runnable, results inline)

### Spike 3 — grid density (answers Q3, drives Q1/Q2)

`spike3_sparsity.py` pulls a real GFS index and classifies every message.

```
total GRIB messages in file: 696
  462  pressure (mb)
  227  single/other
    7  hybrid (model)

PRESSURE-LEVEL: 16 distinct params across 33 levels
  full grid 16 x 33 = 528 ; actual 462  => density 87.5%
  10 params present at ALL 33 levels (HGT, TMP, RH, SPFH, VVEL, DZDT, UGRD, VGRD, ABSV, O3MR)
  6  params present at 22 levels (cloud/precip mixing ratios)

SINGLE-LEVEL: 64 params across 52 distinct 'level' strings
  full grid 64 x 52 = 3328 ; actual 227  => density 6.8%
```

**Finding:** the two regimes could not be more different.

- **Pressure levels are dense (~88%, core vars 100%).** A real `pressure_level`
  dimension is the right structure — it wastes almost nothing and unlocks
  vectorized vertical math. This validates Q1-Option-B *for pressure levels*.
- **Single levels are sparse (~7%).** The 52 "levels" are heterogeneous, mostly
  incomparable surfaces (`surface`, `mean sea level`, `2 m above ground`,
  `planetary boundary layer`, `0-0.1 m below ground`, cloud layers…). Forcing
  these onto one `single_level` axis would be ~93% NaN and semantically wrong
  (the axis values aren't a coordinate you'd ever `.sel()` across). This validates
  Q1-Option-A *for single levels*: keep the suffix names.

The data says the answer is **not** "pick A or B globally" — it's **B for dense
vertical coordinates (pressure, model), A for the sparse grab-bag of single
levels.** Which is exactly the per-group split Q2 proposes.

### Spike 1 — multi-group icechunk round-trip (feasibility for Q2)

`spike1_groups.py` writes a root var + a `/pressure_level` group (extra `level`
dim) + a `/single_level` group into one icechunk store, commits, reads back.

Result: works cleanly. Each group reads via `xr.open_zarr(store, group=…)`, and
`xr.open_datatree(store, engine="zarr")` returns the whole hierarchy at once:

```
Group: /
    temperature_2m  (time, lat, lon)
├── Group: /pressure_level
│       temperature  (time, level, lat, lon)   level = [1000 850 500 250]
└── Group: /single_level
        temperature_2m, wind_u_100m  (time, lat, lon)
```

Notable bonus: **shared coordinates (time/lat/lon) declared at the root are
inherited** by child groups in the DataTree view — they don't have to be
duplicated per group. One small gotcha: `open_datatree` needs an explicit
`engine="zarr"` because the installed `gribberish` backend confuses xarray's
engine autodetection.

### Spike 2 — virtual alias: one source ref at two paths (feasibility for Q4)

`spike2_alias.py` registers a local virtual-chunk container, then sets the
**same** `VirtualChunkSpec` (location/offset/length) on both `temperature_2m`
(root) and `single_level/temperature_2m` (group), commits, reads both.

```
root values == source: True
group values == source: True
root == group (same bytes, one source): True
```

Result: aliasing works end-to-end. Both paths decode to the identical source
bytes. **Cost:** two manifest ref entries instead of one (the ref count, and thus
manifest size, doubles for each aliased variable); the underlying source bytes are
not duplicated. For a back-compat root surface over a few dozen single-level vars
this is cheap; aliasing whole pressure-level stacks would meaningfully grow the
manifest and is probably not worth it.

---

## Synthesis & recommendation

The measurements turn most of the either/or framing into a "both, by regime":

1. **Use real vertical dimensions where the grid is dense, suffix names where it
   is sparse.** Concretely: a `pressure_level` group (and a `model_level` group
   when a source has hybrid levels) with plainly-named variables on a real `level`
   coordinate; single/surface variables stay suffix-named. Spike 3 shows this is
   the structure the data actually has — ~88% dense vs ~7% dense.

2. **Group by level-type** (Q2). It's the only layout that gives each regime its
   natural dimensionality, and spike 1 shows the storage stack fully supports it
   (write per group, read per group or as a DataTree, coords inherited from root).
   The real cost is the base-class work in Q's precondition, not storage.

3. **For back-compat, prefer a root-level "single_level"-style surface that
   reuses today's exact names** so a new virtual dataset can swap in for a
   materialized one (Q5). Whether the back-compat names are *aliases* into a
   canonical group (Q4) or simply *are* the single-level group laid out at the
   root is an open sub-decision — note that for single levels the canonical
   location and the back-compat location are the same names, so a separate
   `single_level/` group would be pure duplication. The alias trick earns its
   manifest cost mainly if we want a clean empty root *and* legacy names.

4. **On "ever any variables at root" (Q8):** the pragmatic reading of spikes 1+3
   is to **keep single-level variables at the root** (that's both the back-compat
   surface and where the sparse grab-bag naturally lives) and **put only the
   dense, real-dimension stacks in named groups** (`pressure_level`,
   `model_level`). That keeps the simple case simple and reserves groups for where
   they add structure. A fully-empty-root, everything-in-groups rule is cleaner in
   theory but buys little and breaks swap-compat.

5. **Migration (Q6):** running old materialized + new virtual in parallel is
   feasible; recommend doing that for the existing products rather than a hard
   cutover, and making the new virtual versions swap-compatible at the root.

### Open decisions to settle before designing the base-class changes

- **Group naming & rule (Q7):** lock the exact group names and a written rule for
  which variables go where, so it's uniform across providers.
- **Root = single_level, or aliases (Q4 vs Q8):** do single-level vars simply live
  at root, or does the root hold aliases into a `single_level/` group?
- **Per-group `dims`/`coords`/`data_vars` in `TemplateConfig`:** how to express a
  multi-group template (e.g. `data_vars` keyed by group, or a small per-group
  sub-config) and how `chunk_key`/`filter_already_present` address
  `group/var`-pathed arrays.
- **`model_level` scope:** GFS pgrb2 exposes only a handful of hybrid messages;
  decide whether model levels come from a different product (e.g. `pgrb2b`/native)
  before committing to the group.
- **Ensemble sizing (Q9):** confirm manifest/chunk sizing with `ensemble_member ×
  pressure_level` for the GEFS-style case.

### Reproducing the spikes

Scripts are committed under `docs/plans/vertical_structure_spikes/`:

```
uv run python docs/plans/vertical_structure_spikes/spike3_sparsity.py  # grid density (network: NODD idx)
uv run python docs/plans/vertical_structure_spikes/spike1_groups.py    # multi-group round-trip
uv run python docs/plans/vertical_structure_spikes/spike2_alias.py     # virtual alias one ref -> two paths
```
