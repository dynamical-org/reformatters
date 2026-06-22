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

### Spike 4 — mixed-dim variables in one flat group (feasibility for Decision A)

`spike4_mixed_dims_flat.py` writes a single-level and a pressure-level variable
into one **root** group (no zarr group), commits, reads back.

```
vars + dims in ONE flat group:
  temperature_2m   ('time', 'lat', 'lon')
  temperature      ('time', 'level', 'lat', 'lon')
single open_zarr, no group= needed.  temperature.sel(level=500) shape: (3, 4, 5)
```

**Finding:** real vertical dimensions don't need zarr groups. A flat dataset holds
heterogeneous-dimension variables fine, so the grouping decision (Q2) is decoupled
from the dimension decision (Q1). Drives Decision A.

### Spike 5 — model-level availability in NODD GFS (feasibility for Decision C)

`spike5_model_level_availability.py` classifies both GFS products' indexes:

```
GFS pgrb2  0p25:  pressure 16 vars x 33 levels ; hybrid/model 6 vars x [1,2]
GFS pgrb2b 0p25:  pressure 15 vars x 22 levels ; hybrid/model 0 vars
```

**Finding:** neither standard NODD GFS product carries usable model levels. A
`model_level` group/dimension is not feasible for GFS from these sources. Drives
Decision C.

---

## Synthesis & recommendation

The measurements turn most of the either/or framing into a "both, by regime":

1. **Use real vertical dimensions where the grid is dense, suffix names where it
   is sparse.** Concretely: a `pressure_level` group (and a `model_level` group
   when a source has hybrid levels) with plainly-named variables on a real `level`
   coordinate; single/surface variables stay suffix-named. Spike 3 shows this is
   the structure the data actually has — ~88% dense vs ~7% dense.

2. **Real vertical dimensions do NOT require zarr groups** (Q2). Spike 4 shows a
   single flat root group holds `temperature_2m(time, y, x)` and
   `temperature(time, pressure_level, y, x)` side by side, read back with one
   `open_zarr` (no `group=`). Spike 1 shows groups *also* work if we want them.
   So the dimension decision (Q1) and the grouping decision (Q2) are independent:
   we can get the dense `pressure_level` win in the existing single-group layout
   and treat groups as a later, optional organizational choice. See expanded
   decision A below.

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

## Expanded open decisions

These are the choices that are genuinely ours to make. Each is laid out as the
real question, the options, and a recommendation.

### Decision A — The machinery change: per-variable dims, not groups

The framing "should we use groups?" buries the actual base-class gap. Look at what
the code keys on:

- `chunk_key()` already reads geometry **per variable** from the template:
  `template_var.dims` and `template_var.encoding["chunks"]`
  (`virtual_region_job.py`). It handles a dim being present or absent in `out_loc`
  generically. **It already supports heterogeneous-dimension variables** — pass it
  a `temperature` whose template var has a `pressure_level` dim and it just works,
  as long as the source ref's `out_loc` carries the level label.
- `filter_already_present()` / `_emit_refs()` address arrays by flat `var.name` at
  the root — fine as long as variables stay in one group.
- The one real blocker is `TemplateConfig`: `update_template()` builds **every**
  variable with the same `self.dims` via
  `make_empty_variable(self.dims, coords, dtype)`, and `self.dims` is a single
  tuple. There is no way today to give `temperature` a `pressure_level` dim while
  `temperature_2m` has none.

So the decision is **per-variable dims (A1)** vs. **zarr groups + DataTree (A2)**,
and spike 4 proves A1 is sufficient for real vertical dimensions.

**A1 — per-variable dims in one flat group (recommended first step).**
Add an optional `dims` to `DataVar` (default = template `dims`), add the
`pressure_level` coordinate to `coords`, and teach `update_template` /
`empty_copy_with_reindex` to honor per-var dims. `chunk_key` needs no change.
`SourceFileCoord.out_loc()` for a pressure message must include its
`pressure_level` label; `representative_var` must pick a (var, level) the file
actually contains.
- Pros: smallest change; one `open_zarr` for readers; single-level vars are byte-
  for-byte unchanged at the root, so swap-compat with the materialized catalog is
  free; `.sel(pressure_level=500)` works; **NaN holes on sparse upper levels cost
  nothing** — a missing (var, level) chunk is simply an absent ref in a virtual
  store, never stored bytes.
- Cons: one flat namespace, so naming discipline matters (decision B); a single
  array set and manifest for the whole dataset.

**A2 — zarr groups + DataTree (defer).**
`TemplateConfig` grows per-group `dims`/`coords`/`data_vars` (e.g. `data_vars`
keyed by group, or a small per-group sub-config); `template_ds` becomes a
DataTree or a dict of datasets; `chunk_key`/filter/emit address `group/var` paths;
`get_template`, `write_metadata`, `empty_copy_with_reindex`, and `sync_dims_to`
all loop over groups; coordination/finalize runs per group.
- Pros: clean separation, per-group chunk policy, readers open only what they want,
  no name collisions.
- Cons: substantial churn across `TemplateConfig`, `RegionJob`,
  `VirtualRegionJob`, `template_utils`, `storage`; readers must know the group; the
  back-compat root either duplicates or uses aliases (decision below).

**Recommendation:** ship A1 now. It delivers the dense-pressure win with minimal
churn and preserves swap-compat. Groups can be layered on later (A2) without
invalidating A1's variable names, and only when a dataset actually needs per-group
chunking or the flat namespace becomes unwieldy (e.g. ensemble × model_level ×
pressure_level). Under A1 the "alias" idea (Q4) and "empty root" idea (Q8) mostly
dissolve: single-level vars simply *are* the root, and pressure vars *are* the
level-dim arrays in the same root — nothing to alias.

**The one case A1 can't express: a variable on two vertical-coordinate types.**
If a dataset publishes `temperature` on *both* pressure levels and model levels,
a flat group can't hold two arrays named `temperature` that differ only in their Z
dim. An array has exactly one Z axis, and a 5-D `temperature(time, pressure_level,
model_level, y, x)` is nonsense (a cell is on one or the other, never both). Three
ways out:

1. **Groups (the clean answer): the group name *is* the vertical-coordinate type.**
   `pressure_level/temperature`, `model_level/temperature`; single/surface vars stay
   at root (suffix-named). This is exactly the case A2 exists for, and it gives a
   crisp escalation rule: **use groups iff a dataset has more than one vertical
   coordinate type for the same variable; otherwise stay flat.** Spike 1 confirms
   root arrays and subgroups coexist in one store, so single-levels-at-root +
   `pressure_level/` + `model_level/` is a valid layout.
2. **Type-suffix the name** (`temperature` + `temperature_model_level`). Keeps the
   flat namespace but is asymmetric (which Z-type wins the bare name?) and reads as
   a special case — rejected.
3. **One generalized `level` dim with auxiliary `level_type`/`level_value`
   coords**, concatenating the two stacks (33 + 127 entries, *not* a cross product,
   so no NaN blow-up). Technically CF-valid, but it reintroduces the
   incomparable-axis ergonomics of the single-level MultiIndex idea below (you
   can't `.sel(pressure_level=500)` cleanly, and slicing across the pressure↔model
   boundary is meaningless) plus awkward CF round-trip — rejected for the same
   reasons.

The useful reframing: **A1 is just the single-vertical-coordinate special case of
A2.** Model the data uniformly as "each variable declares its vertical coordinate"
(`None` for single/surface, `pressure_level`, `model_level`, …); then *render*
flat-at-root when a dataset has ≤1 vertical coordinate type (for swap-compat) and
render as groups when it has more. Designing the var→vertical-coordinate
association into the config now means flat-vs-grouped is a later rendering choice in
*our code*, not a rewrite. (To be clear: re-rendering an *already-published* dataset
is still a breaking change for consumers — a new dataset version, not done lightly.
The cheap-to-change part is our implementation, so we should set the layout policy
deliberately up front.) Note this collision does **not** arise for the immediate
GFS/GEFS pressure-level work (GFS has no usable model levels — decision C), so A1
ships unblocked; the association just keeps the door open.

### Decision B — Naming convention + the written rule — **DECIDED**

Agreed. This is the naming convention of record. With A1 everything shares one
namespace, so the convention is unambiguous and uniform across providers:

- **Level-dim variables use the plain ECMWF name, no suffix:** `temperature`,
  `geopotential_height`, `relative_humidity`, `specific_humidity`, `wind_u`,
  `wind_v`, `vertical_velocity`. A plain name *always means* "indexed by a
  `*_level` dimension."
- **Single/surface variables keep today's suffix convention exactly:**
  `temperature_2m`, `wind_u_10m`, `pressure_surface`,
  `pressure_reduced_to_mean_sea_level`. A suffixed name *always means* "one fixed
  level." This is what makes `temperature` (pressure stack) and `temperature_2m`
  (2 m) coexist without ambiguity.
- **Dimension/coordinate:** `pressure_level`, values in **hPa** (matches the
  legacy `_500hpa` suffix and the idx `mb` units), CF `standard_name`
  `air_pressure`, `axis="Z"`, `positive="down"`. Pick and document an ordering
  (recommend descending, surface→top: 1000 → 1).
- **The placement rule (write it down):** a variable goes on the real
  `pressure_level` dimension iff the source publishes it across a dense set of
  pressure levels (e.g. ≥ ~80% of the union other core vars use); otherwise it
  stays suffix-named. A variable may appear **both** ways (e.g. `temperature` on
  the pressure stack *and* `temperature_2m`).
- **Sparse-upper-level handling:** use one `pressure_level` dimension spanning the
  full 33-level union; variables present only on a subset (spike 3: the 6
  mixing-ratio vars at 22 levels) simply have no ref above their top level. In a
  virtual store that is zero cost (absent chunk → fill value), so there is no
  reason to split the dimension by which vars reach which level.

**Rejected: a MultiIndex `level` dim for single levels** (e.g.
`.sel(level=("height_above_surface", "2m"))`). It doesn't help — the single-level
grid stays ~7% sparse, and in a virtual store sparsity isn't even the cost
(absent refs are free). The real objections are semantic and ergonomic: the
single-level "levels" are *incomparable* (`2 m above ground` vs `mean sea level`
vs `entire atmosphere` vs `boundary layer`), so the one thing a dimension buys —
operating *across* it (slice/interpolate/integrate) — is meaningless for them; you
always pick one named thing. `("height_above_surface","2m")` is more verbose and
less discoverable than `temperature_2m`, and a pandas MultiIndex doesn't round-trip
cleanly to Zarr/CF. The deciding factor stays **density × comparability**: pressure
is the only single regime that is both, which is why it (and only it) gets a real
dimension. Even a numeric `height_above_ground` sub-dim (2/10/80/100 m) is rejected:
the heights are comparable but the cross product is still sparse (temperature only
at 2 m, wind at 10/80/100 m), so single levels stay suffix-flat.

### Decision C — `model_level` scope (defer; provider-specific)

Spike 5 settles this for NOAA GFS: neither `pgrb2.0p25` nor `pgrb2b.0p25` on NODD
carries usable model/hybrid levels (1–2 hybrid messages total; pgrb2b has none).
GFS does not publish a full native-level 0.25° grid on NODD. So:

- **Do not design a universal `model_level` structure now.** Pressure levels are
  universally available and dense; ship `single_level (suffix) + pressure_level
  (real dim)` first.
- Treat `model_level` as **provider-specific and deferred** — HRRR native files,
  IFS model-level products, ICON native levels each differ. Evaluate per provider
  when a concrete dense-native-level source is identified. If/when one is, it is
  the strongest case for A2 groups (a second `model_level` dimension alongside
  `pressure_level` in one flat dataset is also possible but starts to strain the
  flat namespace).

### Decision D — Ensemble & manifest/chunk sizing (Q9)

This is a sizing validation, not a structural choice, but it must be checked before
committing. In a virtual store one ref = one GRIB message = one chunk, and lat/lon
are a single chunk per message, so manifest entries ≈
`#append_dim × #ensemble × #lead × #pressure_level × #vars`.

- Pressure levels multiply the ref count of core vars by ~33. For a GEFS-style
  ensemble (31 members) the pressure block alone is on the order of
  `16 vars × 33 levels × 31 members × ~81 leads ≈ 1.3M refs per init_time` — large,
  but exactly what icechunk manifest splitting is built for.
- The existing `IcechunkVirtualConfig` already splits manifests along the append
  dim (`manifest_append_dim_split`) and caps the chunk-ref cache
  (`num_chunk_refs=1_000_000`, see `storage.py`). **Action:** validate the split
  size and cache budget against the ensemble × pressure projection, and split more
  aggressively if a single append-dim manifest shard exceeds comfortable size.
- **Chunking note:** because a virtual chunk is one whole horizontal field,
  `pressure_level` is chunked **1 per level** (each level is its own message) and
  lat/lon are single-chunk. Encode `pressure_level` chunk size = 1.

### Reproducing the spikes

Scripts are committed under `docs/plans/vertical_structure_spikes/`:

```
uv run python docs/plans/vertical_structure_spikes/spike3_sparsity.py              # grid density (network: NODD idx)
uv run python docs/plans/vertical_structure_spikes/spike1_groups.py                # multi-group round-trip
uv run python docs/plans/vertical_structure_spikes/spike2_alias.py                 # virtual alias one ref -> two paths
uv run python docs/plans/vertical_structure_spikes/spike4_mixed_dims_flat.py       # mixed-dim vars in ONE flat group
uv run python docs/plans/vertical_structure_spikes/spike5_model_level_availability.py  # GFS pgrb2 vs pgrb2b level content
uv run python docs/plans/vertical_structure_spikes/spike6_hrrr_gefs_levels.py      # HRRR wrfprs/wrfnat + GEFS (real .idx)
uv run python docs/plans/vertical_structure_spikes/spike7_ecmwf_levels.py          # ECMWF IFS ENS / AIFS open-data .index
uv run python docs/plans/vertical_structure_spikes/spike8_icon_eu_levels.py        # DWD ICON-EU pressure vs model levels
uv run python docs/plans/vertical_structure_spikes/spike8b_icon_read_one_grib.py   # download+decode one ICON model-level GRIB
uv run python docs/plans/vertical_structure_spikes/spike9_mrms_levels.py           # MRMS 3D products from real bucket
```

(ECMWF/ICON spikes hit live, short-retention open-data servers — update the
hard-coded date if it has aged out.)

## Source-data vertical-level review (all 8 catalog models)

Reviewed which of our catalogued models publish **more than one dense vertical
coordinate type** in the *files we'd virtualize* (the public GRIB archives), which
is what would force a vertical-dimension/grouping decision. **All figures verified
against real source files**, not docs (docs are often stale — see
`source_data_exploration_guide.md` §5): NOAA from NODD `.idx` (spikes 3/5/6), ECMWF
from the live open-data JSON `.index` files (spike 7), DWD ICON-EU from live
directory listings plus one model-level GRIB downloaded and decoded with rasterio
(spikes 8 / 8b), MRMS from the live `noaa-mrms-pds` bucket listing (spike 9). The
real-file pass corrected one doc-sourced figure (AIFS Single is 14 pressure levels,
not 13) and otherwise confirmed the structure.

| Model | Single/surface | Pressure levels (dense) | Model/native levels (dense) | Other dense vertical | Multiple Z-types? |
|---|---|---|---|---|---|
| NOAA GFS | yes | **33** levels, 16 vars | no (1–2 hybrid msgs on NODD) | — | **No** — pressure only |
| NOAA GEFS | yes | modest (0.5° a-file ~12; s-file 0.25° none) | no | — | **No** (but see resolution heterogeneity below) |
| NOAA HRRR | yes | **39** levels, 14 vars (`wrfprs`) | **50** native levels, 20 vars (`wrfnat`) | — | **YES** — pressure + native |
| NOAA MRMS | yes (mostly 2D) | no | no | **33** height levels (0.5–19 km) on 3 vars (`MergedReflectivityQC`, `MergedRhoHV`, `MergedZdr`) | No — one height type |
| ECMWF IFS ENS | yes | **14** levels (1000→10 hPa), 9 params (d, gh, q, r, t, u, v, vo, w) | no (MARS/paid only) | 4 `sol` soil levels | **No** — pressure only |
| ECMWF AIFS Single | yes | **14** levels, 7 params (gh, q, t, u, v, w, z) | no | 2 `sol` soil levels | **No** — pressure only |
| ECMWF AIFS ENS | yes | **14** levels, 6 params (q, t, u, v, w, z) | no (only an *experimental* product) | 2 `sol` soil levels | **No** — pressure only |
| DWD ICON-EU | yes (67 vars) | **20** levels, 7 vars (clc, fi, omega, relhum, t, u, v) | **74** model levels, 10 vars (clc, p, qc, qi, qv, t, tke, u, v, w) | — | **YES** — pressure + model (both: t, u, v, clc) |

**Headline: only HRRR and ICON-EU (2 of 8) genuinely need multiple vertical-level
groups.** Five products are pressure-only (or less); MRMS has a single dense
vertical type (height), so it gets at most one vertical dimension and no
pressure-vs-model decision. (ECMWF's `sol` soil layers are a second vertical type
in principle, but only 2–4 layers on soil vars — not the same atmospheric variable
on two systems, so no group decision.) Crucially, we *know each model's answer up
front* from this review — there is no "we'll be surprised later" risk that would
argue for pre-emptively grouping everything.

## When are zarr groups actually required?

Groups solve more than the vertical-coordinate problem. There are two *independent*
triggers, and a dataset needs groups if **either** holds:

1. **Multiple dense vertical coordinate types for the same variable** (Decision A's
   multi-Z case). `temperature` on both pressure and model levels can't share one
   flat array. → HRRR, ICON-EU.

2. **A dimension whose extent varies across another dimension** (the heterogeneity
   case). A virtual chunk is a raw GRIB message at its native grid — we can't
   resample — so if the horizontal grid size changes along the append/lead axis, the
   data is not rectangular and can't live in one array. **GEFS is the concrete
   example:** its `s` files are 0.25° only through lead 240 h, after which only
   `a`/`b` at 0.5° exist (`gefs_config_models.py:15-19`). Tellingly, our one existing
   virtual GEFS dataset is `forecast_10_day_spatial` = exactly 240 h — capped at the
   resolution boundary to sidestep this. Spanning the full 35 days virtually would
   need two resolution groups (e.g. `/lead_0_240` at 0.25°, `/lead_243_840` at
   0.5°), or two separate datasets.

### Open question (for TSC): always group vertical stacks, or only on conflict?

**Decision needed.** Single/surface variables always live at the root, suffix-named.
The question is where a *vertical stack* (pressure, model, height) lives.

Fixed regardless of the choice: HRRR and ICON-EU publish `temperature` on **both**
pressure and model levels, so the bare name collides and at least one stack **must**
go in a group. This decision is really about the other six (single-vertical-type)
datasets — and thus whether "pressure-level temperature" has a *uniform* location
across the catalog.

**Option A — group only on conflict.** Single-type datasets keep the stack as a
dimension on the root; group only where two types collide.

```
GFS:   /  temperature_2m, temperature(pressure_level), …
HRRR:  /  temperature_2m, …   /pressure_level  temperature   /model_level  temperature
```
- One `open_zarr()` on GFS/IFS yields surface + pressure together; no group knowledge.
- Groups appear only where semantically needed; their presence signals "multiple
  vertical systems."
- Pressure access differs GFS (root) vs HRRR (group) — not uniform.

**Option B — always group stacks.** Root holds only single/surface vars everywhere;
every stack lives in a group named for its coordinate.

```
GFS:   /  temperature_2m, …   /pressure_level  temperature
HRRR:  /  temperature_2m, …   /pressure_level  temperature   /model_level  temperature
```
- Uniform: `pressure_level/temperature` everywhere; a dataset gaining a vertical
  type never relocates existing vars.
- GFS/IFS readers must open the `pressure_level` group (one-shot `open_zarr` gives
  only surface vars); nests 6 of 8 datasets.

Either way, single/surface vars stay at root, so materialized→virtual swap-compat is
the same for both (the suffix→dimension change for pressure breaks those few names
regardless of grouping). Shared data model: every variable declares its vertical
coordinate, so flat-vs-grouped is a rendering choice in our code and the policy can
be applied uniformly. **This is why the policy is worth deciding deliberately now:
changing the layout of an already-published dataset is a breaking change for
consumer code (a new dataset version), not something we would do lightly — even
though our implementation could re-render either way.** (Same call also covers
GEFS's shape-heterogeneity split — resolution windows as groups `/lead_0_240`,
`/lead_243_840`, vs. separate datasets.)

**Author's lean: Option A** — optimize the common, most-used single-type datasets
for one-shot discovery; introduce groups only where a real name conflict requires
them. Pending TSC input.

## Parked / next steps

Decision status and follow-ups (so they aren't lost):

- **A — machinery (per-variable dims):** *parked, to design next.* Draft the
  concrete `DataVar.dims` (optional, default = template `dims`) + `update_template`
  / `empty_copy_with_reindex` change as a small proposal / proof-of-concept diff.
  Model each variable's vertical coordinate (`None` / `pressure_level` /
  `model_level`) so flat-at-root vs. grouped is a rendering choice in our code — A1
  is the single-vertical-coordinate special case of A2. (Cheap to re-render in code;
  *not* cheap to change once published — that's a breaking version bump, so set the
  layout policy (E) deliberately up front.)
- **B — naming convention:** **DECIDED** (see Decision B). Plain ECMWF name ⇒ on a
  `*_level` dim; suffix ⇒ one fixed level; `pressure_level` in hPa, CF
  `air_pressure`, axis Z, positive down, descending; place on the real dim iff
  densely published; single levels stay suffix-flat (MultiIndex rejected).
- **C — `model_level` inclusion:** **RESOLVED — include all available vertical
  levels.** The source review shows only **HRRR (`wrfnat`, 50 levels)** and **DWD
  ICON-EU (74 levels)** publish dense model levels for the same vars as pressure;
  GFS/GEFS/ECMWF have none in their public files. Virtual HRRR and ICON-EU
  (imminent) ship single + `pressure_level` + `model_level`. C is a *data-inclusion*
  question (which vertical types are in the dataset); the *layout* it forces is
  Decision E — for HRRR/ICON-EU the bare name `temperature` collides across pressure
  and model levels, so a group is required.
- **E — group policy (layout):** **OPEN — pending TSC** (see "Open question (for
  TSC): always group vertical stacks, or only on conflict?"). HRRR/ICON-EU must
  group regardless; the call is whether single-vertical-type datasets keep the stack
  as a root dimension (Option A, author's lean) or always nest it in a group (Option
  B). Awaiting technical steering committee input.
