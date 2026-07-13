# NOAA HRRR forecast, 18 hour, virtual — feasibility & implementation plan

Plan for adding an **18-hour** HRRR virtual (spatially-chunked Icechunk) forecast dataset,
modeled on the shipped `noaa-hrrr-forecast-48-hour-virtual` (PR #688; spatial→virtual
rename #731). Archive start is decided: **HRRR v3, `2018-07-13T12:00`**.

## Verdict

**Highly feasible, low risk.** The 18-hour product draws from the *same* bucket, grid, and
variable set as the 48-hour virtual dataset; every read-time codec, `.idx` quirk,
orientation decision, and validator carries over unchanged. Two config differences make it
*simpler* than the model it copies. The work is a small, well-bounded set of edits plus one
shared-code extraction. The only genuine judgement call is the hourly operational-update
design (§6).

## Decisions locked

| | |
|---|---|
| Archive start | HRRR v3, `2018-07-13T12:00` (parity with 48-hour; pre-v3 out of scope) |
| Init cadence | `append_dim_frequency = 1h` — **all 24 hourly cycles** (vs the 48-hour's 4 synoptic) |
| Leads | `0h–18h`, 19 leads (vs 49) |
| `expected_forecast_length` | **constant 18h** for every init — the v3/v4 branch is deleted |
| Structure | shared HRRR-virtual base + thin per-length subclasses (§3) |
| Manifest splits | `{pressure_level: 450, model_level: 400, None: 6000}` — adopt up front (§5) |
| Store | `dataset_version="0.1.0"`, primary-only `NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig` (bucket `s3://dynamical-noaa-hrrr`, already provisioned; no replica, no Source Coop) |

## Source GRIBs

Unchanged from the 48-hour virtual dataset — same `s3://noaa-hrrr-bdp-pds` (NODD,
`us-east-1`, anonymous) `VirtualChunkContainer`, `conus`, `hrrr.tHHz.wrf{sfc,prs,nat}fFF.grib2`
+ `.idx` (skip `wrfsubh`). The only source-scope change is **init hours 00–23 (all 24)** and
**leads f00–f18**.

Empirically confirmed against the live bucket: non-synoptic cycles are structurally
identical to synoptic ones — `t01z`/`t13z`/`t17z`/`t23z` publish all three products + `.idx`
through f18 with byte-identical `.idx` message counts (173 `wrfsfc` / 711 `wrfprs` / 1136
`wrfnat` at f06, matching `t00z`), and f19 does not exist. So the 18-hour variable set equals
the 48-hour set restricted to leads ≤ 18 — nothing is lost, and the existing definitions,
`grib_element_alternatives`, and hour-0 rules apply verbatim.

## 1. What differs from the 48-hour virtual (the whole delta)

Six field values plus one deletion; the ~1,900 lines of variable/coordinate/encoding/factory
definitions are reused verbatim.

- `append_dim_frequency`: `6h` → `1h`.
- `dimension_coordinates()` `lead_time`: `timedelta_range("0h","18h",freq="1h")` (19 leads).
- `expected_forecast_length`: **constant 18h**. Delete the `HRRR_V4_FIRST_INIT` /
  `EXPECTED_FORECAST_LENGTH_V3` / `np.where` branch; `derive_coordinates` returns
  `np.full(ds["init_time"].size, EXPECTED_FORECAST_LENGTH.to_timedelta64())` with
  `EXPECTED_FORECAST_LENGTH = 18h`. That branch only ever governed the synoptic cycles'
  f36→f48 extension, which f18 never touches (verified: v3 synoptic cap is f36, extended to
  f48 at the 2020-12-02T12Z init — orthogonal to f18).
- `coords`: `expected_forecast_length` stats become `min == max == "0 days 18:00:00"`;
  `valid_time` stats max text → `"Present + 18 hours"`.
- `dataset_attributes`: `dataset_id="noaa-hrrr-forecast-18-hour-virtual"`,
  `dataset_version="0.1.0"`, `name="NOAA HRRR forecast, 18 hour, virtual"`,
  `time_resolution="Forecasts initialized every hour"`,
  `forecast_domain="Forecast lead time 0-18 hours ahead"`.
- `append_dim_coordinate_chunk_size()` auto-scales from the frequency (32,120 → 192,720) —
  no manual edit; the checked-in template stays single-chunk.

Everything else — `dims` (identical ROOT/`pressure_level`/`model_level`), `append_dim`,
`append_dim_start`, the 142 root + 14 pressure + 20 model = **176** data variables, the
factories, filters, `PRESSURE_LEVELS`/`MODEL_LEVELS`, `WindowKind`, `_raw_idx_element` — is
reused as-is. Orientation (gribberish south-first / ascending `y`), the one-to-many
accumulation-window match at f1, hour-0 exclusions, and the version-boundary review notes all
carry over unchanged.

## 2. Reuse the shared-base refactor, don't clone

The 48-hour and 18-hour configs differ only in the values in §1; the ~1,900 lines of variable
definitions and the region job are identical. Cloning would leave two ~2,000-line files to
maintain in lockstep — the anti-pattern the code-style rule ("leave one way of doing things")
calls out. Instead extract a shared base, matching the repo's existing flat-shared-module
convention at `noaa/hrrr/` (`template_config.py`, `region_job.py`, `hrrr_config_models.py`).

## 3. The refactor seam

**New `src/reformatters/noaa/hrrr/forecast_virtual_template_config.py`** — the catalog +
base class. Move verbatim out of `forecast_48_hour_virtual/template_config.py`: the constants
(`PRESSURE_LEVELS`, `MODEL_LEVELS`, `_GRID_NY`/`_GRID_NX`, the filters, `WindowKind`,
`_WINDOW_ATTRS`, `_raw_idx_element`), the factories (`_virtual_encoding`, `_data_var`,
`_root_var`, `_pressure_var`, `_model_var`), and the builders (`_root_data_vars`,
`_pressure_data_vars`, `_model_data_vars`). Add base class
`NoaaHrrrForecastVirtualTemplateConfig(NoaaHrrrCommonTemplateConfig)` carrying the shared
`dims`, `append_dim`, `append_dim_start`, `data_vars`, `coords`, `dimension_coordinates`, and
`derive_coordinates`. Per-length hooks each leaf declares: `forecast_length: Timedelta`
(drives the `lead_time` span and the `"Present + N hours"` / expected-length stat strings via
`whole_hours(self.forecast_length)`), `append_dim_frequency`, `dataset_attributes`, and
`_expected_forecast_length_values(ds)` / `_expected_forecast_length_statistics()` defaulting
to a flat `self.forecast_length`. `data_vars` routes through a `_catalog_data_vars()` hook so
a variant can serve a subset of the full catalog (see §9) while the grid assert stays in the
base.

**New `src/reformatters/noaa/hrrr/forecast_virtual_region_job.py`** — move the `_S3_*`
constants and `_REPRESENTATIVE_LEVEL`; rename `NoaaHrrrForecast48HourVirtualSourceFileCoord`
→ **`NoaaHrrrForecastVirtualSourceFileCoord`** (one shared, fully generic coord — no
per-dataset subclass) and `NoaaHrrrForecast48HourVirtualRegionJob` →
**`NoaaHrrrForecastVirtualRegionJob`** base (the `generate_source_file_coords` /
`discover_available` / `file_refs` / `_message_lookup` / `operational_update_jobs` bodies move
unchanged). Replace the hard-coded `pd.Timedelta("14h")` re-sweep window with a
**`operational_update_window: ClassVar[Timedelta]`** classvar (must be a classvar —
`operational_update_jobs` is a classmethod the framework calls via `cls`).

**Keep the concrete 48-hour classes in `forecast_48_hour_virtual/`** as thin subclasses
(`NoaaHrrrForecast48HourVirtualTemplateConfig` sets `forecast_length=48h`,
`append_dim_frequency=6h`, keeps its v3/v4 override; `NoaaHrrrForecast48HourVirtualRegionJob`
sets `operational_update_window=pd.Timedelta("14h")`). This preserves the 48-hour public
class names, its `templates/latest.zarr` path (`template_path()` keys off the concrete
subclass module), and its registry/test imports. Its `dynamical_dataset.py` repoints imports
to the shared region-job module and updates the generic param to
`NoaaHrrrForecastVirtualSourceFileCoord`; a few test imports of `PRESSURE_LEVELS`/coord
classes repoint to the shared modules.

**Byte-identity gate (hard check).** The refactor rebuilds the 48-hour coord stat strings
(`"Present + 48 hours"`, `str(36h)`/`str(48h)`, `init_time` isoformat) via f-strings in the
base, so an imperfect reconstruction would drift the generated bytes. After the refactor run
`uv run main noaa-hrrr-forecast-48-hour-virtual update-template` and require
`git diff --exit-code` **empty** on `forecast_48_hour_virtual/templates/latest.zarr`; treat
any diff as an extraction bug, never commit it. Confirm the two snapshot tests
(`test_update_template_matches_existing_template`, round-trip) pass. Expected outcome: **no
48-hour template regeneration** — the values are unchanged, only their defining module moves.
The region-job extraction is runtime-only and carries zero template risk, so review can focus
narrowly on those coord literals.

The `DynamicalDataset` layer stays per-dataset — 48-hour and 18-hour genuinely diverge on
cron schedule, manifest split, deadline, and validator age, so a shared dataset base would be
a hooks-heavy abstraction re-split on first divergence.

## 4. New 18-hour package

`src/reformatters/noaa/hrrr/forecast_18_hour_virtual/`:

- `__init__.py` — re-export `NoaaHrrrForecast18HourVirtualDataset` (mirror the 48-hour).
- `template_config.py` — `NoaaHrrrForecast18HourVirtualTemplateConfig(NoaaHrrrForecastVirtualTemplateConfig)`
  with `forecast_length=18h`, `append_dim_frequency=1h`, the §1 `dataset_attributes`, and **no**
  expected-length override (the base constant-18h default is correct).
- `region_job.py` — `NoaaHrrrForecast18HourVirtualRegionJob(NoaaHrrrForecastVirtualRegionJob)`
  with `operational_update_window` per §6.
- `dynamical_dataset.py` — `NoaaHrrrForecast18HourVirtualDataset` with the §5 manifest split,
  the §6 cron/validators, and the same `s3://noaa-hrrr-bdp-pds` `VirtualChunkContainer`.
- `templates/latest.zarr/` — generated by
  `uv run main noaa-hrrr-forecast-18-hour-virtual update-template` (fresh; no snapshot to match).

`src/reformatters/__main__.py` — import the dataset and add
`NoaaHrrrForecast18HourVirtualDataset(primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig())`
next to the 48-hour virtual entry. Store path resolves to
`s3://dynamical-noaa-hrrr/noaa-hrrr-forecast-18-hour-virtual/v0.1.0.icechunk`. The
`.github/workflows/manual-create-job-from-cronjob.yml` options are regenerated by the prek
hook (`generate_manual_workflows.py`), not hand-edited.

**Guards confirmed:** `DynamicalDataset._validate_virtual_storage` requires an
`IcechunkVirtualConfig` and all-ICECHUNK storage (both satisfied); the materialized-multi-group
guard is in the non-virtual branch and never fires for a `VirtualRegionJob`, so the
`pressure_level`/`model_level` groups are fine.

## 5. Manifest splits

Adopt **`manifest_append_dim_split(split_size={r"^/pressure_level/": 450, r"^/model_level/": 400, None: 6000}, dim="init_time")`** at store creation. Derived by holding the 48-hour's
proven manifest byte-sizes constant while the lead count drops 49→19 (levels 39/50 unchanged),
so every split scales by 49/19 ≈ 2.58, rounded slightly down (the safe direction — smaller
manifests):

| group | refs/init (18h) | split | full manifest @ ~16.4 B/ref | budget |
|---|---|---|---|---|
| single-level (catch-all `None`) | 19 | 6000 | ~1.8 MiB | ≤ 3 MiB/var |
| `pressure_level` | 19×39 = 741 | 450 | ~5.2 MiB | 5–8 MiB |
| `model_level` | 19×50 = 950 | 400 | ~5.9 MiB | 5–8 MiB |

All groups sit far above the ~1000-refs/manifest zstd-compression floor. Total manifest count
over the v3 archive (~70k hourly inits) `M ≈ 7,400` (vs the 48-hour's ~3,100) — the vertical
groups dominate `M` here (fine vertical splits × 1h cadence), inverting the general
"single-level dominates" heuristic; the lever if commit cost needs cutting is coarsening
`pressure_level`/`model_level` (both have budget headroom), not the catch-all.

Splits are **immutable at store creation** (changing them means a full re-backfill), so these
are chosen conservatively rather than tuned live. Because the 18-hour refs point at
byte-identical URL structure to the 48-hour, the ~16.4 B/ref figure holds a priori — a
re-measure via `repo.list_manifest_files(snapshot_id)` on a staging store is optional
confirmation, **not** a gate that would force re-backfilling production. (Even a 50% B/ref
surprise keeps every group within budget.)

## 6. Operational updates

At hourly cadence the update pod can't run the 48-hour's 1h40m poll (fires would overlap), so
the design must change. Two coherent options, with a latency-vs-ops-noise tradeoff:

- **(A) Race the in-flight init (mirrors the 48-hour, recommended default).** `append_dim_end
  = now`; the `:50` fire polls the current init's f0–f18 as they publish and exits when
  ingested — data visible within seconds of each file publishing (the virtual dataset's core
  value). Empirically f18 completes init+1h16m (typical) to init+1h41m (worst off-synoptic,
  6-cycle sample), i.e. 26–51 min after a `:50` fire. Set `pod_active_deadline ≈ 55m` (covers
  the worst observed with margin, stays under the 60-min gap; the CronJob concurrency policy
  backstops any overlap). Keep the re-sweep window **moderate (~6h)** so a few missed hourly
  runs self-heal while a genuinely-stuck older file isn't re-polled every fire; longer outages
  are recovered by a manual backfill (the documented escape hatch). Residual downside: NODD
  publish stalls > ~50 min cause deadline-kills, and at 24 fires/day that is more Sentry
  job-failure noise than the 48-hour's 4/day (self-healing, not corrupting).
- **(B) Lag the window (escalation if (A)'s alert volume is unacceptable).** `append_dim_end =
  now − ~2h` so the pod only ever ingests fully-published inits, exits in minutes, and takes a
  **wide** re-sweep window (~12h) for strong self-heal with no deadline tension. Cost: data
  visibility lags ~1–2h instead of seconds.

Parametrize the shared region-job base so the choice is a one-line classvar
(`operational_update_window`, plus an optional `operational_update_lag` defaulting to `0h` for
the 48-hour) — start on **(A)** and switch to **(B)** only if deadline-kill alerts prove
noisy. This is the single item worth a maintainer sign-off, since it trades the low-latency
property against operational quiet.

Concrete (A) config:
- `ReformatCronJob` `schedule="50 * * * *"`, `pod_active_deadline=timedelta(minutes=55)`,
  `operational_update_window=pd.Timedelta("6h")`, `cpu="1.5"`, `memory="3.7G"`.
- `ValidationCronJob` `schedule="48 * * * *"` (just after the update's deadline, before the
  next fire), `pod_active_deadline=timedelta(minutes=15)`.
- `validators()`: `check_forecast_current_data(max_latest_init_time_age=timedelta(hours=5))`
  (roomy enough to absorb an occasional deferred init and avoid alert fatigue),
  `CheckVirtualManifestCompleteness(min_present_fraction=(0.0, 0.0, 1.0))` (the two newest
  window positions may be incomplete — the current-hour init is still publishing; older
  positions must be complete; the tuple length ≤ window positions is required),
  `CheckVirtualDecodeHealth()` (defaults; `positions="latest"` targets the newest *present*
  init).

## 7. Tests

Mirror `tests/noaa/hrrr/forecast_48_hour_virtual/` into
`tests/noaa/hrrr/forecast_18_hour_virtual/` (`__init__.py`, `template_config_test.py`,
`region_job_test.py`, `dynamical_dataset_test.py`):

- Group-count assertions stay 142/14/20 (identical var set); `test_levels_exclude_pseudo_level`,
  one-chunk-per-message, K→°C filter, north-first checks port unchanged.
- `region_job_test.py`: template fixture end-time to an hourly value (first init is
  `append_dim_start + 1h = 2018-07-13T13:00`); the operational-update-window assertion changes
  to the new position count; snapshot init `2024-06-01T00:00 @ f6` stays valid (≤ f18).
- `dynamical_dataset_test.py`: snapshot decode values port **verbatim** (same source file,
  same decode — e.g. `temperature_2m = 20.752984619140648` at `2024-06-01T00:00` f6); change
  the k8s-resources deadline assertion to `< timedelta(hours=1)`.
- The registry-driven `tests/common/common_template_config_subclasses_test.py` and
  `datasets_cf_compliance_test.py` auto-cover the new dataset once registered — **no allowlist
  edits** (CF exemptions are keyed by variable/short/long name, not dataset_id, and the var set
  is identical).

Run `uv run main noaa-hrrr-forecast-18-hour-virtual update-template`, then
`uv run pytest tests/common/common_template_config_subclasses_test.py tests/common/datasets_cf_compliance_test.py`
+ the new package tests, then `uv run ruff format && uv run ruff check --fix && uv run ty check`.

## 8. Backfill & rollout

Standard virtual backfill (see `docs/virtual_datasets.md`): suspend the `-update` CronJob for
the duration, set `append_dim_end` to the last *fully published* init (not "now"), run at the
**~10-worker** practical ceiling (contiguous worker assignment is already the `VirtualRegionJob`
default). Expect **~2.3×** the 48-hour backfill wall-clock — source-file count is 24×19×3 =
1,368/day vs 588/day (~70k inits vs ~11.6k), and backfill is bounded by `.idx` file count.
Use `docs/staging.md` to run the new version alongside without disturbing the 48-hour dataset.

## 9. Materialized-mirror 48-hour variant (production-lag experiment)

A third consumer of the seam: a 48-hour virtual dataset restricted to the **materialized
`noaa-hrrr-forecast-48-hour`'s variable set**, to measure the production-lag difference
between a downsized product and the full-variable one. The refactor supports it as a thin
subclass trio — no new framework capability:

- **Template config**: subclass the 48-hour config and override `_catalog_data_vars()` to
  name-filter the full catalog. All of the materialized product's 26 variables are `wrfsfc`
  root variables; 24 map to the virtual catalog by identical name. The two deaccumulated
  rates have no on-read equivalent (virtual chunks can't deaccumulate across leads) and map
  to their raw-window counterparts: `precipitation_surface` → `total_precipitation_surface`
  (1h accumulation) + `precipitation_rate_surface` (native PRATE), and `snowfall_surface` →
  `total_snowfall_run_total_surface` (ASNOW run total). ~27 variables, all single-level.
  Because no variable uses the vertical groups, the subclass also declares root-only `dims`
  and drops the `pressure_level`/`model_level` coords from `coords` (a declared group with
  no variables is rejected by template validation).
- **Region job**: `operational_update_window = 14h` — identical to the full 48-hour so the
  two lag series are directly comparable.
- **Dataset**: same cron schedule/deadline as the full 48-hour (again, for comparability);
  a single uniform manifest split — the full 48-hour's catch-all `2400` applies verbatim
  (same 49 refs/init/array, ~1.8 MiB full manifests), and with only ~30 arrays total
  manifest count `M ≈ 150`, negligible.

`generate_source_file_coords` already derives the product files from the variables
(`{v.internal_attrs.hrrr_file_type ...}`), so an sfc-only variable set automatically polls
**only `wrfsfc`** — no `wrfprs`/`wrfnat` listings or index downloads. The experiment
therefore isolates exactly the two mechanisms that could separate the products' lag:

1. **Per-lead publication wait** — the mirror's lead is visible once `wrfsfc` lands, where
   the full product's lead completes only when the slowest of sfc/prs/nat lands.
2. **Per-tick commit cost** — commit work scales with array count (manifest list
   serialization + active-split rewrites): ~30 arrays vs 176.

Both stores poll the same `wrfsfc` files, so running the experiment alongside the full
dataset adds only trivial duplicate `.idx` fetches. Measurement needs nothing new: the
virtual write loop logs per-batch phase timings (discover/build/emit/commit), icechunk
snapshots carry timestamps and `rebase_attempts`, and NODD object `LastModified` gives the
source-side publication time.

Decisions taken: the dataset id is **`noaa-hrrr-forecast-48-hour-virtual-fast`** (a
reduced-variable virtual variant appends `-fast`, now the documented convention in
AGENTS.md), and both precipitation forms are included (`total_precipitation_surface` +
`precipitation_rate_surface`). The variable subset is derived from the materialized
config at template-build time with an assert on unmapped names, so a variable added to
the materialized product forces an explicit mirror update rather than silently drifting.

## 10. Optional pre-implementation verification

The constant-18h assumption is empirically backed at the archive-start day and spot-checked
across the archive (f18 present, f19 absent on non-synoptic cycles). Residual risk is low
(`expected_forecast_length` is a coordinate value; only an *actual* short run — not an outage —
would make a flat-18h value wrong, and HRRR non-synoptic cycles have run to f18 since v3).
Optional cheap confirmation before deleting the v3/v4 branch: probe one non-synoptic cycle per
archive year for `wrfsfcf18.idx` present / `f19` absent.

## 11. Effort and risk

**Effort:** moderate — the shared-base extraction + migrating the 48-hour onto it, plus the
thin 18-hour package and tests. A few days.

**Risk:** low. No new source, codec, or framework capability; the write loop, atomic commits,
validators, and manifest-splitting config are exercised by a production dataset. The two things
to get right are mechanical and gated: the 48-hour template byte-identity (the `git diff
--exit-code` gate, §3) and the operational cadence (§6, observable on first fires and switchable
via one classvar). The manifest splits are chosen a priori from byte-identical URL structure,
not tuned live.

## Open decisions for maintainers

- ~~The operational-update design (§6)~~ — **decided: (A)**, race-the-init for
  seconds-latency; the lagged-window fallback (B) stays one classvar away if deadline-kill
  alerts prove noisy at 24 fires/day.
- ~~The materialized-mirror variant (§9)~~ — **decided and implemented**:
  `noaa-hrrr-forecast-48-hour-virtual-fast`, with both precipitation forms included.
