# BARRA2 source data exploration

Exploration performed 2026-08-19 against the live NCI THREDDS holdings of the BOM BARRA2
collection (NCI project `ob53`), following [source_data_exploration_guide.md](../source_data_exploration_guide.md).
Every structural claim below was read out of real source files (OPeNDAP `.dds`/`.das`/`.ascii`
or a downloaded NetCDF); claims taken from provider documentation are labelled as such.

## There is no "BARRA-O2"

This exploration was requested for "BARRA-O2". No such product exists. The `source_id`
controlled vocabulary in the collection's own `README.txt` is exactly `BARRA-R2`, `BARRA-RE2`,
`BARRA-C2`, and walking the full THREDDS tree finds no other. The Bureau's reanalysis page and
the NCI Opus documentation list the same three. BARRA2 is atmosphere-and-land only (Unified
Model + JULES); there is no ocean member of the family.

Five published `(source_id, domain_id)` streams exist. Whichever was meant by "BARRA-O2" is
one of these — see [Which product to build](#which-product-to-build).

---

## Dataset: Bureau of Meteorology BARRA2

### Source Information

All five streams come from one collection, one server, one file layout, and one license, so
the access facts are shared. Per-stream differences are in
[Products and grids](#products-and-grids) and [Data variables](#data-variables).

- **Summary of data organization**: one NetCDF-4 file per (product, domain, frequency,
  variable, calendar month). A file holds every time step of that month and the whole
  spatial domain for exactly one variable. Ensemble streams carry a `realization` dimension
  inside the same file; soil variables carry a `depth` dimension. There is no split by
  region, level, or member.
- **File format**: NetCDF-4 classic (HDF5). Data variables are `int32`, zlib level 1 with
  the shuffle filter, and packed with a `scale_factor`/`add_offset` pair **chosen per file**
  (e.g. `tas` on AUS-11 had `add_offset` 266.265625 for 1979-01 and 267.640625 for 2026-03).
  Missing values are `_FillValue = -2147483647` in the packed integers, decoding to NaN.
  A few variables are unpacked `float32` (`orog`, `sftlf`, and 1hr `maxcolwa`, `ztp`,
  `prga`, `helicitymax` on AUST-04).
- **Temporal coverage**: 1979-01-01T00:00Z to 2026-03-31T23:00Z, i.e. 567 monthly files per
  variable. Verified identical for AUS-11, AUS-22, AUST-04, AUST-11 and AUST-22, and for
  every variable spot-checked (`tas`, `pr`, `huss`, `hurs`, `evspsbl`, `evspsblpot`, `prmax`,
  `prc`, `mrfsos`, `wsgsmax`, `rsdsdir`, `CAPE`, `MUCAPE`). The archive is extended one month
  at a time. The bias-adjusted `output-Adjust` branch lags: AUST-11 `tasAdjust` has 564
  files, ending 2025-12.
- **Temporal frequency**: an analysis-style continuous time series, no init/lead structure.
  Per stream: AUS-11 `1hr`/`3hr`/`day`/`mon`/`fx`; AUS-22 the same; AUST-04 adds `20min`;
  AUST-11 `1hr`/`day`/`mon`/`fx`; AUST-22 `3hr` only. Instantaneous variables are stamped on
  the hour; time-aggregated variables are stamped at the **interval midpoint** and carry
  `time_bnds` (see [Time stamping](#time-stamping)).
- **Latency**: month `M` appears roughly 3.5–5 months after the end of `M`, in irregular
  monthly batches. Measured `Last-Modified` for AUS-11 1hr `tas`:

  | month | published | month | published |
  |---|---|---|---|
  | 2025-04 | 2025-08-11 | 2025-10 | 2026-02-13 |
  | 2025-05 | 2025-09-12 | 2025-11 | 2026-03-10 |
  | 2025-06 | 2025-11-13 | 2025-12 | 2026-04-20 |
  | 2025-07 | 2025-11-17 | 2026-01 | 2026-05-14 |
  | 2025-08 | 2025-12-07 | 2026-02 | 2026-06-25 |
  | 2025-09 | 2026-02-05 | 2026-03 | 2026-07-17 |

  All five streams for a given month publish within a few days of each other (2026-03:
  AUS-11, AUST-11, AUS-22 and AUST-22 on 2026-07-17, AUST-04 on 2026-07-20). NCI
  documentation attributes the lag to BARRA2's dependence on ERA5, which ECMWF publishes
  about three months behind.
- **Access notes**:
  - Public, unauthenticated HTTPS from the NCI THREDDS Data Server. Two hostnames serve the
    same tree: `dapds00.nci.org.au` and `thredds.nci.org.au`. `dapds00` was consistently the
    faster of the two in testing. **Note the path differs from the on-disk layout**: the
    `BARRA2` product level is present in the catalogue browse URLs but absent from the
    `fileServer`/`dodsC` data URLs.
  - Services available per file: `fileServer` (whole-file HTTP GET), `dodsC` (OPeNDAP),
    `ncss` (NetCDF Subset Service), `wcs`. `dodsC` `.das`/`.dds` is the cheap way to read
    structure and attributes without transferring data.
  - `fileServer` honours HTTP range requests (`206 Partial Content`), so HDF5 chunk-level
    byte-range reads are possible — relevant if a virtual dataset is ever considered.
  - **Throughput is the main constraint.** Measured from this environment: a single stream
    sustains ~0.3–0.5 MB/s. Concurrency scales close to linearly — 8 parallel range requests
    gave ~2.5 MB/s aggregate, 24 gave ~5.5 MB/s, and a 32-way parallel fetch pulled the
    549 MB 1979-01 `tas` file in 66 s (~8.3 MB/s). Any backfill must parallelise
    aggressively; serial per-file downloads will not finish.
  - NCI account holders can instead read `/g/data/ob53/...` directly on Gadi after
    registering with project `ob53` (provider documentation; not tested here).
  - No cloud mirror (S3/GCS, Zarr, or otherwise) was found. NCI THREDDS is the only public
    route.
- **License**: Creative Commons Attribution 4.0 International
  (https://creativecommons.org/licenses/by/4.0/), stated in
  `.../ob53/BARRA2/license.txt` and in every file's `license` global attribute. Clearly open;
  attribution required. The collection `README.txt` asks for two citations: the NCI
  collection DOI `10.25914/1X6G-2V48` and Su et al. (2025), *J. Southern Hemisphere Earth
  Syst. Sci.* 75, ES25032, `10.1071/ES25032`. The provider labels the data "a research
  product ... that has not been fully evaluated".
- **Browse root**: https://dapds00.nci.org.au/thredds/catalogs/ob53/catalog.html
  (collection `README.txt` and `license.txt`:
  https://dapds00.nci.org.au/thredds/catalog/ob53/BARRA2/catalog.html)
- **URL format**:

```
# data (note: no "BARRA2" path element)
https://dapds00.nci.org.au/thredds/fileServer/ob53/output/reanalysis/{domain_id}/BOM/ERA5/historical/{driving_variant}/{source_id}/v1/{freq}/{variable_id}/latest/{variable_id}_{domain_id}_ERA5_historical_{driving_variant}_BOM_{source_id}_v1_{freq}_{YYYYMM}-{YYYYMM}.nc

# fixed fields (no time suffix)
https://dapds00.nci.org.au/thredds/fileServer/ob53/output/reanalysis/{domain_id}/BOM/ERA5/historical/{driving_variant}/{source_id}/v1/fx/{variable_id}/latest/{variable_id}_{domain_id}_ERA5_historical_{driving_variant}_BOM_{source_id}_v1.nc

# structure/attributes without downloading data
https://dapds00.nci.org.au/thredds/dodsC/{same path}.dds
https://dapds00.nci.org.au/thredds/dodsC/{same path}.das

# directory listing (same path shape as the data URLs)
https://dapds00.nci.org.au/thredds/catalog/ob53/output/reanalysis/{domain_id}/BOM/ERA5/historical/{driving_variant}/{source_id}/v1/{freq}/{variable_id}/latest/catalog.html
```

- **Example URLs**:

```
https://dapds00.nci.org.au/thredds/fileServer/ob53/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/tas/latest/tas_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_197901-197901.nc
https://dapds00.nci.org.au/thredds/fileServer/ob53/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/tas/latest/tas_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_202603-202603.nc
https://dapds00.nci.org.au/thredds/fileServer/ob53/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/fx/orog/latest/orog_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1.nc
https://dapds00.nci.org.au/thredds/fileServer/ob53/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/tas/latest/tas_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_202603-202603.nc
https://dapds00.nci.org.au/thredds/fileServer/ob53/output/reanalysis/AUS-22/BOM/ERA5/historical/eda/BARRA-RE2/v1/1hr/tas/latest/tas_AUS-22_ERA5_historical_eda_BOM_BARRA-RE2_v1_1hr_202603-202603.nc
```

### GRIB Index

Not applicable — the source is NetCDF-4, not GRIB. The equivalent cheap-metadata mechanism is
OPeNDAP `.dds`/`.das`, and the equivalent partial-read mechanism is HTTP range requests against
the HDF5 chunk layout.

### Coordinate Reference System

Identical for all five streams. Each file carries a scalar `crs` variable referenced by every
data variable's `grid_mapping` attribute:

- **Common name**: unrotated geographic latitude–longitude on a **sphere**, not WGS84.
- `crs` attributes: `grid_mapping_name = "latitude_longitude"`, `earth_radius = 6371229.0`.
- **PROJ string**: `+proj=longlat +R=6371229 +no_defs`. There is no exact EPSG code;
  treating it as EPSG:4326 introduces the usual sphere-vs-ellipsoid latitude discrepancy.
- Longitude is stored in a 0–360-style frame that runs past 180° for the Australasian
  domains (AUS-11 reaches 207.39°E = 152.61°W). Latitude is ascending (south to north);
  longitude is ascending.
- No `lat_bnds`/`lon_bnds` are published, so cell-centre vs cell-edge registration cannot be
  confirmed from the files. The values sit on an exact 0.11°/0.22°/0.04° lattice and the
  domain attributes quote the same first/last values, which is consistent with cell centres.

### Products and grids

| `source_id` | `domain_id` | driving variant | grid spacing | lat (n, min, max) | lon (n, min, max) | frequencies | ensemble |
|---|---|---|---|---|---|---|---|
| BARRA-R2 | AUS-11 | `hres` | 0.11° | 646, −57.97, 12.98 | 1082, 88.48, 207.39 | `1hr` `3hr` `day` `mon` `fx` | — |
| BARRA-RE2 | AUS-22 | `eda` | 0.22° | 311, −56.49, 11.71 | 531, 89.53, 206.13 | `1hr` `3hr` `day` `mon` `fx` | 22 |
| BARRA-C2 | AUST-04 | `hres` | 0.04° | 1018, −45.69, −5.01 | 1298, 108.02, 159.90 | `20min` `1hr` `3hr` `day` `mon` `fx` | — |
| BARRA-R2 (convective params) | AUST-11 | `hres` | 0.11° | 372, −45.76, −4.95 | 474, 107.95, 159.98 | `1hr` `day` `mon` `fx` | — |
| BARRA-RE2 (convective params) | AUST-22 | `eda` | 0.22° | 187, −45.71, −4.79 | 237, 108.01, 159.93 | `3hr` | 11 |

Notes:
- AUST-11 is an exact sub-lattice of AUS-11 (−45.76 = −57.97 + 111×0.11, 107.95 = 88.48 +
  177×0.11), so the convective-parameter fields drop straight into an AUS-11-shaped cube
  over the Australian sub-box.
- AUST-22 is likewise an exact sub-lattice of AUS-22 (−45.71 = −56.49 + 49×0.22, 108.01 =
  89.53 + 84×0.22). The 0.22° lattices are their own, not decimations of the 0.11° grids.
- `orog` and `sftlf` are published under `fx` for every domain. They are `float32`,
  unpacked, with no NaN: `orog` is 0 over ocean, `sftlf` is a 0–100 % land fraction.
  Verified `orog` maxima: AUS-11 3696.67 m, AUS-22 3449.72 m, AUST-04 3654.38 m,
  AUST-11 3137.81 m.

### Dimensions & Dimension Coordinates

BARRA2 is an analysis-shaped product: a single continuous `time` dimension, no `init_time` /
`lead_time`. Values below are for BARRA-R2 / AUS-11; substitute the grid row above for the
other streams.

| Dimension | Min | Max | Step | Notes |
|---|---|---|---|---|
| time | 1979-01-01T00:00Z | 2026-03-31T23:00Z | 1 h (`1hr`); 20 min, 3 h, 1 day, 1 month for the other frequencies | `days since 1949-12-01`, `proleptic_gregorian`, UTC. 414,168 hourly steps to date. Instantaneous variables on the hour, aggregated variables on the half hour |
| latitude | −57.97 | 12.98 | 0.11 | `lat`, `float64`, ascending, 646 values, `standard_name = latitude`, `units = degrees_north` |
| longitude | 88.48 | 207.39 | 0.11 | `lon`, `float64`, ascending, 1082 values, exceeds 180° |
| realization | — | — | — | BARRA-RE2 only. **String labels, not integers**: AUS-22 has 22 (`000_0`, `000_1`, `001_0`, … `009_1`, `ctl_0`, `ctl_1`); AUST-22 has 11 (`000_0` … `009_0`, `ctl_0`) |
| depth | 0.05 | 2.0 | irregular | Soil variables `mrsol`, `mrfsol`, `tsl` (`3hr`): 4 levels at 0.05, 0.225, 0.675, 2.0 m. `mrsos`/`mrfsos` are scalar `depth = 0.05 m` |
| pressure | 10 hPa | 1000 hPa | irregular | Not a dimension in the files — encoded in the variable name (`ta850`) with a scalar `pressure` coordinate inside. See [Vertical levels](#vertical-levels) |
| height | 1.5 | 1500 | irregular | Likewise a scalar coordinate: `tas`/`hurs`/`huss` at **1.5 m**, `uas`/`vas`/`sfcWind`/`wsgsmax` at 10 m, and named `…50m`/`100m`/`150m`/`200m`/`250m`/`1500m` variables |

Coordinate values are bit-identical between the 1979-01, 2026-03 and `fx` files
(`np.array_equal` on both `lat` and `lon`), so the historical lat/lon rounding inconsistency
listed in the provider's Known Issues is fixed in the currently published files.

### Vertical levels

Levels are baked into the variable name, one file per level. There is **no dense, uniform
level set** — the available levels differ by variable, frequency and product:

- **Pressure levels (hPa)** on AUS-11 / AUS-22 (identical sets on both):
  - `1hr`: `hus` and `ta` at 200, 300, 400, 500, 600, 700, 850, 925, **950**, 1000;
    `ua`, `va`, `wa`, `zg` at the same set **without 950**; plus `omega500` on its own.
  - `3hr`: `hus`, `ta`, `ua`, `va`, `wa`, `zg` at 10, 20, 30, 50, 70, 100, 150, 250 — the
    stratosphere and upper troposphere only, disjoint from the `1hr` set.
  - `day`/`mon`: all six families at 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500,
    600, 700, 850, 925, 1000 — the union of the two, except that 950 is dropped and 250 is
    added for every family.
- **Pressure levels on AUST-04**: uniform across `hus`, `ta`, `ua`, `va`, `wa`, `wap`, `zg`
  at `1hr` — 200, 300, 400, 500, 600, 700, 750, 800, 850, 900, 925, 950, 975, 1000 — and
  10, 20, 30, 50, 70, 100, 150, 250 at `3hr`. AUST-04 publishes `wap` (omega, Pa s-1) on
  every level in addition to `wa` (m s-1); AUS-11/AUS-22 publish only `wa` plus `omega500`.
- **Heights above ground**: `ta`, `ua`, `va` at 50 m, 100 m, 150 m, 200 m, 250 m, 1500 m —
  the same six on every stream and at `20min` on AUST-04.
- **Soil**: 4 layers at 0.05, 0.225, 0.675, 2.0 m as a real `depth` dimension (`3hr`).

Mapping this into our conventions: on AUS-11/AUS-22 the `1hr` pressure set is ragged (`hus`
and `ta` carry 950 hPa, the wind and height families do not), so a `pressure_level` group
would need either a per-variable level subset or a NaN-padded union; on AUST-04 the `1hr` set
is uniform across all seven families and maps onto a `pressure_level` group cleanly. The named
height-above-ground fields (`ua100m` etc.) match our existing `<var>_<level>` root-level
naming directly.

### Time stamping

Verified on real files, and this is the single most important read-path detail:

- **Instantaneous** variables (`cell_methods = "time: point (interval: 1 hour)"`) are stamped
  on the hour. `tas` 2026-03 runs 2026-03-01T00:00 … 2026-03-31T23:00, 744 steps, no
  `time_bnds`.
- **Time-aggregated** variables (`time: mean`, `time: maximum`, `time: minimum`) are stamped
  at the **midpoint of the interval** and do carry `time_bnds`. `pr` 2026-03 runs
  2026-03-01T00:30 … 2026-03-31T23:30, and `time_bnds[0] = [2026-03-01T00:00,
  2026-03-01T01:00]`. The same holds for `tasmax`, `tasmin`, `tasmean`, `rsds`, `rlds`,
  `clt`, `evspsbl`, `wsgsmax`, etc.

So a materialized cube that puts instantaneous and aggregated variables on one `time` axis
must shift the aggregated variables by half an interval and record the convention chosen.

There is no accumulation to deaccumulate: `pr` is a mean flux in `kg m-2 s-1` over the hour,
not a running total.

### Data Variables

Core-variable availability, read from 2026-03 files on AUS-11 (BARRA-R2) unless noted. All
of these also exist on AUS-22 and AUST-04.

| Variable name | Source `variable_id` | Level | Units | Available from | Notes |
|---|---|---|---|---|---|
| temperature_2m | `tas` | **1.5 m** (not 2 m) | K | 1979-01 | `air_temperature`, instantaneous. Also `tasmax`/`tasmin`/`tasmean` as hourly aggregates |
| wind_u_10m | `uas` | 10 m | m s-1 | 1979-01 | `cell_methods` records bilinear area interpolation. Also `uasmax` |
| wind_v_10m | `vas` | 10 m | m s-1 | 1979-01 | as above |
| wind_u_100m | `ua100m` | 100 m | m s-1 | 1979-01 | logarithmic height interpolation from model levels. Also 50/150/200/250/1500 m |
| wind_v_100m | `va100m` | 100 m | m s-1 | 1979-01 | as above |
| precipitation_surface | `pr` | surface | kg m-2 s-1 | 1979-01 | **Mean flux over the hour**, not an accumulation. Midpoint-stamped with `time_bnds`. `prc` (convective), `prsn` (snowfall), `prmax` (hourly max rate) also available |
| downward_short_wave_radiation_flux_surface | `rsds` | surface | W m-2 | 1979-01 | hourly mean. `rsdsdir` (direct), `rsdscs` (clear sky), `rsus` (upwelling) also available |
| downward_long_wave_radiation_flux_surface | `rlds` | surface | W m-2 | 1979-01 | hourly mean. `rldscs`, `rlus`, `rluscs` also available |
| pressure_surface | `ps` | surface | Pa | 1979-01 | instantaneous |
| pressure_reduced_to_mean_sea_level | `psl` | MSL | Pa | 1979-01 | instantaneous, `air_pressure_at_mean_sea_level` |
| total_cloud_cover_atmosphere | `clt` | atmosphere | % | 1979-01 | hourly **mean**. `cll`/`clm`/`clh` for low/mid/high layers |
| relative_humidity_2m | `hurs` | **1.5 m** | % | 1979-01 | instantaneous |
| specific_humidity_2m | `huss` | **1.5 m** | 1 (kg/kg) | 1979-01 | instantaneous; `units = "1"` |
| dew_point_temperature_2m | — | — | — | **not published** | Derive from `tas` + `hurs`/`huss` if needed |
| wind_speed_10m | `sfcWind` | 10 m | m s-1 | 1979-01 | scalar wind speed, published alongside the components |
| wind_gust_10m | `wsgsmax` | 10 m | m s-1 | 1979-01 | hourly maximum gust; see the known bias below |
| soil moisture (0–10 cm) | `mrsos` | `depth = 0.05 m` | kg m-2 | 1979-01 | **land-only, NaN over ocean** (0.88 NaN fraction on AUS-11) |
| boundary layer height | `zmla` | column | m | 1979-01 | instantaneous |
| CAPE / CIN | `CAPE`, `CIN` | column | J kg-1 | 1979-01 | biased on AUS-11/AUS-22, see below; better values in the AUST-11/AUST-22 convective-parameter streams |

Beyond these the sets are large: 129 `1hr` variables on AUS-11, 209 on AUST-04, 100
convective diagnostics on AUST-11, plus `3hr` (soil, stratospheric pressure levels, fluxes,
snow) and 186–238 `day`/`mon` aggregates. **BARRA-R2 (AUS-11) and BARRA-RE2 (AUS-22) publish
identical variable sets at every frequency.** Exact lists are in the
[appendix](#appendix-exact-variable-lists).

**Temporal availability changes**: none found. Every variable spot-checked is present for
all 567 months, including `evspsbl` and `prhmax`, which the provider's Known Issues page
lists as late additions. What does change with time is the **`variable_version` (processing
version)** of a given month's file:

| variable | 1979-01 | 2000-01 | 2020-01 | 2023-12 | 2024-06 | 2026-03 |
|---|---|---|---|---|---|---|
| `tas` | v20231001 | v20231001 | v20231001 | v20240516 | v20240809 | v20250528 |
| `wsgsmax` | v20231001 | v20231001 | v20231001 | v20240516 | v20240809 | v20250528 |
| `huss` | v20240516 | v20240516 | v20240516 | v20240516 | v20240809 | v20250528 |
| `hurs` | v20240516 | v20240516 | v20240516 | v20240612 | v20240809 | v20250528 |
| `evspsbl` | v20240516 | v20240516 | v20240516 | v20240516 | v20240809 | v20250528 |

So `huss`/`hurs` were reprocessed archive-wide (the provider's "spurious high values" fix)
while `tas` and `wsgsmax` retain their original 2023 processing for the historical period.
Under `latest/` only one version is exposed, so this is informational rather than a choice
we have to make — but it means a re-backfill can pick up silently changed historical values.

### Sample Files Examined

Downloaded in full and opened with xarray/h5netcdf:

- **Start of archive**: `tas_AUS-11_..._1hr_197901-197901.nc` (549 MB) — 1979-01, chunks
  (32, 4, 8), `variable_version` v20231001, `cell_methods "time: point (interval: 1H)"`.
- **Recent data**: `tas_AUS-11_..._1hr_202603-202603.nc` (467 MB) — 2026-03, chunks
  (16, 99, 165), v20250528, `cell_methods "time: point (interval: 1 hour)"`.
- **Aggregated variable**: `pr_AUS-11_..._1hr_202603-202603.nc` (776 MB) — midpoint stamping
  and `time_bnds`; values 0 to 0.007 kg m-2 s-1 (≈25 mm/h), no NaN.
- **Land-only variable**: `mrsos_AUS-11_..._1hr_202603-202603.nc` (134 MB) — NaN over ocean.
- **Convective-scale**: `tas_AUST-04_..._1hr_202603-202603.nc` (803 MB) — 1018 × 1298,
  chunks (12, 130, 168), `scale_factor` 0.015625.
- **Fixed fields**: `orog`/`sftlf` for AUS-11, `orog` for AUS-22, AUST-04, AUST-11.

Inspected via OPeNDAP `.dds`/`.das` without downloading: ~60 AUS-11 `1hr` variables, ~33
AUST-04 `1hr` variables, AUS-22 and AUST-22 ensemble structure, AUS-11 `3hr` soil variables,
and `tas` metadata for every second year 1979–2026 plus every month of 1991–1993.

### Notable Observations

1. **Chunk layout changes mid-archive.** Files for **1979-01 through 1991-08** are chunked
   (32, 4, 8) — 528,768 chunks in a single month of `tas`. From **1991-09 onward** the
   chunking is (16, ~99, ~166), about 2,300 chunks. Reading one full time step took 1.43 s
   from the old layout versus 0.33 s from the new, so spatial-subset reads are ~4× slower
   over the first 12.7 years; a sequential 24-step read was 3.4 s versus 4.0 s, so a
   whole-file read costs about the same either way (~2 min per variable-month). The boundary
   was located exactly: 1991-08 is (32, 4, 8),
   1991-09 is (16, 99, 165). Minor per-file variation exists in the new regime
   ((16, 98, 167), (16, 97, 168), and (15, 100, 174) for 1993-02).
2. **Per-file packing.** `scale_factor`/`add_offset` are re-chosen per file, so the stored
   integers are not comparable across months and every read must decode. Effective precision
   varies a lot by product: AUS-11 `tas` uses `scale_factor` 0.000244140625 (2⁻¹² K) while
   AUST-04 `tas` uses 0.015625 (2⁻⁶ K). Worth checking per variable before choosing
   `keep_mantissa_bits`.
3. **Downloading is the bottleneck, not decoding.** 637 MiB per variable-month averaged over
   15 representative AUS-11 `1hr` variables (range: `wsgsmax` 340 MiB to `hurs` 1099 MiB).
   A 15-variable
   AUS-11 backfill is ≈9.4 GiB/month × 567 months ≈ **5.2 TiB to transfer**. At the ~8 MB/s a
   32-way parallel client achieved, that is ~190 worker-hours of pure download; it needs
   many workers in parallel and per-file retry.
4. **Uncompressed cube sizes** (float32, one variable, hourly, full archive): AUS-11 1.16 TB,
   AUS-22 0.27 TB (before the ×22 ensemble), AUST-04 2.19 TB, AUST-11 0.29 TB.
5. **Ensemble labels are strings.** BARRA-RE2's `realization` coordinate is a NetCDF string
   variable (`"000_0"`, `"ctl_1"`, …), 22 members on AUS-22 and 11 on AUST-22. It encodes an
   ERA5-EDA member and a sub-member, so it does not map onto an integer `ensemble_member`
   dimension without a documented convention.
6. **Metadata defects seen in the files.**
   - `mrsos` has `standard_name = "mass_content_of_water_in_soil_layer\n"` — a literal
     backslash-n in the string. Not a valid CF standard name.
   - Several variables have a doubled `cell_methods`, e.g. `prsn`, `rsus`, `rlus`, `rsdt`,
     `rlut`, `hfls`, `hfss`, `evspsbl`: `"time: mean (interval: 1 hour) time: mean (interval:
     1 hour)"`; `sfcWindmax` and `wsgsmax` similarly double `time: maximum`.
   - `pr` declares `coordinates = "forecast_reference_time forecast_period"` but neither
     variable exists in the file; several others reference `forecast_reference_time` the same
     way. xarray drops them silently.
   - `CAPE`, `twiso`, `twpse`, `visibility`, `fogfraction`, `FZL`, `MLCAPE`, `MUCAPE`,
     `DCAPE`, `wsgs` and others have **no** `cell_methods` at all, so whether they are
     instantaneous or aggregated has to be taken from documentation.
   - `AUST-04` `prra` is "Large Scale Rainfall Rate" but carries `time: maximum`, not
     `time: mean`.
   - The 1979 files write `interval: 1H` where the 2026 files write `interval: 1 hour`, and
     write `geospatial_lat_min` and friends as `float64` where the 2026 files write strings.
     Both are global-attribute-only differences.
7. **Provider Known Issues** (from https://opus.nci.org.au/x/mADADw, not independently
   verified) that bear on how we would describe the archive:
   - **January–February 1979 are spin-up months**, to be used with caution.
   - **Soil moisture has abrupt discontinuities on 1 September** each year, from production
     stream transitions (AUS-11, AUS-22).
   - **`wsgsmax` is computed incorrectly** in UM v11.9–13, with a 1–1.5 m/s bias over land.
     Still open.
   - **`CAPE`/`CIN` on AUS-11 and AUS-22 are biased** by parcel-ascent inaccuracies in UM
     v11.9; the AUST-11/AUST-22 convective-parameter streams are the recommended alternative.
     AUST-04's `CAPE`/`CIN` have their own UM v13.0 bias with replacement data planned.
   - Convective parameters have discontinuities Sept–Dec 2015 (fixed in v20240809).
   - Near-boundary artefacts are expected in every domain — strongest for BARRA-C2 under
     inflow conditions (tropical cyclones), and present as damped variability and artificial
     gradients near the AUS-11/AUS-22 edges.
   - Systematic biases: cold bias in daily maximum screen temperature, warm bias in daily
     minimum, wind speed under-estimated at the high end and over-estimated at the low end,
     tropical cyclone intensity under-estimated.
   These are the sort of facts that belong in a validation report's review notes rather than
   in template metadata.
8. **A bias-adjusted branch exists** at `ob53/output-Adjust/reanalysis/...` — but it contains
   only `tasAdjust` (daily linear rescale against AGCD v1), on AUST-04, AUST-05i and AUST-11,
   and it lags the main archive (last month 2025-12). Probably not worth integrating.
9. **`AUST-05i` appears only in the bias-adjusted branch** — a 0.05° grid, 691 × 886,
   −44.50 to −10.00 lat and 112.00 to 156.25 lon, with `float32` (not `float64`) coordinates
   and no counterpart under `output/`.

### Which product to build

The exploration cannot settle this; it is a Checkpoint A decision. What the data says:

- **BARRA-R2 / AUS-11** is the natural flagship: the full CORDEX-Australasia domain, the
  richest deterministic variable set at 12 km, deterministic (no ensemble dimension), and the
  cheapest of the three at 637 MiB per variable-month. Best first target.
- **BARRA-C2 / AUST-04** is the headline product for Australian users — 4.4 km, 209 `1hr`
  variables including radar reflectivity, lightning flash rate, visibility and fog — but its
  grid has 1.9× the cells of AUS-11 (1.7× the bytes for `tas` 2026-03) and it is
  Australia-only.
- **BARRA-RE2 / AUS-22** adds a 22-member ensemble at 24 km, at 5.9× the bytes of AUS-11 for
  `tas` 2026-03, with a string-valued member coordinate that needs a naming decision.
- **AUST-11 / AUST-22** are convective-diagnostic add-ons on sub-lattices of the R2/RE2 grids,
  not standalone datasets.

Open questions for Checkpoint A:

1. Which stream(s), and the variable set for each.
2. Time stamping: shift the midpoint-stamped aggregated variables onto the hour (and which
   edge), or keep two conventions?
3. Pressure levels: build a `pressure_level` group over the ragged union, restrict to the
   levels every variable shares, or ship only the named height-above-ground and single-level
   variables to start?
4. `tas`/`hurs`/`huss` are at **1.5 m**, not 2 m. Name them `temperature_2m` for
   cross-dataset consistency with our other archives, or `temperature_1_5m` for accuracy?
5. For BARRA-RE2, how should the string `realization` labels map onto `ensemble_member`?
6. Given a ~4-month publication lag and monthly granularity, the operational "update" is a
   once-a-month append of a whole month. Is a monthly cron the right cadence?

---

## Appendix: exact variable lists

Read from the THREDDS catalogue on 2026-08-19. BARRA-R2 (AUS-11) and BARRA-RE2 (AUS-22)
publish **identical** variable sets at every frequency, and `day` and `mon` carry the same
set as each other on every stream.

**BARRA-R2 / AUS-11 and BARRA-RE2 / AUS-22 — `1hr` (129)**

```
CAPE CIN clh clivi cll clm clt clwvi evspsbl evspsblpot hfls hfss hurs hus1000 hus200 hus300
hus400 hus500 hus600 hus700 hus850 hus925 hus950 huss mrfsos mrsos omega500 pr prc prmax prsn
prw ps psl rlds rldscs rlus rluscs rlut rlutcs rsds rsdscs rsdsdir rsdt rsus rsuscs rsut rsutcs
sfcWind sfcWindmax ta1000 ta100m ta1500m ta150m ta200 ta200m ta250m ta300 ta400 ta500 ta50m
ta600 ta700 ta850 ta925 ta950 tas tasmax tasmean tasmin ts twiso twpse ua1000 ua100m ua1500m
ua150m ua200 ua200m ua250m ua300 ua400 ua500 ua50m ua600 ua700 ua850 ua925 uas uasmax uasmean
va1000 va100m va1500m va150m va200 va200m va250m va300 va400 va500 va50m va600 va700 va850 va925
vas vasmax vasmean wa1000 wa200 wa300 wa400 wa500 wa600 wa700 wa850 wa925 wsgsmax zg1000 zg200
zg300 zg400 zg500 zg600 zg700 zg850 zg925 zmla
```

**BARRA-R2 / AUS-11 and BARRA-RE2 / AUS-22 — `3hr` (62)**

```
hus10 hus100 hus150 hus20 hus250 hus30 hus50 hus70 mrfso mrfsol mrro mrros mrso mrsol qfluxu
qfluxv snd snm snw ta10 ta100 ta150 ta20 ta250 ta30 ta50 ta70 tauu tauv tsl ua10 ua100 ua150
ua20 ua250 ua30 ua50 ua70 va10 va100 va150 va20 va250 va30 va50 va70 wa10 wa100 wa150 wa20 wa250
wa30 wa50 wa70 zg10 zg100 zg150 zg20 zg250 zg30 zg50 zg70
```

**BARRA-R2 / AUS-11 and BARRA-RE2 / AUS-22 — `day` and `mon` (186)**

```
CAPE CIN clh clivi cll clm clt clwvi evspsbl evspsblpot hfls hfss hurs hus10 hus100 hus1000
hus150 hus20 hus200 hus250 hus30 hus300 hus400 hus50 hus500 hus600 hus70 hus700 hus850 hus925
huss mrfso mrfsol mrfsos mrro mrros mrso mrsol mrsos omega500 pr prc prhmax prsn prw ps psl rlds
rldscs rlus rluscs rlut rlutcs rsds rsdscs rsdsdir rsdt rsus rsuscs rsut rsutcs sfcWind
sfcWindmax snd snm snw sund ta10 ta100 ta1000 ta100m ta150 ta1500m ta150m ta20 ta200 ta200m
ta250 ta250m ta30 ta300 ta400 ta50 ta500 ta50m ta600 ta70 ta700 ta850 ta925 tas tasmax tasmin
tauu tauv ts tsl twiso twisomax twpse twpsemax ua10 ua100 ua1000 ua100m ua150 ua1500m ua150m
ua20 ua200 ua200m ua250 ua250m ua30 ua300 ua400 ua50 ua500 ua50m ua600 ua70 ua700 ua850 ua925
uas va10 va100 va1000 va100m va150 va1500m va150m va20 va200 va200m va250 va250m va30 va300
va400 va50 va500 va50m va600 va70 va700 va850 va925 vas wa10 wa100 wa1000 wa150 wa20 wa200 wa250
wa30 wa300 wa400 wa50 wa500 wa600 wa70 wa700 wa850 wa925 wsgsmax z0 zg10 zg100 zg1000 zg150 zg20
zg200 zg250 zg30 zg300 zg400 zg50 zg500 zg600 zg70 zg700 zg850 zg925 zmla
```

**BARRA-C2 / AUST-04 — `20min` (34)**

```
clt helicity huss prga prra prsn psl radrefl1km rsdsdif rsdsdir rss ta100m ta1500m ta150m ta200m
ta250m ta50m tas ts ua100m ua1500m ua150m ua200m ua250m ua50m uas va100m va1500m va150m va200m
va250m va50m vas wsgsmax
```

**BARRA-C2 / AUST-04 — `1hr` (209)**

```
BWD03 BWD06 CAPE CIN DCAPE EBWD EILbase EILdepth ESRHl ESRHr FZL LR03 LR75 MLCAPE MLCIN MLLCL
MUCAPE MUCIN MUEL MULPL MULPLmixr MULPLpres MULPLtemp SRH01l SRH01r SRH03l SRH03r clh clivi cll
clm clt clwvi coltotdrym coltotwetm evspsbl evspsblpot flashrate fogfraction helicitymax
helicitymin hfls hfss hurs hus1000 hus200 hus300 hus400 hus500 hus600 hus700 hus750 hus800
hus850 hus900 hus925 hus950 hus975 huss maxcolrefl maxcolwa mrfsos mrro mrros mrsos pr prga
prmax prra prsn prsnmax prw ps psl rlds rldscs rlus rluscs rlut rlutcs rsds rsdscs rsdsdir rsdt
rsus rsuscs rsut rsutcs sfcWind sfcWindmax ta1000 ta100m ta1500m ta150m ta200 ta200m ta250m
ta300 ta400 ta500 ta50m ta600 ta700 ta750 ta800 ta850 ta900 ta925 ta950 ta975 tas tasmax tasmean
tasmin ts tsmean twiso twpse ua1000 ua100m ua1500m ua150m ua200 ua200m ua250m ua300 ua400 ua500
ua50m ua600 ua700 ua750 ua800 ua850 ua900 ua925 ua950 ua975 uas uasmean va1000 va100m va1500m
va150m va200 va200m va250m va300 va400 va500 va50m va600 va700 va750 va800 va850 va900 va925
va950 va975 vas vasmean visibility wa1000 wa200 wa300 wa400 wa500 wa600 wa700 wa750 wa800 wa850
wa900 wa925 wa950 wa975 wap1000 wap200 wap300 wap400 wap500 wap600 wap700 wap750 wap800 wap850
wap900 wap925 wap950 wap975 wsgs wsgsmax zg1000 zg200 zg300 zg400 zg500 zg600 zg700 zg750 zg800
zg850 zg900 zg925 zg950 zg975 zmla ztp
```

**BARRA-C2 / AUST-04 — `3hr` (74)**

```
ares cw hus10 hus100 hus150 hus20 hus250 hus30 hus50 hus70 mrfso mrfsol mrso mrsol qfluxu qfluxv
sfcMoisflx sfcWind10minmean snd snm snw soildrainage ta10 ta100 ta150 ta20 ta250 ta30 ta50 ta70
tauu tauv throughfall tsl ua10 ua100 ua150 ua20 ua250 ua30 ua50 ua70 va10 va100 va150 va20 va250
va30 va50 va70 wa10 wa100 wa150 wa20 wa250 wa30 wa50 wa70 wap10 wap100 wap150 wap20 wap250 wap30
wap50 wap70 zg10 zg100 zg150 zg20 zg250 zg30 zg50 zg70
```

**BARRA-C2 / AUST-04 — `day` and `mon` (238)**

```
CAPE CAPEmax CIN CINmax clh clivi cll clm clt clwvi evspsbl evspsblpot hfls hfss hurs hus10
hus100 hus1000 hus150 hus20 hus200 hus250 hus30 hus300 hus400 hus50 hus500 hus600 hus70 hus700
hus750 hus800 hus850 hus900 hus925 hus950 hus975 huss mrfso mrfsol mrfsos mrro mrros mrso mrsol
mrsos pr prhmax prsn prw ps psl rlds rldscs rlus rluscs rlut rlutcs rsds rsdscs rsdsdir rsdt
rsus rsuscs rsut rsutcs sfcWind sfcWindmax snd snm snw sund ta10 ta100 ta1000 ta100m ta150
ta1500m ta150m ta20 ta200 ta200m ta250 ta250m ta30 ta300 ta400 ta50 ta500 ta50m ta600 ta70 ta700
ta750 ta800 ta850 ta900 ta925 ta950 ta975 tas tasmax tasmin tauu tauv ts tsl twiso twisomax
twpse twpsemax ua10 ua100 ua1000 ua100m ua150 ua1500m ua150m ua20 ua200 ua200m ua250 ua250m ua30
ua300 ua400 ua50 ua500 ua50m ua600 ua70 ua700 ua750 ua800 ua850 ua900 ua925 ua950 ua975 uas va10
va100 va1000 va100m va150 va1500m va150m va20 va200 va200m va250 va250m va30 va300 va400 va50
va500 va50m va600 va70 va700 va750 va800 va850 va900 va925 va950 va975 vas wa10 wa100 wa1000
wa150 wa20 wa200 wa250 wa30 wa300 wa400 wa50 wa500 wa600 wa70 wa700 wa750 wa800 wa850 wa900
wa925 wa950 wa975 wap10 wap100 wap1000 wap150 wap20 wap200 wap250 wap30 wap300 wap400 wap50
wap500 wap600 wap70 wap700 wap750 wap800 wap850 wap900 wap925 wap950 wap975 wsgsmax z0 zg10
zg100 zg1000 zg150 zg20 zg200 zg250 zg30 zg300 zg400 zg50 zg500 zg600 zg70 zg700 zg750 zg800
zg850 zg900 zg925 zg950 zg975 zmla
```

**BARRA-R2 / AUST-11 convective parameters — `1hr` (100)**

```
BWD01 BWD01dir BWD03 BWD03dir BWD06 BWD06dir BWD09 BWD09dir CAPE CIN CRZdepth DCAPE DPL EBWD
EBWDdir EFFCAPE EFFCIN EILbase EILdepth EMLbase EMLdepth EMLlapse ESRHl ESRHr FZL HGZdepth LR01
LR03 LR24 LR36 LR75 MAULbase MAULdepth MAULlapse MLCAPE MLCAPE03 MLCAPEx MLCAPbase MLCAPdepth
MLCIN MLCIN1 MLEL MLLCL MLLFC MLLMB MLLPLmixr MLLPLtemp MLSCLR MLSCRHmean MLVTEmax MLVTEmean
MUCAPE MUCAPE0m20 MUCAPEm10m30 MUCIN MUEL MUELtemp MULCL MULCLtemp MULFC MULMB MULPL MULPLmixr
MULPLpres MULPLtemp MUVTEm10 MUVTEm20 MUVTEmax MUVTEmean PW RH01mean RH03mean RH24mean RH36mean
SF SRH01l SRH01r SRH03l SRH03r SRH0500l SRH0500r U01mean U03mean U0500mean U06mean UANVmean
UEILmean UESMadv USMadv USMdev V01mean V03mean V0500mean V06mean VANVmean VEILmean VESMadv
VSMadv VSMdev WBFZL
```

**BARRA-R2 / AUST-11 convective parameters — `day` and `mon` (4)**

```
CAPE CAPEmax CIN CINmax
```

**BARRA-RE2 / AUST-22 convective parameters — `3hr` (15)**

```
BWD06 EBWD EILbase EILdepth ESRHl LR03 LR36 MUCAPE MUCIN MUEL MULFC MULPLmixr MULPLpres
MULPLtemp SRH03l
```

`fx` on every stream is `orog` and `sftlf`.
