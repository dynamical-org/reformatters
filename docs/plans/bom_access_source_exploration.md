# Source exploration: Bureau of Meteorology ACCESS-G and ACCESS-C

Produced by following [docs/source_data_exploration_guide.md](../source_data_exploration_guide.md).

## Headline finding: there is no AWS-hosted source

The exploration was requested against "AWS ops data". No public AWS/S3 copy of ACCESS-G or
ACCESS-C exists that could be found:

- The AWS Open Data Registry index (1169 datasets, fetched from `registry.opendata.aws`) contains
  no Bureau of Meteorology NWP entry. The only BOM mention in the whole registry is a contributing
  organisation in an AODN wave-buoy collection.
- Anonymous `ListObjectsV2` probes against ~40 plausible bucket names in `us-east-1` and
  `ap-southeast-2` (`bom-nwp`, `bom-ops-data`, `ops-aps3`, `wr45`, `opendata.bom.gov.au`, …) all
  returned `NoSuchBucket`. A control probe against `noaa-gfs-bdp-pds` returned 200, so the probe
  method works.
- `bom.gov.au`'s data-services, data-feeds and ACCESS NWP pages describe FTP/SFTP delivery to
  Registered Users only, with no cloud-object-store option.
- The NCI ISO metadata record for the collection lists no S3 distribution.

**The open source is NCI, over HTTPS, not AWS.** Everything below was verified against real files
served from `thredds.nci.org.au`. If you know of an AWS location, say so — the file-level findings
here (grids, staggering, sentinels, accumulations) carry over to any host of the same APS3 NetCDF
products.

The second, non-open source (BOM Registered User FTP) is documented in its own section because it
is the only channel carrying data after mid-2025 and the only one that de-staggers winds.

---

## Dataset: Bureau of Meteorology, ACCESS APS3 (ACCESS-G / GE / C / CE)

### Source Information — NCI THREDDS (project `wr45`, collection `ops_aps3`)

- **Summary of data organization**: One NetCDF-4 file per variable. The path encodes model,
  stream, init date, init hour, product group and level type; the file holds every output step of
  that run for that one variable. Product groups are `an` (analysis, a single step at T+0),
  `fc` (forecast) and `fcmm` (high-frequency sub-domain, ACCESS-G/C only). Level types are
  `sfc` (single level), `pl` (pressure levels) and `ml` (hybrid-height model levels). Ensembles
  replace the product group with `cf` (control) and `pf/<member>` (perturbed).
- **File format**: NetCDF-4 / HDF5. No GRIB in this collection.
- **Temporal coverage** (first and last populated init date, verified by listing every date
  directory):

  | Model | Path | Coverage |
  |---|---|---|
  | ACCESS-G3 | `access-g/1` | 2019-07-23 → 2025-06-26 (2170 date dirs; late-June 2025 dirs exist but are empty) |
  | ACCESS-G3 parallel suite | `access-g/4003` | 2022-11-07 → 2025-03-25 (790 dates, reduced 42-variable set, `expt_id=4003`) |
  | ACCESS-GE3 | `access-ge/1` | 2019-07-23 → 2025-06-26 (same empty tail as ACCESS-G, plus a stray empty `00` directory) |
  | ACCESS-C3 SY | `access-sy/1` | 2020-09-25 → 2023-09-20, plus an isolated 2024-05-15…17 |
  | ACCESS-C3 VT / AD / BN / PH | `access-{vt,ad,bn,ph}/1` | 2020-07/09 → 2023-09-20 |
  | ACCESS-C3 DN | `access-dn/1` | 2020-09-01 → 2023-09-20 |
  | ACCESS-C3 NQ | `access-nq/1` | 2022-05-06 → 2023-09-20 |
  | ACCESS-CE3 SY/VT/BN | `access-{sye,vte,bne}/1` | 2020-09-01 → 2023-09-20 |
  | ACCESS-CE3 NQ / PH | `access-{nqe,phe}/1` | 2022-05/05 → 2023-09-20 |
  | ACCESS-CE3 AD / DN | `access-{ade,dne}/1` | sparse, 28–29 dates only, 2022-09 → 2023-04 |

  The NCI collection carries **no APS4 data**: there is no `ops_aps4` collection, and the
  collection's own THREDDS notice states the Bureau paused the feed from May 2025 during a
  platform upgrade.

- **Temporal frequency**:
  - ACCESS-G3: 4 runs/day (00, 06, 12, 18 UTC). 00Z and 12Z forecast hourly to T+240; 06Z and 18Z
    hourly to T+84. Analysis is a single step at T+0.
  - ACCESS-GE3: 4 runs/day, 3-hourly from T+3 to T+246 (82 steps) at every init hour. Exception:
    `accum_prcp` is hourly, T+1 to T+246 (246 steps).
  - ACCESS-C3: analyses **hourly** (24 per day, from the hourly 4D-Var cycle); forecasts only at
    00/06/12/18 UTC — hourly to T+36 for 00/06/12Z and to T+42 for 18Z.
  - ACCESS-CE3: forecasts hourly to T+42.
  - `fcmm` (ACCESS-G): 10-minute steps to T+72 (432 steps) over an Australian sub-domain.
- **Latency**: not measurable — the feed stopped in mid-2025, so no file has a publication time
  relative to a live run. The BOM run cadence above is the only schedule that could be verified;
  latency must be measured against the Registered User feed before any operational design.
- **Access notes**: fully anonymous over HTTPS; no login, no token. Two services on the same paths:
  `/thredds/fileServer/…` for whole-file download and `/thredds/dodsC/…` for OPeNDAP
  (`.dds`, `.das`, `.ascii` and subsetted reads). OPeNDAP makes metadata and point sampling cheap.
  **Throughput is the problem**: sustained download measured at ~3 MB/s from a single connection,
  against files of ~1.2 GB (`access-g … fc/sfc/temp_scrn.nc`, 1,168,222,618 bytes) and ~1.6 GB
  (`accum_prcp.nc`). One ACCESS-G 00Z run holds ~85 surface variables, so roughly 100 GB per run
  for surface fields alone. Internal HDF5 chunking is `(1, 768, 1024)` for ACCESS-G surface fields,
  i.e. four chunks per time step, which suits ranged reads if a virtual approach is used.
- **License**: **Creative Commons Attribution 4.0 International**, stated both in the collection's
  ISO record and in `https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/license.txt`.
  Openly redistributable with attribution. DOI `10.25914/608a993391647`.
- **Browse root**: `https://thredds.nci.org.au/thredds/catalog/catalogs/wr45/ops_aps3/ops_aps3.html`
- **URL format**:
```
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/{model}/{stream}/{YYYYMMDD}/{HH}00/{an|fc|fcmm}/{sfc|pl|ml}/{variable}.nc
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/{model}/{stream}/{YYYYMMDD}/{HH}00/{cf|pf/NNN}/{sfc|pl|ml}/{variable}.nc   - ensembles
https://thredds.nci.org.au/thredds/dodsC/wr45/ops_aps3/...                                                                            - same paths over OPeNDAP
https://thredds.nci.org.au/thredds/catalog/wr45/ops_aps3/{...}/catalog.xml                                                            - machine-readable listing
```
- **Example URLs**:
```
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/20240101/0000/fc/sfc/temp_scrn.nc
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/20240101/0000/an/pl/air_temp.nc
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-sy/1/20230918/0000/fc/sfc/accum_prcp.nc
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-ge/1/20240101/0000/pf/001/sfc/temp_scrn.nc
https://thredds.nci.org.au/thredds/dodsC/wr45/ops_aps3/access-g/1/20240101/0000/fc/sfc/temp_scrn.nc.das
```

### GRIB Index

Not applicable — this collection is NetCDF-4 only.

### Coordinate Reference System

- **Common name**: WGS84 geographic, regular latitude/longitude.
- **PROJ string or EPSG**: EPSG:4326. Files carry no CRS variable or `grid_mapping` attribute; the
  only spatial metadata is `lat: degrees_north`, `lon: degrees_east`, both tagged
  `type = "uniform"`. Longitudes run 0–360.

### Dimensions & Dimension Coordinates

**ACCESS-G3** (`modl_vrsn = ACCESS-G`), scalar/theta grid:

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| init_time | 2019-07-23 00Z | 2025-06-26 18Z | 6 h | 00/06/12/18 UTC |
| lead_time | 1 h | 240 h (00/12Z), 84 h (06/18Z) | 1 h | `fc` starts at T+1; T+0 is the `an` file |
| latitude | -89.941406 | 89.941406 | 0.1171875 (= 180/1536) | 1536 cells, descending in file, pixel centres |
| longitude | 0.087890625 | 359.912109 | 0.17578125 (= 360/2048) | 2048 cells, pixel centres |
| pressure_level | 1000 Pa | 100000 Pa | irregular | 27 levels, `an/pl` only (see list below) |
| model_level | 20 m | 80000 m | irregular | 70 hybrid-height theta levels, `atmosphere_hybrid_height_coordinate`, `formula_terms = "a: A_theta b: B_theta orog: topog"`; `A_theta`/`B_theta` ship inside each `ml` file, `topog` is `an/sfc/topog.nc` |

Pressure levels (Pa): 1000, 2000, 3000, 5000, 7000, 10000, 15000, 17500, 20000, 22500, 25000,
27500, 30000, 35000, 40000, 45000, 50000, 60000, 70000, 75000, 80000, 85000, 90000, 92500, 95000,
97500, 100000.

**ACCESS-GE3** (`modl_vrsn = ACCESS-GE`, `total_number_of_forecasts = 18`):

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| init_time | 2019-07-23 00Z | 2025-06-26 18Z | 6 h | |
| lead_time | 3 h | 246 h | 3 h | 82 steps; `accum_prcp` is hourly T+1…T+246 instead |
| latitude | -89.85 | 89.85 | 0.3 | 600 cells |
| longitude | 0.225 | 359.775 | 0.45 | 800 cells |
| ensemble_member | 0 | 17 | 1 | `cf` = member 0, `pf/001`…`pf/017` |

**ACCESS-C3**, 0.0135° (~1.5 km), 36 h forecasts (42 h from 18Z), hourly analyses:

| Domain | lat range | lon range | Shape (lat × lon) |
|---|---|---|---|
| ACCESS-SY (Sydney) | -38.0 … -27.9695 | 144.0 … 156.0285 | 744 × 892 |
| ACCESS-BN (Brisbane) | -31.5 … -21.4695 | 145.0 … 157.0285 | 744 × 892 |
| ACCESS-AD (Adelaide) | -39.5 … -29.4695 | 130.0 … 142.0285 | 744 × 892 |
| ACCESS-PH (Perth) | -37.0 … -26.9695 | 112.0 … 124.0285 | 744 × 892 |
| ACCESS-DN (Darwin) | -18.0 … -7.9695 | 127.0 … 139.0285 | 744 × 892 |
| ACCESS-NQ (North Qld) | -22.5 … -12.4695 | 139.0 … 151.0285 | 744 × 892 |
| ACCESS-VT (Vic/Tas) | -46.0 … -32.9995 | 139.0 … 151.0015 | 964 × 890 |

**ACCESS-CE3**, 0.0198° (~2 km), 42 h forecasts, 12 members (`cf` + `pf/001`…`pf/011`):
ACCESS-SYE 506 × 606 over -38.0…-28.001 / 144.0…155.979; ACCESS-VTE 656 × 606 over
-46.0…-33.031 / 139.0…150.979.

**`fcmm` sub-domain** (ACCESS-G): 700 × 680 on the ACCESS-G grid spacing, lat -64.98047…16.93359,
lon 65.12695…184.48242, 10-minute steps T+0:10 … T+72:00.

We use pixel centres, and these files already are: the ACCESS-G latitudes are offset half a step
from the poles and the longitudes half a step from the prime meridian.

### Data Variables

BOM uses its own short names, not GRIB/CF names. Availability below is from
`access-g/1/20240101/0000/fc/sfc` (85 variables); ACCESS-C carries an equivalent set with the
convection-scheme fields dropped and convective-scale diagnostics added.

| Variable name | Level | Units | Source name | Notes |
|---------------|-------|-------|-------------|-------|
| temperature_2m | 2 m | K | `temp_scrn` | `accum_type = instantaneous` |
| wind_u_10m | 10 m | m s-1 | `uwnd10m` | on the staggered **u** grid — see Notable Observations |
| wind_v_10m | 10 m | m s-1 | `vwnd10m` | on the staggered **v** grid, 1537 latitudes |
| wind_u_100m / wind_v_100m | 100 m | m s-1 | — | not available as a single-level field; model level 3 is 100 m (`fc/ml/wnd_ucmp`, `wnd_vcmp`) |
| precipitation_surface | surface | kg m-2 | `accum_prcp` | running total, never resets within a run |
| downward_short_wave_radiation_flux_surface | surface | W m-2 | `av_swsfcdown` | mean over the preceding hour (`accum_type = mean`, `accum_value = 60` minutes) |
| downward_long_wave_radiation_flux_surface | surface | W m-2 | `av_lwsfcdown` | same averaging |
| pressure_surface | surface | Pa | `sfc_pres` | |
| pressure_reduced_to_mean_sea_level | MSL | Pa | `mslp` | `av_mslp` is the hourly mean variant |
| total_cloud_cover_atmosphere | atmosphere | 1 | `ttl_cld` | fraction 0–1, **not** percent |
| relative_humidity_2m | 2 m | % | `rh_scrn` | "w.r.t. water" |
| specific_humidity_2m | 2 m | kg kg-1 | `qsair_scrn` | |
| dew_point_temperature_2m | 2 m | K | `dewpt_scrn` | `dewpt_scrn_proxy` is a separate approximation |

Other notable single-level fields: `wndgust10m`, `tmax_scrn` / `tmin_scrn` (`accum_type` maximum /
minimum), `abl_ht`, `low_cld` / `mid_cld` / `hi_cld`, `precwtr`, `visibility` and the
`vis_*` / `prob_vis_*` family, `fog_fraction`, eight `cld_base_gt*` ceiling thresholds,
`soil_temp[2-4]` / `soil_mois[2-4]` (four soil layers), `snow_amt_lnd`, `seaice`,
`accum_conv_rain` / `accum_ls_rain` / `accum_conv_snow` / `accum_ls_snow`, `accum_evap`,
`cld_phys_thunder_p`, surface stresses and heat fluxes.

ACCESS-C adds convective-scale diagnostics absent from ACCESS-G: `max_radar_refl_1km`,
`max_maxcol_refl`, `max_maxcol_hail_diam`, `max_updraft_helicity` / `min_updraft_helicity`,
`max_maxcol_vert_wnd`, `max_wndgust10m`, `accm_n_lightning_fl`, `h_eff_ruff`, `tiles_pot_et`.
It drops the convection-scheme split (`accum_rain` / `accum_snow` replace the `conv`/`ls` pairs)
and drops `cld_phys_thunder_p`, `conv_cldbse_pres`, `conv_cldtop_pres`, `precwtr`.

Multi-level variables: `an/pl` carries `air_temp`, `geop_ht`, `relhum`, `vertical_wnd`,
`wnd_ucmp`, `wnd_vcmp` (6 variables, 27 pressure levels). `an/ml` carries 15 variables on 70 model
levels; `fc/ml` carries only `air_temp`, `spec_hum`, `wnd_ucmp`, `wnd_vcmp` and only on the lowest
4 model levels (20, 53.3, 100, 160 m). **There is no `fc/pl`** — pressure-level data exists for the
analysis step only in this collection.

**Temporal availability changes**:
- Files dated 2020-07-01 and earlier carry seven extra legacy header variables alongside the data
  variable: `seg_type`, `base_date`, `base_time`, `valid_date`, `valid_time`, `wrtn_date`,
  `wrtn_time`. Files dated 2020-10-01 and later do not. The exact changeover lies between
  2020-07-01 and 2020-10-01 and was not narrowed further.
- The surface variable count drifts over time and run to run: ACCESS-G `fc/sfc` held 83–86 files
  across April–June 2025, and ACCESS-GE `cf/sfc` held 84 files in 2019 against 81 in 2025. A fixed
  variable list will hit missing files.
- ACCESS-NQ and its ensemble start 2022-05, well after the other C domains (2020-09).
- The parallel-suite stream `access-g/4003` carries only 42 surface variables.

### Sample Files Examined

- **Early archive**: 2019-07-23 00Z, `access-g/1/20190723/0000/fc/sfc/temp_scrn.nc`
- **Mid archive**: 2024-01-01 00Z/06Z/12Z/18Z, `access-g/1/20240101/...` — surface, pressure and
  model levels, `an`, `fc` and `fcmm`
- **Late archive**: 2025-06-20 and 2025-06-26 listings; 2025-06-27…30 confirmed empty
- **Parallel suite**: `access-g/4003/20240101/0000/fc/sfc/temp_scrn.nc`
- **Ensemble**: `access-ge/1/20240101/{0000,0600,1200,1800}/{cf,pf/001}/sfc/...`
- **Convective scale**: `access-sy/1/20230918/{0000,0600,1200,1800}/...` and one file from each of
  ad, bn, dn, nq, ph, vt on the same date; `access-sy/1/20240517` for the isolated 2024 window
- **Convective ensemble**: `access-sye/1/20230918`, `access-vte/1/20230918`

### Notable Observations

**Winds are on staggered Arakawa C grids and are not de-staggered.** Verified on
`access-g/1/20240101/0000/fc/sfc`:

- scalar/theta grid — lat 1536 (89.941406 … -89.941406), lon 2048 starting at 0.087890625
- `uwnd10m` — same 1536 latitudes, but lon 2048 starting at **0.0**, shifted half a cell west
- `vwnd10m` — same longitudes as theta, but **1537** latitudes running 90.0 … -90.0, i.e. on cell
  edges including both poles

Every wind variable must be interpolated onto the theta grid before it can be written alongside the
scalar fields. BOM's own documentation confirms the raw model output is staggered and notes that
the Registered User GRIB2 products interpolate the winds to a uniform grid — the NCI NetCDF files
do not.

**A `+9999.0` sentinel leaks past the declared fill value.** Every variable declares
`_FillValue = missing_value = -9999.0`, but land-surface variables store **`+9999.0`** where they do
not apply. Verified against `an/sfc/lnd_mask.nc` on a 6400-cell sample (exact equality, perfect
agreement with the mask):

- `soil_temp`, `soil_mois`, `soil_mois_cont`, `canopy_wtr_cont` — `9999.0` at every sea cell
- `seaice` — `9999.0` at every land cell

A CF-aware reader masks `-9999.0` and passes `+9999.0` through as data. No `-9999.0` was found in
any sampled field. Note also that `visibility` legitimately exceeds 9998 (max 19170 m observed), so
a threshold test rather than exact equality would corrupt it. `soil_mois_cont` also returned
negative values (-87) over land, which was not investigated.

**Accumulations are run totals that start before the base time.** `accum_prcp` is monotonically
non-decreasing across all 240 forecast steps in ACCESS-G and all 36 in ACCESS-SY — it never resets
within a run. The window starts in the assimilation cycle, not at T+0: the ACCESS-G analysis file
holds a non-zero `accum_prcp` at T+0 (0.0073 kg m-2 at a sampled point) with `accum_value = 180`
minutes, i.e. a 3-hour DA window.

The `accum_value` attribute is the accumulation window of the **first** record, and its unit
disagrees with its own `accum_units = "minutes"` between models. ACCESS-G `fc` gives 240 = 4 hours
(3 h DA + 1 h to the first step) in minutes; ACCESS-SY `fc` gives 5400 and `an` gives 1800, which
only make sense as **seconds** (90 min and 30 min — a 30-minute DA window, matching BOM's
documentation). Do not read this attribute without a per-model unit assumption. `accum_type = mean`
variables (the `av_*` family) are consistent: `accum_value = 60` minutes on both ACCESS-G and
ACCESS-SY, a mean over the preceding hour.

**Other structural notes**:
- The `time` coordinate is validity time, not lead time. Lead time must be derived from the
  global `base_date` / `base_time` attributes. The `forc_minutes` variable that looks like a lead
  time is explicitly attributed `WARNING = "DEPRECATED, DO NOT USE"`.
- Output frequency varies by variable within one run. ACCESS-GE is 3-hourly except `accum_prcp`,
  which is hourly. ACCESS-G `fc/ml` has 162 steps for `air_temp` and 136 for `wnd_ucmp` against 240
  hourly steps for surface fields.
- `ttl_cld` is a fraction with `units = "1"`, unlike the percent convention used by most sources.
- Files carry `Conventions = "CF-1.5,ACDD-1.3"` but no `grid_mapping`, no `standard_name` on data
  variables, and no bounds variables. Each variable carries its UM `stash_code`, which is the
  reliable key for cross-referencing BOM's field documentation.
- ACCESS-C files add `least_significant_digit = 2`, so they are already lossily rounded at source.
- Empty date directories exist inside the covered range (2025-06-15…17 and 2025-06-27…30 in both
  ACCESS-G and ACCESS-GE) — presence of a date directory does not imply data.
- The catalog is enumerable via `catalog.xml` at any level, which is what any backfill should walk
  rather than assuming a date pattern.

---

## Source Information — BOM Registered User feed (second source, not open)

- **Summary of data organization**: `IDYnnnnn.version.fields.levels.base-time.forecast-hour.grid-coords.ext`,
  where `IDYnnnnn` selects model and domain (IDY25000 ACCESS-G global, IDY25001/IDY25006
  Australian/greater-Australian subsets, IDY25400–IDY25405 the ACCESS-C domains), `fields` selects
  a bundle (`all-flds`, `pop-flds`, `group1`–`group4`, `wind`, `prec`, `pres`, `radar`,
  `helicity`, `syncld`, `topog`, `mask`) and `levels` selects a level bundle.
- **File format**: GRIB edition 2 and NetCDF-4.
- **Temporal coverage**: current operations. This is the **only** channel carrying APS4, which
  replaced APS3 and is what runs today. BOM states products are unchanged from APS3 to APS4; the
  APS4 global model assimilates more observations and has improved physics, while the ACCESS-C/CE
  upgrade was technical only.
- **Latency**: not verifiable without a subscription.
- **Access notes**: FTP/SFTP for Registered User subscribers. Bundled multi-variable files rather
  than one file per variable, and **winds are already interpolated to the uniform grid** — the
  single largest processing difference from the NCI files.
- **License**: not open. Subscription terms at `reg.bom.gov.au/other/charges.shtml`; the free
  anonymous-FTP products are separately marked "not for commercial use". This needs a licensing
  conversation before any integration depends on it.
- **Browse root**: `https://www.bom.gov.au/nwp/doc/access/NWPData.shtml` (documentation and sample
  files only; the data itself is behind the subscription).

## What this means for integration

1. **A CC BY 4.0 archive exists and is usable**: ACCESS-G3 hourly to T+240, 2019-07-23 through
   2025-06-26, on a 12 km global grid. That is a complete, openly licensed, ~6-year forecast
   archive.
2. **It is a closed archive.** Nothing arrives after mid-2025, so this can only ever be a
   historical dataset unless the Registered User feed is licensed for the operational tail. The
   APS3→APS4 boundary would sit exactly at that join.
3. **ACCESS-C is much weaker**: it ends 2023-09-20 and each domain is a separate 1.5 km grid, so
   there is no single-dataset framing without either seven separate datasets or a mosaic.
4. **Throughput is the main engineering risk** on a backfill: ~100 GB per ACCESS-G run of surface
   fields at ~3 MB/s from one origin.
5. **Three data quirks must be handled before any write**: de-stagger the winds onto the theta
   grid, mask `+9999.0` on land-surface variables (it is not the declared fill value), and
   deaccumulate `accum_prcp` knowing the window opens 3 hours before the base time.
