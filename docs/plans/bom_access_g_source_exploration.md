# Source Data Exploration: BOM ACCESS-G

Filled-in template per [source_data_exploration_guide.md](../source_data_exploration_guide.md). Every
structural claim below was checked against files served from NCI THREDDS; items that could not be
verified from real files are marked **unverified**.

## ⚠️ Headline finding: the only open source is a closed archive

The one openly licensed ACCESS-G archive found is NCI project `wr45` (`ops_aps3`). It **stopped
receiving data**: the last init time carrying files is **2025-06-26 18Z**, and the NCI collection
record sets maintenance frequency to `notPlanned`. The dataset scan carries this notice verbatim:

> The Bureau of Meteorology is upgrading key platforms and services. From May 2025 there will likely
> be a pause in the feed of new daily data from the Bureau. This may mean that no new data will be
> added to this project. Please note that this change will not affect access to the existing data
> already available under project wr45. If you or your organisation requires access to operational
> data feeds directly from the Bureau, please check the products available and contact
> webreg@bom.gov.au if you have any queries about these services.

Consequences for integration:

- A **fixed historical archive** (2019-07-23 → 2025-06-26) is fully achievable, and the license
  permits redistribution.
- **No operational updates are possible from this source.** Its cron jobs would never find new data.
- BOM's live equivalent is the Registered User subscription feed (`reg.bom.gov.au`), which is
  paid/registered and whose terms are not the CC-BY covering `wr45`. Searching turned up no open
  real-time ACCESS-G source, and no APS4 collection on NCI (`wr45` contains only `ops_aps3`;
  `ops_aps4` and `ops_aps5` are 404).

---

## Dataset: BOM (Australian Bureau of Meteorology) ACCESS-G

### Source Information

- **Summary of data organization**: One NetCDF4 file per **variable**, containing all lead times for
  one init time on the full global grid. Files are grouped by init date → init hour → product
  (`an` analysis / `fc` forecast / `fcmm` regional 10-minute forecast) → level type (`sfc` surface,
  `pl` pressure levels, `ml` model levels). One forecast surface variable is 0.2–1.6 GiB.
- **File format**: NetCDF4 (HDF5), `Conventions = "CF-1.5,ACDD-1.3"`. Internally chunked with
  shuffle + gzip level 1. No GRIB is served from this source.
- **Temporal coverage**: 2019-07-23 → 2025-06-26 (see closed-archive note above). Daily directories
  are **gapless** across that span; per-init completeness gaps are listed under Notable Observations.
- **Temporal frequency**: 4 init times per day (00, 06, 12, 18 UTC).
  - **00Z and 12Z**: hourly lead times **1–240 h**.
  - **06Z and 18Z**: hourly lead times **1–84 h** only.
  - Lead time 0 is absent from `fc`; the `an` (analysis) file for the same init time is the t+0 field.
  - This 240/84 split is stable across the whole archive (checked 2019-07-23, 2024-01-15, 2025-06-26).
- **Latency**: Measured from `Last-Modified` on NCI for 2025-06-25: first files land ~7 h after init
  and the slowest surface variable ~7.5 h (00Z → 09:12–09:31Z; 06Z → 13:01–13:12Z; 12Z → 20:13–20:34Z;
  18Z → 00:58–01:15Z next day). This is NCI archival latency, not BOM's real-time feed latency.
- **Access notes**:
  - `www.bom.gov.au` returns **403** without a browser `User-Agent`; NCI THREDDS does not care.
  - THREDDS offers `fileServer` (whole file), `dodsC` (OPeNDAP), `ncss` (NetCDF Subset Service),
    `wms`, `wcs`. Range requests on `fileServer` work (HTTP 206), so partial reads are possible.
  - OPeNDAP `.dds` / `.das` / `.ascii` are the cheap way to inspect these files — a full forecast
    file is ~1 GiB, and metadata plus a point series costs a few KB.
  - Throughput measured **from this environment through its egress proxy** was ~1.2 MB/s
    single-stream and did not improve with 4 parallel range reads. Real throughput from a cloud
    region should be re-measured before sizing a backfill; if ~1.2 MB/s were representative the
    volumes below would be prohibitive.
  - **No NetCDF engine is currently installed in this repo** (`netCDF4`, `h5netcdf`, `scipy` all
    absent; only `gribberish`, `rasterio`, `zarr`). rasterio/GDAL reads these files via its `netCDF`
    driver — see Notable Observations for the required subdataset syntax.
- **License**: **Creative Commons Attribution 4.0 International (CC-BY 4.0)** —
  `https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/license.txt`, corroborated by the NCI
  collection record (`Creative Commons Attribution 4.0 International`, DOI
  `10.25914/608a993391647`, title "APS3 ACCESS Numerical Weather Prediction (NWP) Models -
  Operational Reference Data Collection"). Open, redistribution permitted, attribution required.
- **Browse root**: https://thredds.nci.org.au/thredds/catalog/wr45/ops_aps3/access-g/catalog.html
- **URL format**:
```
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/{YYYYMMDD}/{HH}00/{an|fc|fcmm}/{sfc|pl|ml}/{variable}.nc
https://thredds.nci.org.au/thredds/dodsC/wr45/ops_aps3/access-g/1/{YYYYMMDD}/{HH}00/{an|fc|fcmm}/{sfc|pl|ml}/{variable}.nc   (OPeNDAP; append .dds/.das/.ascii?var)
https://thredds.nci.org.au/thredds/ncss/grid/wr45/ops_aps3/access-g/1/{YYYYMMDD}/{HH}00/{an|fc|fcmm}/{sfc|pl|ml}/{variable}.nc  (subset service)
https://thredds.nci.org.au/thredds/catalog/wr45/ops_aps3/access-g/1/{YYYYMMDD}/{HH}00/{an|fc|fcmm}/{sfc|pl|ml}/catalog.xml     (listing + file sizes)
```
- **Example URLs**:
```
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/20240115/0000/fc/sfc/temp_scrn.nc
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/20240115/0000/an/sfc/temp_scrn.nc
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/20240115/0000/an/pl/air_temp.nc
https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/20190723/0000/an/sfc/seaice.nc
https://thredds.nci.org.au/thredds/dodsC/wr45/ops_aps3/access-g/1/20250626/0000/fc/sfc/accum_prcp.nc.das
```

There are two experiment streams under `access-g/`: **`1`** (`expt_id = "0001"`, the operational
suite — use this) and **`4003`** (`expt_id = "4003"`, a parallel trial: same grid and same 240 hourly
steps, but only ~42 surface variables, 2022-11-07 → 2025-03-25 with 80 missing days). `4003` adds one
variable `1` lacks, `max_wndgust10m`.

### GRIB Index (if applicable)

- **Index files available**: No — this source serves NetCDF4 only, no `.idx` sidecars.
- **Index style**: N/A.

BOM does publish the same fields as GRIB2 (`https://www.bom.gov.au/nwp/doc/access/GRIB2notes.shtml`),
but only through the Registered User subscription feed, not through `wr45`. **Unverified** — FTP
egress is blocked from this environment, so the GRIB2 sample directories
(`ftp://ftp.bom.gov.au/register/sample/access/grib2/ACCESS-G3/`) could not be inspected.

### Coordinate Reference System

- **Common name**: WGS84 geographic (regular lat/lon). Files carry **no** `grid_mapping` /
  `crs` variable — GDAL reports `crs = None` — so the CRS must be asserted by us.
- **PROJ string or EPSG**: `EPSG:4326`.

### Dimensions & Dimension Coordinates

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| init_time | 2019-07-23 00Z | 2025-06-26 18Z | 6 h | 00/06/12/18 UTC; archive closed after this |
| lead_time | 1 h | 240 h | 1 h | 00Z/12Z only; **06Z/18Z stop at 84 h**. t+0 lives in the `an` file, not `fc`. A few variables are coarser — see Data Variables |
| latitude | -89.94140625 | 89.94140625 | 0.1171875 | 1536 points, stored **descending** (north → south) |
| longitude | 0.087890625 | 359.912109375 | 0.17578125 | 2048 points, ascending, 0–360 |
| pressure_level | 1000 Pa | 100000 Pa | irregular | 27 levels, `an` only (no `fc/pl`). Units **Pa**: 1000, 2000, 3000, 5000, 7000, 10000, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 35000, 40000, 45000, 50000, 60000, 70000, 75000, 80000, 85000, 90000, 92500, 95000, 97500, 100000 |
| model_level (`theta_lvl`) | 20 m | 160 m | irregular | `fc/ml`: 4 levels (20, 53.336, 100, 160 m). `an/ml`: 70 levels |
| model_level (`rho_lvl`) | 10 m | 130 m | irregular | `fc/ml` winds only: 10, 36.664, 76.664, 130 m |
| ensemble_member | — | — | — | Not in ACCESS-G. The ensemble is a separate collection, `access-ge` |

Spatial coordinates are **pixel centers**: 89.94140625 = 90 − 0.1171875/2 and 0.087890625 =
0.17578125/2, so the grid spans the full globe edge-to-edge with cell-centred values.

`time` is stored as `Int32` seconds since the init time (`units = "seconds since {init} 00:00:00"`,
`calendar = "gregorian"`), i.e. it is a valid-time axis. Every file also carries `forc_minutes`,
explicitly attributed `WARNING = "DEPRECATED, DO NOT USE"`.

### Data Variables

Names below are the `fc/sfc` file names. Verified against `20240115/0000` (units/`long_name` read from
`.das`; grids and step counts from `.dds`).

| Variable name | Source variable | Level | Units | Available from | Notes |
|---------------|-----------------|-------|-------|----------------|-------|
| temperature_2m | `temp_scrn` | screen (2 m) | K | 2019-07-23 | instantaneous |
| wind_u_10m | `de_uwnd10m` | 10 m | m s-1 | 2019-07-23 | **use the `de_` (destaggered) variant** |
| wind_v_10m | `de_vwnd10m` | 10 m | m s-1 | 2019-07-23 | **use the `de_` variant** |
| wind_u_100m | `fc/ml` `wnd_ucmp` | `rho_lvl` 76.664 / 130 m | m s-1 | 2019-07-23 | **no exact 100 m level**; needs interpolation or nearest-level choice. Only 136 lead times |
| wind_v_100m | `fc/ml` `wnd_vcmp` | same | m s-1 | 2019-07-23 | as above, **and** on the staggered `lat=1537` grid with no destaggered variant |
| precipitation_surface | `accum_prcp` | surface | kg m-2 | 2019-07-23 | **accumulated from init over the whole run** (verified 9.8e-4 at 1 h → 383.3 at 240 h). Needs deaccumulation |
| downward_short_wave_radiation_flux_surface | `av_swsfcdown` | surface | W m-2 | 2019-07-23 | mean over the preceding step; verified as a per-hour mean (clean diurnal cycle to 0 at night), not a run-cumulative mean |
| downward_long_wave_radiation_flux_surface | `av_lwsfcdown` | surface | W m-2 | 2019-07-23 | per-step mean |
| pressure_surface | `sfc_pres` | surface | Pa | 2019-07-23 | |
| pressure_reduced_to_mean_sea_level | `mslp` | MSL | Pa | 2019-07-23 | `av_mslp` is a per-step mean variant |
| total_cloud_cover_atmosphere | `ttl_cld` | atmosphere | **1 (fraction 0–1)** | 2019-07-23 | ×100 if publishing as % |
| relative_humidity_2m | `rh_scrn` | screen | % | 2019-07-23 | w.r.t. water; observed up to **121%** (supersaturation) |
| specific_humidity_2m | `qsair_scrn` | screen | kg kg-1 | 2019-07-23 | |
| dew_point_temperature_2m | `dewpt_scrn` | screen | K | 2019-07-23 | `dewpt_scrn_proxy` is a separate "near surface proxy dewpoint", not the same field |

Other notable variables available at every init time (all `fc/sfc` unless stated):

- Wind/gust: `wndgust10m` (m s-1); `uwnd_strs` / `vwnd_strs` surface wind stress (N m-2, v staggered).
- Temperature extremes: `tmax_scrn`, `tmin_scrn` (K) — max/min over the preceding step (1 h), verified
  bracketing `temp_scrn` at the same step.
- Precipitation partition: `accum_ls_rain`, `accum_conv_rain`, `accum_ls_snow`, `accum_conv_snow`
  (kg m-2, all accumulated from init); `accum_evap` (accumulated), `accum_evap_sea` (a **rate**,
  kg m-2 s-1, despite the `accum_` prefix).
- Radiation: `av_netswsfc`, `av_netlwsfc`, `av_olr`, `av_swirrtop`, `av_oswrad_flx`,
  `av_sfc_sw_dir`, `av_sfc_sw_dif` (all W m-2, per-step means).
- Turbulent fluxes: `sens_hflx`, `lat_hflx`, `sfc_mois_flx` plus `av_` counterparts.
- Cloud: `low_cld`, `mid_cld`, `hi_cld` (fraction); `cld_base_gt{0p1,1p5,…,7p9}` cloud-base heights in
  **kft** (kilofeet); `conv_cldbse_pres`, `conv_cldtop_pres` (Pa); `cld_phys_thunder_p`.
- Visibility/fog: `visibility`, `vis_excl_prcp`, `vis_conv_pptn`, `vis_ls_pptn` (m);
  `prob_vis_1km_ppt`, `prob_vis_5km_ppt`, `vis_prob` (probabilities); `fog_fraction`.
- Land/surface: `sfc_temp`, `snow_amt_lnd`, `soil_temp`…`soil_temp4`, `soil_mois`…`soil_mois4`,
  `soil_mois_cont`, `canopy_wtr_cont`, `veg_ruff`, `seaice`, `abl_ht`, `precwtr`.
- **`an/sfc` only**: `lnd_mask` (0/1) and `topog` (gpm) — static fields, absent from `fc/sfc`. The
  two sets are otherwise identical: `an/sfc` (87) = `fc/sfc` (86) plus these two, minus `pa229109`.
- Undocumented: `pa229109` and `vis_ls_pptn` carry **no** `units` or `long_name`.

Not all variables share the 240-step hourly axis. Verified step structures for the 00Z run:

| Step structure | Steps | Variables |
|---|---|---|
| hourly 1–240 h | 240 | most `fc/sfc` variables |
| hourly 1–84 h, then 3-hourly to 240 h | 136 | `precwtr`, `fc/ml` `wnd_ucmp`, `fc/ml` `wnd_vcmp` |
| hourly 1–85 h, then a mixed 1/2/3 h pattern to 240 h | 162 | `fc/ml` `air_temp`, `fc/ml` `spec_hum` |
| hourly 1–72 h, 3-hourly to 120 h, 6-hourly to 240 h | 108 | `cld_phys_thunder_p` |

**Temporal availability changes** (`fc/sfc`, 00Z, comparing 2019-07-23 / 2024-01-15 / 2025-06-26 —
83 / 86 / 85 variables):

- Present in 2019 but gone by 2024: `cld_base_gt0p5`, `vis_precip0`.
- Added between 2019 and 2024: `cld_base_gt1p5`, `cld_phys_thunder_p`, `dewpt_scrn_proxy`,
  `pa229109`, `precwtr`.
- Removed between 2024 and 2025: `pa229109`.
- Every variable in the core table above is present for the whole archive.

### Sample Files Examined

- **Start of archive**: 2019-07-23 —
  `.../access-g/1/20190723/0000/{an/sfc/{seaice,soil_temp,lnd_mask},fc/sfc/temp_scrn,an/pl/air_temp}.nc`
- **Recent / last data**: 2025-06-25 and 2025-06-26 —
  `.../access-g/1/2025062{5,6}/{0000,0600,1200,1800}/{an,fc}/sfc/*.nc`
- **Mid-archive reference**: 2024-01-15 — all six groups (`an/{sfc,pl,ml}`, `fc/{sfc,ml}`, `fcmm/sfc`),
  full `.das`/`.dds` sweep of every variable, plus 11 `an/sfc` files downloaded and read.
- **Format transition boundaries**: 2020-07-20/21, 2021-09-29/30, 2022-01-11/12 (bisected, below).
- **Structural sweep**: 54 dates spread across the archive, confirming grid and step count are
  constant (`time=240 lat=1536 lon=2048 source=APS3 modl_vrsn=ACCESS-G` at every one).

### Notable Observations

**1. Winds are on a staggered Arakawa-C grid; use the `de_` variants.** Verified from coordinate values:

| Variable | lat[0] | n lat | lon[0] | Grid |
|---|---|---|---|---|
| `uwnd10m` | 89.941406 | 1536 | **0.0** | staggered in longitude |
| `vwnd10m` | **90.0** | **1537** | 0.087891 | staggered in latitude |
| `de_uwnd10m` | 89.941406 | 1536 | 0.087891 | cell centres ✅ |
| `de_vwnd10m` | 89.941406 | 1536 | 0.087891 | cell centres ✅ |

`av_uwnd10m`/`av_vwnd10m` and `uwnd_strs`/`vwnd_strs` are staggered with **no** destaggered variant, as
are `an/ml` and `fc/ml` `wnd_vcmp` (`lat=1537`, on `rho_lvl`). Any use of those needs our own
destaggering. BOM's GRIB2 product is documented as pre-interpolated to a uniform grid; the NetCDF
product is not.

**2. The declared fill value does not match the data's missing marker.** The real
"undefined / does not apply" sentinel is **+9999.0** in every era, and **-9999.0 never appears in the
data at all**. Verified on 2024-01-15 `an/sfc` against `lnd_mask`:

| Variable | cells == 9999 | corresponds to |
|---|---|---|
| `seaice` | 33.87% | **100%** of land cells, 0% of sea cells |
| `soil_temp` | 66.13% | **100%** of sea cells, 0% of land cells |
| `cld_base_gt0p1` | 13.74% | mixed land/sea — cells with no cloud meeting the threshold |
| `visibility` | 0.01% (292 cells) | mixed; cause not established |

Declared `_FillValue` / `missing_value` by era for `seaice`: `9999.0` (correct) up to **2022-01-11**,
`-9999.0` (wrong) from **2022-01-12** onward. The 2019-era `lnd_mask` instead declares
`9.999999616903162e+35`; no value above 1e30 occurs in any file read.

So an integration must mask on **exact equality to 9999.0** and must not trust rasterio's `nodata`.
A **threshold** test would be a bug: valid `visibility` reaches 20130 m and valid `cld_base_gt0p1`
reaches 16740 kft, both above 9999, so `>= 9999` would silently discard good data. Per the
missing-value rules in AGENTS.md, `seaice`, the `soil_*` layers and the `cld_base_*` family are
"quantity does not apply" cases (land/sea/no-cloud) and warrant a `comment` saying what NaN means.

**3. Three file-format transitions, all bisected to the day.** None changes the grid, the variable
values, or the step structure, but the first affects how a reader must open the file:

| Date | Change |
|---|---|
| **2020-07-21** | Auxiliary variables `seg_type` (a *string* variable), `base_date`, `base_time`, `valid_date`, `valid_time`, `wrtn_date`, `wrtn_time` removed. HDF5 chunk shape `(1, 1536, 2048)` → `(1, 768, 1024)`. |
| **2021-09-30** | `lat`/`lon` dtype `Float32` → `Float64`. Coordinate values agree to float32 precision across the change. |
| **2022-01-12** | Declared `_FillValue`/`missing_value` flips from `+9999` to `-9999` (see item 2). |

Because pre-2020-07-21 files contain a string variable, GDAL exposes them as **subdatasets** and
`rasterio.open(path)` yields **zero bands**. Opening `netcdf:"{path}":{variable}` works uniformly in
every era and returns an identical grid and transform — that is the form to use.

**4. Per-init completeness gaps.** Daily directories are gapless 2019-07-23 → 2025-06-30, but the last
four days are empty or partial and some individual inits are short. Observed for `fc/sfc` at 00Z:
2025-06-15 has 0 files; 2025-06-27 through 2025-06-30 have 0 files; 2025-06-26 12Z has 71 of 85
variables. A backfill needs a per-init, per-variable existence check rather than trusting the date
listing. Stream `4003` additionally misses 80 days including a long run 2023-09-21 → 2023-12-06.

**5. `fcmm` is a regional, 10-minute product, not a global one.** Same resolution, but a 700 × 680
Australia-centred window (lat 16.93 → −64.98, lon 65.13 → 184.48) at **10-minute** steps out to
72 h (432 steps), with 26 surface variables. It is a different datacube from `fc` and should not be
mixed into it.

**6. Analysis is dramatically cheaper than forecast.** Per-init sizes from `catalog.xml`:

| Group | Files | Per init | Per day | Whole archive |
|---|---|---|---|---|
| `an/sfc` (all 87) | 87 | 0.26 GiB | 1.0 GiB | **~2.2 TiB** |
| `an/pl` (all 6) | 6 | 0.32 GiB | 1.3 GiB | ~2.7 TiB |
| `an/ml` (all 15) | 15 | 1.64 GiB | 6.6 GiB | ~13.9 TiB |
| `fcmm/sfc` (all 26) | 26 | 4.62 GiB | 18.5 GiB | ~39 TiB |
| `fc/sfc`, 15 core vars | 15 | 13.0 GiB (240 h run) | ~35 GiB | **~74 TiB** |
| `fc/sfc`, all 86 | 86 | 61.6 GiB (240 h run) | ~166 GiB | ~353 TiB |

A 6-hourly **analysis** dataset covering every surface variable for the full six years reads only
~2 TiB — very tractable. The forecast product is one to two orders of magnitude larger. Because there
is exactly one file per variable, downloads are perfectly selective: bytes fetched equal bytes wanted,
and `ncss` can subset further.

**7. A virtual dataset may be feasible, unlike our GRIB sources.** These are HDF5 files chunked
`(1, 768, 1024)` with `shuffle` + `gzip` level 1 — codecs Zarr can express — and `fileServer`
honours range requests. Referencing HDF5 chunks is a different path from our `gribberish`-based
virtual datasets, so this is a **direction, not a verified capability**; the 2020-07-21 chunk-shape
change also splits the archive into two chunk grids, which a single virtual Zarr array cannot span
without a chunking choice that accommodates both.

**8. Model version.** Every file across the archive reports `source = "APS3"`,
`modl_vrsn = "ACCESS-G"`, `expt_id = "0001"`. The NCI record describes the global component as
**ACCESS-G3, ~12 km**. BOM's documentation page mentions an APS4 upgrade, but no APS4 data appears on
NCI and BOM states products are unchanged from APS3 to APS4 for ACCESS-G output. No resolution, grid,
or step change is detectable anywhere in this archive.

---

## How BOM distributes ACCESS-G, and whether a rolling window exists

There are four distinct routes. Only one is openly licensed, and it is the closed archive above.

| Route | Live? | Gridded ACCESS-G? | Licence | Rolling window |
|---|---|---|---|---|
| NCI THREDDS `wr45/ops_aps3` | **No** — ended 2025-06-26 | Yes, full global | CC-BY 4.0 | None; full 6-year archive retained |
| BOM anonymous FTP `ftp.bom.gov.au/anon/gen/` | Yes | **No** | Personal/internal use only, no redistribution | Yes, hours |
| BOM Registered User cloud (S)FTP | Yes | Yes, full global | Paid subscription, real-time only | Not documented |
| Third-party derived (e.g. Open-Meteo) | No — also stopped 2025-06-27 | Derived, not source | Non-commercial | n/a |

**The public anonymous FTP does have a genuine short rolling window, but carries no gridded NWP.**
Its catalogue (`https://www.bom.gov.au/catalogue/data/SMSRPR09.json`, 2865 products) defines
"Delete time ... the time (in hours) an instance of any product will remain on the FTP", with most
operational products at 2–24 h. Every ACCESS-related entry on it is a **chart image** (`IDX0007`
"M.S.L. ANAL (ACCESS-G)", 36 h; `IDX0002` "M.S.L. PROG (ACCESS-G+36)", 120 h) or a text/marine wind
product — none of the `IDY25xxx` grid files. Its terms also forbid redistribution: "you may
download, use and copy that material for personal use, or use within your organisation but you may
not supply that material to any other person or use it for any commercial purpose."

**The Registered User service is the only live route to the grids.** Per the official
[ACCESS-G NWP Data User Guide v2.1, 1 July 2025](https://www.bom.gov.au/catalogue/Bureau_of_Meteorology_ACCESS-G_user_guide.pdf):

- Delivery is **cloud FTP `ftp-reg.cloud.bom.gov.au` and SFTP `sftp-reg.cloud.bom.gov.au`** — the
  guide states products are "only available via cloud FTP ... and SFTP ... and not via
  ftp.bom.gov.au". Files land in a per-subscriber directory, `/access_g3_nwp4` (NetCDF4) and
  `/access_g3_grib2` (GRIB2).
- **File layout differs fundamentally from NCI's.** Names are
  `IDY25NNN.version.fields.levels.base-time.forecast-hour.grid-coords.ext` (ext `grb2` or `nc4`),
  i.e. one file **per forecast hour** carrying many fields — the inverse of NCI's one file per
  variable carrying all forecast hours. Code written against one layout does not read the other.
- Products: `IDY25000` global all-levels, `IDY25020` global surface-only, `IDY25001`/`IDY25021`
  Australian sub-domain, `IDY25006`/`IDY25026` regional sub-domain.
- **No retention period is stated** for this service in the guide. The product catalogue only says
  "these services are **real-time data only**. For historical data, please see the Bureau's Climate
  and Ocean Data Services", which implies a short window but does not quantify it. **Unverified** —
  FTP egress is blocked from this environment, so neither the live directories nor the public sample
  directories under `ftp://ftp.bom.gov.au/register/sample/access/` could be listed. If this route
  matters, the retention window is a question for webreg@bom.gov.au.
- **Cost** (2026/27 financial year, GST inclusive, payment in advance): `IDBY0001` ACCESS-G full
  global bundle **$16,047/yr**; `IDBY0021` global **surface-only $6,019/yr**; Australian sub-domain
  the same two prices. Plus a one-time $1,335 service establishment fee and $1,282/yr Registered
  User FTP registration. Being a cost-recovery subscription, redistribution rights are not implied —
  they would need to be agreed explicitly, unlike the CC-BY 4.0 covering `wr45`.

**Publication latency on the live feed** (guide Table 7, APS4): the `+000` analysis is available ~6 h
after base time and the complete run ~8 h (00Z run: analysis 0600Z, complete 0800Z). This matches the
~7–8.6 h measured from NCI `Last-Modified` timestamps, so NCI was mirroring with little added delay.

**A third party stopping at the same moment corroborates the cutoff.** Open-Meteo's AWS open-data
bucket carries `data/bom_access_global/`, whose newest object is dated **2025-06-27** — one day after
the last NCI init time. Their data is derived and non-commercially licensed, so it is not a usable
source for us, but it is independent evidence that the open feed, not just NCI's copy of it, ended.

**No other open route exists.** All 64 NCI THREDDS collections were listed; the only operational NWP
one is `wr45` (APS3), and there is no APS4 collection (`ops_aps4` and `ops_aps5` are 404). The
ACCESS-adjacent NCI collections are different products: `bs94` ACCESS Regional, `ia89` 400 m
limited-area, `cj37`/`ob53` BARRA/BARRA2 regional **reanalysis** for Australia, `ux62` ACCESS-S2
seasonal. If open Australian coverage is the actual goal rather than ACCESS-G specifically, BARRA2
(`ob53`) is the collection worth exploring next — it is a reanalysis, still maintained, and openly
licensed.

### The live model is APS4/ACCESS-G4, and it is not the model in the archive

The archive is APS3 throughout (`source = "APS3"` in every file). The guide describes the current
operational model as **APS4 / ACCESS-G4**. Two documented differences would matter if the live feed
were ever integrated alongside the archive:

- **Upper-level time steps**: APS4 hybrid and pressure level fields are "3 hours to 72 hours, 6 hours
  thereafter", whereas the APS3 archive has pressure levels **only at analysis time** and `fc/ml` on
  4 levels with a mixed 1/2/3 h step pattern.
- **Low-level winds**: APS4 publishes "Zonal/Meridional wind at the **50 m rho level**", a single
  level, replacing APS3's four rho levels (10, 36.664, 76.664, 130 m).

The run-length asymmetry is unchanged and confirmed independently by the guide: "00 and 12Z Runs
Up to +240 hours; 06 and 18Z Runs Up to +84 hours".

---

## Open questions for scoping

1. **Is a source that cannot receive operational updates in scope at all?** The archive is closed and
   complete; nothing here can keep a store current. A materialized historical dataset with its update
   cronjob permanently suspended is coherent but unlike our other datasets.
2. **If a live dataset is wanted, is a paid subscription on the table?** The Registered User feed is
   $6,019–$16,047/yr plus fees, carries no stated redistribution right, uses a different file layout
   requiring a second reader, and serves APS4 rather than the archive's APS3 — so it would be a
   separate dataset, not a continuation of this one.
3. **Forecast, analysis, or both?** Analysis is ~2 TiB for every surface variable; the forecast is
   ~74 TiB for 15 variables. The analysis is also the only place `lnd_mask` and `topog` live.
4. **How should the 00/12Z-vs-06/18Z asymmetry be represented?** A `lead_time` axis of 1–240 h leaves
   85–240 h empty for half the inits. Restricting to 00Z/12Z gives a 12-hourly, fully dense cube.
5. **What to do about 100 m wind?** There is no 100 m wind level: `theta_lvl` has 100 m but carries
   only temperature and humidity, while the winds sit on `rho_lvl` at 76.664 m and 130 m, are
   published for only 136 of 240 lead times, and `wnd_vcmp` is on the staggered `lat=1537` grid with
   no destaggered counterpart.
6. **Pressure levels are analysis-only** (`an/pl`, 27 levels, no `fc/pl`), so a `pressure_level` group
   is possible for an analysis dataset but not for a forecast one.
7. Should `fcmm` (regional, 10-minute, 72 h) be a separate dataset, or left out entirely?
8. **Is BARRA2 (`ob53`) a better fit for the underlying goal?** It is an openly licensed, still
   maintained high-resolution regional reanalysis for Australia — worth exploring if the aim is open
   Australian coverage rather than ACCESS-G specifically.
