# CHIRPS source data exploration

Filled-out [source data exploration](../source_data_exploration_guide.md) for **CHIRPS:
Rainfall Estimates from Rain Gauge and Satellite Observations** (Climate Hazards Center,
UC Santa Barbara). Point-in-time exploration on 2026-07-21; every value below was read
from actual files (rasterio) or from the CHC HTTP directory listings, not from
documentation alone.

## Version decision: target CHIRPS v3.0

CHIRPS ships two concurrent major versions on the CHC server:

- **v2.0** — `/products/CHIRPS-2.0/`. The long-standing product, 50°S–50°N. Per the
  v2.0 README, **v2.0 production ends December 2026**.
- **v3.0** — `/products/CHIRPS/v3.0/`. Released 2025-01-14. Extends coverage 10°
  poleward to **60°S–60°N**, uses ~90 station sources (v2.0 ~23), adds gauge-undercatch
  (wind) correction, upgrades the satellite algorithm (CHIRP3), and fills gaps with
  0.25° ERA5 rather than 0.5° CFS. Values run overall wetter than v2.0. This is the
  actively supported version.

**Integrate v3.0.** It is the future-supported product with wider coverage and the same
1981-present span. v2.0 is documented below only as historical context / potential
cross-check.

Within v3.0, daily data exists in three flavors that share one grid but differ in how
the daily CHIRPS pentad totals are disaggregated to days and in coverage/latency:

| Flavor | Path | Downscaling driver | Coverage | Observed latest (2026-07-21) |
|--------|------|--------------------|----------|------------------------------|
| **final / rnl** | `daily/final/rnl/` | ERA5 reanalysis | **1981–present** | 2026-03-31 |
| final / sat | `daily/final/sat/` | NASA IMERG Late v07 | 1998–present | 2026-03-31 |
| **prelim / sat** | `daily/prelim/sat/` | NASA IMERG Late v07 | 2025–present | 2026-07-15 |

The two `final` flavors are published together on a slow (~quarterly-observed) cadence;
`prelim` fills the recent months. For a single homogeneous daily archive back to 1981,
**`final/rnl` is the backbone** and **`prelim/sat` covers the recent tail** the finals
have not yet reached. `final/sat` only adds value from 1998 and lands at the same time
as `final/rnl`, so it is not needed for our purposes.

---

## Dataset: CHIRPS v3.0 (Climate Hazards Center, UC Santa Barbara)

### Source Information — v3.0 final/rnl (primary, historical backbone)

- **Summary of data organization**: One file per day, whole quasi-global grid, single
  variable (precipitation). Also offered as one NetCDF per year (all days stacked).
- **File format**: Cloud-Optimized GeoTIFF (`.cog`, uncompressed, single band, ~17 MB).
  NetCDF4 also available (one file per year, ~23.5 GB/yr — one band per day).
- **Temporal coverage**: 1981-01-01 to present.
- **Temporal frequency**: Daily. Value is the total precipitation for that calendar day.
- **Latency**: Final is slow. As of 2026-07-21 the latest final day is 2026-03-31
  (published 2026-04-20) — roughly a 3–4 month lag in practice, gated by ERA5. Recency
  comes from prelim (see next source).
- **Access notes**: Plain HTTPS directory server, anonymous, range-request capable
  (`/vsicurl/` works for reading COG/NetCDF metadata without full download). COGs are
  not gzipped (unlike v2.0 tifs). Ocean/water is `-9999` (land-only product).
- **Browse root**: https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/
- **URL format**:
```
https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/cogs/{YYYY}/chirps-v3.0.rnl.{YYYY}.{MM}.{DD}.cog
https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/netcdf/byYear/chirps-v3.0.rnl.{YYYY}.days_p05.nc
```
- **Example URLs**:
```
https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/cogs/1981/chirps-v3.0.rnl.1981.01.01.cog
https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/cogs/2020/chirps-v3.0.rnl.2020.01.01.cog
https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/netcdf/byYear/chirps-v3.0.rnl.2025.days_p05.nc
```
A 0.25° version lives alongside under `.../rnl/p25/`.

### Source Information — v3.0 prelim/sat (recent tail)

- **Summary of data organization**: One GeoTIFF per day, same grid and variable as final.
- **File format**: GeoTIFF (`.tif`, uncompressed, single band, ~16 MB). NetCDF also under
  `.../prelim/sat/netcdf/`.
- **Temporal coverage**: 2025-01-01 to present (v3.0 prelim archive only starts in 2025).
- **Temporal frequency**: Daily.
- **Latency**: ~2 days after each pentad ends. Prelim is published on the 2nd, 7th, 12th,
  17th, 22nd, and 27th of each month, each release extending through the pentad that just
  closed (pentads end on the 5th, 10th, 15th, 20th, 25th, and last day). Observed latest
  day 2026-07-15, file mtime 2026-07-17.
- **Access notes**: Prelim files are revised — a given day first appears as prelim, then
  is superseded months later by the final. This is the same early→final revision pattern
  as SWANN; handle it the same way (prefer final, fall back to prelim; on operational
  update re-read a trailing window so days get upgraded final-over-prelim).
- **Browse root**: https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/prelim/sat/
- **URL format**:
```
https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/prelim/sat/{YYYY}/chirps-v3.0.prelim.{YYYY}.{MM}.{DD}.tif
```
- **Example URLs**:
```
https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/prelim/sat/2026/chirps-v3.0.prelim.2026.07.15.tif
```

### GRIB Index (if applicable)
Not applicable — GeoTIFF / NetCDF, no GRIB, no `.idx` files.

### Coordinate Reference System
- **Common name**: WGS84 geographic (lat/lon).
- **PROJ string or EPSG**: EPSG:4326. Verified from the file CRS WKT (`GEOGCS["WGS 84"...
  AUTHORITY["EPSG","4326"]]`).

### Dimensions & Dimension Coordinates

Grid verified by reading `chirps-v3.0.rnl.2020.01.01.cog` and
`chirps-v3.0.prelim.2026.07.15.tif` (both 7200×2400, identical geotransform). We use
pixel centers; the GeoTIFF origin is the top-left **corner** (-180, 60) with a 0.05° step,
so centers are offset by half a pixel.

| Dimension | Min (center) | Max (center) | Step | Notes |
|-----------|-----|-----|------|-------|
| time | 1981-01-01 | present | 1 day | Daily total; NetCDF `time` = first day of the day, `days since 1980-1-1`, gregorian |
| latitude | -59.975 | 59.975 | 0.05° | 2400 cells. GeoTIFF rows run **north→south** (row 0 = 59.975). |
| longitude | -179.975 | 179.975 | 0.05° | 7200 cells, west→east. |

Corner bounds from the file: left=-180.0, right=180.0, top=60.0, bottom=-60.0.
(v2.0 is identical except latitude spans -49.975..49.975 over 2000 cells.)

### Data Variables

CHIRPS is a single-variable product: daily precipitation total.

| Variable name | Level | Units | Available from | Notes |
|---------------|-------|-------|----------------|-------|
| precipitation_surface | surface | mm | 1981-01-01 | Daily accumulated total precip. `-9999` = no data (all ocean/water; land-only estimate). GeoTIFF has no nodata tag set — treat `-9999` as fill. NetCDF declares `_FillValue`/`missing_value` = `-9999`. |

None of the other core template variables (temperature, wind, humidity, radiation,
pressure, cloud cover) exist — CHIRPS is precipitation only.

NetCDF variable metadata (from `chirps-v3.0.rnl.2025.days_p05.nc`, via `/vsicurl/`):
- variable name: `precip` (7200×2400, one band per day, 365 for 2025)
- `units`: `mm/day`
- `long_name`: `Climate Hazards center InfraRed Precipitation with Stations`
- `time`: `days since 1980-1-1 0:0:0`, calendar `gregorian`
- `standard_name`: `convective precipitation rate` — **wrong/misleading**: CHIRPS is a
  total precipitation estimate, not convective, and the value is a daily accumulation
  (mm), not a rate. Do not copy this into our template. Use our existing
  `precipitation_surface` conventions (match the equivalent variable in other datasets);
  CF `standard_name` `precipitation_amount` (units `kg m-2`, equal to mm) fits a daily
  total, or `lwe_thickness_of_precipitation_amount` (units `m`/`mm`).
- Value range sane: sampled days show 0 to ~150–270 mm with a large `0.0` land fraction
  and `-9999` over ocean.

**Temporal availability changes**:
- v3.0 archive begins 1981 (final/rnl) — full period-of-record from the primary source.
- prelim/sat only exists from 2025 onward (v3.0 prelim archive start).
- final/sat only from 1998 (IMERG start) — not needed given final/rnl covers 1981.

### Sample Files Examined

- **Early archive (v2.0)**: 1981-01-01 — `/products/CHIRPS-2.0/global_daily/tifs/p05/1981/chirps-v2.0.1981.01.01.tif.gz` (7200×2000, 50°N/S).
- **v3.0 final/rnl**: 2020-01-01 — `/products/CHIRPS/v3.0/daily/final/rnl/cogs/2020/chirps-v3.0.rnl.2020.01.01.cog` (7200×2400, 60°N/S).
- **v3.0 prelim/sat (recent)**: 2026-07-15 — `/products/CHIRPS/v3.0/daily/prelim/sat/2026/chirps-v3.0.prelim.2026.07.15.tif`.
- **NetCDF metadata**: `/products/CHIRPS-2.0/.../chirps-v2.0.2025.days_p05.nc` and `/products/CHIRPS/v3.0/.../chirps-v3.0.rnl.2025.days_p05.nc` (headers only, via `/vsicurl/`).

### Notable Observations

- **v2.0 vs v3.0 grid**: only the latitude extent differs (v2.0 ±50°, 2000 rows; v3.0
  ±60°, 2400 rows). CRS, longitude grid, 0.05° step, float32 dtype, and `-9999` sentinel
  are identical.
- **File format shift v2.0→v3.0**: v2.0 daily tifs are gzipped (`.tif.gz`); v3.0 finals
  are uncompressed COGs (`.cog`) and v3.0 prelims are uncompressed `.tif`. COG internal
  tiling + `/vsicurl/` means a backfill could read directly without a separate gunzip.
- **Land-only**: roughly half the grid is `-9999` (oceans). Empty-chunk friendly if fill
  is encoded well, but expect large no-data regions.
- **rnl vs sat naming**: filename carries the downscaling flavor
  (`chirps-v3.0.rnl.*`, `chirps-v3.0.sat.*`, `chirps-v3.0.prelim.*`). Mixing final/rnl
  (history) with prelim (recent) means the archive is homogeneous ERA5-downscaled up to
  the final cutoff, then IMERG-downscaled for the trailing months until finals catch up
  and overwrite. Note this discontinuity as a validation review note, not static metadata.
- **Latency tradeoff (measured)**: v2.0 final is *more current* than v3.0 final (v2.0 to
  2026-06-30 vs v3.0 to 2026-03-31) because v2.0 uses faster CFS gap-fill while v3.0
  finals wait on ERA5. We still choose v3.0 (wider coverage, supported past 2026) and lean
  on prelim for recency.
- **Time semantics**: NetCDF global attr `comments: "time variable denotes the first day
  of the given day"` and per-day tif files confirm each file/band is a single calendar
  day's total. `keep_mantissa_bits` convention for a precip flux/rate is 8.

### Integration notes (for a future TemplateConfig / RegionJob)

- **Shape**: analysis dataset, dims `(time, latitude, longitude)`, `append_dim="time"`,
  `append_dim_start=1981-01-01`, `append_dim_frequency=1 day`. Geographic CRS ⇒ latitude/
  longitude dimension names.
- **Materialized** dataset (one variable, root group; rewrites bytes). Closest existing
  analog is `contrib/uarizona/swann/analysis` — same per-day-file, revised early→final
  pattern; reuse its `SourceFileCoord` data-status approach (final → prelim fallback) and
  `read_data` (`rasterio.open`, map `-9999` → fill).
- **Two datasets, split on the final/prelim boundary** (they differ in source flavor,
  coverage, latency, and revision behavior, so they are cleaner as siblings than as one
  store): `chc-chirps-final` (final/rnl, ERA5, 1981-present) and
  `chc-chirps-preliminary` (prelim/sat, IMERG, 2025-present). This maps onto the
  `<provider>-<model>-<variant>` convention with variant = `final` / `preliminary`.
- **No temporal token in the id.** Daily is CHIRPS's finest global resolution — there is
  no hourly or other sub-daily global product (v2.0 had an Africa-only `6-hourly`; v3.0
  has none). The coarser products (pentad, dekad, monthly, annual) are aggregations we'd
  derive or add later, so "daily" would not disambiguate anything today.
- **Provider prefix `chc`**: `CHC` is ambiguous in the wider world — the Canadian
  Hurricane Centre also goes by CHC. But that is Environment and Climate Change Canada,
  which this repo already namespaces as `eccc`, so there is no in-repo collision, and
  CHIRPS is itself an unambiguous product name. `chc` (Climate Hazards Center) is fine;
  `ucsb` is the alternative that most closely matches the existing `uarizona`
  university-based precedent if we prefer maximal external clarity.
- **Variable**: `precipitation_surface`, units `mm`, matching whatever an existing
  precipitation variable in the repo already uses; ignore the source's
  `convective precipitation rate` standard_name.
- **License**: **CC BY 4.0** (public domain, registered with Creative Commons), confirmed
  at https://www.chc.ucsb.edu/data/chirps3. Cite the v3 data repository
  (https://doi.org/10.15780/G2JQ0P) and the v3 paper: Funk, C., Peterson, P., Harrison,
  L. et al. "The Climate Hazards Center Infrared Precipitation with Stations, Version 3."
  Sci Data 13, 718 (2026).
