## Dataset: ECMWF IFS ENS Forecast GRIB Archive (s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/)

### Source Information
- **Summary of data organization**: One date directory per initialization date. Within each directory, up to 5 GRIB files split by member type (control `cf` vs perturbed forecast `pf`) and level type (`sfc` vs `pl`). Perturbed forecast surface data is split into two files (`pf_sfc_0`, `pf_sfc_1`) covering members 1-25 and 26-50 respectively. All lead times and all variables of each type are concatenated within a single file. Each GRIB file has a companion `.idx` file with one JSON record per message, giving byte offsets and lengths for efficient partial reads.
- **File format**: GRIB Edition 1
- **Temporal coverage**: 2016-03-08 to 2024-03-31, with coverage varying by file type (archive is an in-progress backfill — see below)
- **Temporal frequency**: One initialization time per day at 00Z
- **Latency**: Static archive (not operational). The ecmwf-forecasts S3 bucket covers 2024-04-01 onward.
- **Access notes**: Public anonymous S3 read access from `us-west-2.opendata.source.coop`. Use `s3fs.S3FileSystem(anon=True)` (no custom endpoint; this bucket is on AWS S3 directly).
- **Browse root**: `s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/`
- **URL format**:
```
s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/{YYYY-MM-DD}/{file_type}.grib
s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/{YYYY-MM-DD}/{file_type}.grib.idx
```
Where `{file_type}` is one of: `cf_sfc`, `cf_pl`, `pf_sfc_0`, `pf_sfc_1`, `pf_pl`

- **Example URLs**:
```
s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/2016-03-08/cf_sfc.grib
s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/2016-03-08/cf_sfc.grib.idx
s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/2023-01-01/cf_pl.grib
s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/2024-03-31/pf_sfc_1.grib
```

### File Availability by Period (Backfill Progress as of 2026-04-02)

The archive is being filled in progressively. The files present in each date directory vary:

| Date range | cf_pl | cf_sfc | pf_pl | pf_sfc_0 | pf_sfc_1 | Count |
|---|---|---|---|---|---|---|
| 2016-03-08 – 2016-04-07 | ✓ | ✓ | ✓ | ✓ | ✓ | 31 |
| 2016-04-08 – 2017-10-22 | — | — | — | — | — | gap |
| 2017-10-23 – 2023-11-29 | ✓ | — | — | — | — | ~2229 |
| 2023-11-30 – 2024-02-18 | ✓ | ✓ | — | — | — | ~79 |
| 2024-02-19 – 2024-02-26 | ✓ | ✓ | ✓ | — | — | ~8 |
| 2024-02-27 – 2024-03-31 | ✓ | ✓ | ✓ | ✓ | ✓ | ~33 |

Total dates in archive: 2,381. A few isolated missing days exist (2021-07-07, 2022-05-30).

### GRIB Index
- **Index files available**: Yes — one `.grib.idx` per `.grib` file
- **Index style**: ECMWF JSON (one JSON object per line)
- **Fields in index**: `type` (`cf` or `pf`), `param` (ECMWF short name), `step` (hours), `levtype` (`sfc` or `pl`), `_offset` (byte offset), `_length` (byte length), and optionally `number` (ensemble member, present only in pf files) and `levelist` (pressure level in hPa, present only in pl files)
- **Example lines**:
```json
{"type": "cf", "param": "sp", "step": 0, "levtype": "sfc", "_offset": 0, "_length": 2076642}
{"type": "pf", "param": "sp", "step": 0, "levtype": "sfc", "_offset": 0, "_length": 2076642, "number": 1}
{"type": "cf", "param": "z", "step": 0, "levtype": "pl", "_offset": 0, "_length": 2076642, "levelist": 500}
```

**Important difference from operational GRIB2 index**: The `step` field here is a lead time in hours, not a byte position. The existing `ecmwf_grib_index.py` parser does not handle the step dimension — a new reader is needed that uses `(param, step, levtype[, levelist][, number])` as the key to find byte ranges.

### Coordinate Reference System
- **Common name**: Geographic / WGS84-like; earth modeled as sphere
- **PROJ string or EPSG**: Sphere with radius 6,367,470 m (GRIB1 convention), equivalent to the existing dataset's `+a=6371229` sphere in practice for 0.25° data — confirm CRS wkt from rasterio: `GEOGCS["Coordinate System imported from GRIB file", DATUM["unnamed", SPHEROID["Sphere",6367470,0]], PRIMEM["Greenwich",0], UNIT["degree",0.0174532925199433]]`

### Dimensions & Dimension Coordinates

| Dimension | Min | Max | Step | Notes |
|---|---|---|---|---|
| init_time | 2016-03-08T00:00Z | 2024-03-31T00:00Z | 24h (00Z only) | One init per day |
| lead_time | 0h | 360h | 3h for 0–144h, 6h for 150–360h | 85 steps total |
| latitude | -90.0 | 90.0 | 0.25° | 721 pixels; pixel centers; North→South in file |
| longitude | -180.0 | 179.75 | 0.25° | 1440 pixels; pixel centers |
| ensemble_member | 0 | 50 | 1 | 0 = control (cf); 1–50 = perturbed (pf) |
| pressure_level | 500, 850, 925 | hPa | — | Only for `z` and `t` in pl files |

Grid shape: 721 × 1440 (height × width). Pixel center of top-left: (90.0°N, -180.0°E).

### Data Variables

Variables in surface files (`cf_sfc`, `pf_sfc_0`, `pf_sfc_1`):

| Variable name | ECMWF param | GRIB element | GRIB_COMMENT (from rasterio) | Units in file | Notes |
|---|---|---|---|---|---|
| pressure_surface | `sp` | `SP` | `Surface pressure [Pa]` | Pa | |
| temperature_2m | `2t` | `2T` | `2 metre temperature [C]` | K (tag wrong; values 213–309 are Kelvin) | GRIB1 unit tag unreliable |
| wind_u_10m | `10u` | `10U` | `10 metre u wind component [m/s]` | m/s | |
| wind_v_10m | `10v` | `10V` | `10 metre v wind component [m/s]` | m/s | |
| wind_u_100m | `100u` | `var246 of table 228 of center ECMWF` | `undefined [-]` | m/s (inferred) | GRIB1 local param, no standard metadata |
| wind_v_100m | `100v` | `var247 of table 228 of center ECMWF` | `undefined [-]` | m/s (inferred) | GRIB1 local param, no standard metadata |
| precipitation_surface | `tp` | `TP` | `Total precipitation [m]` | m (accumulation from T+0) | Deaccumulate to rate; scale ×1000 for mm |
| downward_long_wave_radiation_flux_surface | `strd` | `STRD` | `Surface thermal radiation downwards [W*s/m^2]` | J/m² (accumulation from T+0) | Deaccumulate to W/m² rate |
| downward_short_wave_radiation_flux_surface | `ssrd` | `SSRD` | `Surface solar radiation downwards [W*s/m^2]` | J/m² (accumulation from T+0) | Deaccumulate to W/m² rate |
| pressure_reduced_to_mean_sea_level | `msl` | `MSL` | `Mean sea level pressure [Pa]` | Pa | |
| dew_point_temperature_2m | `2d` | `2D` | `2 metre dewpoint temperature [C]` | K (tag wrong; values 210–300 are Kelvin) | GRIB1 unit tag unreliable |
| categorical_precipitation_type_surface | `ptype` | `PTYPE` | `Precipitation type [0=No precipitation; ...]` | categorical | Same encoding as operational GRIB2 |
| wind_gust_10m | `10fg` | `var49 of table 128 of center ECMWF` | `undefined [-]` | m/s (inferred; range 0.5–42 m/s) | GRIB1 local param; no `step_type` metadata from rasterio |
| total_cloud_cover_atmosphere | `tcc` | `TCC` | `Total cloud cover (0 - 1) [-]` | fraction 0–1 | Scale ×100 for percent |

Variables in pressure level files (`cf_pl`, `pf_pl`):

| Variable name | ECMWF param | Levels | GRIB_COMMENT | Units in file | Notes |
|---|---|---|---|---|---|
| geopotential_500/850/925hpa | `z` | 500, 850, 925 hPa | `Geopotential (at the surface = orography) [m^2/s^2]` | m²/s² | Divide by g≈9.80665 for geopotential height in m |
| temperature_500/850/925hpa | `t` | 500, 850, 925 hPa | `Temperature [C]` | °C (tag and values agree; range –48 to +36°C at 500–925 hPa) | |

**Checked availability**: All 14 surface variables and both pressure level variables are present from 2016-03-08 through at least 2024-03-31 in dates that have the respective file.

**Temporal availability notes**:
- `10fg` and `tcc` present in GRIB1 archive from 2016-03-08; in operational GRIB2 data these have `date_available` restrictions (`2024-11-13` and `2025-11-21` respectively) because they were added to the operational feed later. The GRIB1 archive always had them.
- `100u`/`100v` present from 2016-03-08 but GRIB1 does not carry standard metadata for them (local param table).

### Sample Files Examined

- **Earliest archive**: 2016-03-08 — `cf_sfc.grib`, `cf_pl.grib`, `pf_sfc_0.grib`, `pf_sfc_1.grib`, `pf_pl.grib`
- **cf_pl only period**: 2020-01-15 — only `cf_pl.grib` and `cf_pl.grib.idx`
- **Recent (partial backfill)**: 2024-03-31 — full set of all 5 file types present
- **First date with cf_sfc after gap**: ~2023-11-30

### Notable Observations

1. **Backfill in progress**: The archive is being filled backwards from the present toward 2016. At the time of exploration, most dates only have `cf_pl.grib` (control forecast, pressure levels only). Full ensemble surface data is available for the earliest ~31 dates and most recent ~44 dates in the archive.

2. **GRIB1 unit tags unreliable for temperature**: Rasterio reports `[C]` for `2t` and `2d` but the actual values (213–309 K) are clearly Kelvin. This is a known GRIB1 metadata issue. Pressure-level temperature `t` reports `[C]` and the values (–48 to +36) genuinely are in Celsius.

3. **Geopotential in m²/s²**: The `z` variable is geopotential (not geopotential height). Must divide by g = 9.80665 m/s² to get geopotential height in meters. The operational GRIB2 data uses `gh` (already in geopotential height meters) — this is a key difference.

4. **Radiation accumulation**: `ssrd` and `strd` are accumulated from T+0 (same as operational data). Unit is W·s/m² (= J/m²). Deaccumulate to average rate in W/m² using the lead_time step interval. At step 0 values are always 0.

5. **Precipitation accumulation**: `tp` accumulated from T+0 in meters. Deaccumulate to mm/s rate (multiply by 1000 to convert m→mm, divide by step interval).

6. **File-per-date vs file-per-lead-time**: The operational ECMWF open data (`ecmwf-forecasts`) has one GRIB2 file per lead time. This archive has one GRIB1 file per date/type containing all 85 lead times. The existing `ecmwf_grib_index.py` reader assumes per-lead-time files — a new reader is needed that indexes by `(param, step, levtype[, levelist][, number])`.

7. **100u/100v local parameters**: These are ECMWF GRIB1 local parameters (table 228, params 246/247). Rasterio has no metadata for them beyond the element being "undefined". Values are in m/s and range plausibly for 100m winds.

8. **wind_gust_10m (10fg)**: Also a local ECMWF GRIB1 parameter (table 128, param 49). Rasterio shows it as undefined. Values range 0.5–42 m/s which is physically consistent with wind gusts.

9. **pf files split at members 1–25 / 26–50**: The two perturbed forecast surface files cover non-overlapping member ranges. The pl file (`pf_pl`) contains all 50 perturbed members in one file.

10. **CRS sphere radius**: GRIB1 files use 6,367,470 m; the existing reformatter uses 6,371,229 m (WMO standard). This difference (~4 km) is negligible for 0.25° resolution data.

11. **No 12Z initializations**: Only 00Z per day in this archive.

---

## Exploration Process Summary

Files examined via `s3fs.S3FileSystem(anon=True)` on `us-west-2.opendata.source.coop`. Individual GRIB messages were downloaded using byte ranges from `.idx` files and inspected with `rasterio`. The index files were parsed with `json.loads` per line. All 2,381 dates were enumerated to characterize file availability by period.
