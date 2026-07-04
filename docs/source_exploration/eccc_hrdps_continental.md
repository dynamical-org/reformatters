# Source Data Exploration: ECCC HRDPS (Continental 2.5 km)

Completed following [docs/source_data_exploration_guide.md](../source_data_exploration_guide.md).
All figures below were verified against live files on 2026-07-04 (init `20260704T00Z`) unless noted.

---

## Dataset: Environment and Climate Change Canada (ECCC) — HRDPS Continental

The High Resolution Deterministic Prediction System (HRDPS) is ECCC's 2.5 km
convection-permitting atmospheric NWP model. The "continental" domain covers
Canada and the northern continental US. It is an **atmosphere-only** model — ECCC
runs separate systems for ocean/ice (CIOPS/RIOPS/GIOPS) and precipitation analysis
(HRDPA/RDPA/CaPA), so there is no ocean/wave grouping within HRDPS itself.

### Source Information
- **Summary of data organization**: One GRIB2 file **per variable, per vertical level, per forecast hour**, laid out in directories by run hour then forecast hour. (No combined multi-variable files, and no GRIB `.idx` index files — a variable *is* a file.)
- **File format**: GRIB2 (plus a handful of derived GeoJSON/JSON products, e.g. `HighLowPressure`, which we would not ingest)
- **Temporal coverage**:
  - **MSC Datamart (`dd.weather.gc.ca` / `dd.meteo.gc.ca`)**: rolling ~30-day window only (see Data Retention).
  - **`hpfx.collab.science.gc.ca`**: rolling window, observed back ~51 days (see Data Retention).
  - **Longer archive**: [CaSPAr](https://caspar-data.ca/) (Canadian Surface Prediction Archive, U. Waterloo) holds HRDPS from **~May 2017 to present**. This is the source to combine with the live datamart for a long archive. Note the grid changed in **November 2023** (HRDPS v7): current files are **rotated lat-lon**; earlier files were **polar-stereographic** (`ps2.5km`). Any pre-Nov-2023 data from CaSPAr will be on the old grid and must be handled as a separate structural regime.
- **Temporal frequency**: 4 runs/day at **00, 06, 12, 18 UTC**. Each run produces **hourly** steps from **PT000H to PT048H** (49 lead times, 48 h horizon). All four runs are full 48 h runs (verified identical structure across 00/06/12 runs).
- **Latency**: Full run published ~3–4 h after init. Observed for `20260704T00Z`: fh000 at 02:59 UTC (~3.0 h), fh024 at 03:24 UTC (~3.4 h), fh048 at 03:47 UTC (~3.8 h after init).
- **Access notes**: Plain HTTPS directory listings (Apache autoindex), no auth. Because there are no index files and each variable is its own file, "partial download by variable" means fetching the whole (small, ~1–4 MB) file for that variable/level/hour.
- **Browse root**: <https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km/> (also date-partitioned under `https://dd.weather.gc.ca/{YYYYMMDD}/WXO-DD/model_hrdps/...`)
- **URL format**:
```
https://dd.weather.gc.ca/{YYYYMMDD}/WXO-DD/model_hrdps/continental/2.5km/{HH}/{fff}/{YYYYMMDD}T{HH}Z_MSC_HRDPS_{VAR}_{LEVELTYPE-LEVEL}_RLatLon0.0225_PT{fff}H.grib2
```
where `{HH}` ∈ {00,06,12,18}, `{fff}` ∈ {000..048}. Post-processed "Weather Elements on Grid" products use the tag `MSC_HRDPS-WEonG` in place of `MSC_HRDPS`.
- **Example URLs**:
```
https://dd.weather.gc.ca/20260704/WXO-DD/model_hrdps/continental/2.5km/00/001/20260704T00Z_MSC_HRDPS_TMP_AGL-2m_RLatLon0.0225_PT001H.grib2
https://dd.weather.gc.ca/20260704/WXO-DD/model_hrdps/continental/2.5km/00/000/20260704T00Z_MSC_HRDPS_ABSV_ISBL_0500_RLatLon0.0225_PT000H.grib2
https://dd.weather.gc.ca/20260704/WXO-DD/model_hrdps/continental/2.5km/00/024/20260704T00Z_MSC_HRDPS-WEonG_VISIFG_Sfc_RLatLon0.0225_PT024H.grib2
```

### GRIB Index
- **Index files available**: No. There are no `.idx`/JSON byte-index files. Each GRIB2 file contains a single variable/level message, so index files are unnecessary.
- **Index style**: N/A

### Coordinate Reference System
- **Common name**: Rotated latitude-longitude (rotated pole) grid, spherical earth. (Switched from polar-stereographic in Nov 2023.)
- **PROJ string or EPSG**: No EPSG code. Rotated-pole per GRIB convention, read from the file:
  - Ellipsoid: sphere, radius **6371229 m**
  - Southern pole of the rotated grid at **lat −36.08852°, lon −114.694858°**, axis rotation 0°
  - GRIB center = 54 (Montreal)

### Dimensions & Dimension Coordinates

Values read from `20260704T00Z_MSC_HRDPS_TMP_AGL-2m ... PT001H.grib2` via rasterio.

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| init_time | — | — | 6 h | 00/06/12/18 UTC daily |
| lead_time | 0 h | 48 h | 1 h | 49 steps (PT000H–PT048H) |
| x (rotated lon) | −14.8325° | 42.3175° | 0.0225° | 2540 columns (rotated coords, not geographic) |
| y (rotated lat) | −12.3138° | 16.7113° | 0.0225° | 1290 rows (rotated coords) |
| pressure_level | 50 hPa | 1015 hPa | irregular | 28 levels for the main upper-air vars (see below) |

Grid: **2540 × 1290** points, 0.0225° spacing (~2.5 km). This is a **projected grid** in the reformatters sense (rotated pole), so spatial dims should be **y / x**, not latitude/longitude. We use pixel centers.

**Pressure levels (ISBL, hPa)** — for the densely-sampled upper-air vars (TMP, RH, SPFH, HGT, UGRD, VGRD, WIND, WDIR, DEPR — 28 levels each):
`50, 100, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 970, 985, 1000, 1015`.
A few vars carry only a sparse subset: ABSV and VVEL on `{250, 500, 700, 850, 1000}` (5 levels); LFTX/SHWINX/MU-VT-LI only on 500 hPa.

**Above-ground levels (AGL)**: 2 m (temperature/humidity family), 10 m (winds/gusts), and 40 m / 80 m / 120 m (winds, density, dewpoint). Note: there is **no 100 m wind** — the model exposes 40/80/120 m instead.

### Data Variables

Core variables from the guide's checklist (raw `MSC_HRDPS` GRIB `GRIB_ELEMENT` in parentheses):

| Variable name | Level | Units (in file) | Available from | Notes |
|---------------|-------|-------|----------------|-------|
| temperature_2m | 2 m | °C (TMP) | fh000 | AGL-2m |
| wind_u_10m | 10 m | m/s (UGRD) | fh000 | AGL-10m |
| wind_v_10m | 10 m | m/s (VGRD) | fh000 | AGL-10m |
| wind_u_100m | 100 m | — | — | **Not available**; nearest are AGL-40m/80m/120m |
| wind_v_100m | 100 m | — | — | **Not available**; nearest are AGL-40m/80m/120m |
| precipitation_surface | surface | kg/m² (APCP) | fh000 | Run-total accumulation; also per-window `APCP-Accum{1,3,6,12,24}h`. Verify accumulation reset behavior before use. |
| downward_short_wave_radiation_flux_surface | surface | W/m² (DSWRF) | fh000 | Sfc (also DSWRF at NTAT) |
| downward_long_wave_radiation_flux_surface | surface | W/m² (DLWRF) | fh000 | Sfc |
| pressure_surface | surface | Pa (PRES) | fh000 | Sfc |
| pressure_reduced_to_mean_sea_level | MSL | Pa (PRMSL) | fh000 | MSL |
| total_cloud_cover_atmosphere | atmosphere | % (TCDC) | fh000 | Tagged as `Sfc` level type in filename |
| relative_humidity_2m | 2 m | % (RH) | fh000 | AGL-2m |
| specific_humidity_2m | 2 m | kg/kg (SPFH) | fh000 | AGL-2m |
| dew_point_temperature_2m | 2 m | °C (DPT) | fh000 | AGL-2m |

**84 distinct raw HRDPS GRIB elements** are published (plus ~13 WEonG post-processed elements). Beyond the core set above, notable available fields include: HGT/ABSV/VVEL (upper-air geopotential/vorticity/vertical velocity), CAPE, HLCY (helicity), HPBL (boundary-layer height), GUST/GUST-Max/GUST-Min, SNOD/SDWE/SDEN (snow), SOILW/TSOIL (soil moisture/temp on DBS depth layers), skin temperature (SKINT), radiation balance (NSWRS/NLWRS/USWRF/ULWRF/NTAT fluxes), PTYPE/PRATE, and comfort indices (Humidex/WCHIL/UTCI/UVIndex).

**Temporal availability changes within a run**:
- Accumulated/rate/window fields (APCP, the `WEA{RN,SN,FR,PE}` precip-type accumulations, GUST-Max/Min) are absent or zero at fh000 and appear from fh001 — hence file count grows from **350 files at fh000 → 414 at fh048**.

### File groupings & daily volume

HRDPS does not split into ocean/atmosphere domains; the meaningful groupings are
**(a) product** (raw model vs. post-processed WEonG) and **(b) vertical level type**.
Sizes are summed from the Apache listing (human-readable, ~±0.1 MB/file rounding).

**Per run (49 forecast hours): 20,021 files, ~32.5 GB.**
**Per day (×4 runs): ~80,100 files, ~130 GB.**

By product tag (per run → per day):

| Product | Files/run | GB/run | Files/day | GB/day |
|---------|-----------|--------|-----------|--------|
| `MSC_HRDPS` (raw NWP) | 18,389 | 29.5 | 73,556 | ~118 |
| `MSC_HRDPS-WEonG` (Weather Elements on Grid, post-processed) | 1,632 | 3.0 | 6,528 | ~12 |

By vertical level type (per run → per day; spans both products):

| Level type | Files/run | GB/run | Files/day | GB/day |
|------------|-----------|--------|-----------|--------|
| Pressure levels (ISBL) | 12,985 | 19.9 | 51,940 | ~79 |
| Surface (Sfc) | 4,390 | 7.0 | 17,560 | ~28 |
| Above-ground level (AGL 2/10/40/80/120 m) | 2,058 | 4.7 | 8,232 | ~19 |
| Depth below surface (DBS, soil) | 147 | 0.36 | 588 | ~1.4 |
| Nominal top of atmosphere (NTAT) | 147 | 0.21 | 588 | ~0.8 |
| Fixed height layer (ETAL, for VWSH) | 49 | 0.17 | 196 | ~0.7 |
| Isobaric layer (ISBY) | 147 | 0.07 | 588 | ~0.3 |
| Entire atmosphere (EATM) | 49 | 0.07 | 196 | ~0.3 |
| Mean sea level (MSL) | 49 | 0.05 | 196 | ~0.2 |

Pressure-level fields dominate: ISBL is ~61% of both file count and volume. Individual GRIB2 files are small (~1–4 MB; the 2540×1290 TMP AGL-2m sample was 3.6 MB).

### Data Retention

Observed directory spans on 2026-07-04:
- **`dd.weather.gc.ca`**: dates `20260605` … `20260705` → **~30-31 day** rolling window (matches MSC's documented 30-day retention policy for the Datamart).
- **`hpfx.collab.science.gc.ca`**: dates `20260514` … `20260705` → **~51-53 days** observed, with HRDPS confirmed present at the earliest date (`20260514`). hpfx is the higher-throughput mirror and in practice keeps HRDPS noticeably longer than the nominal 30 days.

Implication for integration: neither public server is a long archive. For a multi-year
backfill, combine **CaSPAr (~2017→present)** for history with the **datamart/hpfx**
for the live tail, and try hpfx first (throughput) falling back to dd (reliability).

### Sample Files Examined
- **Recent data**: 2026-07-04 00Z run, e.g. `.../00/001/20260704T00Z_MSC_HRDPS_TMP_AGL-2m_RLatLon0.0225_PT001H.grib2` (downloaded & inspected with rasterio: 2540×1290, rotated pole, °C).
- **Full-run listing**: all 49 forecast hours of the 2026-07-04 00Z run (for volume/variable census).
- **Retention checks**: earliest dd date `20260605`, earliest hpfx date `20260514` (HRDPS confirmed present).

### Notable Observations
- **Projected grid → use y/x dims.** Rotated-pole grid (sphere R=6371229 m; southern pole −36.08852°, −114.694858°). File `bounds`/`transform` are in rotated-grid degrees, not geographic lat/lon.
- **Nov 2023 grid change (HRDPS v7).** Current = rotated lat-lon; pre-Nov-2023 (only reachable via CaSPAr) = polar-stereographic. Treat as two structural regimes.
- **One variable per file, no index files.** Simplifies "download only what you need" (whole small file per var), but means many (~80k/day) small HTTP requests for a full ingest.
- **Units are per-file** (e.g. temperature in °C, not K) and must be read from `GRIB_UNIT`; convert to SI/CF on ingest.
- **No 100 m wind**; 40/80/120 m AGL winds are provided instead.
- **WEonG is post-processed**, not raw model output — keep separate from raw fields if ingested.
- **Accumulation fields** (APCP and precip-type WEA* families) need their accumulation/reset semantics verified before deaccumulation.
