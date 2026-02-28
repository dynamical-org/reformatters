## Dataset: NASA GPM IMERG (Integrated Multi-satellitE Retrievals for GPM)

IMERG is NASA's global surface precipitation product at 0.1° × 0.1° / 30-minute resolution,
produced by the Global Precipitation Measurement (GPM) mission. It comes in three runs
(Early, Late, Final) with different latencies and quality levels.

Current version: **V07B** (released July 2023, reprocessed through November 2024).
V07B supersedes all prior IMERG versions and also replaces the TRMM-era TMPA products
(3B42, 3B43). Archive extends back to January 1998.

---

## Source 1: NASA PPS Near Real-Time Server (jsimpson) — Lowest Latency

Covers **Early Run** (~4 h latency) and **Late Run** (~12–14 h latency).

- **Summary of data organization**: One HDF5 file per 30-minute granule, organized by
  YYYYMM subdirectory. Early and Late runs are in separate directory trees. Each file
  contains all variables for that granule (global coverage, all latitudes).
- **File format**: HDF5 (`.RT-H5` extension on NRT server). Also GeoTIFF GIS products,
  but HDF5 is preferred for integration.
- **Temporal coverage**:
  - Early Run: 2000-06 to present (rolling retention window on NRT server — older data
    moves to arthurhou/GES DISC archive)
  - Late Run: 2000-06 to present
- **Temporal frequency**: One file per 30 minutes (48 files/day per run)
- **Latency**:
  - Early Run: ~4 hours after observation time. By ~3 h after observation, ~85% of
    microwave data has arrived at PPS. Uses forward-propagation morphing only.
  - Late Run: ~12–14 hours after observation time. Waits 11 h to collect the next
    satellite overpass. Uses both forward and backward propagation morphing.
- **Access notes**:
  - Requires registration at https://registration.pps.eosdis.nasa.gov/ with
    "Near-Realtime Products" box checked.
  - Registered email address is used as **both username and password** (lowercased).
  - Authentication via HTTP Basic Auth over HTTPS.
  - Text-file listing of all files at `/text/` endpoint enables programmatic discovery
    without crawling HTML directories.
- **Browse root**:
  - Early: https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/early/
  - Late: https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/late/
  - Text listing root: https://jsimpsonhttps.pps.eosdis.nasa.gov/text/
- **URL format**:
```
# Early Run HDF5
https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/early/{YYYYMM}/3B-HHR-E.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V07B.RT-H5

# Late Run HDF5
https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/late/{YYYYMM}/3B-HHR-L.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V07B.RT-H5

# Text listing for a month (use this to discover exact filenames)
https://jsimpsonhttps.pps.eosdis.nasa.gov/text/imerg/early/{YYYYMM}/
https://jsimpsonhttps.pps.eosdis.nasa.gov/text/imerg/late/{YYYYMM}/
```
- **Example URLs**:
```
# Early Run — 2024-01-15 00:00 UTC granule
https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/early/202401/3B-HHR-E.MS.MRG.3IMERG.20240115-S000000-E002959.0000.V07B.RT-H5

# Late Run — 2020-06-01 00:00 UTC granule (observed in search results, V06B era)
https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/late/200006/3B-HHR-L.MS.MRG.3IMERG.20000601-S000000-E002959.0000.V06B.RT-H5

# Text listing for January 2020 Early Run
https://jsimpsonhttps.pps.eosdis.nasa.gov/text/imerg/early/202001/
```
- **Filename field breakdown**:
  - `YYYYMMDD` — date of observation
  - `S{HHMMSS}` — granule start time (e.g., `S000000` = 00:00:00 UTC)
  - `E{HHMMSS}` — granule end time (e.g., `E002959` = 00:29:59 UTC)
  - `{MMMM}` — minutes elapsed since midnight UTC (0000, 0030, 0060, …, 1410)
  - `V07B` — algorithm version
  - `.RT-H5` — real-time HDF5 extension (NRT server); GES DISC archive uses `.HDF5`

---

## Source 2: NASA PPS Research Server (arthurhou) — Final Run Archive

Covers **Final Run** (~3.5 month latency after observation month).

- **Summary of data organization**: One HDF5 file per 30-minute granule, organized by
  `/gpmdata/YYYY/MM/DD/imerg/` path. Research-quality gauge-calibrated product.
- **File format**: HDF5 (`.HDF5` extension)
- **Temporal coverage**: 1998-01 to present (with ~3.5 month lag). Full TRMM-era record
  available from V07B reprocessing.
- **Temporal frequency**: 48 files/day (30-minute granules)
- **Latency**: ~3.5 months (Final Run is produced monthly using rain-gauge calibration data)
- **Access notes**:
  - Requires registration at https://registration.pps.eosdis.nasa.gov/
  - Email address is username and password (lowercased).
  - FTP access was discontinued January 19, 2021. Use FTPS or HTTPS only.
  - Text listing available at https://arthurhou.pps.eosdis.nasa.gov/text/
- **Browse root**: https://arthurhou.pps.eosdis.nasa.gov/gpmdata/
- **URL format**:
```
https://arthurhou.pps.eosdis.nasa.gov/gpmdata/{YYYY}/{MM}/{DD}/imerg/3B-HHR.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V07B.HDF5
```
- **Example URLs**:
```
# Final Run — 2024-01-15 00:00 UTC granule
https://arthurhou.pps.eosdis.nasa.gov/gpmdata/2024/01/15/imerg/3B-HHR.MS.MRG.3IMERG.20240115-S000000-E002959.0000.V07B.HDF5

# Final Run — 1998-01-01 00:00 UTC (earliest data)
https://arthurhou.pps.eosdis.nasa.gov/gpmdata/1998/01/01/imerg/3B-HHR.MS.MRG.3IMERG.19980101-S000000-E002959.0000.V07B.HDF5
```

---

## Source 3: NASA GES DISC / S3 — Archive with Cloud Access

All three runs are mirrored to the **GES DISC** (Goddard Earth Sciences Data and Information
Services Center) and exposed via both HTTPS and an AWS S3 bucket. The S3 path requires
Earthdata login credentials but supports same-region (`us-west-2`) direct access with
temporary credentials — the same pattern used by our existing NASA SMAP integration.

- **Summary of data organization**: One HDF5 file per 30-minute granule, organized by
  year and day-of-year (`YYYY/DDD/`). Separate product directories for each run.
- **File format**: HDF5 (`.HDF5`). Daily and monthly aggregates also available as `.nc4`.
- **Temporal coverage**: 1998-01-01 to present for all three runs
- **Temporal frequency**: 48 files/day
- **Latency**: Same as the originating run (Early ~4 h, Late ~14 h, Final ~3.5 months).
  GES DISC receives files shortly after PPS produces them.
- **Access notes**:
  - **HTTPS**: Requires NASA Earthdata Login. Same credential system as our SMAP
    integration (`get_authenticated_session()`). Free registration at
    https://urs.earthdata.nasa.gov/
  - **S3 direct access**: Temporary credentials at
    https://data.gesdisc.earthdata.nasa.gov/s3credentials. Must access from
    AWS `us-west-2`. Earthdata Login required for credential issuance.
  - **S3 access is NOT fully public/open** — it is "controlled access" via Earthdata
    Login but the data itself is free.
  - S3 is listed on the AWS Open Data Registry (for all three runs) which is how it is
    discoverable, even though it requires authentication.
- **Browse root**:
  - Final: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/
  - Early: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/
  - Late: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.07/
- **URL format**:
```
# HTTPS — Final Run (GPM_3IMERGHH)
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/{YYYY}/{DDD}/3B-HHR.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V07B.HDF5

# HTTPS — Early Run (GPM_3IMERGHHE)
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/{YYYY}/{DDD}/3B-HHR-E.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V07B.HDF5

# HTTPS — Late Run (GPM_3IMERGHHL)
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.07/{YYYY}/{DDD}/3B-HHR-L.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V07B.HDF5

# S3 — Final Run
s3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHH.07/{YYYY}/{DDD}/3B-HHR.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V07B.HDF5

# S3 — Early Run
s3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHHE.07/{YYYY}/{DDD}/3B-HHR-E.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V07B.HDF5

# S3 — Late Run
s3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHHL.07/{YYYY}/{DDD}/3B-HHR-L.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V07B.HDF5
```
- **Example URLs**:
```
# Final Run — 2024-01-15 (day 015) 00:00 UTC, HTTPS
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2024/015/3B-HHR.MS.MRG.3IMERG.20240115-S000000-E002959.0000.V07B.HDF5

# Final Run — same file via S3
s3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHH.07/2024/015/3B-HHR.MS.MRG.3IMERG.20240115-S000000-E002959.0000.V07B.HDF5

# Final Run — 1998-01-01 (day 001), earliest data
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/1998/001/3B-HHR.MS.MRG.3IMERG.19980101-S000000-E002959.0000.V07B.HDF5
```
- **AWS Open Data Registry entries**:
  - Final half-hourly: https://registry.opendata.aws/nasa-gpm3imerghh/
  - Late half-hourly: https://registry.opendata.aws/nasa-gpm3imerghhl/
  - Early half-hourly: https://registry.opendata.aws/nasa-gpm3imerghhe/
  - Final daily: https://registry.opendata.aws/nasa-gpm3imergdf/
  - Late daily: https://registry.opendata.aws/nasa-gpm3imergdl/
  - Early daily: https://registry.opendata.aws/nasa-gpm3imergde/
- **S3 bucket region**: `us-west-2`
- **S3 temp credentials endpoint**: https://data.gesdisc.earthdata.nasa.gov/s3credentials

---

### GRIB Index (if applicable)
Not applicable. IMERG uses HDF5 format, not GRIB.

---

### Coordinate Reference System
- **Common name**: WGS84 geographic (regular lat/lon grid)
- **PROJ string or EPSG**: EPSG:4326

---

### Dimensions & Dimension Coordinates

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| time | 1998-01-01 00:00 UTC | present | 30 min | 1 value per file (dim size = 1) |
| lon | -179.95° | 179.95° | 0.1° | 3600 values; pixel centers |
| lat | -89.95° | 89.95° | 0.1° | 1800 values; pixel centers |

**Important dimension ordering note**: Dimensions in V07 HDF5 files are ordered
`(time, lon, lat)` — **not** `(time, lat, lon)`. The `lon` axis comes before `lat`.
This is a known gotcha; verify when reading files. (Noted in V06 release notes and
still applies in V07.)

**HDF5 group structure**: All data variables and coordinates are under the `Grid` group.
When reading via rasterio or xarray, specify `group="Grid"` or use the HDF5 path
`HDF5:{file}:/Grid/{variable}`.

**Spatial coverage**: Nominally global (90°N–90°S), but polar coverage has sparse
microwave data. The `precipitationCal` / `precipitation` field uses IR fill-in between
60°N–60°S; poleward of 60° there are data gaps (filled with `HQprecipitation` where
available, otherwise missing).

---

### Data Variables

IMERG is a precipitation-only product. The standard variables (in the `Grid` HDF5 group)
in V07 are:

All variables are under the HDF5 `/Grid/` group. In V07, several secondary variables
were moved into a `/Grid/Intermediate/` subgroup (see rename table below).

| HDF5 path (V07) | Level | Units | Notes |
|-----------------|-------|-------|-------|
| `Grid/precipitation` | surface | mm/hr | **Primary variable.** Multi-satellite + gauge-calibrated (Final) or climatologically calibrated (Early/Late). Was `precipitationCal` in V06. |
| `Grid/randomError` | surface | mm/hr | Random error estimate for `precipitation`. |
| `Grid/probabilityLiquidPrecipitation` | surface | % | Probability precipitation is liquid phase (not snow/ice). New in V07. |
| `Grid/PrecipitationQualityIndex` | surface | — | Quality index 0–1. |
| `Grid/Intermediate/precipitationUncal` | surface | mm/hr | Multi-satellite without gauge calibration. Differs from `precipitation` only in Final Run. |
| `Grid/Intermediate/MWprecipitation` | surface | mm/hr | Microwave-only (was `HQprecipitation`). Has significant spatial gaps. |
| `Grid/Intermediate/MWprecipSource` | surface | categorical | Which sensor provided the MW estimate (was `HQprecipSource`). |
| `Grid/Intermediate/MWobservationTime` | surface | minutes | Time of MW observation within the half-hour (was `HQobservationTime`). |
| `Grid/Intermediate/IRprecipitation` | surface | mm/hr | IR-only geostationary estimate. Less reliable. |
| `Grid/Intermediate/IRinfluence` | surface | fraction | Weight of IR in the merged estimate (was `IRkalmanFilterWeight`). |

**V06 → V07 variable renames:**

| V06 path | V07 path |
|----------|----------|
| `Grid/precipitationCal` | `Grid/precipitation` |
| `Grid/precipitationUncal` | `Grid/Intermediate/precipitationUncal` |
| `Grid/HQprecipitation` | `Grid/Intermediate/MWprecipitation` |
| `Grid/HQprecipSource` | `Grid/Intermediate/MWprecipSource` |
| `Grid/HQobservationTime` | `Grid/Intermediate/MWobservationTime` |
| `Grid/IRkalmanFilterWeight` | `Grid/Intermediate/IRinfluence` |

Code targeting V06 files must handle both the old names and paths, or restrict to V07+ only.

**Temporal availability changes**:
- TRMM-era data (1998-01 to 2014-02): Reprocessed and available in V07B; uses
  constellation partners available at the time (fewer satellites than post-2014).
- GPM Core Observatory era (2014-03 to present): Full constellation including GPM-CO.
- V06 data available through ~2023; V07B reprocessing completed ~November 2024.
  Only V07B should be used going forward — older versions superseded.
- Daily and monthly aggregates are also available (separate product IDs) but half-hourly
  is the source for our integration.

---

### Sample Files Examined

Not yet examined directly — file access requires Earthdata login registration.
The following files should be examined before integration:

- **Early archive (TRMM era)**: 1998-01-01, `3B-HHR.MS.MRG.3IMERG.19980101-S000000-E002959.0000.V07B.HDF5`
- **GPM era start**: 2014-03-01
- **V06→V07 transition**: 2023-07-07 (when V07B was released for Early/Late; Final
  reprocessing completed ~November 2024)
- **Recent data**: 2025-01-01

---

### Notable Observations

1. **Two authentication systems**: PPS (jsimpson/arthurhou) uses email-as-password HTTP
   Basic Auth; GES DISC uses NASA Earthdata OAuth/cookie-based auth (same as our SMAP
   integration). For operational Early/Late updates, PPS NRT is the primary source. For
   the archive backfill, GES DISC HTTPS or S3 is preferred for bandwidth.

2. **S3 is controlled access, not open**: Despite being listed on the AWS Open Data
   Registry, the S3 bucket `gesdisc-cumulus-prod-protected` requires Earthdata Login
   credentials. Direct S3 access requires temporary credentials and must be performed
   from AWS `us-west-2`. This is the same pattern as our NASA SMAP integration.

3. **Dimension ordering gotcha**: IMERG HDF5 files use `(time, lon, lat)` ordering,
   not the typical `(time, lat, lon)`. Rasterio reads the data correctly but verify
   before assuming standard ordering.

4. **HDF5 group**: All data is under the `/Grid` group. rasterio path:
   `HDF5:{local_file}:/Grid/precipitation`

5. **V06 vs V07 variable names**: Primary precipitation field renamed `precipitationCal`
   → `precipitation`. If supporting historical re-downloads, handle both.

6. **NRT server retention**: The jsimpson NRT server has a rolling retention window (not
   a permanent archive). Older data is available from arthurhou (Final) or GES DISC.
   For our integration: use jsimpson for operational Early/Late updates; use GES DISC
   for backfill of any run.

7. **No index files**: Unlike GRIB2 sources, IMERG HDF5 files have no byte-range index.
   Files must be downloaded in full (~5–10 MB each based on 3600×1800 grid at float32).

8. **File size estimate**: 3600 × 1800 grid × ~10 variables × 4 bytes ≈ ~260 MB
   uncompressed; compressed HDF5 files are typically much smaller. Actual file sizes
   should be verified by downloading sample files.

---

## Source Recommendation

| Use case | Recommended source | Why |
|----------|-------------------|-----|
| Operational Early updates | PPS NRT (`jsimpsonhttps`) | Lowest latency (~4h), direct from producer |
| Operational Late updates | PPS NRT (`jsimpsonhttps`) | ~14h latency, direct from producer |
| Operational Final updates | GES DISC HTTPS | PPS arthurhou also works, GES DISC has better tooling |
| Backfill (all runs) | GES DISC S3 (`gesdisc-cumulus-prod-protected`) | High-bandwidth S3 access, longest archive (1998-present), same auth as SMAP |
| Fallback for backfill | GES DISC HTTPS | If outside AWS us-west-2; same Earthdata auth |

**Recommended integration strategy**: Mirror the NDVI CDR pattern — use PPS NRT for
operational updates (lowest latency), fall back to GES DISC HTTPS/S3 for archive access
and if NRT files are unavailable. Both sources require Earthdata/PPS registration but the
auth mechanisms differ; the GES DISC auth can reuse our existing SMAP `earthdata_auth.py`
infrastructure.

---

## Version History

| Version | Release date | Notes |
|---------|-------------|-------|
| V03 | Early 2014 | Initial GPM IMERG release, GPM era only |
| V04 | March 2016 | Improved GMI calibration, GPM era only |
| V05 | ~2017–2018 | Accuracy improvements, first widespread research use |
| V06A | March 2019 | First record spanning TRMM + GPM eras (2000-06 onward); retracted within days due to snow/ice masking bug |
| V06B | March 2019 | Bug fix for snow/ice masking; primary operational version through mid-2023 |
| V07A | July 7, 2023 | Current generation: gridding offset corrected, improved intercalibration, IR algorithm upgraded, removed frozen-surface PMW masking; primary field renamed `precipitation`. Early/Late updated first. |
| V07B | January 2024 | Re-reprocessing of V07A to fix 162 orbits of defective GPROF estimates across all passive microwave satellites |
| V07 Final RP | ~November 2024 | Retrospective reprocessing of Final Run to V07B completed |

Only **V07B** data should be used. Older versions are superseded and NASA no longer
maintains them. V07 also supersedes the TRMM-era TMPA products (3B42, 3B43).

**Known operational risk (2025):** DoD/FNMOC shut down F16, F17, F18 SSMIS data flow
on June 30, 2025, removing 3 conically-scanning radiometers from the GPM constellation.
This will reduce accuracy of all IMERG products from that date forward.

---

## Next Steps

1. Register for Earthdata Login and PPS NRT access.
2. Download and examine sample HDF5 files from:
   - Early archive: 1998-01-01 (TRMM era)
   - Recent: 2024-01-15
   - Using: `python -c "import h5py; f = h5py.File('file.HDF5', 'r'); print(list(f['Grid'].keys()))"`
   - Or rasterio: `HDF5:{path}:/Grid/precipitation`
3. Verify dimension ordering (`time, lon, lat` vs `time, lat, lon`).
4. Confirm exact variable names in V07B files (watch for `precipitation` vs `precipitationCal`).
5. Measure actual compressed file sizes to estimate storage and transfer costs.
6. Decide which runs to integrate (Early only for lowest latency? Early + Final for
   quality? All three?).
