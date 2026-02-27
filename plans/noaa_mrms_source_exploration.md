## Dataset: NOAA MRMS (Multi-Radar Multi-Sensor) — Hourly QPE

MRMS is a NOAA system that fuses data from ~180 WSR-88D (NEXRAD) radars, gauge networks,
satellite, and NWP model fields to produce high-resolution (~1 km, 2-minute) precipitation
estimates and other diagnostic products over CONUS and adjacent territories. It became
operationally deployed at NCEP in September 2014; a predecessor (NMQ/Q2) ran at NSSL before that.
MRMS v12.0 launched October 14, 2020, introducing the `MultiSensor_QPE` product family.

This exploration focuses on **hourly** fields. The core product is the hourly MultiSensor QPE
accumulation; PrecipRate, PrecipFlag, and merged reflectivity are also relevant.

---

### Source 1 — AWS Open Data (noaa-mrms-pds S3 bucket)

- **Summary of data organization**: One gzip-compressed GRIB2 file per product per valid time,
  organized by region subdirectory and product subdirectory. The height/level (in km) is encoded
  in both the path and filename. Each file contains exactly one GRIB2 message (one variable, one
  time step). No `.idx` index files are provided — they aren't needed since each file is a single
  message.
- **File format**: GRIB2, gzip-compressed (`.grib2.gz`). Decompress before reading with rasterio.
  ~270–600 KB compressed per hourly QPE file.
- **Temporal coverage**: 2020-10-14 (20:00 UTC) to present, near real-time
- **Temporal frequency**:
  - Hourly QPE products (MultiSensor_QPE, RadarOnly_QPE): one file per hour on the hour
  - 2-minute products (PrecipRate, PrecipFlag, reflectivity): one file every 2 minutes with second-precision UTC timestamps
- **Latency**: Data appears in S3 within 2–3 hours of valid time (verified: 23:00 UTC QPE Pass2 data for 2026-02-26 was in S3 by ~00:00 UTC next day). The NOAA NCEP HTTP server is authoritative and has lower latency (see Source 3).
- **Access notes**: Public, no authentication. HTTP access at `https://noaa-mrms-pds.s3.amazonaws.com/`. Region: `us-east-1`. SNS notifications for new objects available at `arn:aws:sns:us-east-1:123901341784:NewMRMSObject`.
- **Browse root**: https://noaa-mrms-pds.s3.amazonaws.com/index.html
- **Regions available in bucket**: `CONUS/`, `ALASKA/`, `CARIB/`, `GUAM/`, `HAWAII/`, `CONUS_5KM/`, `ANC/`, `ConvectProb/`, `ProbSevere/`
- **URL format**:
```
https://noaa-mrms-pds.s3.amazonaws.com/{REGION}/{PRODUCT}_{LEVEL_KM}/{YYYYMMDD}/MRMS_{PRODUCT}_{LEVEL_KM}_{YYYYMMDD-HHMMSS}.grib2.gz
```
- **Example URLs**:
```
# Hourly MultiSensor QPE Pass2 (best accuracy, ~60-min latency)
https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00/20210115/MRMS_MultiSensor_QPE_01H_Pass2_00.00_20210115-120000.grib2.gz

# Hourly MultiSensor QPE Pass1 (near-real-time, ~20-min latency)
https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass1_00.00/20210115/MRMS_MultiSensor_QPE_01H_Pass1_00.00_20210115-120000.grib2.gz

# 2-minute PrecipRate
https://noaa-mrms-pds.s3.amazonaws.com/CONUS/PrecipRate_00.00/20201014/MRMS_PrecipRate_00.00_20201014-000000.grib2.gz

# Earliest available date: 2020-10-14
https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00/20201014/MRMS_MultiSensor_QPE_01H_Pass2_00.00_20201014-200000.grib2.gz
```

---

### Source 2 — Iowa Mesonet MTArchive (mtarchive.geol.iastate.edu)

This is the best source for data **before the S3 bucket begins (pre-2020-10-14)**.

- **Summary of data organization**: One gzip-compressed GRIB2 file per product per valid time,
  in the same naming convention as Source 1. Archives a curated subset of ~15 MRMS products (not
  all 241 CONUS products). Data appears to be sourced from NCEP operational feed.
- **File format**: GRIB2, gzip-compressed (`.grib2.gz`). Same format as S3.
- **Temporal coverage**: October 2014 (MRMS v11.x operational launch at NCEP) to present
- **Temporal frequency**: Hourly for QPE products (same as S3)
- **Latency**: Near real-time (minutes to low hours behind production)
- **Access notes**: Public HTTP, no authentication required
- **Browse root**: https://mtarchive.geol.iastate.edu/
- **URL format**:
```
https://mtarchive.geol.iastate.edu/{YYYY}/{MM}/{DD}/mrms/ncep/{PRODUCT}/MRMS_{PRODUCT}_{LEVEL_KM}_{YYYYMMDD-HHMMSS}.grib2.gz
```
- **Example URLs**:
```
# Post-v12.0 (Oct 2020 onward) — MultiSensor_QPE products
https://mtarchive.geol.iastate.edu/2024/01/15/mrms/ncep/MultiSensor_QPE_01H_Pass2/MRMS_MultiSensor_QPE_01H_Pass2_00.00_20240115-000000.grib2.gz

# Pre-v12.0 (2014–2020) — GaugeCorr_QPE (predecessor to MultiSensor_QPE)
https://mtarchive.geol.iastate.edu/2019/06/15/mrms/ncep/GaugeCorr_QPE_01H/MRMS_GaugeCorr_QPE_01H_00.00_20190615-120000.grib2.gz

# RadarOnly_QPE available across full archive
https://mtarchive.geol.iastate.edu/2016/06/15/mrms/ncep/RadarOnly_QPE_01H/MRMS_RadarOnly_QPE_01H_00.00_20160615-120000.grib2.gz
```

**Products available by era** (from directory listings; curated subset only):

| Era | QPE products | Other products |
|-----|-------------|----------------|
| 2014–Oct 2020 (v11.x) | `GaugeCorr_QPE_01H`, `GaugeCorr_QPE_24H`, `GaugeCorr_QPE_72H`, `RadarOnly_QPE_01H`, `RadarOnly_QPE_24H`, `RadarOnly_QPE_72H` | `PrecipFlag`, `PrecipRate`, `RadarQualityIndex`, `SeamlessHSR`, `RotationTrack1440min`, `FLASH/`, `MESH/` |
| Oct 2020–present (v12.x) | `MultiSensor_QPE_01H_Pass2`, `MultiSensor_QPE_24H_Pass2`, `MultiSensor_QPE_72H_Pass2`, `RadarOnly_QPE_01H`, `RadarOnly_QPE_24H`, `RadarOnly_QPE_72H` | `PrecipFlag`, `PrecipRate`, `RadarQualityIndex`, `SeamlessHSR`, `RotationTrack1440min`, `FLASH/`, `MESH/`, `MESH_Max_1440min/`, `ProbSevere/` |

**Critical product discontinuity at v12.0 (Oct 14, 2020)**:
- `GaugeCorr_QPE_01H` (radar QPE with gauge bias correction) → replaced by `MultiSensor_QPE_01H_Pass1` / `_Pass2` (full multi-sensor fusion including Mountain Mapper and NWP QPF). These are scientifically different products; the multi-sensor version is superior. A combined archive would have a step-change in data quality at this date.
- Note: `MultiSensor_QPE_01H_Pass1` is **not** archived by IEM — only Pass2 is. For Pass1 in the pre-2020 period, there is no equivalent archive (GaugeCorr_QPE is the closest analog for Pass2, not Pass1).

---

### Source 3 — NOAA NCEP Direct HTTP (mrms.ncep.noaa.gov)

- **Summary**: Operational real-time server. Lowest latency. Rolling ~48-hour window only — not an archive.
- **File format**: GRIB2, gzip-compressed. Same file format and naming as S3.
- **Temporal coverage**: Rolling ~48 hours of current data
- **Temporal frequency**: Same as S3
- **Latency** (verified 2026-02-26):
  - `MultiSensor_QPE_01H_Pass1`: ~16 minutes after valid time (e.g., 23:00 UTC product posted at 23:15–23:16 UTC)
  - `MultiSensor_QPE_01H_Pass2`: ~55–60 minutes after valid time (e.g., 22:00 UTC product posted at 22:57 UTC)
- **Access notes**: Public HTTP, no authentication. A `.latest.grib2.gz` convenience symlink is provided in each product directory.
- **Browse root**: https://mrms.ncep.noaa.gov/2D/
- **URL format**:
```
https://mrms.ncep.noaa.gov/2D/{PRODUCT}/MRMS_{PRODUCT}_{LEVEL_KM}_{YYYYMMDD-HHMMSS}.grib2.gz
https://mrms.ncep.noaa.gov/2D/{PRODUCT}/MRMS_{PRODUCT}.latest.grib2.gz
```
- **Example URLs**:
```
https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260226-220000.grib2.gz
https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass1/MRMS_MultiSensor_QPE_01H_Pass1_00.00_20260226-230000.grib2.gz
https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/MRMS_MultiSensor_QPE_01H_Pass2.latest.grib2.gz
```

**Available 2D product directories on mrms.ncep.noaa.gov** (selected):
```
MultiSensor_QPE_01H_Pass1    MultiSensor_QPE_01H_Pass2    MultiSensor_QPE_03H_Pass1/2
MultiSensor_QPE_06H_Pass1/2  MultiSensor_QPE_12H_Pass1/2  MultiSensor_QPE_24H_Pass1/2
MultiSensor_QPE_48H_Pass1/2  MultiSensor_QPE_72H_Pass1/2
RadarOnly_QPE_01H            RadarOnly_QPE_03H             RadarOnly_QPE_06H
RadarOnly_QPE_12H            RadarOnly_QPE_24H             RadarOnly_QPE_48H
RadarOnly_QPE_72H            RadarOnly_QPE_15M             RadarOnly_QPE_Since12Z
PrecipRate                   PrecipFlag                    MergedReflectivityAtLowestAltitude
MergedReflectivityQC         MergedReflectivityQCComposite MergedBaseReflectivityQC
RadarQualityIndex             GaugeInflIndex_01H_Pass1/2   WarmRainProbability
BrightBandTopHeight          BrightBandBottomHeight        EchoTop_18/30/50/60
VIL                          MESH                          POSH
```

---

### Source 4 — Iowa Mesonet GIS Archive (images only)

- **Format**: PNG rendered images + WLD world files. Not GRIB2. **Not viable for numerical data.**
- Coverage: ~2014/2015 to present at https://mesonet.agron.iastate.edu/archive/data/{YYYY}/{MM}/{DD}/GIS/mrms/

---

### GRIB Index

- **Index files available**: No, on any source.
- **Rationale**: Each MRMS GRIB2 file contains exactly one GRIB2 message (one variable, one timestep). There is nothing to index — just download the whole (small) file.

---

### Coordinate Reference System

- **Common name**: Geographic lat/lon with IAU 1965 spheroid (not WGS84)
- **CRS from actual files** (rasterio WKT):
  ```
  GEOGCS["Coordinate System imported from GRIB file",
    DATUM["unnamed", SPHEROID["Spheroid imported from GRIB file", 6378160, 298.253916296469]],
    PRIMEM["Greenwich", 0],
    UNIT["degree", 0.0174532925199433, AUTHORITY["EPSG","9122"]],
    AXIS["Latitude", NORTH], AXIS["Longitude", EAST]]
  ```
- The spheroid (6378160 m, 1/f ≈ 298.25) is the IAU 1965 spheroid used in legacy NOAA radar processing. Difference from WGS84 is sub-pixel at 0.01° resolution.
- Dimension names: `latitude` and `longitude` (geographic CRS, not projected)

---

### Dimensions & Dimension Coordinates

Verified by rasterio inspection of files from 2020-10-14 and 2026-02-26 (identical grid):

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| time | 2014-10 (IEM) / 2020-10-14 (S3) | present | 1 hour (QPE) / 2 min (instantaneous) | S3 archive starts 2020-10-14 |
| latitude | 20.005° N | 54.995° N | 0.01° (~1.1 km) | Pixel centers; 3500 rows |
| longitude | -129.995° E | -60.005° E | 0.01° (~0.9 km) | Pixel centers; 7000 columns |

Rasterio pixel-edge bounds: left=-130.0, bottom=20.0, right=-60.0, top=55.0
Grid: 7000 × 3500 (width × height); same for all products examined.
The CONUS grid is stable across the entire AWS archive period (2020–present).

---

### Data Variables

MRMS GRIB2 uses local parameter table (GRIB discipline 209, center 161/NSSL).
Each file is a single variable at a single time. Below are the variables most relevant for
hourly precipitation applications.

#### Primary QPE — Hourly, verified on NCEP + S3

| Variable | GRIB_ELEMENT | GRIB_UNIT | Frequency | Description |
|----------|-------------|-----------|-----------|-------------|
| MultiSensor_QPE_01H_Pass1 | `MultiSensor_QPE_01H_Pass1` | `[mm]` | 60-min | Gauge+radar+satellite+NWP; ~20-min latency. Available Oct 2020–present (S3 + NCEP). |
| MultiSensor_QPE_01H_Pass2 | `MultiSensor_QPE_01H_Pass2` | `[mm]` | 60-min | Additional gauge QC pass; ~60-min latency. Best accuracy. Available Oct 2020–present (S3 + NCEP + IEM). |
| GaugeCorr_QPE_01H | (predecessor) | `[mm]` | 60-min | Gauge bias-corrected radar QPE. Available Oct 2014–Oct 2020 via IEM only. |
| RadarOnly_QPE_01H | `RadarOnly_QPE_01H` | `[mm]` | 2-min updates | Hourly radar accumulation, no gauges. Available Oct 2014–present (all sources). |
| RadarOnly_QPE_15M | `RadarOnly_QPE_15M` | `[mm]` | 15-min | 15-minute radar accumulation. |

GRIB comment in actual file: `'Multi-sensor accumulation 1-hour (2-hour latency) [mm]'` — note the GRIB_COMMENT says "2-hour latency" but actual operational latency is ~60 min; the comment appears to be a documentation artifact.

#### Instantaneous / 2-minute products

| Variable | GRIB_ELEMENT | GRIB_UNIT | Frequency | Description |
|----------|-------------|-----------|-----------|-------------|
| PrecipRate | `PrecipRate` | `[mm/hr]` | 2-min | Instantaneous radar precipitation rate |
| PrecipFlag | `PrecipFlag` | `[flag]` | 2-min | Precipitation type: 0=none, 1=warm stratiform, 3=snow, 6=convective, 7=hail, 10=cold stratiform, 91=tropical |
| MergedReflectivityAtLowestAltitude | — | `[dBZ]` | 2-min | Reflectivity at lowest unblocked tilt (0.50 km level) |
| MergedReflectivityQC | — | `[dBZ]` | 2-min | QC'd 3D reflectivity (multiple levels: 0.50–2.00 km on NCEP) |

#### Quality / Ancillary — Hourly

| Variable | Units | Frequency | Description |
|----------|-------|-----------|-------------|
| GaugeInflIndex_01H_Pass1 / _Pass2 | non-dim (0–1) | 60-min | Fraction of QPE cell influenced by gauges |
| RadarAccumulationQualityIndex_01H | non-dim | 60-min | Quality of 1-hour radar accumulation |
| RadarQualityIndex | non-dim (0–1) | 2-min | Instantaneous radar data quality |
| WarmRainProbability | % | 60-min | Probability of warm-rain (no-ice) precipitation process |

**Temporal availability changes**:
- `GaugeCorr_QPE_01H` available 2014–2020-10-13 (IEM only); replaced by `MultiSensor_QPE_01H`
- `MultiSensor_QPE_01H_Pass1` / `_Pass2`: available from 2020-10-14 onward
- Pre-v12.0 products may have slightly different grid extent or metadata (not verified; the IEM archive should be checked carefully at the transition date)

---

### Sample Files Examined

- **Earliest S3**: 2020-10-14 20:00 UTC — `MRMS_MultiSensor_QPE_01H_Pass2_00.00_20201014-200000.grib2.gz`
  Grid: 7000×3500, 0.01°, bounds [-130, 20, -60, 55], units mm ✓
- **Recent S3/NCEP**: 2026-02-26 22:00 UTC — `MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260226-220000.grib2.gz`
  Grid: identical to 2020 file ✓; GRIB_COMMENT: `"Multi-sensor accumulation 1-hour (2-hour latency) [mm]"`
- **PrecipRate**: 2020-10-14 00:00:00 UTC — same grid (7000×3500, 0.01°), units mm/hr ✓

---

### Notable Observations

1. **One variable per file**: Each MRMS GRIB2 file contains exactly one band. Byte-range downloads
   are not needed — just download the whole compressed file (~200–600 KB).

2. **Grid is stable**: The CONUS grid (7000×3500, 0.01°, [-130, -60, 20, 55]) is identical
   across all products and unchanged between 2020 and 2026. No grid-change handling needed
   for the S3 archive period.

3. **Product discontinuity at v12.0**: The most important discontinuity is the rename of
   `GaugeCorr_QPE_01H` → `MultiSensor_QPE_01H_Pass2` on Oct 14, 2020. The multi-sensor product
   incorporates Mountain Mapper analysis and NWP QPF in addition to gauge correction, so it is
   scientifically better but not directly comparable to `GaugeCorr_QPE`. A combined archive
   would span two scientifically distinct product generations.

4. **Pass1 vs Pass2**:
   - Pass1 (~20-min latency): real-time gauge set (~10% of available gauges). Good for operational
     use requiring low latency.
   - Pass2 (~60-min latency): larger gauge set with QC (~60% of gauges). Higher accuracy.
     Preferred for archives and post-event analysis.
   - IEM MTArchive only archives Pass2, not Pass1, for the pre-2020 equivalent (`GaugeCorr_QPE`).

5. **Source priority for integration**:
   - **Backfill (Oct 2020–present)**: AWS S3 is the primary source — full product set, fast
     object storage, long retention.
   - **Backfill (Oct 2014–Oct 2020)**: Iowa Mesonet MTArchive for `GaugeCorr_QPE_01H` and
     `RadarOnly_QPE_01H`. This requires separate code handling for the older product name.
   - **Operational updates**: Try S3 first (good throughput); fall back to NCEP HTTP for the
     most recent hours not yet propagated to S3 (especially for Pass1's ~20-min latency window).

6. **Iowa Mesonet MTArchive coverage gap**: IEM archives only ~15 curated products (not all 241
   CONUS products). Specifically, `GaugeInflIndex`, `WarmRainProbability`, and many reflectivity
   products are not available in the pre-2020 IEM archive.

7. **CONUS only**: The hourly QPE products are CONUS-specific. Alaska and Hawaii products are
   separately organized in S3 and may have different spatial extents and temporal availability.
   Not explored here.

8. **No ensemble dimension**: MRMS is deterministic (single analysis). No ensemble members.

---

### Source Comparison Summary

| Source | Temporal Coverage | Frequency | Format | Key QPE Products | Access | Latency |
|--------|-------------------|-----------|--------|-----------------|--------|---------|
| **AWS S3** `noaa-mrms-pds` | 2020-10-14 – present | Hourly (QPE) / 2-min | GRIB2.gz | All 241 CONUS products incl. MultiSensor QPE Pass1/Pass2 | Anonymous HTTP/S3 | 2–3 hours |
| **IEM MTArchive** `mtarchive.geol.iastate.edu` | 2014-10 – present | Hourly | GRIB2.gz | ~15 products: QPE, PrecipRate, PrecipFlag; `GaugeCorr_QPE` pre-2020 | Anonymous HTTP | Near real-time |
| **NCEP direct** `mrms.ncep.noaa.gov` | Rolling ~48 hours | Hourly (QPE) / 2-min | GRIB2.gz | All 140+ 2D products | Anonymous HTTP | 16–60 min |
| **Iowa Mesonet GIS** `mesonet.agron.iastate.edu` | ~2014 – present | 2-min | PNG images | None (images only) | Anonymous HTTP | N/A |
| **NCEI HAS** | 2014 – present | Hourly | GRIB2 | Historical, request-based | Request workflow | Days (tape) |

---

### Recommended Variables for a Hourly Precipitation Analysis Dataset

| Priority | Variable | Source | Reasoning |
|----------|----------|--------|-----------|
| High | `MultiSensor_QPE_01H_Pass2` | S3 (Oct 2020+), IEM (as `GaugeCorr_QPE_01H` pre-2020) | Gold standard hourly QPE |
| High | `MultiSensor_QPE_01H_Pass1` | S3 + NCEP (Oct 2020+) | Near-real-time operational QPE |
| Medium | `RadarOnly_QPE_01H` | S3 + IEM (Oct 2014+) | Radar-only baseline; longest archive |
| Medium | `PrecipFlag` | S3 + IEM (Oct 2014+) | Precipitation type classification |
| Medium | `PrecipRate` | S3 + IEM (Oct 2014+) | Instantaneous rate for sub-hourly use |
| Medium | `MergedReflectivityAtLowestAltitude` | S3 + NCEP (Oct 2020+) | Base reflectivity for storm detection |
| Lower | `GaugeInflIndex_01H_Pass2` | S3 (Oct 2020+) | QPE quality metadata |
| Lower | `RadarQualityIndex` | S3 + IEM (Oct 2014+) | Radar data quality flag |
| Lower | `WarmRainProbability` | S3 (Oct 2020+) | Precip type disambiguation |
