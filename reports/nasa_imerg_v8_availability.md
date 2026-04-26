# Source Data Exploration: NASA IMERG V08 (Early / Late / Final)

**Date of investigation**: 2026-04-26
**Question**: Is IMERG Version 08 data available yet for the three latency runs (Early, Late, Final), focused on the core precipitation variables?

## TL;DR

**No IMERG V08 data is publicly available as of 2026-04-26**, in any of the three runs (Early, Late, Final). IMERG remains at V07 only.

This was verified directly against the NASA GES DISC data server, which is the authoritative public distribution point. While many *other* GPM Level-3 products (DPR radar, GPROF radiometer, SLH latent heating, etc.) do have `.08` directories created in March–April 2026, the IMERG product family does *not*. V08 for IMERG is described in the latest V07 release notes only as forward-looking work ("the CORRA and GPROF teams are examining this issue for V08") with no announced production date.

No sample files were obtainable, so the structural sections of the standard exploration template are deliberately left empty below — there is nothing yet to characterize.

## Evidence

### 1. GES DISC GPM_L3 directory listing

Listed `https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/` on 2026-04-26. The IMERG product directories present:

| Product | Description | V07 dir | V08 dir |
|---|---|---|---|
| `GPM_3IMERGHHE` | Half-hourly Early Run | present (mod 2026-01-01) | **absent (HTTP 404)** |
| `GPM_3IMERGHHL` | Half-hourly Late Run | present (mod 2026-01-01) | **absent (HTTP 404)** |
| `GPM_3IMERGHH`  | Half-hourly Final Run | present (mod 2025-06-13) | **absent (HTTP 404)** |
| `GPM_3IMERGDE`  | Daily Early Run | present (mod 2026-01-02) | **absent (HTTP 404)** |
| `GPM_3IMERGDL`  | Daily Late Run | present (mod 2026-01-02) | **absent (HTTP 404)** |
| `GPM_3IMERGDF`  | Daily Final Run | present (mod 2025-06-23) | **absent (HTTP 404)** |
| `GPM_3IMERGM`   | Monthly Final Run | present (mod 2025-06-13) | **absent (HTTP 404)** |

For comparison, on the same server other GPM Level-3 product families *do* have V08 directories created recently (e.g. `GPM_3DPR.08/` mod 2026-03-24, `GPM_3DPRD.08/` mod 2026-04-20, `GPM_3GPROFGPMGMI.08/` mod 2026-03-18, multiple `GPM_3GPROF*SSMIS_CLIM.08/` and `GPM_3GPROFNOAA*ATMS_CLIM.08/` dirs mod 2026-04-11 to 2026-04-25). So the absence of IMERG `.08` directories is not a server issue — V08 IMERG is simply not being distributed yet.

Probed URLs and responses:

```
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.08/   -> HTTP 404
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.08/  -> HTTP 404
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.08/  -> HTTP 404
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.08/   -> HTTP 404
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.08/    -> HTTP 404
```

### 2. PPS NRT (`jsimpsonhttps`) for Early/Late real-time

```
https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/early/  -> HTTP 401 (auth required, no public listing)
https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/late/   -> HTTP 401 (auth required, no public listing)
```

Browsing requires an EarthData/PPS registered account; the publicly browsable mirror at GES DISC (above) is the canonical view of what is actually being produced. No V08 NRT path or announcement was found.

### 3. Latest IMERG release notes (V07, 20 Nov 2024)

`https://gpm.nasa.gov/sites/default/files/2024-12/IMERG_V07_ReleaseNotes_241126.pdf` is the most recent release notes document published on the GPM site. The only V08 mention in the entire document is a single forward-looking sentence in the context of describing a known V07 issue:

> "We identified this issue too recently to develop, implement, and test a fix in V07 IMERG. … The CORRA and GPROF teams are examining this issue for V08."

No V08 release-notes PDF exists at predictable paths on `gpm.nasa.gov` or `docserver.gesdisc.eosdis.nasa.gov` (all 404). No V08 announcement appears on the GPM "Data News" page (latest entries dated April 2025). The IMERG main page (`https://gpm.nasa.gov/data/imerg`) references only V06 and V07.

### 4. Conference / planning context

A 2025 AMS conference abstract titled "IMERG V08 and Beyond (Invited)" describes V08 as "currently in progress" and lists planned changes (orbit-boost shifts, SmallSat ingestion, ML algorithms, uncertainty estimates). It does not give a production or release date. A separate note in V08 search results indicates the related Version 08 DPR L3 radar products will not release until April 2026, with single-frequency 2AKu/2AKa V08 "shortly" thereafter — IMERG is downstream of those calibration products, so an IMERG V08 production start in the near term is unlikely until those upstream V08 inputs settle.

## Recommendation

Defer any IMERG V08 integration work. Re-check `https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/` periodically for the appearance of `GPM_3IMERGHHE.08/`, `GPM_3IMERGHHL.08/`, and `GPM_3IMERGHH.08/` (or the daily/monthly equivalents), and watch the GPM Data News page and `gpm.nasa.gov/sites/default/files/.../IMERG_V08_ReleaseNotes*.pdf` for an official release notes PDF.

If a V07 IMERG integration is still desired in the meantime, that is feasible today using the existing `GPM_3IMERGHHE.07` (Early), `GPM_3IMERGHHL.07` (Late), and `GPM_3IMERGHH.07` (Final) products at GES DISC — but per request, the V07 archive is not characterized in this report.

---

## Exploration template (V08 — not yet populatable)

The standard template fields below are intentionally blank because no V08 files exist to inspect.

### Dataset: NASA GPM IMERG V08 (Early / Late / Final)

#### Source Information
- **Summary of data organization**: unknown — no V08 files published.
- **File format**: not yet determined for V08; V07 IMERG L3 uses HDF5 (`.HDF5`), with NetCDF, GeoTIFF, and OPeNDAP also offered at GES DISC. V08 is expected to follow the same conventions but this is unverified.
- **Temporal coverage**: not started.
- **Temporal frequency**: V07 produces half-hourly, daily, and monthly granules. V08 expected to mirror this, unverified.
- **Latency** (V07, expected to carry over): Early ~4 h, Late ~14 h, Final ~3.5 months.
- **Access notes**: GES DISC requires NASA Earthdata Login; PPS NRT (`jsimpsonhttps.pps.eosdis.nasa.gov`) requires a separate PPS account.
- **Browse root** (where V08 *would* appear): `https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/`
- **URL format** (extrapolated from V07 pattern, **unverified for V08**):
```
# V07 pattern - V08 would presumably substitute .08 and Vnn-> V08
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.08/{YYYY}/{DDD}/3B-HHR-E.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V08X.HDF5
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.08/{YYYY}/{DDD}/3B-HHR-L.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V08X.HDF5
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.08/{YYYY}/{DDD}/3B-HHR.MS.MRG.3IMERG.{YYYYMMDD}-S{HHMMSS}-E{HHMMSS}.{MMMM}.V08X.HDF5
```

#### GRIB Index
- N/A — IMERG is HDF5, not GRIB.

#### Coordinate Reference System
- Unknown for V08. V07 uses WGS84 geographic at 0.1° × 0.1°, global (-180..180, -90..90, pixel-center). Likely unchanged in V08 but unverified.

#### Dimensions & Dimension Coordinates

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| time      |     |     |      | unknown — no V08 files |
| latitude  |     |     |      | unknown |
| longitude |     |     |      | unknown |

#### Data Variables (core precipitation only — what we care about)

| Variable name (V07) | Level | Units | Available in V08 | Notes |
|---------------------|-------|-------|------------------|-------|
| `precipitation`     | surface | mm/hr | unknown | V07 calibrated multi-satellite estimate (was `precipitationCal` pre-V07). |
| `precipitationUncal` | surface | mm/hr | unknown | V07 uncalibrated estimate. |
| `randomError`       | surface | mm/hr | unknown | V07 random error estimate. |
| `probabilityLiquidPrecipitation` | surface | % | unknown | V07. |
| `precipitationQualityIndex` | surface | dimensionless | unknown | V07. |

V08 variable names, units, and `_FillValue` cannot be confirmed without files. Note the V06 → V07 rename of `precipitationCal` → `precipitation`; further renames in V08 are possible.

#### Sample Files Examined
- None — no V08 files exist to download.

#### Notable Observations
- Other GPM Level-3 V08 directories appeared at GES DISC between 2026-03-18 and 2026-04-25 (DPR, SLH, multiple GPROF radiometer products), so the V08 reprocessing campaign is clearly underway upstream — but IMERG V08 specifically is not yet being distributed.
- The next concrete checkpoints to monitor: appearance of any `GPM_3IMERG*.08/` directory at GES DISC, and publication of an `IMERG_V08_ReleaseNotes*.pdf` on `gpm.nasa.gov`.
