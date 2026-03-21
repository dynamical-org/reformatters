## Dataset: NOAA RRFS (Rapid Refresh Forecast System)

RRFS is NOAA's next-gen convection-allowing model replacing RAP, HRRR, NAM, HREF, HiResW, and NARRE. Built on the FV3 dynamical core (UFS). Went operational in early-to-mid March 2026.

### Source Information

- **Summary of data organization**: One GRIB2 file per lead time, init time, domain, and product type (natlev or prslev). All variables for that product type are in a single file.
- **File format**: GRIB2 (with `.idx` index files)
- **Temporal coverage**: 2026-03-11 to present (~10 days retained in bucket)
- **Temporal frequency**: Hourly init times. 00/06/12z cycles run to 84h; all other hours run to 18h. 18z likely also runs to 84h (incomplete data observed for current day). Lead time step is 1 hour throughout.
- **Latency**: Not yet measured.
- **Access notes**: Public S3 bucket, no authentication required. Files are large (~6 GB for prslev NA, ~9 GB for natlev NA per lead time). GRIB index files enable byte-range partial downloads.
- **Browse root**: https://noaa-rrfs-pds.s3.amazonaws.com/index.html
- **URL format**:
```
https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.{YYYYMMDD}/{HH}/rrfs.t{HH}z.prslev.3km.f{FFF}.na.grib2
https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.{YYYYMMDD}/{HH}/rrfs.t{HH}z.prslev.3km.f{FFF}.na.grib2.idx
https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.{YYYYMMDD}/{HH}/rrfs.t{HH}z.natlev.3km.f{FFF}.na.grib2
```
- **Example URLs**:
```
https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.20260320/00/rrfs.t00z.prslev.3km.f006.na.grib2
https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.20260320/00/rrfs.t00z.prslev.3km.f006.na.grib2.idx
```

### GRIB Index
- **Index files available**: Yes
- **Index style**: NOAA (colon-separated)
- **Example lines**:
```
739:4678831439:d=2026032000:TMP:2 m above ground:6 hour fcst:
753:4776210918:d=2026032000:UGRD:10 m above ground:6 hour fcst:
762:4841638018:d=2026032000:APCP:surface:0-6 hour acc fcst:
834:5231052773:d=2026032000:DSWRF:surface:5-6 hour ave fcst:
```

### Coordinate Reference System
- **Common name**: Not yet verified from file (likely Lambert Conformal Conic for the native 3km NA grid — needs confirmation with rasterio)

### Dimensions & Dimension Coordinates

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| init_time | 2026-03-11T00Z | present | 1h | Hourly cycles |
| lead_time | 0h | 84h | 1h | 84h at 00/06/12/18z, 18h otherwise |
| y | | | | 3km NA grid, exact dims TBD (need rasterio) |
| x | | | | 3km NA grid, exact dims TBD (need rasterio) |
| pressure_level | 2 mb | 1000 mb | varies | 49 levels in prslev files |
| hybrid_level | 1 | 65 | 1 | 65 levels in natlev files |

### Data Variables

From the `prslev.3km.na` index (992 total records, ~130 unique variable names):

| Variable name | Level | Units | Available | Notes |
|---------------|-------|-------|-----------|-------|
| temperature_2m | 2 m above ground | K | Yes (TMP) | |
| wind_u_10m | 10 m above ground | m/s | Yes (UGRD) | |
| wind_v_10m | 10 m above ground | m/s | Yes (VGRD) | |
| wind_u_100m | 100 m above ground | m/s | Not seen | Not in idx |
| wind_v_100m | 100 m above ground | m/s | Not seen | Not in idx |
| precipitation_surface | surface | kg/m2 | Yes (APCP) | Both 0-N hour acc and hourly acc available |
| downward_short_wave_radiation_flux_surface | surface | W/m2 | Yes (DSWRF) | Both instantaneous and 1h average |
| downward_long_wave_radiation_flux_surface | surface | W/m2 | Yes (DLWRF) | Both instantaneous and 1h average |
| pressure_surface | surface | Pa | Yes (PRES) | |
| pressure_reduced_to_mean_sea_level | MSL | Pa | Yes (MSLET) | |
| total_cloud_cover_atmosphere | atmosphere | % | Yes (TCDC) | Multiple levels available |
| relative_humidity_2m | 2 m above ground | % | Yes (RH) | |
| specific_humidity_2m | 2 m above ground | kg/kg | Yes (SPFH) | |
| dew_point_temperature_2m | 2 m above ground | K | Yes (DPT) | |

Additional notable variables: GUST (surface), CAPE, CIN, PWAT, VIS, HAIL, LTNG, REFC (composite reflectivity), snow vars (WEASD, SNOD, SNOWC, ASNOW), soil vars (SOILW, TSOIL, SOILL), LHTFL, SHTFL, USWRF, ULWRF at surface and TOA.

### Bucket Structure

```
noaa-rrfs-pds/
├── rrfs_a/                              # Primary operational output
│   ├── rrfs.YYYYMMDD/HH/               # Deterministic RRFS
│   │   ├── rrfs.tHHz.prslev.3km.fFFF.{na,conus,ak}.grib2      # Pressure/surface (main interest)
│   │   ├── rrfs.tHHz.prslev.2p5km.fFFF.{hi,pr}.grib2           # Hawaii/PR at 2.5km
│   │   ├── rrfs.tHHz.natlev.3km.fFFF.na.grib2                  # Native levels (65 hybrid, ~9 GB)
│   │   ├── forecast/INPUT/              # Model restart files (not useful)
│   │   └── lbcs/                        # GFS lateral boundary conditions (not useful)
│   ├── rrfsens.YYYYMMDD/HH/m00N/       # Ensemble members (BUFR only, no GRIB2)
│   └── refs.YYYYMMDD/HH/               # REFS ensemble GIF images only
├── rrfs_public/
│   ├── rrfs.YYYYMMDD/HH/               # BUFR soundings only (no GRIB2)
│   ├── refs.YYYYMMDD/HH/enspost/       # REFS ensemble post-processed (GRIB2)
│   │   └── refs.tHHz.{conus,ak}.{mean,sprd,prob,avrg,lpmm,pmmn,eas}.fFF.grib2
│   └── firewx.YYYYMMDD/HH/             # Fire weather nest (1.5km, LCC, 0-36h)
└── retro_output_final/, rrfs_retro_maps/, rrfsv1_eval_meg/  # Retro/experimental (ignore)
```

### Comparison to Announced Plans

| Aspect | Planned | Observed | Match? |
|--------|---------|----------|--------|
| Init frequency | Hourly | Hourly | Yes |
| Extended cycles | 00/06/12/18z to 60h | 00/06/12/18z to 84h | Exceeds plan |
| Short cycles | Other hours to 18h | Other hours to 18h | Yes |
| Ensemble members | 9-10 | 5 (BUFR only) | Fewer, no gridded output |
| Ensemble post products | Yes | Yes (mean/sprd/prob, CONUS+AK, to 60h) | Yes |
| Resolution | 3km NA | 3km NA | Yes |
| Replaces | RAP, HRRR, NAM, HREF | — | — |

### Notable Observations
- Bucket retention is only ~10 days. No historical archive available here.
- Two stray July 2025 `refs` dates exist (likely testing artifacts).
- The `prslev` files come in regional subsets (NA, CONUS, AK, HI, PR) in addition to the full domain, which could be useful for smaller-scope integrations.
- Ensemble gridded output (individual members as GRIB2) is not available in this bucket. Only BUFR soundings per member and post-processed statistics (mean/spread/prob) as GRIB2.
- APCP (precipitation) has both bucket accumulations (0-N hour) and hourly accumulations (N-1 to N hour) — deaccumulation may not be needed.
- Radiation fields (DSWRF, DLWRF) have both instantaneous and 1-hour average variants.
- CRS and exact grid dimensions still need verification by downloading a sample file and inspecting with rasterio.
- 100m wind components were not found in the prslev index.

### Sample Files Examined
- **Recent data**: 2026-03-20 00z, `prslev.3km.f006.na.grib2.idx` (992 records), `natlev.3km.f006.na.grib2.idx` (1730 records)
