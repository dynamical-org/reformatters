# HRRR Analysis Historical Data Investigation

## Dataset: NOAA HRRR Analysis - Historical Extension

### Executive Summary

✅ **Extension is feasible**: The HRRR analysis dataset can be extended backward from the current start date of 2018-09-16.

**Recommended extension**: Start from **2014-10-01** (4+ years of additional data)
- 16 out of 20 current variables available
- 4 variables either missing or require code changes (see details below)
- Stable data availability
- Minimal grid changes (negligible < 0.02 pixel shift)

### Source Information

- **Summary of data organization**: Same as current - one GRIB2 file per init time and file type (sfc), with all variables as bands
- **File format**: GRIB2 (consistent since 2014-09-30)
- **Temporal coverage**: 2014-09-30 (operational start) to present
- **Temporal frequency**: Hourly
- **Latency**: Archival data, all historical hours available
- **Access notes**: AWS Open Data, free egress
- **Browse root**: https://noaa-hrrr-bdp-pds.s3.amazonaws.com/
- **URL format**:
```
https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{YYYYMMDD}/conus/hrrr.t{HH}z.wrfsfcf{FF}.grib2
https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{YYYYMMDD}/conus/hrrr.t{HH}z.wrfsfcf{FF}.grib2.idx
```
- **Example URLs**:
```
https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20141001/conus/hrrr.t00z.wrfsfcf00.grib2
https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20141001/conus/hrrr.t00z.wrfsfcf00.grib2.idx
```

### GRIB Index

- **Index files available**: Yes
- **Index style**: NOAA (colon-separated)
- **Example line**:
```
49:33617865:d=2014100100:TMP:2 m above ground:anl:
```

### Coordinate Reference System

- **Common name**: Lambert Conformal Conic
- **PROJ string**: `+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs=True`
- **CRS WKT**: PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],...]]
- **Grid consistency**: ✅ CRS identical across all versions
- **Minor grid shift**: HRRRv3 (2018+) has a ~53m shift in origin compared to HRRRv1/v2, but this is negligible (< 0.02 pixels)

### Dimensions & Dimension Coordinates

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| time | 2014-10-01T00:00 | Present | 1 hour | Consistent |
| y | -1588806 m | 1588194 m | 3000 m | 1059 pixels |
| x | -2699020 m | 2697980 m | 3000 m | 1799 pixels |

Grid shape: **1059 x 1799** pixels (consistent across all versions)
Pixel size: **3000 m x 3000 m** (consistent)

### Data Variables - Availability Analysis

Checked availability of all 20 current variables across HRRR versions:

| Variable name | HRRRv1 (2014-10-01) | HRRRv2 (2016-08-23) | HRRRv3 (2018-09-16) | Notes |
|---------------|---------------------|---------------------|---------------------|-------|
| composite_reflectivity | ✅ | ✅ | ✅ | Available since day 1 |
| temperature_2m | ✅ | ✅ | ✅ | Available since day 1 |
| wind_u_10m | ✅ | ✅ | ✅ | Available since day 1 |
| wind_v_10m | ✅ | ✅ | ✅ | Available since day 1 |
| precipitation_surface | ✅ | ✅ | ✅ | Available since day 1 |
| precipitable_water_atmosphere | ✅ | ✅ | ✅ | Available since day 1 |
| total_cloud_cover_atmosphere | ✅ | ✅ | ✅ | Available since day 1 |
| downward_short_wave_radiation_flux_surface | ✅ | ✅ | ✅ | Available since day 1 |
| downward_long_wave_radiation_flux_surface | ❌ | ❌ | ✅ | **Added in HRRRv3** |
| pressure_reduced_to_mean_sea_level | ⚠️ | ⚠️ | ✅ | **PRMSL in v1/v2, MSLMA in v3** |
| percent_frozen_precipitation_surface | ✅ | ✅ | ✅ | Available since day 1 |
| pressure_surface | ✅ | ✅ | ✅ | Available since day 1 |
| categorical_ice_pellets_surface | ✅ | ✅ | ✅ | Available since day 1 |
| categorical_snow_surface | ✅ | ✅ | ✅ | Available since day 1 |
| categorical_freezing_rain_surface | ✅ | ✅ | ✅ | Available since day 1 |
| categorical_rain_surface | ✅ | ✅ | ✅ | Available since day 1 |
| relative_humidity_2m | ❌ | ❌ | ✅ | **Added in HRRRv3**, but SPFH & DPT available |
| geopotential_height_cloud_ceiling | ✅ | ✅ | ✅ | Available since day 1 |
| wind_u_80m | ✅ | ✅ | ✅ | Available since day 1 |
| wind_v_80m | ✅ | ✅ | ✅ | Available since day 1 |

**Summary**:
- **16/20 variables**: Available consistently from 2014-10-01
- **1 variable**: Changed GRIB name (PRMSL → MSLMA), requires code update
- **2 variables**: Added in HRRRv3, must be excluded from historical backfill
- **1 variable**: Alternative available (SPFH/DPT instead of RH), optional enhancement

### Temporal Availability Changes

**HRRRv1 (2014-09-30 to 2016-08-22)**:
- First operational day (2014-09-30): Only 57 bands, many variables missing
- From 2014-10-01: 102 bands consistently
- Missing: DLWRF, RH at 2m
- Different name: PRMSL (instead of MSLMA)

**HRRRv2 (2016-08-23 to 2018-07-11)**:
- Transition period (2016-08-23 to ~2016-09-01): Many missing hours due to operational changeover
- Same variable availability as HRRRv1
- 102 bands consistently

**HRRRv3 (2018-07-12 onwards)**:
- Current dataset starts 2018-09-16 (avoiding high missing-data rate in first 2 months)
- Added: DLWRF, RH at 2m
- Changed: PRMSL → MSLMA
- 148 bands (many new variables beyond our 20)

### Sample Files Examined

- **First operational day**: 2014-09-30, https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20140930/conus/hrrr.t00z.wrfsfcf00.grib2
  - 57 bands only, incomplete variable set
- **Early HRRRv1**: 2014-10-01, https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20141001/conus/hrrr.t00z.wrfsfcf00.grib2
  - 102 bands, stable variable set
- **HRRRv2 start**: 2016-08-23, https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20160823/conus/hrrr.t00z.wrfsfcf00.grib2
  - Many missing hours during transition
- **HRRRv2 stable**: 2016-09-01, https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20160901/conus/hrrr.t00z.wrfsfcf00.grib2
  - Consistent hourly availability
- **Current start**: 2018-09-16, https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20180916/conus/hrrr.t00z.wrfsfcf00.grib2
  - 148 bands, full current variable set

### Notable Observations

1. **URL pattern unchanged**: File naming convention has been consistent since 2014-09-30, exact same pattern as current implementation

2. **Grid nearly identical**: Minor grid origin shift (53m in x, 0.08m in y) between HRRRv1/v2 and HRRRv3, but this is < 0.02 pixels and likely negligible for practical purposes

3. **GRIB name change for MSL pressure**: 
   - HRRRv1/v2: `PRMSL:mean sea level`
   - HRRRv3: `MSLMA:mean sea level`
   - Both represent mean sea level pressure, just different GRIB encoding

4. **Data quality issues documented in literature**:
   - 2014-09-30: Incomplete (first operational day)
   - 2014-2015: "Missing a considerable number of time steps" per web sources
   - January 2018: Major archive restoration after storage failure
   - Missing files repaired as of March 2022

5. **Tested data availability**: Random sampling of 15 dates from 2014-2018 shows good availability except during model version transitions

6. **Index files**: Consistently available with same NOAA colon-separated format

## Implementation Recommendations

### Option 1: Conservative Extension (Recommended)
**Start date: 2014-10-01**

**Rationale**:
- 4+ years of additional data (vs current 2018-09-16 start)
- Stable 102-band structure from day 1
- 16/20 variables available without code changes
- Avoids incomplete first operational day (2014-09-30)
- Documented online quality issues are less concerning after examining actual files

**Required code changes**:
1. Update `append_dim_start` in `src/reformatters/noaa/hrrr/analysis/template_config.py`:
   ```python
   append_dim_start: Timestamp = pd.Timestamp("2014-10-01T00:00")
   ```

2. **Handle PRMSL vs MSLMA** - Add version-aware GRIB element lookup:
   - For init_times < 2018-07-12: Look for `PRMSL` at "mean sea level"
   - For init_times >= 2018-07-12: Look for `MSLMA` at "mean sea level"

3. **Exclude 2 variables from historical backfill**:
   - `downward_long_wave_radiation_flux_surface` (not available before 2018-07-12)
   - `relative_humidity_2m` (not available before 2018-07-12)
   
   Implementation approach: Add `available_from` field to `NoaaHrrrDataVar` and filter variables based on time period being processed.

4. **Optional enhancement**: For historical period, could derive `relative_humidity_2m` from available `specific_humidity_2m` (SPFH) and temperature, but not required.

### Option 2: Maximum Extension  
**Start date: 2014-09-30 (not recommended)**

Same as Option 1 but includes first operational day with only 57 bands and 10/20 variables. The incomplete first day adds complexity for minimal benefit (1 day of data).

### Option 3: Wait for HRRRv3
**Start date: 2018-07-12 (current HRRRv3 start)**

- Extends current dataset by only 2 months
- All 20 variables available
- Minimal code changes needed
- Doesn't capitalize on 4 years of available historical data

## Recommended Implementation Approach

1. **Phase 1**: Implement version-aware variable handling
   - Add `available_from: Optional[Timestamp]` to variable definitions
   - Update region job to filter variables based on time period
   - Handle PRMSL/MSLMA name change in GRIB element lookup

2. **Phase 2**: Test with small backfill
   - Run backfill for 2014-10-01 to 2014-10-07 (1 week)
   - Verify all 16 variables read correctly
   - Check for missing data patterns

3. **Phase 3**: Production backfill
   - Backfill 2014-10-01 to 2018-09-15 (current start - 1 hour)
   - Monitor for missing files and handle gracefully

4. **Phase 4**: Consider adding excluded variables
   - Evaluate if DLWRF and RH are important enough to:
     - Start dataset at 2018-07-12 for those 2 variables only, OR
     - Create separate variables with later start dates

## Data Quality Considerations

Based on actual file inspection (not just documentation):
- ✅ File structure consistent and intact
- ✅ Grid structure consistent (minor sub-pixel shift acceptable)
- ✅ Variable availability stable from 2014-10-01
- ✅ Random date sampling shows good data availability
- ⚠️ Model version transitions have gaps (2016-08-23 to 2016-09-01)
- ⚠️ First operational day incomplete (2014-09-30)

**Verdict**: Historical data quality is sufficient for production use from 2014-10-01 onward.

## Conclusion

✅ **Extending the HRRR analysis dataset to 2014-10-01 is feasible and recommended.**

The historical archive is well-structured, consistently formatted, and contains 16 of our 20 current variables. The required code changes are straightforward (version-aware variable filtering and GRIB name handling). This extension would add over 4 years of valuable historical weather data to the dataset.

The trade-off of losing 2 variables (DLWRF, RH) in the historical period is acceptable given:
1. 16 core variables including temperature, winds, precipitation are available
2. Alternative humidity variables exist (SPFH, DPT)
3. Both excluded variables are available in the entire operational period (2018+)
4. Users interested in these variables still have 6+ years of data available

---

**Investigation Date**: 2026-02-02
**Files Examined**: 7 GRIB2 files + index files across 2014-2018
**Dates Tested**: 15+ random dates spanning 2014-2018
