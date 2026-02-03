# Source Data Exploration: ECMWF AIFS ENS

**Related to:** Issue #413

## Dataset: ECMWF AIFS ENS (AI-Integrated Forecasting System - Ensemble)

### Source Information
- **Summary of data organization**: Two files per lead time and init time: one control forecast file (cf) containing 103 fields, and one perturbed forecast file (pf) containing 103 fields × 50 ensemble members. All lead times for a given init time are stored in separate files.
- **File format**: GRIB2
- **Temporal coverage**: 2025-07-02 to present (operational data)
  - Test data available: 2025-06-20 to 2025-06-29 (aifs-ens_testdata)
  - Archive appears to be complete and continuous from operational start
- **Temporal frequency**:
  - Init times: 00z, 06z, 12z, 18z (4 times per day)
  - Lead times: 0h to 360h in 6-hour steps (61 forecast hours total, 15 days)
  - Output timestep: 6 hours
- **Latency**: Not yet determined (requires monitoring recent forecasts)
- **Access notes**:
  - S3 bucket: `s3://ecmwf-forecasts/` (public, no authentication required)
  - Supports anonymous access via s3fs
  - GRIB index files available for efficient partial downloads
  - Total data volume per init time: ~520 GB (cf files ~5 GB + pf files ~250 GB per init time)
- **Browse root**: No web interface found, but bucket is browsable via S3 tools
- **URL format**:
```
s3://ecmwf-forecasts/{YYYYMMDD}/{HH}z/aifs-ens/0p25/enfo/{YYYYMMDD}{HH}0000-{LLL}h-enfo-cf.grib2
s3://ecmwf-forecasts/{YYYYMMDD}/{HH}z/aifs-ens/0p25/enfo/{YYYYMMDD}{HH}0000-{LLL}h-enfo-cf.index
s3://ecmwf-forecasts/{YYYYMMDD}/{HH}z/aifs-ens/0p25/enfo/{YYYYMMDD}{HH}0000-{LLL}h-enfo-pf.grib2
s3://ecmwf-forecasts/{YYYYMMDD}/{HH}z/aifs-ens/0p25/enfo/{YYYYMMDD}{HH}0000-{LLL}h-enfo-pf.index
```
Where:
- `{YYYYMMDD}`: Date (e.g., 20260202)
- `{HH}`: Init hour (00, 06, 12, 18)
- `{LLL}`: Lead time in hours (0, 6, 12, ..., 354, 360)
- `cf`: control forecast (1 member)
- `pf`: perturbed forecast (50 ensemble members)

- **Example URLs**:
```
s3://ecmwf-forecasts/20260202/00z/aifs-ens/0p25/enfo/20260202000000-0h-enfo-cf.grib2
s3://ecmwf-forecasts/20260202/00z/aifs-ens/0p25/enfo/20260202000000-0h-enfo-cf.index
s3://ecmwf-forecasts/20260202/00z/aifs-ens/0p25/enfo/20260202000000-120h-enfo-pf.grib2
s3://ecmwf-forecasts/20260202/00z/aifs-ens/0p25/enfo/20260202000000-120h-enfo-pf.index
```

### GRIB Index
- **Index files available**: Yes
- **Index style**: ECMWF (JSON format, one JSON object per line)
- **Example line**:
```json
{"domain": "g", "date": "20260202", "time": "0000", "expver": "0001", "class": "ai", "type": "cf", "stream": "enfo", "step": "0", "levelist": "600", "levtype": "pl", "param": "q", "model": "aifs-ens", "_offset": 0, "_length": 705443}
```
- **Index fields**:
  - `domain`: "g" (global)
  - `date`: Init date (YYYYMMDD)
  - `time`: Init time (HHMM)
  - `expver`: Experiment version (0001)
  - `class`: "ai" (AIFS)
  - `type`: "cf" (control) or "pf" (perturbed)
  - `stream`: "enfo" (ensemble forecast)
  - `step`: Lead time in hours
  - `levtype`: "pl" (pressure level), "sfc" (surface), or specific heights
  - `levelist`: Pressure level in hPa (for levtype=pl)
  - `param`: ECMWF parameter short name
  - `number`: Ensemble member number (1-50, only in pf files)
  - `model`: "aifs-ens"
  - `_offset`: Byte offset in GRIB file
  - `_length`: Length in bytes

### Coordinate Reference System
- **Common name**: WGS84 geographic (regular lat/lon grid)
- **PROJ string or EPSG**:
  - Custom CRS from GRIB: Spherical Earth with radius 6371229m
  - PROJ: `+proj=longlat +R=6371229 +no_defs`
  - Essentially equivalent to EPSG:4326 for most purposes

### Dimensions & Dimension Coordinates

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| init_time | 2025-07-02 00:00 | present | 6 hours | Four init times per day: 00z, 06z, 12z, 18z |
| lead_time | 0h | 360h | 6 hours | 61 forecast hours total (0, 6, 12, ..., 354, 360) |
| latitude | -90.125° | 90.125° | 0.25° | 721 points (pixel centers) |
| longitude | -180.125° | 179.875° | 0.25° | 1440 points (pixel centers, wraps at dateline) |
| ensemble_member | 0 (control) | 50 | 1 | 51 total members: 1 control (cf file) + 50 perturbed (pf file, numbered 1-50) |
| pressure_level | 50 hPa | 1000 hPa | Variable | 13 levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa |

Grid details:
- Global coverage: 1440 × 721 grid
- 0.25° resolution (~25-28 km at mid-latitudes)
- Pixel-centered coordinates
- Longitude wraps at dateline (-180/+180)

### Data Variables

Core surface variables available (from pf files with ECMWF short names):

| Variable name | ECMWF param | Level | Units | Available from | Notes |
|---------------|-------------|-------|-------|----------------|-------|
| temperature_2m | 2t | 2 m | °C | 2025-07-02 | |
| dew_point_temperature_2m | 2d | 2 m | °C | 2025-07-02 | |
| wind_u_10m | 10u | 10 m | m/s | 2025-07-02 | |
| wind_v_10m | 10v | 10 m | m/s | 2025-07-02 | |
| wind_u_100m | 100u | 100 m | m/s | 2025-07-02 | |
| wind_v_100m | 100v | 100 m | m/s | 2025-07-02 | |
| total_precipitation | tp | surface | kg/(m²·s) | 2025-07-02 | **Rate**, not accumulated |
| snowfall | sf | surface | kg/(m²·s) | 2025-07-02 | **Rate**, not accumulated |
| downward_short_wave_radiation_flux_surface | ssrd | surface | W/m² | 2025-07-02 | Instantaneous flux |
| downward_long_wave_radiation_flux_surface | strd | surface | W/m² | 2025-07-02 | Instantaneous flux |
| pressure_surface | sp | surface | Pa | 2025-07-02 | |
| pressure_reduced_to_mean_sea_level | msl | MSL | Pa | 2025-07-02 | |
| total_cloud_cover | tcc | atmosphere | % | 2025-07-02 | |
| low_cloud_cover | lcc | atmosphere | % | 2025-07-02 | |
| medium_cloud_cover | mcc | atmosphere | % | 2025-07-02 | |
| high_cloud_cover | hcc | atmosphere | % | 2025-07-02 | |
| total_column_water | tcw | atmosphere | kg/m² | 2025-07-02 | |
| skin_temperature | skt | surface | °C | 2025-07-02 | |
| soil_temperature | sot | 1 m depth | °C | 2025-07-02 | Also available at surface (0 cm) |
| geopotential_surface | z | surface | m²/s² | 2025-07-02 | Surface geopotential/orography |
| runoff_water_equivalent | rowe | surface | kg m⁻² s⁻¹ | 2025-07-02 | Water runoff rate |

Pressure level variables (available at 13 levels: 50-1000 hPa):

| Variable name | ECMWF param | Levels | Units | Notes |
|---------------|-------------|--------|-------|-------|
| geopotential | z | 50-1000 hPa | m²/s² | 13 levels |
| temperature | t | 50-1000 hPa | °C | 13 levels |
| wind_u | u | 50-1000 hPa | m/s | 13 levels |
| wind_v | v | 50-1000 hPa | m/s | 13 levels |
| specific_humidity | q | 50-1000 hPa | kg/kg | 13 levels |
| vertical_velocity | w | 50-1000 hPa | Pa/s | 13 levels |

Static/orography fields:
- Land-sea mask (lsm)
- Standard deviation of orography (sdor)
- Slope of orography (slor)

**Field counts per file:**
- Control forecast (cf): 103 fields per lead time
- Perturbed forecast (pf): 103 fields × 50 members = 5150 fields per lead time

**Temporal availability changes**:
- AIFS ENS operational start: 2025-07-02
- No known variable changes since operational start
- Test data exists from 2025-06-20 (pre-operational)

### Sample Files Examined

- **Early archive**: 2025-07-02 00z, s3://ecmwf-forecasts/20250702/00z/aifs-ens/0p25/enfo/
  - First operational AIFS ENS forecast
- **Recent data**: 2026-02-02 00z, s3://ecmwf-forecasts/20260202/00z/aifs-ens/0p25/enfo/
  - Latest available (as of exploration date)
- **Test data**: 2025-06-20, s3://ecmwf-forecasts/aifs-ens_testdata/20250620/
  - Pre-operational test forecasts

### Notable Observations

1. **File organization**: Unlike traditional ensemble forecasts where all members might be in one file, AIFS ENS splits control (cf) and perturbed (pf) forecasts into separate files. The pf file contains all 50 ensemble members interleaved.

2. **Precipitation as rates**: Precipitation and snowfall are provided as instantaneous rates (kg/(m²·s)), not accumulated amounts. This differs from many operational forecast models (like IFS, GFS) that provide accumulated precipitation. To get accumulation, integration over time is required.

3. **Radiation as fluxes**: Solar and thermal radiation are instantaneous fluxes (W/m²), not accumulated energy.

4. **Temperature units**: All temperatures in GRIB files are in Celsius (not Kelvin), which is unusual for ECMWF GRIB files. Will need unit conversion to Kelvin for CF compliance.

5. **Consistent structure**: File structure, grid dimensions, and variable availability are identical between the earliest (July 2025) and most recent (February 2026) files examined. No structural changes detected.

6. **Ensemble member numbering**: Control forecast is separate (cf file), perturbed members numbered 1-50 (in pf file). Total of 51 ensemble members.

7. **Index file efficiency**: JSON index files enable selective variable/level downloads, which is critical given the large pf file sizes (~4 GB per file).

8. **Resolution**: 0.25° is relatively coarse compared to some regional models but standard for global ensemble forecasts. Similar to IFS ENS resolution.

9. **Vertical coverage**: 13 pressure levels provide good coverage from upper troposphere (50 hPa) to surface (1000 hPa). Includes key levels for weather (850, 700, 500 hPa).

10. **Missing standard variables**:
    - No relative humidity at 2m directly (only dew point and specific humidity available)
    - No convective vs. stratiform precipitation separation
    - No visibility, ceiling, or flight level variables

11. **Cloud cover structure**: Total, low, medium, and high cloud cover all available. Medium (800 hPa) and high (450 hPa) cloud cover appear to be at single levels, not layer-averaged.

12. **Bucket access**: S3 bucket allows anonymous access, which simplifies data retrieval (no authentication required).

13. **AIFS model**: This is ECMWF's AI-based forecasting system, a data-driven model trained on historical IFS analyses and forecasts. It's an alternative to the traditional physics-based IFS model.

14. **Operational timeline**: Very recent operational deployment (July 2025), so archive is relatively short (~7 months as of Feb 2026). Archive will grow continuously.

15. **Data completeness**: Spot checks show consistent 4x daily init times and complete lead time coverage. No obvious gaps observed in operational period.

## Implementation Considerations

### Dataset Type
This should be implemented as a **forecast dataset** with dimensions:
- init_time (append dimension)
- lead_time
- ensemble_member
- latitude
- longitude
- [pressure_level for 3D variables]

### Source Files
- Two file types per (init_time, lead_time) combination
- Will need to read both cf and pf files to get all 51 ensemble members
- Can use index files to selectively download only needed variables/levels

### Key Challenges

1. **Large file sizes**: pf files are ~4 GB each. Need efficient chunking strategy.
2. **Variable extraction**: Must parse JSON index files to locate specific variables/levels/members
3. **Unit conversions**: Temperature C→K, possibly others
4. **Rate to accumulation**: If users want accumulated precipitation, need to integrate rates over lead time
5. **Ensemble dimension**: Need to combine control (from cf) and perturbed members (from pf) into single ensemble dimension
6. **ECMWF parameter mapping**: Need to map ECMWF short names to CF-compliant variable names

### Suggested Priority Variables
For initial implementation, focus on these commonly-used surface variables:
- 2m temperature (2t)
- 2m dew point (2d)
- 10m winds (10u, 10v)
- Mean sea level pressure (msl)
- Total precipitation (tp)
- Total cloud cover (tcc)
- Downward solar radiation (ssrd)
- 100m winds (100u, 100v) - increasingly important for wind energy

Could expand later to include:
- Pressure level variables (z, t, u, v, q at key levels like 850, 500, 250 hPa)
- Additional surface variables (skin temp, soil temp, etc.)

### Alternative Sources
The same S3 bucket also contains:
- `aifs-single`: AIFS single/deterministic forecast (non-ensemble)
- `ifs`: Traditional IFS model forecasts

For the longest archive, we should also consider:
- ECMWF's MARS archive (tape storage, requires API access)
- Historical IFS ENS data for pre-2025 ensemble forecasts

However, the current source (S3 bucket) is excellent for:
- Reliability (ECMWF-maintained)
- Access speed (object storage)
- No authentication required

### Recommended Next Steps
1. Implement template config for AIFS ENS forecast dataset
2. Start with control forecast (cf) only for initial implementation (simpler, smaller files)
3. Add ensemble members (pf files) in a second phase
4. Begin with surface variables only, add pressure levels later
5. Monitor a few forecast cycles to determine actual latency
6. Consider whether to integrate precipitation rates or keep as rates
