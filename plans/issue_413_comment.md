# ECMWF AIFS ENS Data Source Exploration Report

I've completed a comprehensive exploration of the ECMWF AIFS ENS data available at `s3://ecmwf-forecasts/`. The detailed report is available in the PR from branch `claude/explore-ecmwf-aifs-data-gaSEY`.

## Key Findings

### Data Availability
- **Operational since**: 2025-07-02
- **Temporal frequency**: 4 init times per day (00z, 06z, 12z, 18z)
- **Lead times**: 0-360 hours in 6-hour steps (15-day forecasts)
- **Ensemble size**: 51 members (1 control + 50 perturbed)

### Data Structure
- **Format**: GRIB2 with JSON index files
- **Resolution**: 0.25° global (1440×721 grid)
- **File organization**: Separate control forecast (cf) and perturbed forecast (pf) files per lead time
  - cf files: ~80 MB (103 fields for control member)
  - pf files: ~4 GB (103 fields × 50 ensemble members)
- **Access**: Anonymous S3 access, no authentication required

### Available Variables
**Surface variables** (23 total including):
- 2m temperature, dew point
- 10m and 100m winds
- Precipitation (as **rate**, not accumulated)
- Solar and thermal radiation (instantaneous fluxes)
- Cloud cover (total, low, medium, high)
- Pressure (surface and MSL)
- Soil temperature, skin temperature
- And more...

**Pressure level variables** (13 levels from 50-1000 hPa):
- Geopotential, temperature, winds (u/v)
- Specific humidity, vertical velocity

### Notable Observations
1. **Precipitation as rates**: Unlike most forecast models, precipitation and snowfall are provided as instantaneous rates (kg/(m²·s)) rather than accumulated amounts
2. **Temperature units**: Celsius (unusual for GRIB), will need conversion to Kelvin for CF compliance
3. **Index files**: JSON format enabling efficient selective downloads of specific variables/levels/members
4. **Consistent structure**: No changes detected between July 2025 and February 2026 samples
5. **AI-based model**: ECMWF's data-driven forecasting system, trained on historical IFS data

### Implementation Recommendations

**Suggested approach**:
1. Implement as forecast dataset with dimensions: init_time, lead_time, ensemble_member, latitude, longitude
2. Phase 1: Control forecast only (simpler, smaller files)
3. Phase 2: Add ensemble members from pf files
4. Start with key surface variables, add pressure levels later

**Priority variables** for initial implementation:
- 2m temperature and dew point
- 10m and 100m winds
- Mean sea level pressure
- Total precipitation and cloud cover
- Downward solar radiation

**Key challenges**:
- Large pf file sizes (~4 GB each)
- Need to parse JSON index files for variable extraction
- Unit conversions (C→K)
- Combining control and perturbed members into unified ensemble dimension
- Decision on whether to integrate precipitation rates into accumulations

Full detailed report available in `plans/ecmwf_aifs_ens_exploration.md` in the branch.
