# Validation run — `ecmwf-aifs-ens-forecast` `v0.1.0`

Started: 2026-05-07T12:53:18 local

## Summary

### For further review

- **Sparse missing timestamps across all 17 variables, concentrated in the early backfill window (Jul–Sep 2025).** 116 missing timestamps across 34 (variable, point) combinations, all isolated dots in [`combined_nulls.png`](combined_nulls.png) (and the per-variable `nulls_*.png`). Reporter noted S3 `503`s during backfill, which matches the temporal clustering of the gaps and the lack of any structural pattern (no fixed lead time, no fixed init hour, distribution differs per variable). Retry strings are pre-rendered in [`missing_timestamps.txt`](missing_timestamps.txt). Recommend a targeted `--filter-contains` retry rather than a full re-backfill.
- **`downward_short_wave_radiation_flux_surface` shows occasional small spikes (≤18 W m⁻²) at Point 2 (lat=−85.00) in late August / early September while GEFS stays at 0** ([`temporal_downward_short_wave_radiation_flux_surface.png`](temporal_downward_short_wave_radiation_flux_surface.png)). Possible — late Antarctic winter / early spring with low-angle / twilight diffuse, and AIFS may be more sensitive than GEFS — but worth a sanity check that the variable isn't picking up a small offset. Magnitudes are tiny relative to mid-latitude values, so even if it's a slight bias it's not data-corruption-level.
- **`pressure_surface` validation min (4.93×10⁴ Pa) is ~1700 Pa below the GEFS reference min (5.10×10⁴ Pa)** at the spatial snapshot ([`spatial_pressure_surface.png`](spatial_pressure_surface.png)). Histograms otherwise overlap and the global map is coherent; this is likely model orography/elevation differences (AIFS keeps high-elevation pixels GEFS smooths), not a unit or coordinate bug. Worth confirming by spot-checking a Himalayan or Andean grid cell.
- **`temperature_2m` at Point 2 (Antarctic) shows AIFS ~5–10°C colder than GEFS for several days** ([`temporal_temperature_2m.png`](temporal_temperature_2m.png)). No phase shift, no unit issue — likely a known model-vs-model bias at extreme southern latitudes, but flagging because magnitude is the largest of any variable.

### What looks good

- **All spatial maps are correctly oriented** (north up, east right, longitudes −180 → +180), continents land on the expected coordinates, and AIFS maps line up cleanly with the GEFS reference where the reference is available ([`combined_spatial.png`](combined_spatial.png), individual `spatial_*.png`).
- **Histograms overlap the GEFS reference** for every variable available in both datasets — no horizontal offset (no unit bug) and no width mismatch (no quantization or smoothing bug). Pressure, MSLP, temperature_2m, winds (10 m and 100 m, U and V), short-wave and long-wave radiation, precipitation, and cloud cover all match.
- **No sentinel values bleeding through** (no 9999 / −9999 / 1e20 spikes in any spatial colorbar).
- **No visible quantization banding** in spatial maps or time series — `keep_mantissa_bits` looks adequately sized.
- **Diurnal phase matches** for short-wave radiation at Point 1 (Canadian Arctic) — sun-up / sun-down hours align with GEFS, ruling out a time-coordinate (UTC vs local, off-by-one lead) bug.
- **Vertical-level variables are physically consistent**: geopotential_height_500hpa > 850hpa > 925hpa (✓), temperature_925hpa > temperature_850hpa near the surface (✓).
- **Stats and ranges are physically plausible** across all variables (temperatures −71 → +46 °C, MSLP 9.46×10⁴ → 1.04×10⁵ Pa, precipitation rate ~10⁻⁵ kg m⁻² s⁻¹, wind speeds within ±30 m s⁻¹).
- **The structural first-lead NaN for accumulation/average variables is being excluded from null counts** as expected — `precipitation_surface`, `downward_short_wave_radiation_flux_surface`, and `downward_long_wave_radiation_flux_surface` only show the small handful of unexpected gaps.

## Datasets

| Role | Name | ID / Version | URL |
|---|---|---|---|
| Validation | ECMWF AIFS ENS forecast | `ecmwf-aifs-ens-forecast` `v0.1.0` | `s3://dynamical-ecmwf-aifs-ens/ecmwf-aifs-ens-forecast/v0.1.0.icechunk` |
| Reference  | NOAA GEFS analysis | `noaa-gefs-analysis` `0.1.2` | `https://data.dynamical.org/noaa/gefs/analysis/latest.zarr` |

## Run parameters

- Validation dataset type: **forecast**
- Validation time range: init_time 2025-07-02T00:00 → 2026-04-29T18:00
- Reference time range:  time 2000-01-01T00:00 → 2026-05-07T06:00
- Time scope: full dataset
- Ensemble member: 19
- Point 1: lat=64.2500, lon=-95.5000
- Point 2: lat=-85.0000, lon=151.2500
- Spatial comparison time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- Timeseries period: Forecast init_time: 2025-08-29T00:00

## Combined plots

- nulls: [`combined_nulls.png`](combined_nulls.png)
- spatial: [`combined_spatial.png`](combined_spatial.png)
- temporal: [`combined_temporal.png`](combined_temporal.png)

## Missing timestamps

**116** missing timestamps across **34** (variable, point) combinations.

Full list: [`missing_timestamps.txt`](missing_timestamps.txt)

| Variable | Point | Count |
|---|---|---|
| `dew_point_temperature_2m` | P1 | 2 |
| `dew_point_temperature_2m` | P2 | 2 |
| `downward_long_wave_radiation_flux_surface` | P1 | 2 |
| `downward_long_wave_radiation_flux_surface` | P2 | 2 |
| `downward_short_wave_radiation_flux_surface` | P1 | 3 |
| `downward_short_wave_radiation_flux_surface` | P2 | 3 |
| `geopotential_height_500hpa` | P1 | 2 |
| `geopotential_height_500hpa` | P2 | 2 |
| `geopotential_height_850hpa` | P1 | 6 |
| `geopotential_height_850hpa` | P2 | 6 |
| `geopotential_height_925hpa` | P1 | 6 |
| `geopotential_height_925hpa` | P2 | 6 |
| `precipitation_surface` | P1 | 2 |
| `precipitation_surface` | P2 | 2 |
| `pressure_reduced_to_mean_sea_level` | P1 | 4 |
| `pressure_reduced_to_mean_sea_level` | P2 | 4 |
| `pressure_surface` | P1 | 5 |
| `pressure_surface` | P2 | 5 |
| `temperature_2m` | P1 | 5 |
| `temperature_2m` | P2 | 5 |
| `temperature_850hpa` | P1 | 1 |
| `temperature_850hpa` | P2 | 1 |
| `temperature_925hpa` | P1 | 1 |
| `temperature_925hpa` | P2 | 1 |
| `total_cloud_cover_atmosphere` | P1 | 3 |
| `total_cloud_cover_atmosphere` | P2 | 3 |
| `wind_u_100m` | P1 | 5 |
| `wind_u_100m` | P2 | 5 |
| `wind_u_10m` | P1 | 3 |
| `wind_u_10m` | P2 | 3 |
| `wind_v_100m` | P1 | 5 |
| `wind_v_100m` | P2 | 5 |
| `wind_v_10m` | P1 | 3 |
| `wind_v_10m` | P2 | 3 |

## Variables overview

| Variable | Units | Long name | Nulls @ P1 | Nulls @ P2 | Plots |
|---|---|---|---|---|---|
| `dew_point_temperature_2m` | degree_Celsius | 2 metre dewpoint temperature | 2/73688 | 2/73688 | [nulls](nulls_dew_point_temperature_2m.png) · [spatial](spatial_dew_point_temperature_2m.png) · [temporal](temporal_dew_point_temperature_2m.png) |
| `downward_long_wave_radiation_flux_surface` | W m-2 | Surface downward long-wave radiation flux | 5/72480 | 5/72480 | [nulls](nulls_downward_long_wave_radiation_flux_surface.png) · [spatial](spatial_downward_long_wave_radiation_flux_surface.png) · [temporal](temporal_downward_long_wave_radiation_flux_surface.png) |
| `downward_short_wave_radiation_flux_surface` | W m-2 | Surface downward short-wave radiation flux | 6/72480 | 6/72480 | [nulls](nulls_downward_short_wave_radiation_flux_surface.png) · [spatial](spatial_downward_short_wave_radiation_flux_surface.png) · [temporal](temporal_downward_short_wave_radiation_flux_surface.png) |
| `geopotential_height_500hpa` | m | Geopotential height | 2/73688 | 2/73688 | [nulls](nulls_geopotential_height_500hpa.png) · [spatial](spatial_geopotential_height_500hpa.png) · [temporal](temporal_geopotential_height_500hpa.png) |
| `geopotential_height_850hpa` | m | Geopotential height | 6/73688 | 6/73688 | [nulls](nulls_geopotential_height_850hpa.png) · [spatial](spatial_geopotential_height_850hpa.png) · [temporal](temporal_geopotential_height_850hpa.png) |
| `geopotential_height_925hpa` | m | Geopotential height | 6/73688 | 6/73688 | [nulls](nulls_geopotential_height_925hpa.png) · [spatial](spatial_geopotential_height_925hpa.png) · [temporal](temporal_geopotential_height_925hpa.png) |
| `precipitation_surface` | kg m-2 s-1 | Precipitation rate | 5/72480 | 5/72480 | [nulls](nulls_precipitation_surface.png) · [spatial](spatial_precipitation_surface.png) · [temporal](temporal_precipitation_surface.png) |
| `pressure_reduced_to_mean_sea_level` | Pa | Pressure reduced to MSL | 4/73688 | 4/73688 | [nulls](nulls_pressure_reduced_to_mean_sea_level.png) · [spatial](spatial_pressure_reduced_to_mean_sea_level.png) · [temporal](temporal_pressure_reduced_to_mean_sea_level.png) |
| `pressure_surface` | Pa | Surface pressure | 6/73688 | 6/73688 | [nulls](nulls_pressure_surface.png) · [spatial](spatial_pressure_surface.png) · [temporal](temporal_pressure_surface.png) |
| `temperature_2m` | degree_Celsius | 2 metre temperature | 6/73688 | 6/73688 | [nulls](nulls_temperature_2m.png) · [spatial](spatial_temperature_2m.png) · [temporal](temporal_temperature_2m.png) |
| `temperature_850hpa` | degree_Celsius | Temperature | 1/73688 | 1/73688 | [nulls](nulls_temperature_850hpa.png) · [spatial](spatial_temperature_850hpa.png) · [temporal](temporal_temperature_850hpa.png) |
| `temperature_925hpa` | degree_Celsius | Temperature | 1/73688 | 1/73688 | [nulls](nulls_temperature_925hpa.png) · [spatial](spatial_temperature_925hpa.png) · [temporal](temporal_temperature_925hpa.png) |
| `total_cloud_cover_atmosphere` | percent | Total cloud cover | 3/73688 | 3/73688 | [nulls](nulls_total_cloud_cover_atmosphere.png) · [spatial](spatial_total_cloud_cover_atmosphere.png) · [temporal](temporal_total_cloud_cover_atmosphere.png) |
| `wind_u_100m` | m s-1 | 100 metre U wind component | 5/73688 | 5/73688 | [nulls](nulls_wind_u_100m.png) · [spatial](spatial_wind_u_100m.png) · [temporal](temporal_wind_u_100m.png) |
| `wind_u_10m` | m s-1 | 10 metre U wind component | 3/73688 | 3/73688 | [nulls](nulls_wind_u_10m.png) · [spatial](spatial_wind_u_10m.png) · [temporal](temporal_wind_u_10m.png) |
| `wind_v_100m` | m s-1 | 100 metre V wind component | 5/73688 | 5/73688 | [nulls](nulls_wind_v_100m.png) · [spatial](spatial_wind_v_100m.png) · [temporal](temporal_wind_v_100m.png) |
| `wind_v_10m` | m s-1 | 10 metre V wind component | 3/73688 | 3/73688 | [nulls](nulls_wind_v_10m.png) · [spatial](spatial_wind_v_10m.png) · [temporal](temporal_wind_v_10m.png) |

## Per-variable details

### `dew_point_temperature_2m`

**Metadata**

- units: `degree_Celsius`
- long_name: 2 metre dewpoint temperature
- short_name: 2d
- standard_name: dew_point_temperature
- step_type: instant

**Spatial comparison**

- plot: `spatial_dew_point_temperature_2m.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=-71, max=31.5, mean=4.159
- reference:  variable not available in reference dataset

**Temporal comparison**

- plot: `temporal_dew_point_temperature_2m.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=-1.273, max=17, mean=4.877
- validation @ P2 (lat=-85.00, lon=151.25): min=-61.25, max=-39.25, mean=-52.14
- reference:  variable not available in reference dataset

**Nulls**

- P1 nulls: 2/73688 — 2 missing: 2025-08-05T12:00:00, 2025-10-10T18:00:00
- P2 nulls: 2/73688 — 2 missing: 2025-08-05T12:00:00, 2025-10-10T18:00:00

### `downward_long_wave_radiation_flux_surface`

**Metadata**

- units: `W m-2`
- long_name: Surface downward long-wave radiation flux
- short_name: sdlwrf
- standard_name: surface_downwelling_longwave_flux_in_air
- step_type: avg

**Spatial comparison**

- plot: `spatial_downward_long_wave_radiation_flux_surface.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=58.25, max=478, mean=322.2
- reference:  min=61.5, max=508, mean=316.3

**Temporal comparison**

- plot: `temporal_downward_long_wave_radiation_flux_surface.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=247, max=364, mean=315.7
- validation @ P2 (lat=-85.00, lon=151.25): min=67, max=140, mean=91.93
- reference  @ P1: min=235, max=366, mean=297.2
- reference  @ P2: min=76, max=174, mean=110

**Nulls**

- P1 nulls: 5/72480 — 2 missing: 2025-08-05T06:00:00, 2025-09-11T06:00:00
- P2 nulls: 5/72480 — 2 missing: 2025-08-05T06:00:00, 2025-09-11T06:00:00

### `downward_short_wave_radiation_flux_surface`

**Metadata**

- units: `W m-2`
- long_name: Surface downward short-wave radiation flux
- short_name: sdswrf
- standard_name: surface_downwelling_shortwave_flux_in_air
- step_type: avg

**Spatial comparison**

- plot: `spatial_downward_short_wave_radiation_flux_surface.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=0, max=912, mean=152.4
- reference:  min=0, max=920, mean=162.4

**Temporal comparison**

- plot: `temporal_downward_short_wave_radiation_flux_surface.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=0, max=340, mean=110.2
- validation @ P2 (lat=-85.00, lon=151.25): min=0, max=17.75, mean=2.619
- reference  @ P1: min=0, max=520, mean=112.6
- reference  @ P2: min=0, max=10, mean=0.1405

**Nulls**

- P1 nulls: 6/72480 — 3 missing: 2025-08-06T00:00:00, 2025-08-08T12:00:00, 2025-08-18T12:00:00
- P2 nulls: 6/72480 — 3 missing: 2025-08-06T00:00:00, 2025-08-08T12:00:00, 2025-08-18T12:00:00

### `geopotential_height_500hpa`

**Metadata**

- units: `m`
- long_name: Geopotential height
- short_name: gh
- standard_name: geopotential_height
- step_type: instant

**Spatial comparison**

- plot: `spatial_geopotential_height_500hpa.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=4674, max=5994, mean=5582
- reference:  min=0, max=5990, mean=5575

**Temporal comparison**

- plot: `temporal_geopotential_height_500hpa.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=5336, max=5678, mean=5510
- validation @ P2 (lat=-85.00, lon=151.25): min=4812, max=5076, mean=4947
- reference  @ P1: min=5360, max=5698, mean=5549
- reference  @ P2: min=4830, max=5074, mean=4937

**Nulls**

- P1 nulls: 2/73688 — 2 missing: 2025-08-05T12:00:00, 2025-10-10T18:00:00
- P2 nulls: 2/73688 — 2 missing: 2025-08-05T12:00:00, 2025-10-10T18:00:00

### `geopotential_height_850hpa`

**Metadata**

- units: `m`
- long_name: Geopotential height
- short_name: gh
- standard_name: geopotential_height
- step_type: instant

**Spatial comparison**

- plot: `spatial_geopotential_height_850hpa.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=827.2, max=1666, mean=1418
- reference:  variable not available in reference dataset

**Temporal comparison**

- plot: `temporal_geopotential_height_850hpa.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=1280, max=1498, mean=1403
- validation @ P2 (lat=-85.00, lon=151.25): min=1092, max=1274, mean=1188
- reference:  variable not available in reference dataset

**Nulls**

- P1 nulls: 6/73688 — 6 missing: 2025-07-20T00:00:00, 2025-07-27T06:00:00, 2025-08-06T00:00:00, 2025-08-18T00:00:00, 2025-09-23T06:00:00, 2026-03-22T12:00:00
- P2 nulls: 6/73688 — 6 missing: 2025-07-20T00:00:00, 2025-07-27T06:00:00, 2025-08-06T00:00:00, 2025-08-18T00:00:00, 2025-09-23T06:00:00, 2026-03-22T12:00:00

### `geopotential_height_925hpa`

**Metadata**

- units: `m`
- long_name: Geopotential height
- short_name: gh
- standard_name: geopotential_height
- step_type: instant

**Spatial comparison**

- plot: `spatial_geopotential_height_925hpa.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=171.4, max=947, mean=722
- reference:  variable not available in reference dataset

**Temporal comparison**

- plot: `temporal_geopotential_height_925hpa.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=588.8, max=819.5, mean=718.3
- validation @ P2 (lat=-85.00, lon=151.25): min=464.9, max=635.5, mean=554.8
- reference:  variable not available in reference dataset

**Nulls**

- P1 nulls: 6/73688 — 6 missing: 2025-07-20T00:00:00, 2025-07-27T06:00:00, 2025-08-06T00:00:00, 2025-08-18T00:00:00, 2025-09-23T06:00:00, 2026-03-22T12:00:00
- P2 nulls: 6/73688 — 6 missing: 2025-07-20T00:00:00, 2025-07-27T06:00:00, 2025-08-06T00:00:00, 2025-08-18T00:00:00, 2025-09-23T06:00:00, 2026-03-22T12:00:00

### `precipitation_surface`

**Metadata**

- units: `kg m-2 s-1`
- long_name: Precipitation rate
- short_name: prate
- standard_name: precipitation_flux
- step_type: avg

**Spatial comparison**

- plot: `spatial_precipitation_surface.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=0, max=0.002899, mean=2.664e-05
- reference:  min=0, max=0.004639, mean=2.771e-05

**Temporal comparison**

- plot: `temporal_precipitation_surface.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=0, max=0.0002289, mean=2.015e-05
- validation @ P2 (lat=-85.00, lon=151.25): min=0, max=1.445e-06, mean=2.409e-08
- reference  @ P1: min=0, max=0.0003757, mean=2.279e-05
- reference  @ P2: min=0, max=2.778e-05, mean=1.783e-06

**Nulls**

- P1 nulls: 5/72480 — 2 missing: 2025-08-05T06:00:00, 2025-09-11T06:00:00
- P2 nulls: 5/72480 — 2 missing: 2025-08-05T06:00:00, 2025-09-11T06:00:00

### `pressure_reduced_to_mean_sea_level`

**Metadata**

- units: `Pa`
- long_name: Pressure reduced to MSL
- short_name: prmsl
- standard_name: air_pressure_at_mean_sea_level
- step_type: instant

**Spatial comparison**

- plot: `spatial_pressure_reduced_to_mean_sea_level.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=9.456e+04, max=1.036e+05, mean=1.009e+05
- reference:  min=9.459e+04, max=1.051e+05, mean=1.01e+05

**Temporal comparison**

- plot: `temporal_pressure_reduced_to_mean_sea_level.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=9.942e+04, max=1.024e+05, mean=1.01e+05
- validation @ P2 (lat=-85.00, lon=151.25): min=9.885e+04, max=1.015e+05, mean=1.003e+05
- reference  @ P1: min=9.984e+04, max=1.028e+05, mean=1.013e+05
- reference  @ P2: min=9.792e+04, max=1.032e+05, mean=1.011e+05

**Nulls**

- P1 nulls: 4/73688 — 4 missing: 2025-08-06T00:00:00, 2025-08-08T12:00:00, 2025-08-18T12:00:00, 2025-09-04T06:00:00
- P2 nulls: 4/73688 — 4 missing: 2025-08-06T00:00:00, 2025-08-08T12:00:00, 2025-08-18T12:00:00, 2025-09-04T06:00:00

### `pressure_surface`

**Metadata**

- units: `Pa`
- long_name: Surface pressure
- short_name: sp
- standard_name: surface_air_pressure
- step_type: instant

**Spatial comparison**

- plot: `spatial_pressure_surface.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=4.928e+04, max=1.039e+05, mean=9.665e+04
- reference:  min=5.101e+04, max=1.036e+05, mean=9.663e+04

**Temporal comparison**

- plot: `temporal_pressure_surface.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=9.933e+04, max=1.023e+05, mean=1.009e+05
- validation @ P2 (lat=-85.00, lon=151.25): min=6.845e+04, max=7.03e+04, mean=6.947e+04
- reference  @ P1: min=9.926e+04, max=1.023e+05, mean=1.008e+05
- reference  @ P2: min=6.797e+04, max=7.014e+04, mean=6.926e+04

**Nulls**

- P1 nulls: 6/73688 — 5 missing: 2025-08-06T06:00:00, 2025-08-31T00:00:00, 2025-09-09T00:00:00, 2026-03-22T18:00:00, 2026-03-27T06:00:00
- P2 nulls: 6/73688 — 5 missing: 2025-08-06T06:00:00, 2025-08-31T00:00:00, 2025-09-09T00:00:00, 2026-03-22T18:00:00, 2026-03-27T06:00:00

### `temperature_2m`

**Metadata**

- units: `degree_Celsius`
- long_name: 2 metre temperature
- short_name: 2t
- standard_name: air_temperature
- step_type: instant

**Spatial comparison**

- plot: `spatial_temperature_2m.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=-67, max=45.5, mean=8.784
- reference:  min=-61.5, max=44.25, mean=8.843

**Temporal comparison**

- plot: `temporal_temperature_2m.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=3.688, max=20.12, mean=8.631
- validation @ P2 (lat=-85.00, lon=151.25): min=-56, max=-34, mean=-47.68
- reference  @ P1: min=1, max=18.5, mean=7.128
- reference  @ P2: min=-50.25, max=-30.12, mean=-41.2

**Nulls**

- P1 nulls: 6/73688 — 5 missing: 2025-08-06T06:00:00, 2025-08-31T00:00:00, 2025-09-09T00:00:00, 2026-03-22T18:00:00, 2026-03-27T06:00:00
- P2 nulls: 6/73688 — 5 missing: 2025-08-06T06:00:00, 2025-08-31T00:00:00, 2025-09-09T00:00:00, 2026-03-22T18:00:00, 2026-03-27T06:00:00

### `temperature_850hpa`

**Metadata**

- units: `degree_Celsius`
- long_name: Temperature
- short_name: t
- standard_name: air_temperature
- step_type: instant

**Spatial comparison**

- plot: `spatial_temperature_850hpa.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=-40.5, max=35.75, mean=4.827
- reference:  variable not available in reference dataset

**Temporal comparison**

- plot: `temporal_temperature_850hpa.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=-8.375, max=16.38, mean=1.352
- validation @ P2 (lat=-85.00, lon=151.25): min=-42.5, max=-20.38, mean=-30.37
- reference:  variable not available in reference dataset

**Nulls**

- P1 nulls: 1/73688 — 1 missing: 2025-08-08T18:00:00
- P2 nulls: 1/73688 — 1 missing: 2025-08-08T18:00:00

### `temperature_925hpa`

**Metadata**

- units: `degree_Celsius`
- long_name: Temperature
- short_name: t
- standard_name: air_temperature
- step_type: instant

**Spatial comparison**

- plot: `spatial_temperature_925hpa.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=-36.75, max=42.25, mean=7.94
- reference:  variable not available in reference dataset

**Temporal comparison**

- plot: `temporal_temperature_925hpa.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=-3, max=20.38, mean=4.219
- validation @ P2 (lat=-85.00, lon=151.25): min=-38.5, max=-16.38, mean=-26.44
- reference:  variable not available in reference dataset

**Nulls**

- P1 nulls: 1/73688 — 1 missing: 2025-08-08T18:00:00
- P2 nulls: 1/73688 — 1 missing: 2025-08-08T18:00:00

### `total_cloud_cover_atmosphere`

**Metadata**

- units: `percent`
- long_name: Total cloud cover
- short_name: tcc
- standard_name: cloud_area_fraction
- step_type: instant

**Spatial comparison**

- plot: `spatial_total_cloud_cover_atmosphere.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=0, max=100, mean=67.11
- reference:  min=0, max=100, mean=63.3

**Temporal comparison**

- plot: `temporal_total_cloud_cover_atmosphere.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=3, max=100, mean=87.22
- validation @ P2 (lat=-85.00, lon=151.25): min=0, max=100, mean=47.34
- reference  @ P1: min=0, max=100, mean=67.27
- reference  @ P2: min=0, max=100, mean=66.83

**Nulls**

- P1 nulls: 3/73688 — 3 missing: 2025-08-06T00:00:00, 2025-08-26T06:00:00, 2026-03-23T00:00:00
- P2 nulls: 3/73688 — 3 missing: 2025-08-06T00:00:00, 2025-08-26T06:00:00, 2026-03-23T00:00:00

### `wind_u_100m`

**Metadata**

- units: `m s-1`
- long_name: 100 metre U wind component
- short_name: 100u
- standard_name: eastward_wind
- step_type: instant

**Spatial comparison**

- plot: `spatial_wind_u_100m.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=-28.5, max=30.25, mean=0.1609
- reference:  min=-28, max=31, mean=0.105

**Temporal comparison**

- plot: `temporal_wind_u_100m.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=-9.5, max=20.5, mean=4.022
- validation @ P2 (lat=-85.00, lon=151.25): min=-10.75, max=11.25, mean=-0.5256
- reference  @ P1: min=-9.375, max=15.5, mean=1.649
- reference  @ P2: min=-8.75, max=4.688, mean=-1.91

**Nulls**

- P1 nulls: 5/73688 — 5 missing: 2025-07-26T18:00:00, 2025-08-06T18:00:00, 2025-08-13T12:00:00, 2025-09-26T00:00:00, 2025-11-26T12:00:00
- P2 nulls: 5/73688 — 5 missing: 2025-07-26T18:00:00, 2025-08-06T18:00:00, 2025-08-13T12:00:00, 2025-09-26T00:00:00, 2025-11-26T12:00:00

### `wind_u_10m`

**Metadata**

- units: `m s-1`
- long_name: 10 metre U wind component
- short_name: 10u
- standard_name: eastward_wind
- step_type: instant

**Spatial comparison**

- plot: `spatial_wind_u_10m.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=-22, max=23.25, mean=0.1177
- reference:  min=-24.5, max=23.5, mean=0.09414

**Temporal comparison**

- plot: `temporal_wind_u_10m.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=-7.875, max=16.5, mean=3.049
- validation @ P2 (lat=-85.00, lon=151.25): min=-6.312, max=7.938, mean=0.9338
- reference  @ P1: min=-8.25, max=11.38, mean=1.106
- reference  @ P2: min=-5.125, max=4.312, mean=-0.01894

**Nulls**

- P1 nulls: 3/73688 — 3 missing: 2025-07-26T00:00:00, 2025-08-08T18:00:00, 2025-08-29T18:00:00
- P2 nulls: 3/73688 — 3 missing: 2025-07-26T00:00:00, 2025-08-08T18:00:00, 2025-08-29T18:00:00

### `wind_v_100m`

**Metadata**

- units: `m s-1`
- long_name: 100 metre V wind component
- short_name: 100v
- standard_name: northward_wind
- step_type: instant

**Spatial comparison**

- plot: `spatial_wind_v_100m.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=-25, max=23.5, mean=0.5348
- reference:  min=-29, max=26.25, mean=0.5484

**Temporal comparison**

- plot: `temporal_wind_v_100m.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=-19, max=11.62, mean=-1.789
- validation @ P2 (lat=-85.00, lon=151.25): min=0.6484, max=19.5, mean=13.43
- reference  @ P1: min=-14, max=13.75, mean=-1.946
- reference  @ P2: min=-1.266, max=17, mean=9.609

**Nulls**

- P1 nulls: 5/73688 — 5 missing: 2025-07-26T18:00:00, 2025-08-06T18:00:00, 2025-08-13T12:00:00, 2025-09-26T00:00:00, 2025-11-26T12:00:00
- P2 nulls: 5/73688 — 5 missing: 2025-07-26T18:00:00, 2025-08-06T18:00:00, 2025-08-13T12:00:00, 2025-09-26T00:00:00, 2025-11-26T12:00:00

### `wind_v_10m`

**Metadata**

- units: `m s-1`
- long_name: 10 metre V wind component
- short_name: 10v
- standard_name: northward_wind
- step_type: instant

**Spatial comparison**

- plot: `spatial_wind_v_10m.png`
- time: init=2025-08-09T12:00, lead=30h (reference at 2025-08-10T18:00)
- validation: min=-18.5, max=19, mean=0.5074
- reference:  min=-23.5, max=22.5, mean=0.4886

**Temporal comparison**

- plot: `temporal_wind_v_10m.png`
- period: Forecast init_time: 2025-08-29T00:00
- validation @ P1 (lat=64.25, lon=-95.50): min=-16.25, max=8.125, mean=-1.616
- validation @ P2 (lat=-85.00, lon=151.25): min=4.125, max=13.12, mean=9.564
- reference  @ P1: min=-11, max=9.125, mean=-1.587
- reference  @ P2: min=0.1426, max=12.5, mean=6.731

**Nulls**

- P1 nulls: 3/73688 — 3 missing: 2025-07-26T00:00:00, 2025-08-08T18:00:00, 2025-08-29T18:00:00
- P2 nulls: 3/73688 — 3 missing: 2025-07-26T00:00:00, 2025-08-08T18:00:00, 2025-08-29T18:00:00
