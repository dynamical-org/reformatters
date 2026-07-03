This dataset validation report plots a sample of values from the NOAA HRRR forecast, 48 hour, spatial dataset over time and across space, comparing where possible to a previously validated reference dataset. It also reports the quantity of unavailable values and their associated timestamps. These analyses are one layer of a multi-layered dataset validation process we perform at dynamical.org and also provide users a preview of the dataset contents.

Report generation start time: 2026-07-03T14:32:35 UTC

## Datasets

| Role | Name | ID | Version | URL |
|---|---|---|---|---|
| Validation | NOAA HRRR forecast, 48 hour, spatial | `noaa-hrrr-forecast-48-hour-spatial` | `v0.5.0` | `s3://dynamical-noaa-hrrr/noaa-hrrr-forecast-48-hour-spatial/v0.5.0.icechunk` |
| Reference  | NOAA GEFS analysis | `noaa-gefs-analysis` | `0.1.2` | `s3://dynamical-noaa-gefs/noaa-gefs-analysis/v0.1.2.icechunk/` |

## Run parameters

| Parameter | Value |
|---|---|
| Validation dataset type | forecast |
| Validation time range | init_time 2026-06-30T00:00 → 2026-07-03T12:00 |
| Reference time range | time 2000-01-01T00:00 → 2026-07-03T06:00 |
| Point 1 | lat=27.0676, lon=-116.3027 |
| Point 2 | lat=49.1042, lon=-69.9023 |
| Spatial comparison time | init=2026-07-03T00:00, lead=28h (reference at 2026-07-03T06:00) |
| Timeseries period | Forecast init_time: 2026-06-30T12:00 |
| Vertical level | middle level per vertical dim |
| Store type | virtual — availability via manifest_scan.py; value series sampled |

## Combined plots

- Value time series (full period): [`combined_value_timeseries.png`](combined_value_timeseries.png)
- Spatial and distributions: [`combined_spatial.png`](combined_spatial.png)
- Time series: [`combined_temporal.png`](combined_temporal.png)

## Unavailable timestamps

Not value-scanned (virtual store). Whole-archive availability is checked by `manifest_scan.py`; include its output when publishing this report.

## Per-variable details

### `composite_reflectivity`

**Metadata**

| Field | Value |
|---|---|
| units | `dBZ` |
| long_name | Maximum/Composite radar reflectivity |
| short_name | refc |
| standard_name | equivalent_reflectivity_factor |
| step_type | instant |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | -10 | -7.583 | 0 | 1.688 |
| P2 | -10 | -4.179 | 0 | 30.38 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference not available)

| Source | min | mean | max |
|---|---|---|---|
| Validation | -10 | -6.928 | 73 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | -10 | -8.587 | 1.062 |
| P2 Validation | -10 | -3.444 | 41.62 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `dew_point_temperature_2m`

**Metadata**

| Field | Value |
|---|---|
| units | `degree_Celsius` |
| long_name | 2 metre dewpoint temperature |
| short_name | 2d |
| standard_name | dew_point_temperature |
| step_type | instant |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | 14.08 | 15.78 | 0 | 17.36 |
| P2 | 10.31 | 16.21 | 0 | 21.98 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference not available)

| Source | min | mean | max |
|---|---|---|---|
| Validation | -81.03 | 14.91 | 27.6 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | 14.06 | 15.18 | 16.45 |
| P2 Validation | 11.72 | 16.6 | 21.55 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `downward_short_wave_radiation_flux_surface`

**Metadata**

| Field | Value |
|---|---|
| units | `W m-2` |
| long_name | Surface downward short-wave radiation flux |
| short_name | sdswrf |
| standard_name | surface_downwelling_shortwave_flux_in_air |
| step_type | instant |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | 0 | 226.1 | 0 | 714 |
| P2 | 0 | 176.3 | 0 | 837 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference at 2026-07-03T06:00)

| Source | min | mean | max |
|---|---|---|---|
| Validation | 0 | 0.4241 | 85.29 |
| Reference | 0 | 48.87 | 280 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | 0 | 212.1 | 982 |
| P1 Reference | 0 | 296.7 | 1004 |
| P2 Validation | 0 | 191.9 | 597.2 |
| P2 Reference | 0.01599 | 213.4 | 700 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `model_level/temperature`

**Metadata**

| Field | Value |
|---|---|
| units | `degree_Celsius` |
| long_name | Temperature |
| short_name | t |
| standard_name | air_temperature |
| step_type | instant |
| sampled level | model_level=26 |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | -35.27 | -32.92 | 0 | -31.51 |
| P2 | -41.25 | -38.12 | 0 | -36.27 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference not available)

| Source | min | mean | max |
|---|---|---|---|
| Validation | -45.98 | -36.79 | -28.36 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | -35.01 | -33.83 | -32.7 |
| P2 Validation | -42.5 | -37.68 | -35.28 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `pressure_level/geopotential_height`

**Metadata**

| Field | Value |
|---|---|
| units | `m` |
| long_name | Geopotential height |
| short_name | gh |
| standard_name | geopotential_height |
| step_type | instant |
| sampled level | pressure_level=525 |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | 5487 | 5511 | 0 | 5537 |
| P2 | 5260 | 5362 | 0 | 5412 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference not available)

| Source | min | mean | max |
|---|---|---|---|
| Validation | 5283 | 5498 | 5572 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | 5484 | 5499 | 5513 |
| P2 Validation | 5361 | 5397 | 5423 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `pressure_level/relative_humidity`

**Metadata**

| Field | Value |
|---|---|
| units | `percent` |
| long_name | Relative humidity |
| short_name | r |
| standard_name | relative_humidity |
| step_type | instant |
| sampled level | pressure_level=525 |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | 3.715 | 28.59 | 0 | 78.72 |
| P2 | 5.032 | 40.68 | 0 | 86.09 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference not available)

| Source | min | mean | max |
|---|---|---|---|
| Validation | 0.4858 | 41.33 | 99.61 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | 2.065 | 7.855 | 15.17 |
| P2 Validation | 8.559 | 54.31 | 99.51 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `pressure_level/temperature`

**Metadata**

| Field | Value |
|---|---|
| units | `degree_Celsius` |
| long_name | Temperature |
| short_name | t |
| standard_name | air_temperature |
| step_type | instant |
| sampled level | pressure_level=525 |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | -6.194 | -3.157 | 0 | -1.781 |
| P2 | -13.87 | -8.003 | 0 | -4.378 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference not available)

| Source | min | mean | max |
|---|---|---|---|
| Validation | -14.17 | -5.625 | -1.23 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | -4.123 | -2.855 | -2.018 |
| P2 Validation | -12.62 | -8.415 | -5.13 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `snow_water_equivalent_surface`

**Metadata**

| Field | Value |
|---|---|
| units | `m` |
| long_name | Snow depth water equivalent |
| short_name | sd |
| standard_name | lwe_thickness_of_surface_snow_amount |
| step_type | instant |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | 0 | 0 | 0 | 0 |
| P2 | 0 | 0 | 0 | 0 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference not available)

| Source | min | mean | max |
|---|---|---|---|
| Validation | 0 | 0.0001943 | 5 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | 0 | 0 | 0 |
| P2 Validation | 0 | 0 | 0 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `temperature_2m`

**Metadata**

| Field | Value |
|---|---|
| units | `degree_Celsius` |
| long_name | 2 metre temperature |
| short_name | 2t |
| standard_name | air_temperature |
| step_type | instant |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | 18.34 | 18.83 | 0 | 19.18 |
| P2 | 12.58 | 18.91 | 0 | 26.25 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference at 2026-07-03T06:00)

| Source | min | mean | max |
|---|---|---|---|
| Validation | -3.926 | 22.42 | 38.82 |
| Reference | 1.383 | 20.34 | 32.5 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | 17.54 | 18.57 | 19.32 |
| P1 Reference | 18.5 | 18.74 | 19 |
| P2 Validation | 14.18 | 19.08 | 25.86 |
| P2 Reference | 13.5 | 19.94 | 28.75 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `wind_gust_surface`

**Metadata**

| Field | Value |
|---|---|
| units | `m s-1` |
| long_name | Wind speed (gust) |
| short_name | gust |
| standard_name | wind_speed_of_gust |
| step_type | instant |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | 5.132 | 6.686 | 0 | 8.429 |
| P2 | 4.177 | 8.569 | 0 | 14.01 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference not available)

| Source | min | mean | max |
|---|---|---|---|
| Validation | 0.08062 | 6.024 | 49.08 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | 4.862 | 6.257 | 6.882 |
| P2 Validation | 1.796 | 6.469 | 10.58 |

**Nulls** — P1: n/a (none); P2: n/a (none)

### `wind_u_10m`

**Metadata**

| Field | Value |
|---|---|
| units | `m s-1` |
| long_name | 10 metre U wind component |
| short_name | 10u |
| standard_name | eastward_wind |
| step_type | instant |

**Point time series statistics for the full period (2026-06-30T00:00 - 2026-07-03T12:00) — sampled (pinned lead/level/member)**

| Point | min | mean | std | max |
|---|---|---|---|---|
| P1 | -1.839 | 1.098 | 0 | 2.854 |
| P2 | -5.038 | 2.089 | 0 | 6.036 |

**Spatial** — snapshot at init=2026-07-03T00:00, lead=28h (reference at 2026-07-03T06:00)

| Source | min | mean | max |
|---|---|---|---|
| Validation | -17.46 | 0.1511 | 33.54 |
| Reference | -13.12 | -0.1988 | 9.625 |

**Temporal** — period Forecast init_time: 2026-06-30T12:00

| Source | min | mean | max |
|---|---|---|---|
| P1 Validation | -0.497 | 1.209 | 3.411 |
| P1 Reference | 0.9609 | 1.932 | 2.812 |
| P2 Validation | -4.501 | -0.6696 | 3.548 |
| P2 Reference | -3.594 | 1.581 | 6.812 |

**Nulls** — P1: n/a (none); P2: n/a (none)
