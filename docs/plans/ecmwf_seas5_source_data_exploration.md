# Source Data Exploration: ECMWF SEAS5

Completed following [docs/source_data_exploration_guide.md](../source_data_exploration_guide.md).

SEAS5 is ECMWF's fifth-generation seasonal forecast system (operational since
November 2017). One ensemble forecast is initialized on the 1st of every month at
00 UTC and runs 7 months ahead, with a matching 1981–2016 reforecast (hindcast)
set used for calibration. As of 1 October 2025 SEAS5 is openly licensed
(CC-BY-4.0 / Copernicus Licence).

**TL;DR for integration.** The only source we could access and examine directly is
the **Planette C3S mirror**, a public, anonymous, Icechunk-native store on AWS
covering **1981–2025 at 1°, daily, 51 (forecast) / 25 (hindcast) members** — an
excellent fit for our stack and the longest archive, but it is a third-party
mirror that **appears stalled since October 2025** and omits several core surface
variables (radiation, surface pressure, cloud cover, 10/100 m humidity). The
producer sources (ECMWF MARS / dissemination `mmsf`, and the Copernicus CDS) carry
the full ~100-variable set at up to 0.4°, but require authentication and are
lower throughput. A production integration will likely **combine Planette (history,
high throughput) with the CDS/MARS producer feed (recent months, completeness)**.
The advertised real-time open-data `mmsf` stream is **not currently functional**
(see Source 4).

---

## Dataset: ECMWF SEAS5

### Source Information — Source 1: Planette C3S Seasonal Forecast (AWS, Icechunk) — *examined directly*

- **Summary of data organization**: One Icechunk repository per `(variable, year)`.
  Each store holds, for one calendar year, every monthly init in that year × all
  ensemble members × 215 daily lead steps × lat × lon. Variables and pressure
  levels are split into separate stores (47 stores per year, one per
  variable/level). Dimensions inside each store:
  `(init_time, number, lead, lat, lon)`.
- **File format**: Icechunk repository (Zarr v3 data model; store contains
  `chunks/ manifests/ refs/ snapshots/ transactions/`). Read with
  `icechunk` + `xarray`. This is the same format family we already use.
- **Temporal coverage**: 1981-01-01 to **2025-10-01** (latest init observed).
  Hindcast 1981–2016, forecast 2017–present. Pressure-level temperature variables
  (`t100`…`t1000`) currently only go through **2024**.
- **Temporal frequency**: One init per month (1st of month, 00 UTC). 215 daily
  lead steps per init (~7 months / 215 days).
- **Latency**: **Stale / unreliable.** Latest init is 2025-10-01 and the 2025
  store's snapshots were last modified 2025-10-17. No new data in ~8 months as of
  2026-06-22, despite the registry advertising "monthly" updates. Treat as a
  historical archive, not an operational feed.
- **Access notes**: Public **anonymous** S3, region `us-east-2`, no credentials.
  Open with:
  ```python
  import icechunk, xarray as xr
  storage = icechunk.s3_storage(
      bucket="planette-c3s-seasonal-forecasts",
      prefix="seas5/sys51/t2m/day/1latx1lon/seas5_sys51_t2m_day_1latx1lon_2025.zarr",
      region="us-east-2", anonymous=True,
  )
  ds = xr.open_zarr(icechunk.Repository.open(storage).readonly_session("main").store,
                    consolidated=False)
  ```
- **Browse root**: https://planette-c3s-seasonal-forecasts.s3.amazonaws.com/index.html#seas5
  (AWS Open Data registry: https://registry.opendata.aws/planette_c3s_seasonal_forecast_data/)
- **URL / path format**:
```
s3://planette-c3s-seasonal-forecasts/seas5/sys51/{var}/day/1latx1lon/seas5_sys51_{var}_day_1latx1lon_{YYYY}.zarr
```
- **Example paths**:
```
seas5/sys51/t2m/day/1latx1lon/seas5_sys51_t2m_day_1latx1lon_1981.zarr   # hindcast, 25 members
seas5/sys51/t2m/day/1latx1lon/seas5_sys51_t2m_day_1latx1lon_2025.zarr   # forecast, 51 members
seas5/sys51/pr/day/1latx1lon/seas5_sys51_pr_day_1latx1lon_2025.zarr
seas5/sys51/z500/day/1latx1lon/seas5_sys51_z500_day_1latx1lon_2025.zarr
```

### Source Information — Source 2: ECMWF MARS / dissemination (`set-v`, `stream=mmsf`) — *producer, from docs*

- **Summary of data organization**: GRIB2 from the producer. Seasonal forecast
  fields under `stream=mmsf` (individual members), plus monthly-mean / anomaly
  streams (`msmm`, `mmsa`). Surface (`levtype=sfc`) and pressure-level
  (`levtype=pl`) fields. `type=fc`, `origin=ecmwf`. Operational data is
  `class=od`; the 1981–2016 reforecasts are a separate MARS retrieval.
- **File format**: GRIB2 (model-level data is GRIB2; this replaced GRIB1 from
  System 4).
- **Temporal coverage**: Operational SEAS5 from 2017-11 to present; reforecasts
  1981–2016. The producer is the authoritative, complete record.
- **Temporal frequency**: Monthly init, 1st at 00 UTC. 7-month range (some
  products extend to 13 months at reduced membership).
- **Latency**: Products released on **the 5th of each month at 12 UTC**.
- **Resolution**: Native O320 reduced-Gaussian (~35 km) / TCO319 spectral;
  disseminated on regular lat/lon at **0.4°×0.4°** or **0.75°×0.75°** — higher
  than the 1° Planette/CDS product.
- **Members**: 51 (forecast), 25 (reforecast). 100+ parameters including the
  surface variables missing from the C3S subset (radiation fluxes, surface
  pressure, cloud cover, etc.).
- **Access notes**: ECMWF Web API / MARS (`ecmwf-api-client`) or CDS. Requires an
  ECMWF account + API key. Now CC-BY-4.0. Lower throughput than object storage.
- **Browse root**: https://www.ecmwf.int/en/forecasts/datasets/set-v

### Source Information — Source 3: Copernicus Climate Data Store (C3S seasonal) — *producer-backed, from docs*

- **Summary of data organization**: The C3S "Seasonal forecast daily/subdaily
  data" collections, `originating_centre=ecmwf`, `system=51`. This is the
  **same underlying data Planette mirrors into Icechunk** (1° daily). Useful as a
  reliable, producer-backed feed for recent months that Planette has not ingested.
- **File format**: GRIB or NetCDF (requestor's choice).
- **Temporal coverage**: 1981–present (hindcast 1981–2016 @ 25 members, forecast
  2017–present @ 51 members).
- **Resolution**: 1°×1° global (regridded from native).
- **Access notes**: CDS API (`cdsapi`), free registration + API key; requests are
  queued/rate-limited. Authoritative and reliable but not high-throughput.
- **Browse root**: https://cds.climate.copernicus.eu (search "seasonal forecast").

### Source Information — Source 4: ECMWF real-time Open Data (`mmsf` stream) — *advertised but NON-FUNCTIONAL*

- **Status**: Documented but **could not be accessed** — every probe returned 404.
  The `ecmwf-opendata` client and the open-data docs describe a seasonal `mmsf`
  stream, but the files are not present on `data.ecmwf.int/forecasts/...` and the
  AWS `ecmwf-forecasts` bucket only mirrors medium-range streams
  (`enfo`, `oper`, `waef`, `wave`) — no `mmsf` directory for any date.
- **Documented (non-working) path** the client builds (stream `mmsa` → `mmsf` in
  the URL, step in months):
```
https://data.ecmwf.int/forecasts/{YYYYMMDD}/00z/ifs/0p25/mmsf/{YYYYMMDD}000000-{N}m-mmsf-fc.grib2
```
- **Evidence**: probed `fcmonth` 1–7, `type` in {fc, fcmean, em, es, ef, cf, pf},
  `resol` in {0p25, 0p4-beta, 1p0, 0p5}, and init dates monthly from 2025-10-01
  through 2026-06-01 — all 404. Consistent with open issue
  `ecmwf/ecmwf-opendata#12` ("Unable to access seasonal forecasts stream mmsf").
- **Conclusion**: Do not rely on the real-time open-data portal for SEAS5 today.
  Revisit later — if/when it starts publishing, it would be the ideal
  high-throughput, no-auth, producer source.

### GRIB Index (if applicable)

Not applicable to the examined source (Source 1 is Icechunk/Zarr, not GRIB). The
producer sources (2 & 3) deliver GRIB2; their index conventions were not examined
here because access requires an API key. The native ECMWF GRIB index style is the
JSON-per-line "ECMWF style" used by our existing IFS integration.

### Coordinate Reference System

- **Common name**: Geographic lat/lon (WGS84-like), regular grid.
- **Source 1 grid (verified from GRIB metadata in the store)**: `gridType =
  regular_ll`, `iDirectionIncrementInDegrees = 1.0`,
  `jDirectionIncrementInDegrees = 1.0`, `Nx = 360`, `Ny = 180`. Latitude runs
  **north→south** (`latitudeOfFirstGridPoint = 89.5`, last `= -89.5`); longitude
  stored 0.5→359.5 and presented by xarray rolled to −179.5→179.5. **Pixel
  centers at half-degrees** (X.5°), not edge-aligned.
- **PROJ string or EPSG**: Not stamped in the product metadata. ECMWF/WMO
  convention is a sphere of radius **6,371,229 m** (the value our other ECMWF
  datasets use); **not verified** from this product, so confirm before encoding a
  `spatial_ref`.

### Dimensions & Dimension Coordinates

(Verified from Source 1 stores.)

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| init_time | 1981-01-01 | 2025-10-01 (observed) | 1 month | 1st of each month, 00 UTC. `standard_name=forecast_reference_time`. |
| lead | 1 day | 215 days | 1 day | **Units differ by variable** — see gotcha below. 215 daily steps ≈ 7 months. |
| number (ensemble) | 0 | 50 (forecast) / 24 (hindcast) | 1 | **25 members 1981–2016, 51 members 2017–present.** `standard_name=realization`. |
| latitude (`lat`) | -89.5 | 89.5 | 1.0 | Descending (N→S). Pixel centers at X.5°. |
| longitude (`lon`) | -179.5 | 179.5 | 1.0 | Pixel centers at X.5°. Stored 0.5–359.5 in GRIB. |

- `valid_time(init_time, lead)` is provided as a derived 2-D coordinate. The first
  daily value is at **init + 1 day** (e.g. init 2025-06-01 → first valid 2025-06-02;
  lead 215 → 2026-01-02). There is no init-day (lead 0) value.
- Scalar coords present in each store: `surface`/`isobaricInhPa`/`depthBelowLandLayer`
  (level), `year`.

We use pixel centers for spatial coordinates — matches this source.

### Data Variables

Core checklist (availability in **Source 1 / Planette C3S 1°**; the producer
sources 2 & 3 additionally carry the surface variables marked ✗):

| Variable name | Level | Units | Available from | Notes |
|---------------|-------|-------|----------------|-------|
| temperature_2m | 2 m | K | ✓ `t2m` | paramId 167, `2t`, instant |
| wind_u_10m | 10 m | m s⁻¹ | ✓ `u10m` | paramId 165, `10u` |
| wind_v_10m | 10 m | m s⁻¹ | ✓ `v10m` | paramId 166, `10v` |
| wind_u_100m | 100 m | m s⁻¹ | ✗ | not in C3S subset (`u100`/`v100` here are 100 **hPa**, not 100 m) |
| wind_v_100m | 100 m | m s⁻¹ | ✗ | not in C3S subset |
| precipitation_surface | surface | m | ✓ `pr` | paramId 228, `tp`. **Daily total in metres, resets each day** (not cumulative). lead in hours. |
| downward_short_wave_radiation_flux_surface | surface | W m⁻² | ✗ | not in C3S subset; present in native SEAS5/MARS |
| downward_long_wave_radiation_flux_surface | surface | W m⁻² | ✗ | not in C3S subset; present in native SEAS5/MARS |
| pressure_surface | surface | Pa | ✗ | only MSL provided (no `sp`) |
| pressure_reduced_to_mean_sea_level | MSL | Pa | ✓ `slp` | paramId 151, `msl` |
| total_cloud_cover_atmosphere | atmosphere | % | ✗ | not in C3S subset |
| relative_humidity_2m | 2 m | % | ✗ | derive from `t2m` + `t2d` if needed |
| specific_humidity_2m | 2 m | kg kg⁻¹ | ✗ | `q` only at pressure levels |
| dew_point_temperature_2m | 2 m | K | ✓ `t2d` | paramId 168, `2d` |

**Other surface variables available in Source 1** (verified):

| Var | paramId / short | Units | Notes |
|-----|-----------------|-------|-------|
| `sf` | 144 / `sf` | m of water equivalent | Snowfall (daily accumulation; lead in hours) |
| `sst` | 34 / `sst` | K | Sea surface temperature |
| `stl1` | 139 / `stl1` | K | Soil temperature level 1 (`depthBelowLandLayer`) |
| `t2m_max` | 51 / `mx2t24` | K | Max 2 m temp in last 24 h (lead in hours) |
| `t2m_min` | 52 / `mn2t24` | K | Min 2 m temp in last 24 h (lead in hours) |
| `tcwv` | 137 / `tcwv` | kg m⁻² | Total column water vapour (lead in hours) |
| `tau_x` | 180 / `ewss` | N m⁻² s | **Time-integrated** eastward surface stress |
| `tau_y` | 181 / `nsss` | N m⁻² s | **Time-integrated** northward surface stress |

**Pressure-level variables available in Source 1** (verified). Levels in hPa:

| Family | paramId / short | Units | Levels (hPa) | Notes |
|--------|-----------------|-------|--------------|-------|
| `q###` specific humidity | 133 / `q` | kg kg⁻¹ | 100,200,300,400,500,700,850,925,1000 | |
| `t###` temperature | 130 / `t` | K | 100,200,300,400,500,700,850,925,1000 | **only 1981–2024** |
| `u##` U wind | 131 / `u` | m s⁻¹ | 10,200,500,850 | `u10` is **10 hPa**, not 10 m |
| `v###` V wind | 132 / `v` | m s⁻¹ | 10,100,200,500,850 | `v10` is 10 hPa |
| `z###` geopotential | 129 / `z` | m² s⁻² | 10,200,300,500,700,850 | **Geopotential, not height** — divide by g=9.80665 for geopotential height |

**Temporal availability changes**:
- **Ensemble size**: 25 members (reforecast, 1981–2016) → 51 members (forecast,
  2017–present). A combined archive has a ragged `number` dimension across the
  2016/2017 boundary.
- **Pressure-level temperature** (`t100`…`t1000`): present 1981–**2024** only at
  examination time (surface variables go to 2025).
- **Archive currency**: Source 1 has not advanced past 2025-10 (see Latency).

### Sample Files Examined

All from Source 1 (`s3://planette-c3s-seasonal-forecasts`, anonymous, us-east-2):

- **Early / hindcast**: `seas5/sys51/t2m/day/1latx1lon/seas5_sys51_t2m_day_1latx1lon_1981.zarr`
  — 12 inits (Jan–Dec 1981), **25** members, lead 1–215.
- **Recent / forecast**: `seas5/sys51/t2m/day/1latx1lon/seas5_sys51_t2m_day_1latx1lon_2025.zarr`
  — 10 inits (Jan–Oct 2025), **51** members, lead 1–215. Snapshots last modified
  2025-10-17.
- **Variables sampled for metadata/units**: `pr, sf, slp, sst, stl1, t2d, t2m,
  t2m_max, t2m_min, tcwv, tau_x, tau_y, u10m, v10m, u10, v10, v100, u200, z500,
  q850, t850, t100` across 1981/2024/2025.
- **Producer GRIB files (Sources 2 & 3)**: *not* downloaded — both require an
  API key unavailable in this environment. Their specs above come from ECMWF/C3S
  documentation and are flagged as such per the guide.

### Notable Observations

- **Great format fit, awkward layout.** The examined source is Icechunk-native
  (matches our stack), but it is sharded into ~47 stores/year × 45 years
  (~2,000 repositories). An integration must iterate and concatenate these.
- **`lead` units are inconsistent across variables** — the single most important
  gotcha. Instantaneous fields (`t2m, slp, sst, u10m, u200, z500, q850, t###`)
  use **days** `[1…215]`; accumulated/statistic fields (`pr, sf, t2m_max,
  t2m_min, tcwv`) use **hours** `[24…5160]`. Both encode the same 215 daily steps
  and identical `valid_time`. **Key off `valid_time` (or normalize lead) when
  combining variables** — do not concatenate `lead` blindly.
- **Unit conversions needed to match our conventions**: `pr`/`sf` are daily
  accumulations in metres (scale ×1000 → mm; note they already reset daily, so no
  deaccumulation); `z###` is geopotential in m² s⁻² (÷9.80665 → geopotential
  height); `tau_x`/`tau_y` are time-integrated stresses.
- **CF metadata is not populated** in the source (`standard_name='unknown'`) — we
  assign CF `standard_name`/`units` ourselves following repo conventions, reusing
  existing variable names from the IFS ENS dataset where they match
  (`temperature_2m`, `wind_u_10m`, `precipitation_surface`, etc.).
- **Reliability vs. completeness vs. throughput** (the guide's priority tension):
  Source 1 wins on throughput/access and archive length but is a third-party
  mirror that is currently stalled and lacks several core surface variables;
  Sources 2/3 win on completeness and producer-reliability but need auth and are
  slower. Expect to **combine** them.
- **Resolution ceiling**: 1° from Sources 1 & 3. For ~0.4° we must go to MARS
  (Source 2). 1° (~111 km) is coarse but normal for seasonal forecasts.

### Recommendation for integration

1. **Primary / backfill**: Planette C3S Icechunk (Source 1) for the 1981–2025
   history — high throughput, no auth, native format. Build the `lead`
   normalization and unit conversions described above.
2. **Operational updates**: pull recent months from the **CDS** (Source 3) or
   **MARS** (Source 2), since Planette lags. This is the same data Planette
   mirrors, so backfill and updates stay consistent at 1°.
3. **Optional higher-resolution / extra-variable track**: MARS at 0.4° for
   radiation, surface pressure, cloud, 2 m humidity, and 10/100 m winds if those
   variables are required.
4. **Re-check the real-time open-data `mmsf` stream periodically** — if it goes
   live it would simplify operations considerably.

This mirrors the multi-source pattern in our IFS ENS integration (high-throughput
mirror first, producer fallback).
