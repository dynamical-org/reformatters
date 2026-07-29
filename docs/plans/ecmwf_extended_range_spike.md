# ECMWF IFS Extended Range through ECDS: decision spike

## Executive summary

**Recommendation: CONDITIONAL GO for a revised 1.5° product; NO-GO for the originally assumed 0.25° product through ECDS.** The credentialed acceptance gate passed. Twenty-seven accepted requests downloaded and passed exact inventory validation, including archive-start and intermediate dates, control plus 100 perturbed members, every candidate field, and one full 46-lead initialization. A complete 27-field-level development Zarr reopened and queried successfully.

The condition is a product decision, not a remaining transport blocker: live ECDS files are global 121 × 240 at 1.5°, not the documented 0.25° archived grid assumed by the original proposal. If 1.5° is useful, proceed to production design and obtain ECMWF confirmation of the intended unattended request rate and attribution presentation. If 0.25° is required, stop this path and seek scheduled dissemination or another source.

The acceptance run also established the operational scale. The full initialization contained 125,442 GRIB messages and 6,985,939,316 source bytes. A streaming, ten-member-chunk development transform produced an 8,958,032,476-byte Zarr in 784.1 seconds. Expected implementation effort remains approximately 3–5 engineering weeks for a production adapter, template, validators, deployment, and bounded backfill; this is a planning range, not a fixed quote.

### Proposed bounded production scope

* ECMWF real-time forecasts only; exclude hindcasts/reforecasts.
* Begin at 2023-06-28, the first ECMWF initialization exposed by the live catalogue constraints and verified in this run.
* Publish daily initializations after the mandatory 48-hour access delay, checking actual per-date availability before creating each append region.
* Daily lead samples through 1,104 hours (46 days), selected from the parameter-dependent 6-hourly, 24-hourly, or daily-average output.
* Publish the observed global 1.5° grid: 121 descending latitudes from 90° to −90° and 240 longitudes from −180° to 178.5°.
* Start with the surface manifest below and fixed-level root variables at 500, 700, and 850 hPa. Fixed-level names avoid adding materialized multi-group support during the first integration.

## What was inspected and what is reusable

The existing 15-day IFS ENS integration already supplies canonical variable models and metadata, init/lead/member/latitude/longitude coordinates, `valid_time`, regional chunking, rasterio GRIB decoding, scale conversion, lead-time deaccumulation, materialized Zarr writes, Icechunk atomic commits, and forecast-current/recent-NaN validators. Its tests cover source coordinates, availability cutovers, absent step zero, strict metadata and shape checks, scaling, deaccumulation, operational updates, and real-file integration.

ECDS needs a source adapter rather than another transformation stack. The adapter must own authenticated submission, durable request IDs, one-shot polling, result expiry, request sharding, resumable downloads, local GRIB indexing, and strict validation of the expected variable × level × member × step product.

The current integration is specifically coupled to public Open Data or the internal MARS mirror by URL naming, `.index` sidecars, byte-range reads, GCS-to-S3 fallback, one init/lead/member source-coordinate granularity, a `(721, 1440)` grid, 51 members, and fixed 3/6-hour lead coordinates. Its operational deadline assumes directly downloadable objects. None of these are safe ECDS assumptions.

There is also an important completeness gap: failed individual downloads can remain fill values while finalization trims to an initialization with any successful coordinate. Extended Range must withhold the entire initialization unless its pre-write inventory is complete; recent-NaN validation after publication is too late.

## Source findings

### Observed in this run

* A standard `.cdsapirc` with accepted S2S terms authenticated successfully. Personal access tokens are sent in the `PRIVATE-TOKEN` header.
* ECDS submission returned a job ID and status location. Successful status responses exposed `links[rel=results]`; the results endpoint returned the signed download URL at `asset.value.href`.
* Twenty-seven accepted requests downloaded and passed strict variable × level × member × lead inventory checks. They covered 2023-06-28, 2024-06-27, 2026-07-23, and 2026-07-24.
* The current selector is `2_m_temperature`. The earlier `2m_temperature` failure used a stale selector and is not evidence of a backend mapping defect.
* Exact all-member files contained perturbation numbers 1–100; the separate control requests supplied member 0.
* The full 2026-07-24 candidate initialization used 14 shards and contained exactly 125,442 messages: 12 single-level fields plus five pressure variables at 500, 700, and 850 hPa, across 101 members and 46 daily leads.
* Every returned file used a 121 × 240 global 1.5° grid. GRIB coordinates decode to descending latitude 90° to −90°, longitude −180° to 178.5°, and an earth radius of 6,367,470 m.
* A duplicate request returned byte-identical output with SHA-256 `85abedb05adcffcb405879ef136cb8e019dad5d8713f382625ac23900dafaad9`.
* Restarting from a seeded partial file received HTTP 206 and reproduced the original GRIB. The HTTP 200 fallback correctly restarts rather than appending.
* One large all-member pressure submission received HTTP 502 before a request ID existed. Resubmission succeeded. No accepted job failed or returned an incomplete inventory.
* The download-state counter originally found 4,603 `GRIB` byte strings in a structurally valid 4,600-message file. It now follows each GRIB message length and validates every trailer rather than counting embedded payload bytes.

### Authoritative documentation findings, not file observations

The live ECDS catalogue exposes the exact collection identifier [`s2s-forecasts`](https://ecds.ecmwf.int/api/catalogue/v1/collections/s2s-forecasts). Its current form and constraint resources expose `origin`, `forecast_type`, `level_type`, `variable`, `year`, `month`, `day`, `time`, and `leadtime_hour`; the official [API example](https://ecds.ecmwf.int/how-to-api) uses `cdsapi.Client().retrieve("s2s-forecasts", request, target)`. These are live schema responses, not returned-GRIB observations.

The authoritative [S2S model table](https://confluence.ecmwf.int/display/S2S/Models) records CY48R1 from 2023-06-27 with 100 perturbed members plus one control, daily runs through day 46, TCo319 (~32 km), and a 0.25° archived grid. The live ECMWF catalogue begins on 2023-06-28 and its returned files are 1.5°. CY49R1 begins 2024-11-12 with the same headline dimensions but replaces CAPE with MUCAPE, a schema boundary inside the target archive. Live constraints list pressure levels 10, 50, 100, 200, 300, 500, 700, 850, 925, and 1000 hPa and daily pressure-field leads through 1,104 hours.

Authentication requires an ECMWF account, accepted dataset terms, and a `$HOME/.cdsapirc` token. The official page recommends `cdsapi>=0.7.2`. The incubating [`ecmwf-datastores-client`](https://ecmwf.github.io/ecmwf-datastores-client/) exposes asynchronous submission and status, but production should pin a tested release. Result retention and numeric concurrency/rate limits remain undocumented by this run.

## Candidate request and production sharding

A smoke payload should start from this official shape and be generated separately for control and perturbed forecast types:

```json
{
  "origin": "ecmwf",
  "forecast_type": "control_forecast",
  "level_type": "single_level",
  "variable": ["10_m_u_component_of_wind"],
  "year": ["2026"],
  "month": ["07"],
  "day": ["24"],
  "leadtime_hour": ["024", "048", "072"],
  "time": ["00:00"],
  "data_format": "grib"
}
```

Production-size shards should separate surface from pressure levels, control from perturbed members if required by the API, and split variables/steps until observed queue and response sizes are bounded. Shard boundaries must be tuned from measurements, not guessed.

## Candidate variable manifest

Every row below passed live inventory validation for control and 100 perturbed members. The full 46-lead transform validated the combined 27 field-level outputs.

| ECDS variable | level | observed decoded units | Dynamical name | processing |
|---|---|---|---|---|
| `2_m_temperature` | 2 m | °C | `temperature_2m` | daily average |
| `2_m_dewpoint_temperature` | 2 m | °C | `dew_point_temperature_2m` | daily average |
| `10_m_u_component_of_wind` | 10 m | m s-1 | `wind_u_10m` | instantaneous |
| `10_m_v_component_of_wind` | 10 m | m s-1 | `wind_v_10m` | instantaneous |
| `mean_sea_level_pressure` | mean sea level | Pa | `pressure_reduced_to_mean_sea_level` | instantaneous |
| `surface_pressure` | surface | Pa | `pressure_surface` | instantaneous |
| `total_precipitation` | surface | kg m-2 accumulated | `precipitation_surface` | difference to kg m-2 s-1 |
| `total_cloud_cover` | atmosphere | percent | `total_cloud_cover_atmosphere` | daily average |
| `surface_solar_radiation_downwards` | surface | J m-2 accumulated | `downward_short_wave_radiation_flux_surface` | difference to W m-2 |
| `surface_thermal_radiation_downwards` | surface | J m-2 accumulated | `downward_long_wave_radiation_flux_surface` | difference to W m-2 |
| `surface_runoff` | surface | kg m-2 accumulated | `runoff_surface` | difference to kg m-2 s-1 |
| `soil_moisture_top_20_cm` | 0–20 cm | kg m-3 | `soil_moisture_0_20cm` | daily average |
| `temperature` | 500/700/850 hPa | °C | `temperature_{level}hpa` | instantaneous |
| `u_component_of_wind` | 500/700/850 hPa | m s-1 | `wind_u_{level}hpa` | instantaneous |
| `v_component_of_wind` | 500/700/850 hPa | m s-1 | `wind_v_{level}hpa` | instantaneous |
| `specific_humidity` | 500/700/850 hPa | 1 | `specific_humidity_{level}hpa` | instantaneous |
| `geopotential_height` | 500/700/850 hPa | m | `geopotential_height_{level}hpa` | instantaneous |

The accumulated fields increase from initialization and are differenced across daily leads. GRIB packing creates small negative precipitation/runoff differences; the development transform clamps rates above the repository's established invalid-negative thresholds to zero. Soil moisture and runoff are land-only and share a 66.18% global missing mask. Soil moisture reaches −6.8×10⁻¹⁰ and cloud cover 100.0012% from packing; production should clamp these to their physical ranges and test the observed clamp fraction.

## Retrieval prototype and request matrix

Run the tool as separate cheap invocations so a scheduler, pod, or replacement process can resume state:

```bash
# Configure url and key in ~/.cdsapirc and accept the S2S terms first.
uv run src/scripts/ecmwf_extended_range_spike.py --state data/ecmwf-er/recent-smoke.json submit --payload payload.json
uv run src/scripts/ecmwf_extended_range_spike.py --state data/ecmwf-er/recent-smoke.json poll --maximum-polls 1
uv run src/scripts/ecmwf_extended_range_spike.py --state data/ecmwf-er/recent-smoke.json download --target data/ecmwf-er/recent-smoke.grib
```

State writes use a temporary sibling and atomic rename. Poll failures are recorded with bounded exponential backoff. Downloads use `.partial`, request a byte range on restart, only append after HTTP 206, validate leading `GRIB` and terminal `7777`, count messages, atomically rename, and write a completion marker. A production adapter must additionally parse every GRIB key, validate content length/checksum or stable validator, and compare the exact expected Cartesian inventory before completion.

The machine-readable measurements now record every live case without signed URLs. State persistence, live result discovery, structural GRIB counting, and both HTTP 206 append and HTTP 200 restart behavior are covered by unit tests. Production should continue using short-lived one-shot poll processes; the acceptance run showed no benefit to keeping a worker allocated while ECDS queues a job.

## Development Zarr proof

The complete real development Zarr contains dimensions `(init_time=1, lead_time=46, ensemble_member=101, latitude=121, longitude=240)` and 27 data variables. Xarray recognized `valid_time` and `spatial_ref` as coordinates, verified `valid_time = init_time + lead_time`, and selected an East Africa box with shape `(1, 46, 101, 21, 17)`.

The 6,985,939,316 source bytes became 8,958,032,476 Zarr bytes in 784.078 seconds. The writer streams GRIB and groups ten ensemble members per Zarr chunk; the resulting tree has 13,703 files. A one-member-per-chunk attempt was interrupted after its small-file throughput degraded, which is direct evidence against that layout.

All non-land variables had zero missing values. Soil moisture and runoff had the same expected 66.18% land mask. Temperature, wind, pressure, humidity, geopotential height, radiation, precipitation, and cloud-cover ranges were physically plausible after unit conversion and deaccumulation. A 772.101-second repeated build reproduced the same byte count and full-tree SHA-1 `215e037503e6ffc413ce38b87cb545616331753b`.

## Reliability measurements

| Metric | Observation |
|---|---|
| authenticated accepted requests | 27 downloaded and inventory-valid |
| accepted-job success rate | 27/27; one additional submission received HTTP 502 before a job ID and succeeded on resubmission |
| queue time | min 11.705 s; median 64.286 s; p95 216.431 s; max 299.646 s |
| server processing | min 2.884 s; median 12.592 s; p95 71.636 s; max 201.415 s |
| downloads | 7,128,215,624 bytes across all acceptance cases in 410.549 s total; largest download 154.799 s |
| update-cadence fit | daily operation has ample margin after the mandatory 48-hour delay |
| restart/retry/idempotency | HTTP 206 resume and HTTP 200 restart passed; duplicate output was byte-identical; deterministic Zarr rerun passed |
| incomplete-success behavior | no incomplete successful response observed; every completion still requires strict inventory validation |

## Storage and compute

The measured candidate initialization requires 6.986 GB of source staging and 8.958 GB of development Zarr storage. The output/source ratio is 1.282. The simple streaming transform took 13.1 minutes on this workstation; production RegionJobs should parallelize the 27 variables and member blocks.

Assuming every daily initialization from 2023-06-28 through 2026-07-24 exists gives 1,123 initializations. Linear projection from the measured Zarr is 10.060 decimal TB for that archive and 3.270 TB of annual growth. At the proposal's `$0.015/GB-month`, those are approximately `$150.90/month` for the initial archive and `$49.05/month` of storage added per year. These figures exclude Icechunk metadata, replicas, requests, compute, staging overlap, missing dates, and variable-dependent compression changes across model cycles.

The measured request plan uses 14 source shards per initialization, or 15,722 accepted jobs for the 1,123-initialization projection. Do not multiply median queue time by that count as a backfill duration: concurrent submissions overlapped, ECDS appeared to serialize some large jobs, and no authoritative concurrency limit was found. Run a bounded multi-initialization throughput test before pricing backfill wall time.

## Redistribution and operational terms

The current dataset-specific [S2S licence](https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/licences/s2s-licence/s2s-licence_c6d3b10fff18f2016c2bd4071748c73609d92c0d3cb4938132012457bdf77900.pdf) places ECMWF-origin S2S data under CC BY 4.0. It facially permits sharing and adaptation, including commercially, subject to appropriate attribution, a licence link, an indication of modifications, the requested S2S acknowledgement/citation, no endorsement, and no additional restrictions. The catalogue requests its DOI (10.21957/5ac361bf), Vitart et al. (2017), `Copyright © [year] ECMWF`, ECDS source, licence/disclaimer, and modification notice. A public transformed Zarr therefore appears permissible, but this is not legal advice.

Automated API access is the documented client pattern, and ECMWF-origin forecasts become accessible after a licence-mandated 48-hour delay. No authoritative numeric ECDS S2S concurrency/fair-use limit was found; the catalogue warns that costly requests receive lower priority and recommends smaller requests. Before public redistribution, ask ECMWF to confirm that (1) the exact attribution presentation is sufficient, (2) scheduled unattended access at the measured rate is acceptable, and (3) the transformed service imposes no incompatible downstream restriction.

Paid pre-scheduled delivery would replace unpredictable on-demand queueing and is the fallback if the acceptance gate fails. The internal `meta#135` analysis and “Storage scenarios” spreadsheet were unavailable here, so assigning a dissemination volume band or claiming economic plausibility would be invented. Obtain an ECMWF quote using the measured narrow manifest; do not price broad profiles or 6-hourly output by default.

## Risks and decision gate

Top risks:

1. **Resolution mismatch:** ECDS returned 1.5° files, so it cannot deliver the originally assumed 0.25° product.
2. **Backfill throughput:** one initialization is operationally comfortable, but 15,722 projected backfill jobs need a bounded multi-date concurrency test and an ECMWF automation-rate discussion.
3. **Static source details:** result retention is still unmeasured, and model-cycle boundaries require date-aware availability tests even though the selected 27-field manifest passed start, intermediate, and recent samples.
4. **Physical-range cleanup:** production transformations must clamp the measured soil-moisture/cloud-cover packing artifacts and validate the land mask and deaccumulation clamp fractions.
5. **Redistribution confirmation:** the licence appears compatible with a transformed public Zarr, but attribution presentation and unattended access rate should be confirmed with ECMWF.

The technical acceptance gate is complete. Proceed only if the product owner accepts 1.5°. The next implementation gate is a production-quality ECDS source adapter plus a small multi-initialization staging backfill; it is not another single-file spike.

## Proposed `dynamical-org/meta#144` comment

> **ECMWF Extended Range passed the technical ECDS acceptance gate, with one product-level catch: live files are global 1.5°, not the assumed 0.25°.** Twenty-seven accepted requests passed strict inventory checks across archive-start, intermediate, and recent dates. The full candidate initialization contained 125,442 messages, 101 members, 46 daily leads, and 27 field-level outputs; 6.986 GB of GRIB became an 8.958 GB Zarr in 13.1 minutes and reopened/query-tested successfully. Queue time was 64.3 s median and 299.6 s worst; one pre-job HTTP 502 succeeded on resubmission, HTTP Range recovery passed, and duplicate output was byte-identical. Recommend GO only if a 1.5° product is useful. If 0.25° is required, seek dissemination or another source. Before pricing production, confirm unattended request rate/attribution with ECMWF and run a bounded multi-initialization throughput test.

## Artifacts

* Report: `docs/plans/ecmwf_extended_range_spike.md`
* Request measurements/matrix: `docs/plans/ecmwf_extended_range_request_measurements.json`
* Retrieval tool: `src/scripts/ecmwf_extended_range_spike.py`
* Inventory and development transform: `src/scripts/ecmwf_extended_range_acceptance.py`
* Tests: `tests/scripts/ecmwf_extended_range_spike_test.py` and `tests/scripts/ecmwf_extended_range_acceptance_test.py`
