# ECMWF IFS Extended Range through ECDS: decision spike

## Executive summary

**Recommendation: CONDITIONAL GO, with no fixed-price production commitment until the authenticated acceptance gate below passes.** The repository has most of the expensive transformation machinery, and a durable asynchronous retrieval prototype demonstrates the required local control flow under simulated responses. This run could not answer the primary source question, however: the environment had no CDS/ECDS credential, so it submitted no live request and downloaded no GRIB. Consequently there are **no measured bytes, queue times, success rate, transformation costs, or end-to-end development Zarr from ECDS**. Documentation is not a substitute for those decision criteria.

The minimum acceptance gate is three real dates (near June 2023, intermediate, and recent), one complete proposed initialization, ten total request shards, interruption/retry exercises, a strict GRIB inventory, and an end-to-end Zarr. A fixed implementation price should wait for that run. If it fails because operational ECDS retrieval is unreliable, seek a quote for scheduled dissemination rather than making the public dataset depend on manual recovery.

Expected remaining human effort is roughly 2–5 engineering days for the credentialed acceptance run, then 3–5 engineering weeks for a production source adapter, strict inventory/decode integration, template and validation work, operational deployment, and a bounded backfill. This is a planning range rather than a fixed estimate; observed queue behavior and file layout are the largest multipliers.

### Proposed bounded production scope

* ECMWF real-time forecasts only; exclude hindcasts/reforecasts.
* Begin at 2023-06-27, the documented CY48R1 start and first daily 101-member configuration, subject to real-file verification.
* Publish daily initializations after the mandatory 48-hour access delay, checking actual per-date availability before creating each append region.
* Daily lead samples through 1,104 hours (46 days), selected from the parameter-dependent 6-hourly, 24-hourly, or daily-average output.
* Use the documented global 0.25° archived grid. Coordinate orientation and longitude convention remain real-file observations.
* Start with the surface manifest below and fixed-level root variables at 500, 700, and 850 hPa. Fixed-level names avoid adding materialized multi-group support during the first integration.

## What was inspected and what is reusable

The existing 15-day IFS ENS integration already supplies canonical variable models and metadata, init/lead/member/latitude/longitude coordinates, `valid_time`, regional chunking, rasterio GRIB decoding, scale conversion, lead-time deaccumulation, materialized Zarr writes, Icechunk atomic commits, and forecast-current/recent-NaN validators. Its tests cover source coordinates, availability cutovers, absent step zero, strict metadata and shape checks, scaling, deaccumulation, operational updates, and real-file integration.

ECDS needs a source adapter rather than another transformation stack. The adapter must own authenticated submission, durable request IDs, one-shot polling, result expiry, request sharding, resumable downloads, local GRIB indexing, and strict validation of the expected variable × level × member × step product.

The current integration is specifically coupled to public Open Data or the internal MARS mirror by URL naming, `.index` sidecars, byte-range reads, GCS-to-S3 fallback, one init/lead/member source-coordinate granularity, a `(721, 1440)` grid, 51 members, and fixed 3/6-hour lead coordinates. Its operational deadline assumes directly downloadable objects. None of these are safe ECDS assumptions.

There is also an important completeness gap: failed individual downloads can remain fill values while finalization trims to an initialization with any successful coordinate. Extended Range must withhold the entire initialization unless its pre-write inventory is complete; recent-NaN validation after publication is too late.

## Source findings

### Observed in this run

* Neither `ECDS_API_ENDPOINT` nor `ECDS_API_KEY` was exposed to the process, and no CDS credential file was present. No authenticated request was attempted. This was rechecked after the environment secrets were identified by name.
* The checked-in prototype defaults to the process endpoint for the live `s2s-forecasts` collection. The retrieve endpoint and response body still require an authenticated smoke test.
* The prototype's submission/status body compatibility is intentionally small. The official client or an observed response should determine the final response schema.

### Authoritative documentation findings, not file observations

The live ECDS catalogue exposes the exact collection identifier [`s2s-forecasts`](https://ecds.ecmwf.int/api/catalogue/v1/collections/s2s-forecasts). Its current form and constraint resources expose `origin`, `forecast_type`, `level_type`, `variable`, `year`, `month`, `day`, `time`, and `leadtime_hour`; the official [API example](https://ecds.ecmwf.int/how-to-api) uses `cdsapi.Client().retrieve("s2s-forecasts", request, target)`. These are live schema responses, not returned-GRIB observations.

The authoritative [S2S model table](https://confluence.ecmwf.int/display/S2S/Models) records CY48R1 from 2023-06-27 with 100 perturbed members plus one control, daily runs through day 46, TCo319 (~32 km), and a 0.25° archived grid. CY49R1 begins 2024-11-12 with the same headline dimensions but replaces CAPE with MUCAPE, a schema boundary inside the target archive. Live constraints list pressure levels 10, 50, 100, 200, 300, 500, 700, 850, 925, and 1000 hPa and daily pressure-field leads from 0 through 1,104 hours.

Authentication requires an ECMWF account, accepted dataset terms, and a `$HOME/.cdsapirc` token. The official page recommends `cdsapi>=0.7.2`. The incubating [`ecmwf-datastores-client`](https://ecmwf.github.io/ecmwf-datastores-client/) exposes asynchronous submission and status, but production should pin a tested release. Actual member encoding, per-date file inventory, GRIB edition, coordinate orientation, result retention, latency, concurrency, and rate limits remain unobserved.

## Candidate request and production sharding

A smoke payload should start from this official shape and be generated separately for control and perturbed forecast types:

```json
{
  "origin": "ecmwf",
  "forecast_type": "control_forecast",
  "level_type": "single_level",
  "variable": ["2m_temperature"],
  "year": ["2026"],
  "month": ["07"],
  "day": ["24"],
  "leadtime_hour": ["024", "048", "072"],
  "time": "00:00:00",
  "data_format": "grib"
}
```

Production-size shards should separate surface from pressure levels, control from perturbed members if required by the API, and split variables/steps until observed queue and response sizes are bounded. Shard boundaries must be tuned from measurements, not guessed.

## Candidate variable manifest

Every row is a **candidate to verify in actual S2S GRIB**. Parameter IDs are ECMWF GRIB table identifiers commonly used by the existing integration; availability, units, statistical processing, and equivalence are not asserted until inventory validation passes.

| ECDS/MARS parameter | short name / ID | level | expected source units | Dynamical name | processing | archive status |
|---|---|---|---|---|---|---|
| 2 metre temperature | `2t` / 167 | 2 m | K | `temperature_2m` | instantaneous | unverified |
| 2 metre dewpoint | `2d` / 168 | 2 m | K | `dew_point_temperature_2m` | instantaneous | unverified |
| 10 metre U wind | `10u` / 165 | 10 m | m s-1 | `wind_u_10m` | instantaneous | unverified |
| 10 metre V wind | `10v` / 166 | 10 m | m s-1 | `wind_v_10m` | instantaneous | unverified |
| mean sea-level pressure | `msl` / 151 | surface | Pa | `pressure_reduced_to_mean_sea_level` | instantaneous | unverified |
| surface pressure | `sp` / 134 | surface | Pa | `pressure_surface` | instantaneous | unverified |
| total precipitation | `tp` / 228 | surface | m | `precipitation_surface` | deaccumulate and convert to rate | unverified |
| total cloud cover | `tcc` / 164 | atmosphere | 1 | `total_cloud_cover_atmosphere` | instantaneous | unverified |
| surface solar radiation downwards | `ssrd` / 169 | surface | J m-2 | `downward_short_wave_radiation_flux_surface` | deaccumulate to W m-2 | unverified |
| surface thermal radiation downwards | `strd` / 175 | surface | J m-2 | `downward_long_wave_radiation_flux_surface` | deaccumulate to W m-2 | unverified |
| runoff | `ro` / 205 | surface | m | `runoff_surface` | deaccumulate to rate | unverified |
| volumetric soil water layer 1 | `swvl1` / 39 | soil layer 1 | m3 m-3 | `soil_moisture_0_7cm` | instantaneous; verify layer | unverified |
| temperature | `t` / 130 | 500/700/850 hPa | K | `temperature_{level}hpa` | instantaneous | unverified |
| U wind | `u` / 131 | 500/700/850 hPa | m s-1 | `wind_u_{level}hpa` | instantaneous | unverified |
| V wind | `v` / 132 | 500/700/850 hPa | m s-1 | `wind_v_{level}hpa` | instantaneous | unverified |
| specific humidity | `q` / 133 | 500/700/850 hPa | kg kg-1 | `specific_humidity_{level}hpa` | instantaneous | unverified |
| geopotential | `z` / 129 | 500/700/850 hPa | m2 s-2 | `geopotential_height_{level}hpa` | divide by standard gravity if canonical field is height | unverified |

Relative humidity (`r` / 157) is a fallback if specific humidity is absent; do not publish both without a use case. Wind gust, 100 m winds, categorical precipitation type, and every other existing 15-day field are unsupported for planning purposes until observed. Soil moisture layer bounds and accumulated-field reset semantics require message-level verification.

## Retrieval prototype and request matrix

Run the tool as separate cheap invocations so a scheduler, pod, or replacement process can resume state:

```bash
export ECDS_API_ENDPOINT=... # environment secret; never commit it
export ECDS_API_KEY=...      # environment secret; never commit it
uv run src/scripts/ecmwf_extended_range_spike.py --state data/ecmwf-er/recent-smoke.json submit --payload payload.json
uv run src/scripts/ecmwf_extended_range_spike.py --state data/ecmwf-er/recent-smoke.json poll --maximum-polls 1
uv run src/scripts/ecmwf_extended_range_spike.py --state data/ecmwf-er/recent-smoke.json download --target data/ecmwf-er/recent-smoke.grib
```

State writes use a temporary sibling and atomic rename. Poll failures are recorded with bounded exponential backoff. Downloads use `.partial`, request a byte range on restart, only append after HTTP 206, validate leading `GRIB` and terminal `7777`, count messages, atomically rename, and write a completion marker. A production adapter must additionally parse every GRIB key, validate content length/checksum or stable validator, and compare the exact expected Cartesian inventory before completion.

The machine-readable matrix records seven required cases plus intermediate, second-recent, and simulated-transient cases. Authenticated cases are `blocked_credentials`; restart and state persistence are unit-tested with simulated HTTP responses. Empty observations are deliberate and must not be interpreted as zero latency or 100% reliability.

## Development Zarr proof

The unit test writes a deterministic 101-member × 46-daily-lead synthetic initialization twice, reopens it, checks `valid_time`, and queries an East Africa box. It proves local dimension/order/query/idempotent-write mechanics only. It does **not** prove ECDS retrieval, GRIB decoding, real units, accumulation semantics, missingness, pressure levels, chunk efficiency, or actual complete-initialization transformation. Those remain acceptance-gate work.

## Reliability measurements

| Metric | Observation |
|---|---|
| authenticated requests | 0 |
| success rate | not measured |
| typical/worst queue time | not measured |
| typical/worst end-to-end latency | not measured |
| update-cadence fit | unknown |
| real restart/retry/idempotency | unknown; simulated state recovery only |
| incomplete-success behavior | unknown; strict inventory requirement identified |

## Storage and compute

There were no actual downloaded or generated representative files, so raw GRIB bytes per initialization, staging bytes, production Zarr bytes, compression ratio, peak memory, CPU time, transform wall time, requests per initialization, annual growth, initial archive size, backfill duration, and dollar cost are **not measured**. Supplying numeric values from theoretical dimensions would violate the spike requirement to use actual files.

After one complete initialization, calculate:

* initial archive bytes = measured compressed bytes per initialization × observed initialization count from stable-era start;
* annual growth = measured bytes × observed yearly initialization frequency;
* monthly storage cost = decimal GB × `$0.015`;
* staging peak = completed shard bytes + partial shard bytes + transformation workspace;
* backfill requests = initialization count × observed shards per initialization;
* backfill wall time from measured queue throughput, download bandwidth, and transform throughput, explicitly reporting the maximum bottleneck.

Sensitivity runs should repeat the same real initialization for 6-hourly versus daily steps, surface-only versus selected pressure levels, selected versus broad pressure profiles, and current-era versus all schema eras. Queue time is expected to be the unique ECDS risk, but calling the backfill queue-limited before measurements would be speculation. The “Storage scenarios” spreadsheet was not present in this repository, so its compute assumptions could not be applied.

## Redistribution and operational terms

The current dataset-specific [S2S licence](https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/licences/s2s-licence/s2s-licence_c6d3b10fff18f2016c2bd4071748c73609d92c0d3cb4938132012457bdf77900.pdf) places ECMWF-origin S2S data under CC BY 4.0. It facially permits sharing and adaptation, including commercially, subject to appropriate attribution, a licence link, an indication of modifications, the requested S2S acknowledgement/citation, no endorsement, and no additional restrictions. The catalogue requests its DOI (10.21957/5ac361bf), Vitart et al. (2017), `Copyright © [year] ECMWF`, ECDS source, licence/disclaimer, and modification notice. A public transformed Zarr therefore appears permissible, but this is not legal advice.

Automated API access is the documented client pattern, and ECMWF-origin forecasts become accessible after a licence-mandated 48-hour delay. No authoritative numeric ECDS S2S concurrency/fair-use limit was found; the catalogue warns that costly requests receive lower priority and recommends smaller requests. Before public redistribution, ask ECMWF to confirm that (1) the exact attribution presentation is sufficient, (2) scheduled unattended access at the measured rate is acceptable, and (3) the transformed service imposes no incompatible downstream restriction.

Paid pre-scheduled delivery would replace unpredictable on-demand queueing and is the fallback if the acceptance gate fails. The internal `meta#135` analysis and “Storage scenarios” spreadsheet were unavailable here, so assigning a dissemination volume band or claiming economic plausibility would be invented. Obtain an ECMWF quote using the measured narrow manifest; do not price broad profiles or 6-hourly output by default.

## Risks and decision gate

Top risks:

1. **Unobserved source contract:** authenticated request syntax, archive calendar, schema eras, full manifest, member completeness, and result retention are not established.
2. **Unobserved operational behavior:** queue latency, throughput, limits, resumability, idempotency, and structurally incomplete successful responses are not measured.
3. **Redistribution and cost:** dataset-specific terms need confirmation, while actual storage/compute and paid-delivery economics have no measured basis.

Recommend GO only after all of these pass unattended:

1. Inventory recent, intermediate, and near-start GRIBs; establish edition, grid, members, steps, parameters, levels, statistics, units, and schema cutovers.
2. Complete ten representative shards including a full initialization, an older full request, a duplicate, a killed download, and an injected transient failure. Record all measurement fields.
3. Transform one complete real initialization; compare accumulated fields numerically; verify fixed pressure levels, missingness, coordinate orientation, regional query, chunks, and idempotent rerun.
4. Project storage/cost only from those files and verify latency fits the observed initialization cadence.
5. Obtain written ECMWF confirmation for public transformed redistribution and intended automation rate if the catalogue terms are ambiguous.

## Proposed `dynamical-org/meta#144` comment

> **ECMWF Extended Range is a conditional go, not yet ready for a fixed-price commitment.** Most GRIB normalization, deaccumulation, Zarr writing, metadata, and validation can be reused from IFS ENS, but ECDS needs a durable asynchronous source adapter and strict whole-initialization completeness checks. I recommend a bounded first product: ECMWF real-time forecasts only, daily lead samples through the available extended horizon, all members, a useful surface set, and temperature/wind/humidity/geopotential at 500/700/850 hPa represented as fixed-level variables; start at the first live-verified stable 101-member date. This spike environment had no ECDS credentials, so it produced no defensible bytes/init, archive-size, compute, queue-latency, or reliability measurements. Remaining source risk is therefore high: live schema/calendar, unattended queue behavior, and dataset-specific redistribution terms. Budget human effort for an authenticated acceptance run plus source-adapter/integration work; do not quote production until three dates, ten shards, one complete real Zarr, restart/retry tests, measured storage, and ECMWF terms confirmation pass.

## Artifacts

* Report: `docs/plans/ecmwf_extended_range_spike.md`
* Request measurements/matrix: `docs/plans/ecmwf_extended_range_request_measurements.json`
* Retrieval tool: `src/scripts/ecmwf_extended_range_spike.py`
* Development-only mechanics proof: `tests/scripts/ecmwf_extended_range_spike_test.py`
