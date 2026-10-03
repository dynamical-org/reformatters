# NOAA RRFS and REFS virtual forecasts

The deterministic 84-hour and 18-hour datasets include every unique quantity from the CONUS `2dfld` and `prslev` files. The subhourly dataset includes all 39 quantities from `2dfld.3km.subh`; one hourly file supplies four successive 15-minute forecast steps. The REFS member dataset includes the 58 surface quantities and eight pressure-level quantities from the CONUS `2dfldnomads` and `prslevnomads` families. Member 0 reads the corresponding deterministic files; members 1–5 read their own perturbed-member files. These are six unique RRFS runs, while the full REFS ensemble products also incorporate previous-cycle RRFS runs and current/previous-cycle HRRR.

The common catalog declares exact element, level, window and trailing selectors. Aerosol species, particle-size selectors, probability bounds and generating-process tags are part of message identity. A required message missing from an available index fails completeness. Duplicate source messages supply one archive quantity; intentionally zero-length ARI/FFG placeholders supply no reference.

## Vertical coordinates

Dense pressure fields use `pressure_level`; dense geometric heights use `height_above_mean_sea_level`. Soil temperature, total volumetric moisture and liquid volumetric moisture use `depth_below_ground`, with source point depths 0, 0.01, 0.04, 0.1, 0.3, 0.6, 1, 1.6 and 3 metres. The depth coordinate has CF `standard_name="depth"`, `units="m"`, `positive="down"` and `axis="Z"`. These are point depths, so no layer bounds are inferred. Sparse heights and the six pressure-difference boundary layers remain root variables with source-level suffixes.

The CONUS Lambert conformal grid uses the common NOAA CONUS grid utility. Source grid-component flags identify grid-relative winds, so component metadata uses CF `x_wind` and `y_wind`.

## Source interpretation

ARI and FFG exceedance fields are binary categorical flags, with 0=no and 1=yes from GRIB table 4.222. Zero also represents unavailable guidance. ARI thresholds are recurrence intervals in years; FFG's `>1` source label is not a duration. Run-total FFG exists at leads 1, 3, 6 and 12 hours; run-total ARI additionally exists at 24 hours. UPP emits all-zero, zero-length placeholders at other leads; those positions remain structural absence in the archive.

RUC land-surface classes use STATSGO soil codes 1–19 and modified IGBP vegetation codes 1–21, including lakes as vegetation code 21. UPP maps undefined source classes to code 0. The variable flag attributes carry the complete code mappings.

Missing source bitmap cells decode to NaN. Cloud-ceiling fields contain bitmap-valid values exactly equal to GDAL's nominal 9999 nodata marker; declaring 9999 as archive fill would remove valid measurements. The archive preserves these measurements. NaN ceiling means no applicable cloud ceiling as well as unavailable source data, as described in the variable comment.

The installed gribberish 1.8.0 decoder does not honor missing management for GRIB data representation template 5.2. Deterministic surface specific humidity, potential evaporation rate and potential evaporation, and perturbed-member aerosol optical thickness and wildfire potential have affected all-missing source messages. Their metadata intent is NaN, with no fill255 workaround: 255 can be physically valid for other fields and packings. Strict expected-failure tests preserve tiny real CONUS messages and compare with GDAL. Correct decoding is a release prerequisite. All-NaN validation allowances apply only to the three deterministic fields and the two perturbed-member fields for which source evidence establishes that state; they never excuse decode errors or absent references.

## Suspended operational resources

Update and validation jobs are suspended. The source exploration measured subhourly completion near initialization +1h46m, deterministic lead18 near +2h21m, and long deterministic/member leads near +3h32m. Updates start just before the measured first-file publication and poll through the final leads. Initial source-derived settings are:

| Dataset | Fire offset from initialization | Pod deadline | Validation offset / current-data delay |
|---|---|---|---|
| subhourly | 75minutes | 55minutes | 135minutes |
| 18-hour | 100minutes | 60minutes | 165minutes |
| 84-hour | 100minutes | 135minutes | 240minutes |
| members | 75minutes | 160minutes | 240minutes |

The hourly subhourly cron fires at minute15. Its processing window includes an unpublished next initialization, but the newest ingested initialization is normally the previous one: its poll ends before the next cycle normally begins publishing. Completeness requires at least5% of the newest ingested initialization (one of18files), and100% of every older initialization. Each update stops polling five minutes before the pod deadline to leave time for inline validation; scheduled validation starts five minutes after the pod deadline. Deadline/runtime sizing must be measured during the isolated backfill/validation stage before enabling jobs.

Manifest splits use 90 initializations for pressure arrays and 300 for root, AMSL and depth arrays. At approximately 16 bytes per reference, an 84-hour pressure array has 85×45×90 references (about 5.3 MiB); member pressure has 61×14×6×90 (about 7.0 MiB). The 84-hour AMSL/depth catch-all is about 3.9/3.5 MiB, while root arrays are about 0.4 MiB and subhourly about 0.3 MiB. These estimates describe per-array reader cost; operational commits rewrite active manifests across all arrays. Measure commit latency before enabling jobs.

## Verification

Index fixtures exercise complete message/level/window matching, minute packing, exact selectors, early/recent source coverage and supported lead eligibility. Isolated local Icechunk tests backfill and update each of the four datasets and pin every processed variable's initial and updated values, all subhourly quarter-hour slots, and all six members. Independent GDAL range reads verify quarter-hour and perturbed-member snapshots. No test writes production stores or submits cluster work.
