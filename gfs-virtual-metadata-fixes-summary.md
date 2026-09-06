# GFS virtual metadata fixes — PR #1018 (draft)

https://github.com/dynamical-org/reformatters/pull/1018
Branch `gfs-virtual-metadata-fixes` off `origin/main` (c0454df0), 8 commits, one
statement each, both templates regenerated in every commit.

Metadata only: no stored value changes, no change to which source message any chunk
references, so neither store needs re-ingestion. `virtual_template_config.py` is shared
by `noaa-gfs-analysis-virtual` and `noaa-gfs-forecast-virtual`; fix 8 is analysis-only.

## The eight fixes

| Commit | Fix |
|---|---|
| c4d0062c | Convective cloud pressure pair get their own NaN wording — the source does not provide a convective cloud base/top pressure at that cell, which can happen even where `convective_cloud_cover` is nonzero. The eight layer-cloud variables are unchanged. |
| b6df9251 | `haines_index_surface` declares `flag_values=(2,3,4,5,6)` and `flag_meanings="very_low_potential very_low_potential low_potential moderate_potential high_potential"`. |
| 3051a9b3 | `soil_type_surface` drops `standard_name="soil_type"`; no flags added. |
| ccd1e5a8 | `albedo_surface`: exactly 0 where the averaging window received no sunlight, exclude those zeros from a time-mean albedo. `fill_value` stays NaN. |
| e940c10b | The eight soil-moisture variables: near-1 values over permanent land ice are placeholders rather than soil moisture measurements, ending `Mask values >= 0.9.` |
| c1e4334d | `ventilation_rate_planetary_boundary_layer`: precision as coarse as 1,000 m2 s-1, a decoded 0 is rounded down, not missing. Nothing added to `vertical_speed_shear_*`. |
| e365de66 | The two UV-B fluxes drop the disproved 280-315 nm band with no replacement number; they now name the UV-B portion of the downward shortwave flux. |
| c29e531a | The five instantaneous variables the source does not publish at lead 0 say, in the analysis only, that their 00/06/12/18 UTC values come from the previous forecast cycle. |

## Corrections applied after the independent review

- **UV-B**: the 263-345 nm replacement was reverted before the PR; no number is
  asserted, pending the UPP/RRTMG source check.
- **Storm-relative helicity**: the "globally right-moving storm motion solution" comment
  was dropped entirely. The variable carries no comment, as before.
- **Soil moisture**: the comment now carries the mask instruction, worded without
  claiming equivalence to a soil-type-16 mask, without claiming the threshold clips
  nothing, and without claiming no real soil cell in the archive reaches 0.9.

## Implementation notes

Fix 8 is analysis-specific, so it lives in
`src/reformatters/noaa/gfs/analysis_virtual/template_config.py`, not the shared catalog:
`_with_window_comment` became `_with_source_comment`, which prepends the window sentence
to windowed variables as before and the previous-cycle sentence to instantaneous
variables that lack hour-0 values. Those two sets are disjoint, and the second is exactly
the five named variables — pinned by a new test.

## Tests

- `tests/common/datasets_cf_compliance_test.py` — added `soil_type_surface` to
  `ALLOWED_MISSING_STANDARD_NAME` with the reason; deleted the `("soil_type", "1")`
  entry from `CF_UNITS_VARIANCES_ALLOWLIST`, whose premise (a dataset declaring that
  standard name) no longer exists.
- `tests/noaa/gfs/forecast_virtual/template_config_test.py` — updated the verbatim
  `albedo_surface` comment assertion to the composed window + sunlight sentence.
- `tests/noaa/gfs/analysis_virtual/template_config_test.py` — new test pinning the five
  instantaneous variables without hour-0 values and their previous-cycle comment.

No test was deleted or weakened; no test premise was removed.

## Checks run

`uv run ruff format --check`, `uv run ruff check`, `uv run ty check` — pass.
`uv run pytest -m "not slow" -n 4` — 2395 passed.
Slow modules, one invocation each: `tests/noaa/gfs/analysis_virtual/dynamical_dataset_test.py` (5),
`tests/noaa/gfs/analysis_virtual/region_job_test.py` (12),
`tests/noaa/gfs/forecast_virtual/region_job_test.py` (95),
`tests/noaa/gfs/forecast_virtual/dynamical_dataset_test.py` (5) — all pass.

## Deliberately not included (in the PR description)

Owner decisions or value changes: `sunshine_duration_surface` lead selection,
`total_ozone_atmosphere` units, the coverage-vs-coherence contract behind fix 8, the
deep-soil precision change (a Review note, not metadata), and the stale-index guard fix.

Pending source verification: the UV-B wavelength band (280-315 nm removed as disproved,
correct band awaiting the UPP/RRTMG definition) and the storm-relative-helicity
storm-motion convention (awaiting UPP's `CALHEL`).

## Not done

Nothing from the brief is outstanding.
