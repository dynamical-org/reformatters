# Dataset validation

Validate a reformatted dataset by producing and inspecting per-variable plots against a reference dataset. Use this after any backfill, after adding a new variable, or when investigating a regression.

The tool is deliberately visual – view the images. Statistics alone cannot catch many of the most common data quality problems (flipped maps, coordinate-axis errors, quantization banding, unit conversion bugs that only bite at certain lead times).

## 1. Run

```bash
uv run src/scripts/validation/plots.py run-all <DATASET_URL>
```

`DATASET_URL` is the complete, direct URL to the dataset (`bucket-prefix/dataset-id/version`). Examples:

- `s3://dynamical-noaa-hrrr/noaa-hrrr-analysis/v0.1.0.icechunk`
- `s3://dynamical-dwd-icon-eu/dwd-icon-eu-forecast-5-day/v0.2.0.icechunk`

The bucket prefix can be found in `src/reformatters/__main__.py`. The dataset id and version are in the `TemplateConfig.dataset_attributes`.

Expect the run to take ~30–60 seconds per variable, mostly bounded by S3 reads (a ~20-variable dataset finishes in ~10–15 minutes). Progress is logged: one line per variable per plot type.

When the run completes, stdout prints the path of `validation_summary.md` (relative to the repo root). Open that file first.

### Options

- `--reference-url` Reference dataset for side-by-side comparison (default: NOAA GEFS analysis). Change if your dataset is outside the reference's temporal or spatial coverage. Variables not present in the reference still get validation-only plots and stats.
- `--variable <name>` (repeatable, alias `-v`) Restrict to specific variables. Especially useful while iterating on a single new variable — brings total runtime down to under a minute. To compare related variables side-by-side, pass `--variable` multiple times: `-v downward_short_wave_radiation_flux_surface -v total_cloud_cover_atmosphere`.
- `--start-date <YYYY-MM-DD>` / `--end-date <YYYY-MM-DD>` Restrict the append dimension. Useful for reproducing an issue in a specific window.
- `--init-time` / `--lead-time` (forecast) or `--time` (analysis) Pick the exact spatial snapshot instead of a random one. Use this to reproduce a spatial anomaly deterministically.
- `--output-dir <path>` Write into an existing directory instead of a new one (useful if rerunning a single plot type into an existing run dir).

You can also run a single plot type with `compare-spatial`, `compare-timeseries`, or `report-nulls`. The primary entry point `run-all` runs all three of those steps and is the only command that produces the `validation_summary.md` index.

## 2. Output layout

Each run writes to a fresh directory under `data/output/`:

```
data/output/<dataset-id>/<version>_<YYYY-MM-DDTHH-MM>/
├── validation_summary.md           # start here
├── combined_nulls.png              # all variables, one image
├── combined_spatial.png            # all variables, one image
├── combined_temporal.png           # all variables, one image
├── nulls_<var>.png                 # one per variable
├── spatial_<var>.png               # one per variable
├── temporal_<var>.png              # one per variable
└── missing_timestamps.txt          # only if any nulls were detected
```

The directory path itself is dense enough to identify the run: dataset id + version + minute-precision timestamp. Multiple runs of the same dataset group under a shared `<dataset-id>/` parent.

### What each plot type shows

- **`nulls_<var>.png`** — null fraction over time at two spatial points. A flat line at 0 on both panels is the healthy outcome. Catches missing source files, partial writes, or a variable that's silently all-NaN at a subset of timestamps. Missing data is expected for some variables; note the pattern.
- **`spatial_<var>.png`** — 3 panels at one time step: reference map (left), validation map (middle), value distribution histogram (right). If the variable isn't in the reference dataset, the left panel reads "Variable not available" and the histogram plots only the validation distribution. Catches flipped / rotated / mis-projected maps, wrong coordinate extents, unit-scale mismatches (different histograms), and over-quantization (visible as banding).
- **`temporal_<var>.png`** — time series at two spatial points, validation (red) vs reference (blue). If the variable isn't in the reference, only the validation series is plotted. Catches time misalignment, diurnal-cycle phase errors, unit mismatches, trend-level biases, missing/incorrect deaccumulation, and projection errors (uncorrelated / offset timeseries).
- **`combined_*.png`** — every variable stacked into one tall image per plot type. Use these for a first scroll through the run (seeing multiple variables together helps spot cross-variable inconsistencies — e.g. radiation peaking while cloud cover is also high — and is fast to open in a single viewer). The per-variable PNGs are higher-resolution for detailed inspection.

### `validation_summary.md`

The entry point for every run. It contains, in order:

- Validation and reference dataset identity (name, id, version, URL), time ranges, and scope.
- Run parameters (ensemble member, spatial points, chosen init/lead/time for the spatial plot, timeseries period).
- Links to the three `combined_*.png` images.
- Missing-timestamp summary (count, pointer to `missing_timestamps.txt` if any were detected).
- A variables table with nulls at the two points and links to the three per-variable images.
- Per-variable details: metadata (units, long/short/standard name, step type), spatial + temporal min/max/mean for both validation and reference, and nulls.

## 3. Step-by-step inspection

Work in this order.

### 3a. Read `validation_summary.md` entirely

Open `validation_summary.md` first. It provides text-based information which can help identify issues that are harder to view in plots.

- [ ] **Datasets block**: confirm the validation dataset id + version match what you intend to review. Note the reference dataset and its time range — if the reference doesn't cover your validation window, temporal + spatial comparisons will be empty (expected, not a bug).
- [ ] **Run parameters**: confirm the ensemble member (if any), the chosen spatial time (init/lead for forecasts), and the timeseries period. The spatial plot is a single snapshot — if you see something weird, you can pass `--init-time`/`--lead-time`/`--time` to reproduce deterministically.
- [ ] **Missing timestamps**: if any, open `missing_timestamps.txt`. The file lists exact timestamps by (variable, point) and provides a ready-to-paste `--filter-contains` argument string for a targeted backfill retry.
- [ ] **Variables overview table**: scan the null counts. Unexpected non-zero values are your first lead.
- [ ] **Per-variable details**: for each variable, compare validation stats to reference stats. Validation min/max that is orders of magnitude off from the reference is a near-certain unit mismatch.

### 3b. Read every PNG — do not skip this

Statistics miss the visual failure modes. Open the images and walk through the checklist in section 4.

A good working rhythm:

1. Open the three `combined_*.png` files first. Scroll through each to get an overview of all variables together — this quickly surfaces patterns across variables (e.g. radiation peaks coinciding with cloud cover minima) and spots any variable that looks dramatically off relative to its neighbors.
2. For each variable that looked fine in combined or that you want to inspect more closely, open `nulls_<var>.png`, `spatial_<var>.png`, and `temporal_<var>.png`. The per-variable PNGs are higher-resolution. Filenames are consistent, so a pattern like `*_<var>.png` opens all of them at once in most viewers.
3. Cross-check against the variable's row in `validation_summary.md` (units, long_name, stats).
4. Apply the checklist below. Note anomalies as `<variable> + <file> + what's wrong` so they can be acted on.

### 3c. Investigate anomalies

For any anomaly, reproduce it deterministically so a fix can be verified:

- If spatial: rerun `run-all` with `--variable <name> --init-time <t> --lead-time <h>` (forecast) or `--time <t>` (analysis).
- If temporal: the timeseries period is randomized — it's in `validation_summary.md`. Narrow with `--start-date` / `--end-date`.
- If nulls: use the `retry-filter:` line in `missing_timestamps.txt` to backfill just those timestamps.

### 3d. Update `validation_summary.md`

When your review is complete, update `validation_summary.md` with notable findings. Insert a `## Summary` section at the top of the file containing two subsections: `### For further review` (definite and possible issues, each with a link to the image(s) where the issue is apparent) and `### What looks good` (a brief summary).

## 4. Data quality checklist

Look for each of these in every image.

### Geometry and coordinates (from `spatial_<var>.png`)

- [ ] **Map is oriented correctly.** North at the top, east on the right. If the reference map shows a continent and the validation map shows its mirror image or upside-down copy, data and coordinates are out of sync (`pcolormesh` is getting a flipped latitude or longitude axis relative to the data array).
- [ ] **Longitude axis is increasing with east on the right.** If there's a vertical seam in the middle of the map, the data likely uses a 0–360 convention and needs conversion. If features are shifted 180° east/west, the longitude convention is wrong. We expect global datasets to go from −180 to +180 (not 0–360).
- [ ] **Coastlines/land features land at expected coordinates.** Locate a known landmark (e.g. the UK, Italy, the Great Lakes, southern Africa). Verify it sits at the correct lat/lon in the validation map. If it's shifted, the coordinate arrays are offset relative to the data array (off-by-one, wrong origin, or wrong grid spacing).
- [ ] **Spatial extent matches the declared grid.** Domain bounds in the plot should match what the dataset's `TemplateConfig` declares. An extent truncation or overshoot means the coordinates or slicing is wrong.

### Values (from `spatial_<var>.png` histogram and `validation_summary.md` stats)

- [ ] **Validation histogram overlaps the reference histogram** for variables that are also in the reference dataset. A large horizontal offset is a unit or scale bug. A large width difference is a quantization or smoothing bug.
- [ ] **Validation min/max is physically plausible.** Temperature in °C roughly −50 to +50. Pressure in Pa is ~50000–110000. Precipitation rate mm/s is tiny (10⁻⁵). Wind speed m/s is 0–100. Check `units` in `validation_summary.md` vs observed range.
- [ ] **No obvious quantization banding in the spatial map.** Large flat patches of identical values or "staircasing" in smooth gradients indicates `keep_mantissa_bits` is too low.
- [ ] **No suspicious sentinel values showing through.** Values like 9999 / -9999 / 32767 / 1e20 appearing as a color extreme mean a source nodata value was not translated to NaN.

### Time alignment (from `temporal_<var>.png`)

- [ ] **Diurnal cycle is in phase with the reference.** Shortwave radiation, 2m temperature, and similar variables should peak at the same local hour as the reference. Phase-shift is a time-coordinate bug (e.g. UTC vs local, or off-by-one lead time).
- [ ] **Trend magnitudes match.** The validation and reference should have similar day-to-day ranges. A consistent offset points to a calibration or unit issue; a consistent scale difference points to a unit conversion.
- [ ] **No unexpected flatlines or spikes.** Flatlines at one value for many steps can be a read error or a stuck sentinel; isolated spikes can be unit bugs at specific lead times (common in accumulated variables).
- [ ] **Accumulated variables reset as expected.** Precipitation and radiation accumulators should typically reset each forecast — check the `step_type` in `validation_summary.md` and confirm the shape.

### Missing data (from `nulls_<var>.png` and `missing_timestamps.txt`)

- [ ] **Null fraction is 0 or explained.** Any non-zero null fraction should have a reason: source data unavailable before a date for a specific variable, known source outage, ocean point for a land-only variable. Unexplained nulls are the bug.
- [ ] **Missing pattern is not structural.** Nulls concentrated at specific lead times, specific hours of day, or specific forecast cycles suggest a processing or indexing bug, not a random source outage.

Note on `step_type` ≠ `instant` variables (accumulation / average / max): the first lead time of each forecast is structurally NaN (there is no prior window to accumulate/average over). The tool excludes that slice from the null count and from `missing_timestamps.txt`, so a "0 / N nulls — none" line for an accumulated variable means *no unexpected* nulls — the structural analysis-step NaN is not counted.

For a sampling of unexplained nulls, go manually fetch source data files and inspect them. Do they have the data we expect, or is the data really not available at the source?

### Cross-dataset checks

- [ ] **Reference availability.** If the reference dataset doesn't cover your window, `validation_summary.md` will show `variable not available in reference dataset` — that is not a bug in the validation dataset, but spatial / temporal comparisons lose their signal. Consider a different reference or re-run on a time range the reference covers.
- [ ] **Ensemble member is plausible.** For ensemble datasets `validation_summary.md` records the randomly-selected member. Rerun once more to confirm that a different member also looks right.
