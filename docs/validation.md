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

To look up the URLs for a dataset, run `uv run main <dataset-id> dataset-urls`. It prints the primary and replica URLs; pass `--format json` for a machine-readable form.

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
└── unavailable_timestamps.txt      # only if any nulls were detected
```

The directory path itself is dense enough to identify the run: dataset id + version + minute-precision timestamp. Multiple runs of the same dataset group under a shared `<dataset-id>/` parent.

### What each plot type shows

- **`nulls_<var>.png`** — null fraction over time at two spatial points. A flat line at 0 on both panels is the healthy outcome. Catches missing source files, partial writes, or a variable that's silently all-NaN at a subset of timestamps. Unavailable data is expected for some variables; note the pattern.
- **`spatial_<var>.png`** — 3 panels at one time step: reference map (left), validation map (middle), value distribution histogram (right). If the variable isn't in the reference dataset, the left panel reads "Variable not available" and the histogram plots only the validation distribution. Catches flipped / rotated / mis-projected maps, wrong coordinate extents, unit-scale mismatches (different histograms), and over-quantization (visible as banding).
- **`temporal_<var>.png`** — time series at two spatial points, validation (red) vs reference (blue). If the variable isn't in the reference, only the validation series is plotted. Catches time misalignment, diurnal-cycle phase errors, unit mismatches, trend-level biases, missing/incorrect deaccumulation, and projection errors (uncorrelated / offset timeseries).
- **`combined_*.png`** — every variable stacked into one tall image per plot type. Use these for a first scroll through the run (seeing multiple variables together helps spot cross-variable inconsistencies — e.g. radiation peaking while cloud cover is also high — and is fast to open in a single viewer). The per-variable PNGs are higher-resolution for detailed inspection.

### `validation_summary.md`

The entry point for every run. It contains, in order:

- Validation and reference dataset identity (name, id, version, URL), time ranges, and scope.
- Run parameters (ensemble member, spatial points, chosen init/lead/time for the spatial plot, timeseries period).
- Links to the three `combined_*.png` images.
- Unavailable-timestamp summary (count, earliest/latest unavailable per (variable, point), pointer to `unavailable_timestamps.txt` if any were detected).
- A variables table with nulls at the two points and links to the three per-variable images.
- Per-variable details: metadata (units, long/short/standard name, step type), spatial + temporal min/max/mean for both validation and reference, and nulls.

## 3. Step-by-step inspection

Work in this order.

### 3a. Read `validation_summary.md` entirely

Open `validation_summary.md` first. It provides text-based information which can help identify issues that are harder to view in plots.

- [ ] **Datasets block**: confirm the validation dataset id + version match what you intend to review. Note the reference dataset and its time range — if the reference doesn't cover your validation window, temporal + spatial comparisons will be empty (expected, not a bug).
- [ ] **Run parameters**: confirm the ensemble member (if any), the chosen spatial time (init/lead for forecasts), and the timeseries period. The spatial plot is a single snapshot — if you see something weird, you can pass `--init-time`/`--lead-time`/`--time` to reproduce deterministically.
- [ ] **Unavailable timestamps**: if any, open `unavailable_timestamps.txt`. The file lists exact timestamps by (variable, point) and provides a ready-to-paste `--filter-contains` argument string for a targeted backfill retry. The summary table in `validation_summary.md` also shows the earliest and latest unavailable timestamp per (variable, point), making it easy to spot common patterns across variables about when data is (un)available.
- [ ] **Variables overview table**: scan the null counts. Unexpected non-zero values are your first lead.
- [ ] **Per-variable details**: for each variable, compare validation stats to reference stats. Validation min/max that is orders of magnitude off from the reference is a near-certain unit mismatch.

### 3b. Read every PNG — do not skip this

Statistics miss the visual failure modes. Open the images and walk through the checklist in section 4.

Every per-variable PNG must be reviewed: for each variable, open all three of `nulls_<var>.png`, `spatial_<var>.png`, and `temporal_<var>.png`. Do not stop after a representative sample — issues can be variable-specific (a unit bug at one level, a flipped map for one field) and only surface when every plot is checked. The only plots you may skip are the three `combined_*.png` files, and only as noted below.

A good working rhythm:

1. If you are a human, open the three `combined_*.png` files first. Scroll through each to get an overview of all variables together — this quickly surfaces patterns across variables (e.g. radiation peaks coinciding with cloud cover minima) and spots any variable that looks dramatically off relative to its neighbors. Skip this step if you are an AI assistant, the `combined_*.png` image pixel dimensions exceeds standard limits and attempting to read them can blow up your context or stall the session.
2. Open `nulls_<var>.png`, `spatial_<var>.png`, and `temporal_<var>.png` for **every** variable in the dataset — not a sample. The per-variable PNGs are higher-resolution. Filenames are consistent, so a pattern like `*_<var>.png` opens all three at once in most viewers.
3. Cross-check against the variable's row in `validation_summary.md` (units, long_name, stats).
4. Apply the checklist below. Note anomalies as `<variable> + <file> + what's wrong` so they can be acted on.

### 3c. Investigate anomalies

For any anomaly, reproduce it deterministically so a fix can be verified:

- If spatial: rerun `run-all` with `--variable <name> --init-time <t> --lead-time <h>` (forecast) or `--time <t>` (analysis).
- If temporal: the timeseries period is randomized — it's in `validation_summary.md`. Narrow with `--start-date` / `--end-date`.
- If nulls: use the `retry-filter:` line in `unavailable_timestamps.txt` to backfill just those timestamps.

### 3d. Update `validation_summary.md`

When your review is complete, update `validation_summary.md` with notable findings. Insert a `## Summary` section at the top of the file containing two subsections: `### For further review` (definite and possible issues, each with a link to the image(s) where the issue is apparent) and `### What looks good` (a brief summary).

## 4. Data quality checklist

Look for each of these in every image.

### Geometry and coordinates (from `spatial_<var>.png`)

- [ ] **Map is oriented correctly.** North at the top, east on the right. If the reference map shows a continent and the validation map shows its mirror image or upside-down copy, data and coordinates are out of sync (`pcolormesh` is getting a flipped latitude or longitude axis relative to the data array).
- [ ] **Longitude axis is increasing with east on the right.** If there's a vertical seam in the middle of the map, the data likely uses a 0–360 convention and needs conversion. If features are shifted 180° east/west, the longitude convention is wrong. We expect global datasets to go from −180 to +180 (not 0–360).
- [ ] **Coastlines/land features land at expected coordinates.** Locate a known landmark (e.g. the UK, Italy, the Great Lakes, southern Africa). Verify it sits at the correct lat/lon in the validation map. If it's shifted, the coordinate arrays are offset relative to the data array (off-by-one, wrong origin, or wrong grid spacing).
- [ ] **Spatial extent matches the declared grid.** Domain bounds in the plot should match what the dataset's `TemplateConfig` declares. An extent truncation or overshoot means the coordinates or slicing is wrong.

### Spatial plots (from `spatial_<var>.png` histogram and `validation_summary.md` stats)

- [ ] **Validation histogram overlaps the reference histogram** for variables that are also in the reference dataset. A large horizontal offset is a unit or scale bug. A large width difference is a quantization or smoothing bug.
- [ ] **Validation min/max is physically plausible.** Temperature in °C roughly −50 to +50. Pressure in Pa is ~50000–110000. Precipitation rate mm/s is tiny (10⁻⁵). Wind speed m/s is 0–100. Check `units` in `validation_summary.md` vs observed range.
- [ ] **No obvious quantization banding in the spatial map.** Large flat patches of identical values or "staircasing" in smooth gradients indicates `keep_mantissa_bits` is too low.
- [ ] **No suspicious sentinel values showing through.** Values like 9999 / -9999 / 32767 / 1e20 appearing as a color extreme mean a source nodata value was not translated to NaN.
- [ ] **Whole plot matches meteorological expectations.** Look closely for subtly or obviously wrong new types of problems not enumerated here. Visual plots are a ley layer of our defense in depth approach to catching data quality issues. We can't list every possible issue, rather use your meterological knowlege to first define what you expect to see and compare that to what you actually see in the plots.

### Time series plots (from `temporal_<var>.png`)

- [ ] **Diurnal cycle is in phase with the reference.** Shortwave radiation, 2m temperature, and similar variables should peak at the same local hour as the reference. Phase-shift is a time-coordinate bug (e.g. UTC vs local, or off-by-one lead time).
- [ ] **Trend magnitudes match.** The validation and reference should have similar day-to-day ranges. A consistent offset points to a calibration or unit issue; a consistent scale difference points to a unit conversion.
- [ ] **No unexpected flatlines or spikes.** Flatlines at one value for many steps can be a read error or a stuck sentinel; isolated spikes can be unit bugs at specific lead times (common in accumulated variables).
- [ ] **Accumulated variables reset as expected.** Precipitation and radiation accumulators should typically reset each forecast — check the `step_type` in `validation_summary.md` and confirm the shape.
- [ ] **No obvious quantization in time series.** Time series which are snapped or binned to a limited set of values or "staircasing" in what should be smooth time series indicates `keep_mantissa_bits` is too low.
- [ ] **Whole plot matches meteorological expectations.** Look closely for subtly or obviously wrong new types of problems not enumerated here. Visual plots are a ley layer of our defense in depth approach to catching data quality issues. We can't list every possible issue, rather use your meterological knowlege to first define what you expect to see and compare that to what you actually see in the plots.

### Unavailable data (from `nulls_<var>.png` and `unavailable_timestamps.txt`)

- [ ] **Null fraction is 0 or explained.** Any non-zero null fraction should have a reason: source data unavailable before a date for a specific variable, known source outage, ocean point for a land-only variable. Unexplained nulls are the bug.
- [ ] **Unavailable pattern is not structural.** Nulls concentrated at specific lead times, specific hours of day, or specific forecast cycles suggest a processing or indexing bug, not a random source outage. Use the earliest/latest unavailable columns in the summary table to spot patterns shared across variables (e.g. a consistent earliest-unavailable date points to source coverage starting later).

Note on `step_type` ≠ `instant` variables (accumulation / average / max): the first lead time of each forecast is structurally NaN (there is no prior window to accumulate/average over). The tool excludes that slice from the null count and from `unavailable_timestamps.txt`, so a "0 / N nulls — none" line for an accumulated variable means *no unexpected* nulls — the structural analysis-step NaN is not counted.

For a sampling of unexplained nulls, go manually fetch source data files and inspect them. Do they have the data we expect, or is the data really unavailable at the source?

### Cross-dataset checks

- [ ] **Reference availability.** If the reference dataset doesn't cover your window, `validation_summary.md` will show `variable not available in reference dataset` — that is not a bug in the validation dataset, but spatial / temporal comparisons lose their signal. Consider a different reference or re-run on a time range the reference covers.
- [ ] **Ensemble member is plausible.** For ensemble datasets `validation_summary.md` records the randomly-selected member. Rerun once more to confirm that a different member also looks right.

## 5. Publishing the report

Once a run is reviewed, render it to a static HTML report and publish it. Two paths exist:

- **Draft** — every render goes here, timestamped, kept forever. Use for sharing a single run for review without committing to it.
- **Stable** — the canonical report for a dataset, overwritten by each new publish. Embedded in the dynamical-stac catalog and linked from dynamical.org.

Both paths live in the `dataset-validation-reports` Cloudflare R2 bucket, served publicly at `https://dataset-validation-reports.dynamical.org`. Drafts and previously-published reports are archived forever — only the file at the stable path is overwritten.

### 5a. Render the HTML report

```bash
uv run src/scripts/validation/plots.py render-report <run-dir>
```

Reads `<run-dir>/validation_summary.md` and writes `<run-dir>/validation_report.html` next to it. Self-contained HTML (inline CSS/JS, no build step), images referenced at relative paths so the rendered file works locally and after upload.

The HTML mirrors the markdown 1:1 with two viewing affordances:

- **Left TOC** (slide-out hamburger on mobile) lists the top-level sections plus a Variables group with a checkbox per variable. "All / none" toggles let you compare a subset of variables side-by-side. Each variable section has `id="var-<name>"` so URLs can deep-link (e.g. `validation_report.html#var-temperature_2m`).
- **Images** are full-width on mobile and wrapped in a link to the underlying file so tap/click opens the full-resolution image.

`combined_*.png` images are linked from the "Combined plots" section (not inlined — they're large).

`upload` re-renders before uploading, so this command is only needed for local-only previews.

### 5b. Upload — drafts and publish

```bash
uv run src/scripts/validation/plots.py upload <run-dir>             # draft
uv run src/scripts/validation/plots.py upload <run-dir> --publish   # publish to stable + archive
```

`upload` re-renders the HTML, then uploads the entire run directory (`validation_summary.md`, all `*.png`, `unavailable_timestamps.txt`, `validation_report.html`) to R2.

Without `--publish`:

```
<dataset-id>/drafts/<version>_<YYYY-MM-DDTHH-MM>/
```

Use this to share a single run for review without committing to it. Drafts are kept forever; iterate by re-running `run-all` (a new timestamped run dir) and re-uploading.

With `--publish`:

```
<dataset-id>/latest/                                  # stable, overwritten each publish
<dataset-id>/published/<version>_<YYYY-MM-DDTHH-MM>/  # archive copy, kept forever
```

The stable path is what the dynamical-stac catalog links to. The archive copy preserves what `latest/` was before this publish; previously published reports are never lost.

Prints the public URL of `validation_report.html` on completion (e.g. `https://dataset-validation-reports.dynamical.org/<dataset-id>/latest/validation_report.html`).

### 5c. Update the summary in place before publishing

Per [3d](#3d-update-validation_summarymd), the run already has a `## Summary` block at the top of `validation_summary.md` written during review. Before running `upload --publish`, edit that block in place: drop `### For further review` items you've followed up on, add notes about specific known issues, and update `### What looks good` if your view has changed. `upload` re-renders the HTML automatically.

Before publishing a non-draft report, rewrite the `## Summary` text for a public dataset consumer audience. Drafts can use internal shorthand for fast iteration, but the published report is read by external dataset users. Spell out variable names, expand acronyms, and avoid internal jargon such as "P1"/"P2" (use the explicit lat/lon or describe the point), ticket numbers, internal codenames, or process shorthand. Each item should make sense to someone who has never seen the run directory or our review process.

### 5d. Wire into dynamical-stac

After the first publish for a dataset, add the report URL to the catalog so it surfaces on dynamical.org. In the `dynamical-org/dynamical-stac` repo:

1. Add `validation_report_href` to the relevant `CatalogItem` in `src/catalog.py`. Value is the `latest/` URL.
2. The STAC generator surfaces it as an asset with role `validation-report` (type `text/html`).
3. Run `./scripts/generate` and commit `stac/`.

This is a one-time wiring per dataset. Subsequent `publish-stable` runs update the report contents at the same URL — no STAC change needed.

### Configuration

`publish-draft` and `publish-stable` read R2 credentials from these environment variables:

- `R2_VALIDATION_REPORTS_ENDPOINT_URL`
- `R2_VALIDATION_REPORTS_ACCESS_KEY_ID`
- `R2_VALIDATION_REPORTS_SECRET_ACCESS_KEY`

Scoped to the `dataset-validation-reports` bucket. Set them in the environment before running the publish commands.
