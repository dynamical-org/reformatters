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

Expect the run to take ~30–60 seconds per variable, mostly bounded by S3 reads (a ~20-variable dataset finishes in ~10–15 minutes). Progress is logged: one line per variable per plot type. Virtual stores are slower and their runtime scales differently — see [Virtual and multi-level datasets](#virtual-and-multi-level-datasets) below.

Long runs and remote sessions: a full virtual `run-all` takes on the order of an hour or two (see below), and `upload` of a many-variable run pushes hundreds of images. On a managed/remote session (e.g. Claude Code on the web), launch `run-all` detached in the background with a monitor that periodically emits a progress line (this keeps the session from being reclaimed mid-run) and that signals both completion (`validation_summary.md` written) and failure (the run process exited without writing it), so a crash surfaces immediately instead of looking like "still running". Budget ~6–8 GB RAM: a full virtual run-all holds several GB of decoded fields.

When the run completes, stdout prints the path of `validation_summary.md` (relative to the repo root). Open that file first.

### Options

- `--reference-url` Reference dataset for side-by-side comparison. Defaults to the NOAA GEFS analysis icechunk asset URL discovered from the STAC catalog at `https://stac.dynamical.org/catalog.json`. Change if your dataset is outside the reference's temporal or spatial coverage. Variables not present in the reference still get validation-only plots and stats.
- `--variable <name>` (repeatable, alias `-v`) Restrict to specific variables. Especially useful while iterating on a single new variable — brings total runtime down to under a minute. To compare related variables side-by-side, pass `--variable` multiple times: `-v downward_short_wave_radiation_flux_surface -v total_cloud_cover_atmosphere`.
- `--start-date <YYYY-MM-DD>` / `--end-date <YYYY-MM-DD>` Restrict the append dimension. Useful for reproducing an issue in a specific window.
- `--init-time` / `--lead-time` (forecast) or `--time` (analysis) Pick the exact spatial snapshot instead of a random one. Use this to reproduce a spatial anomaly deterministically.
- `--output-dir <path>` Write into an existing directory instead of a new one (useful if rerunning a single plot type into an existing run dir).
- `--level <value>` For datasets with vertical-level groups, the level value to sample (e.g. `500` for 500 mb). Selects the nearest level on each variable's own vertical dim; the default is the middle level. Single-level variables ignore it.
- `--point <lat,lon>` (repeatable, up to 2) Pin the two spatial points used by the value and time series plots to specific coordinates instead of the random default; the nearest grid cell is used. Pick points that will provide clear validation signal, for example one point on land and another in the ocean for a global dataset, or two spaced-apart but on-land points for a regional dataset. Avoid points very near the poles and avoid points where there are expected geographic coverage gaps. The unpinned random default already draws both points from the middle 50% of each spatial dimension (avoiding the outer-25% margins), but pinning is the deterministic choice for a published report. Also honored by the standalone `value-timeseries` and `compare-timeseries` commands so a verification re-render can reuse the same points.

You can also run a single plot type with `compare-spatial`, `compare-timeseries`, `availability`, or `value-timeseries`. The primary entry point `run-all` runs all of those steps and is the only command that produces the `validation_summary.md` index.

## Virtual and multi-level datasets

Two structural differences change how these tools behave; both are detected automatically from the store URL, but it helps to know what's happening.

### Virtual stores invert the read-cost model

A [virtual dataset](virtual_datasets.md) stores chunks as references into source GRIB files, decoding on read. One chunk is one GRIB message — a whole spatial field at a single (init, lead, member, level). So the cost model is the **opposite** of a materialized store: a spatial snapshot is cheap (one decode), while a point time series across the append dimension is catastrophic (one decode per position — millions for a full archive). `run-all` adapts:

- **Availability is manifest-probed, not value-scanned.** On a materialized store `availability` reads `.isnull()` at two points across all time. On a virtual store that would decode every chunk it touches, and presence is anyway a *manifest* question (does the reference exist?), not a value question. So `run-all` (and the standalone `availability` command) runs the manifest scan instead — exhaustive per source file and per variable across the whole archive (see `manifest_scan` below) — and renders the same availability artifacts either way.
- **`value-timeseries` is sampled.** Instead of reducing mean ± std over lead × member at every position (which decodes the whole column), it strides the append dimension and decodes **one message per sampled position**, pinning a representative (lead, member, level) slice so the whole-period series is a single, comparable slice. A chunk is a full spatial field, so both run points are read from the same decode. This still surfaces drift and unit step-changes; per-variable loads run a few ahead of the plotting loop. The summary marks the series "sampled" and records the pinned lead/member/level.
- **Spatial and time series comparisons are unchanged** — both already scope to a single snapshot / single init, the cheap access pattern.

Note that spatial downsampling (`_downsample_for_plot`) only shrinks the plot, not the read: gribberish decodes the full message regardless.

Virtual runs decode many full-field messages, so scope a run with `--variable`/`-v` or `--start-date`/`--end-date` when you don't need the whole archive. Each variable's decode count is logged before its read starts.

Runtime scales roughly linearly with variable count: budget **~0.4 minutes per variable end-to-end** for a full virtual run-all (measured: 176 variables in ~71 minutes on the HRRR virtual store, including the concurrent whole-archive manifest scan and the sampled decode scan). So a ~150-variable virtual dataset takes **~1–2 hours**, depending on archive length (the manifest scan cost grows with the number of expected source files) and S3 latency. Scope with `-v`/`--start-date` while iterating to bring this down to well under a minute.

### Whole-archive scans (manifest completeness, decode health)

These are the thorough, post-backfill correctness pass for a virtual dataset, distinct from the cheap operational `-validate` checks. Both reach the dataset's own region job (for expected source files and manifest chunk-key logic) by resolving the registered dataset from the store's `dataset_id` attribute, so both take a store URL like every other command. `run-all` runs both automatically for a virtual store, so one command produces the complete report — the manifest scan runs alongside the decode + plot phases in the same process, but they contend for I/O, so its cost is not fully hidden; the standalone commands are the strict post-backfill gates (exit code).

```bash
uv run src/scripts/validation/plots.py availability <DATASET_URL> [--start-date <date>] [--end-date <date>] [--min-fraction 1.0]
uv run src/scripts/validation/plots.py decode-scan <DATASET_URL> [--start-date <date>] [--end-date <date>] [--max-samples 20]
```

- **manifest scan** (`availability` on a virtual store) — the strict completeness gate, probing ref existence (no decode) job by job with two measures — the scan streams, folding each region job's probes as they complete, so peak memory stays flat regardless of window length and a whole multi-year archive can be scanned on a modest host. **Per source file**: every expected file's representative ref; the one-file-one-commit invariant ([virtual_datasets.md](virtual_datasets.md)) extends that probe to every ref the file contributed, so a file that never ingested (any lead / member / level-group file) is exactly the gap this catches. Any incomplete position is listed in `unavailable_timestamps.txt` to target a backfill. **Per variable**: each variable's ref at one present source file per position (smallest nonzero lead; middle level for group vars), which catches a variable missing from otherwise-ingested files — e.g. a variable the model only started producing partway through the archive shows as a 0→1 availability step. What neither measure catches is a level missing **within** an ingested file at an unprobed level; only the decode scan's level sampling / NaN checks can surface that. Expected files respect the store's committed `expected_forecast_length` coordinate when the dataset has one, so leads that never existed upstream (e.g. a 36-hour-era init in a 48-hour archive) are not flagged. Exits non-zero if any position is below `--min-fraction` of expected source files. The scan window ends at the store's committed extent, so not-yet-published positions are never flagged; narrow further with `--start-date`/`--end-date`.
- **decode scan** (`decode-scan`) — sampled decode health. Samples evenly spaced append-dim regions (every variable group at each sampled region), decodes a bounded sample of present references — across lead times, members, and levels — and fails if any sampled chunk errors or decodes entirely NaN. Results render as the "Decode health (sampled)" section of `validation_summary.md` (the standalone command also writes `decode_scan_summary.md`). **This is a sample, not an exhaustive sweep**: a reference that decodes to garbage outside the sample is not caught (a literal every-chunk decode across all leads × members × levels is hours).

The guarantee split is the thing to keep straight: **completeness is exhaustive at file granularity and at (variable, position) granularity**, **decode health is sampled**. "Validated" therefore means every expected source file was ingested, every variable has a ref at every probed position, and a representative sample decodes cleanly — not that every value in the archive was decoded and checked. State this in the published summary so users read it correctly.

### Vertical-level groups

A dataset with vertical groups (e.g. `pressure_level`, `model_level`) exposes group variables by store path, like `pressure_level/temperature`, each carrying its level dimension. The plots **sample one representative level per variable** (the middle level by default), recorded in the summary and the plot title — the same philosophy as sampling one ensemble member and one spatial snapshot. Use `--level <value>` to inspect a specific level deterministically. Different groups have different level dims and lengths; each variable is sampled on its own dim. The manifest scan's file pass is the exception — it covers every source file (and, through the per-file atomic commit, every level those files contributed); its per-variable pass probes the middle level, matching the plots' sampling.

Group variables are addressed by their store path: `-v pressure_level/temperature`, not `-v temperature`.

## 2. Output layout

Each run writes to a fresh directory under `data/output/`:

```
data/output/<dataset-id>/<version>_<YYYY-MM-DDTHH-MM>/
├── validation_summary.md           # start here
├── availability_heatmap.png        # all variables × append dim, one image
├── availability_<var>.png          # one per variable
├── value_timeseries_<var>.png      # one per variable
├── spatial_<var>.png               # one per variable
├── temporal_<var>.png              # one per variable
└── unavailable_timestamps.txt      # only if any data is missing
```

The directory path itself is dense enough to identify the run: dataset id + version + minute-precision timestamp. Multiple runs of the same dataset group under a shared `<dataset-id>/` parent.

### What each plot type shows

- **`availability_heatmap.png`** — every variable × the full append dim, colored by fraction of data available (green = available, red = missing, grey = not probed because no source file is present at that position). This is the availability overview: era boundaries (a variable that starts partway through the archive), whole-position gaps shared across variables, and per-variable holes are all visible in one image. Availability is manifest-probed on a virtual store (exhaustive) and value-scanned at the two run points on a materialized store.
- **`availability_<var>.png`** — the same series as one trace, one per variable. Reviewers may skip it for a variable whose summary line reads complete; it is always written so every variable's section renders consistently.
- **`value_timeseries_<var>.png`** — the variable's value over the **entire** dataset time range at two spatial points: the per-timestep mean (line, left y-axis). On a materialized forecast / ensemble dataset the per-timestep standard deviation across the non-time dims (lead time, ensemble member) is drawn as a second, lighter line on a secondary right y-axis, with a `mean` / `std dev` legend. It is omitted (mean line only, no second axis or legend) when there is nothing to spread over: analysis datasets (a single value per timestep) and virtual stores (the series is a single pinned lead/member/level slice, so each timestep is one value). On a materialized store this reuses the point data already loaded for the availability scan; a virtual store does its own sampled read. Unlike `temporal_<var>.png` (a short window vs the reference), this spans all time and is meant to surface slow drifts and sudden discontinuities — e.g. a source that begins emitting a variable in different units shows up as a sharp step change in the mean line; where a std line is present, a distribution change shows up as a step change in it.
- **`spatial_<var>.png`** — 3 panels at one time step: reference map (left), validation map (middle), value distribution histogram (right). If the variable isn't in the reference dataset, the left panel reads "Variable not available" and the histogram plots only the validation distribution. Catches flipped / rotated / mis-projected maps, wrong coordinate extents, unit-scale mismatches (different histograms), and over-quantization (visible as banding).
- **`temporal_<var>.png`** — time series at two spatial points, validation (red) vs reference (blue). If the variable isn't in the reference, only the validation series is plotted. Catches time misalignment, diurnal-cycle phase errors, unit mismatches, trend-level biases, missing/incorrect deaccumulation, and projection errors (uncorrelated / offset timeseries).

Plots are per-variable; `availability_heatmap.png` is the only all-variable image. Compare variables against each other by opening several `*_<var>.png` of the same type together.

### `validation_summary.md`

The entry point for every run. It contains, in order:

- Validation and reference dataset identity (name, id, version, URL), time ranges, and scope.
- Run parameters: spatial points used by the value scans (all ensemble members), and for each of the spatial and time series sections the picked ensemble member plus the chosen init/lead/time (spatial) or timeseries period.
- Availability section: how availability was measured, the heatmap, a table of incomplete variables (complete/total positions, first/last incomplete), and a link to the unavailable-timestamps file (`unavailable_timestamps.txt`).
- Per-variable details: metadata (units, long/short/standard name, step type), full-period value min/mean/std/max at each point, spatial + temporal min/max/mean for both validation and reference, and an availability line (positions complete, plus null counts at the two points on materialized stores).

## 3. Step-by-step inspection

Work in this order. AI assistants reviewing a dataset with more than ~25 variables must use the batched process in [3f](#3f-batched-review-for-many-variable-datasets-ai-assistants) — a complete-variables dataset's plots do not fit in one agent context — with 3a-3e as the per-batch and aggregation steps.

### 3a. Read `validation_summary.md` entirely

Open `validation_summary.md` first. It provides text-based information which can help identify issues that are harder to view in plots.

- [ ] **Datasets block**: confirm the validation dataset id + version match what you intend to review. Note the reference dataset and its time range — if the reference doesn't cover your validation window, temporal + spatial comparisons will be empty (expected, not a bug).
- [ ] **Run parameters**: confirm the ensemble member recorded under each of the spatial and time series subsections (if the dataset has ensembles — the same member is used for both, and the null analysis intentionally runs over all members), the chosen spatial time (init/lead for forecasts), and the timeseries period. The spatial plot is a single snapshot — if you see something weird, you can pass `--init-time`/`--lead-time`/`--time` to reproduce deterministically.
- [ ] **Availability**: read the method line, the heatmap, and the incomplete-variables table. Any incomplete variable is your first lead. Open `unavailable_timestamps.txt` — it lists the positions missing source files, to target a backfill. Use the first/last incomplete columns to spot patterns shared across variables (e.g. a shared first-incomplete date points to source coverage starting later).
- [ ] **Per-variable details**: for each variable, compare validation stats to reference stats. Validation min/max that is orders of magnitude off from the reference is a near-certain unit mismatch.

### 3b. Read every PNG — do not skip this

Statistics miss the visual failure modes. Open the images and walk through the checklist in section 4.

Every per-variable PNG must be reviewed: for each variable, open `value_timeseries_<var>.png`, `spatial_<var>.png`, and `temporal_<var>.png`; open `availability_<var>.png` too unless the variable's summary line reads complete. Do not stop after a representative sample — issues can be variable-specific (a unit bug at one level, a flipped map for one field) and only surface when every plot is checked.

A good working rhythm:

1. Open `availability_heatmap.png` first for the all-variable availability overview (era boundaries, shared gaps) — the only all-variable image. Per-variable comparison happens in step 2.
2. Open `value_timeseries_<var>.png`, `spatial_<var>.png`, `temporal_<var>.png` (plus `availability_<var>.png` for any incomplete variable) for **every** variable in the dataset — not a sample. Filenames are consistent, so a pattern like `*_<var>.png` opens all of them at once in most viewers; opening a family together surfaces cross-variable patterns (e.g. radiation peaks coinciding with cloud cover minima).
3. Cross-check against the variable's row in `validation_summary.md` (units, long_name, stats).
4. Apply the checklist below. Note anomalies as `<variable> + <file> + what's wrong` so they can be acted on.

### 3c. Investigate anomalies

For any anomaly, reproduce it deterministically so a fix can be verified:

- If spatial: rerun `run-all` with `--variable <name> --init-time <t> --lead-time <h>` (forecast) or `--time <t>` (analysis).
- If temporal: the timeseries period is randomized — it's in `validation_summary.md`. Narrow with `--start-date` / `--end-date`.
- If a discontinuity in `value_timeseries_<var>.png`: narrow the full-period plot to the transition with `value-timeseries <DATASET_URL> --variable <name> --start-date / --end-date`, then fetch the source files on either side of the step to confirm a unit/scale change at the source.
- If availability gaps: use the positions listed in `unavailable_timestamps.txt` to backfill just those timestamps.

### 3d. Dig into each follow-up item

Before writing the summary, work through every item you flagged during the review and gather enough evidence to either confirm it as an issue, downgrade it to a documented quirk, or resolve it. The goal is that nothing reaches `### For further review` without you having looked at it twice. Use the single-step entry points (`compare-spatial`, `compare-timeseries`, `availability`) rather than re-running the full `run-all` — they're faster and let you target the exact slice that's in question. Each follow-up category below has a required verification step — completing it is what lets you resolve the item or move it from `### For further review` to `### Review notes`.

**If you can name a verification, you must run it.** Any time you can write down a specific command, time range, lat/lon, or alternate snapshot that would confirm or rule out an item, run it yourself before stopping. The single-step tools are cheap (under a minute per re-render) and exist for this. Conclusions like "worth re-running on another snapshot", "should re-confirm against the raw GRIB", or "warrants checking at a 3-hourly slot" are unfinished verifications — they must be executed and the bullet updated with what the verification revealed, not left as a to-do for the next reader.

- **Rerun a single plot type targeted at the variable, time, and location in question** to confirm an anomaly is real and not an artifact of the randomly chosen snapshot or timeseries window. For example, `uv run src/scripts/validation/plots.py compare-spatial <DATASET_URL> --variable <name> --time <t> --output-dir <run-dir>` to re-render one spatial plot at a chosen time, or `compare-timeseries` with `--start-date` / `--end-date` and the same lat/lon as the original run to zoom in on a temporal anomaly. Re-rendering into the existing `--output-dir` overwrites the original PNG so the summary's links stay valid.
  - **Gotcha — a `-v`-filtered re-render shrinks `availability_heatmap.png` and desyncs the summary stats.** A single-step command rebuilds `availability_heatmap.png` from **only** the variables passed in that invocation, and only `run-all` writes `validation_summary.md` — so a subset re-render into the run dir leaves the heatmap and the summary's stats stale. To fix a report whose random snapshot missed part-of-archive variables, re-run the **full** `run-all` with the snapshot pinned (`--init-time` / `--lead-time` for forecasts, `--time` for analyses, or `--start-date` / `--end-date`) to a date where every variable is present. Reserve `-v`-filtered single-step runs for throwaway investigation in a scratch `--output-dir`, not for updating the report you intend to publish.
- **For unexpected unavailable timesteps, you must fetch a representative sample of the unavailable timestamps from the upstream archive before attributing the gap to an upstream cause** — outages happen, but so do ingestion errors, and an LLM or a hurried reviewer will tend to default to "outage" without checking. A single spot-check is not enough: sample across the affected init cycles and lead times so a per-file bug (like a stale sidecar on individual leads) doesn't get mistaken for a clean outage. Verify both the source data file (e.g. the GRIB) **and** any sidecar the reformatter depends on (e.g. the `.idx` byte index for GRIB-based datasets). Compare what you find against what the reformatted archive has at the same timestamp to determine whether to retry the backfill (the positions listed in `unavailable_timestamps.txt`) or document the gap as a confirmed source outage with the URL(s) you checked.
- **For suspected unit, scale, or coordinate bugs**, cross-check against a third independent source (the raw GRIB/NetCDF file, a public viewer such as NOAA's nowCOAST or ECMWF's open charts, or another reformatted archive) — the GEFS reference is convenient but it's only one comparison and shares some biases with GFS-derived datasets.
- **For ensemble datasets**, rerun once more without `--variable` filters so a different ensemble member is selected; an anomaly that only appears for one member is structurally different from one that appears across members.

Track what you found per item so the eventual `### For further review` entries cite the evidence (filename, timestamp, source URL) rather than just describing what you saw in the original snapshot.

### 3e. Update `validation_summary.md`

When your review is complete, insert a `## Summary` section into `validation_summary.md`, placed immediately below the `Report generation start time: …` line and directly above the `## Datasets` section (i.e. right after the report's introductory paragraph and timestamp, not above them). This keeps the report's identifying header — the intro sentence and generation time — first, with your summary as the first content section that follows. This section is the part of the report that downstream readers actually scan — the rest is reference material — so its job is to answer "is this dataset ready to use, and is there anything I should know?" at a glance.

**Opening sentences (always).** Begin with one or two sentences stating where the dataset stands. In an early draft this might just be "Initial validation pass; see `### For further review` for open items." In a final draft and in published reports it should affirm that the dataset has been reviewed and is ready for use, and briefly call out anything from `### Review notes` that needs special care from a user (ideally there is nothing and the sentence is just "This dataset has been reviewed and is ready for use.").

**Subsections (up to three, included as needed).** Each is a bulleted list.

- **`### For further review`** — issues an expert dataset creator should look into before the report is published, each with a link to the image(s) where the issue is apparent. Audience is the internal reviewer. **Must be removed from final-draft and published reports** — any item that ends up worth surfacing to users belongs under `### Review notes` instead.
- **`### What looks good`** — short, positive summary confirming the checks in [section 4](#4-data-quality-checklist) that passed. Keep this section brief and don't duplicate detail already covered by the doc.
- **`### Review notes`** — user-facing notes about quirks or known gaps in the dataset that a downstream consumer would want to know (e.g. "Source data was unavailable for 16 hours on 2022-11-29 → 2022-11-30 across all variables; backfill is not possible from the upstream archive."). Items from `### For further review` that are investigated and turn out to be real but acceptable should be moved here and reworded for an external audience. Omit this section entirely if there is nothing to report. Only **time-windowed** characteristics of this archive instance belong here — version-boundary behavior changes, historical low-quality windows, source outages. **Intrinsic, always-true** variable facts (masking sentinels like -999/-50, unit quirks, what the variable is) do not go here; they belong in the variable's `comment` attr so they travel with the data. See the comment-vs-review-note rule in [AGENTS.md](../AGENTS.md) (Metadata conventions).

**Verification gate on `### Review notes`.** Any entry that attributes a gap, sentinel, or anomaly to an **upstream cause** (source outage, upstream archive gap, model physics, source-GRIB precision) must be backed by direct evidence: a fetched source file (per the unavailable-timestamp tactic in [3d](#3d-dig-into-each-follow-up-item)) or an inspection of the reference dataset at the same timestamps. If you cannot produce that evidence, the item stays in `### For further review`, not `### Review notes` — "looks like an outage" without verification is exactly the failure mode this gate prevents.

**Phrasing gate on `### For further review`.** Bullets in this section must describe an **open question with the evidence already gathered**, not a proposed verification that has not been performed. If a bullet contains "worth re-running", "should re-confirm", "warrants checking", "would be nice to verify", or any other phrasing that names a verification you could run, you have not finished §3d — go run it, then either resolve the item or rewrite the bullet around what the verification revealed.

### 3f. Batched review for many-variable datasets (AI assistants)

A complete-variables dataset (e.g. `noaa-hrrr-forecast-48-hour-virtual`: 177 variables, so ~700 per-variable PNGs) cannot be reviewed in a single agent context: at roughly 300-500 tokens per plot image plus each variable's stats block and reasoning, a full pass is on the order of a million tokens before any follow-up work. At ~25 variables a single context works; well above that, split the review across subagents. The lead (orchestrating) agent must not open plot images itself — every image read happens inside a batch or verification subagent whose context is discarded after it returns.

**Lead agent process:**

1. Produce or locate the run directory (`run-all` — for a virtual store it runs the manifest scan and the sampled decode scan itself, so availability and decode health are already in the report).
2. From `validation_summary.md` read **only** the header: the Datasets block, Run parameters, and the Availability section (everything above `## Per-variable details`). Availability findings come from that section's table and files, not from plot review. Never read the per-variable details wholesale — at 177 variables that section alone is tens of thousands of tokens.
3. List the variables (`ls <run-dir>/spatial_*.png` maps slugs back to names) and partition them into **theme-coherent batches of 10-20**: keep families together (all radiation, all reflectivity/precipitation, all cloud, each vertical group's variables together) so within-batch cross-variable checks — radiation peaking while cloud cover is high, temperature vs dew point consistency — remain possible. Note which batch got which family so cross-batch questions have an owner.
4. Spawn one subagent per batch, in parallel, with the prompt contract below.
5. Aggregate the structured verdicts. Look specifically for cross-batch patterns the subagents cannot see: the same timestamp flagged in several batches (run-level ingest issue), one member of a physically-linked pair flagged while the other batch reported its partner clean, level-adjacent anomalies in a vertical group split across batches.
6. Dispatch each SUSPECT/FAIL finding to a **verification subagent** per [3d](#3d-dig-into-each-follow-up-item): give it the exact single-step re-render command (scratch `--output-dir`), or the upstream GRIB/idx URLs to fetch, and the specific question to answer. It returns confirmed/refuted plus evidence. The 3d rule is unchanged: if you can name a verification, run it — via a subagent.
7. Write the `## Summary` per [3e](#3e-update-validation_summarymd) from the structured findings, citing the plot filenames and evidence the subagents returned.

**Batch subagent prompt contract.** Each batch subagent gets: the run directory path; its explicit variable list; the reproduction parameters from the run header it needs (spatial snapshot time, timeseries period, sampled level); any dataset-specific expectations (e.g. hour-0-NaN accumulated variables, projected y/x grid, known source quirks); and instructions to, per variable, read its PNGs and its heading section in `validation_summary.md` and apply the full [section 4](#4-data-quality-checklist) checklist. Be explicit that the three per-variable plots — `value_timeseries_`, `spatial_`, and `temporal_<slug>.png` — are **always** opened for every variable (including availability-complete ones; the spatial map is the only place the flipped-map / sentinel-bleed / histogram-overlap checks can be made, and stats alone will not catch them), and that **only** `availability_<slug>.png` is conditional (opened when the variable's availability line is not complete). Slug = variable name with `/` → `__`. Word the contract so a subagent cannot read "conditional" as applying to the spatial/temporal plots — a common misread that silently drops the geometry checks for well-behaved fields. It must return compactly (≤ ~40 lines), e.g.:

```
VERDICTS
temperature_2m: OK
composite_reflectivity: SUSPECT — histogram mass at -10 dBZ sentinel-like spike
FINDINGS (SUSPECT/FAIL only)
- var: composite_reflectivity; plot: spatial_composite_reflectivity.png; what: ...;
  where: init=..., lead=...; suspected-cause: ...; next-verification: <exact command>
BATCH OBSERVATIONS
- radiation family diurnal phase consistent across all 4 vars in this batch
```

Verdicts + precise pointers, not narration: the lead acts on `next-verification` lines and never re-reads the images behind an OK.

**Sizing.** Budget 10-15 variables per batch subagent — a 177-variable dataset is 12-18 parallel batches, then a handful of verification subagents. Keep batches under ~15 variables; a subagent that also does follow-up re-renders needs the headroom.

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
- [ ] **Whole plot matches meteorological expectations.** Look closely for subtly or obviously wrong new types of problems not enumerated here. Visual plots are a key layer of our defense in depth approach to catching data quality issues. We can't list every possible issue, rather use your meteorological knowledge to first define what you expect to see and compare that to what you actually see in the plots.

### Time series plots (from `temporal_<var>.png`)

- [ ] **Diurnal cycle is in phase with the reference.** Shortwave radiation, 2m temperature, and similar variables should peak at the same local hour as the reference. Phase-shift is a time-coordinate bug (e.g. UTC vs local, or off-by-one lead time).
- [ ] **Trend magnitudes match.** The validation and reference should have similar day-to-day ranges. A consistent offset points to a calibration or unit issue; a consistent scale difference points to a unit conversion.
- [ ] **No unexpected flatlines or spikes.** Flatlines at one value for many steps can be a read error or a stuck sentinel; isolated spikes can be unit bugs at specific lead times (common in accumulated variables).
- [ ] **Accumulated variables reset as expected.** Precipitation and radiation accumulators should typically reset each forecast — check the `step_type` in `validation_summary.md` and confirm the shape.
- [ ] **No obvious quantization in time series.** Time series which are snapped or binned to a limited set of values or "staircasing" in what should be smooth time series indicates `keep_mantissa_bits` is too low.
- [ ] **Whole plot matches meteorological expectations.** Look closely for subtly or obviously wrong new types of problems not enumerated here. Visual plots are a key layer of our defense in depth approach to catching data quality issues. We can't list every possible issue, rather use your meteorological knowledge to first define what you expect to see and compare that to what you actually see in the plots.

### Full-period value time series (from `value_timeseries_<var>.png`)

- [ ] **No sharp discontinuities in the mean line.** A sudden step up or down in the value across the full time range — not explained by a real seasonal/physical transition — is the signature of a source data change such as a unit switch (e.g. a GRIB file that begins emitting Kelvin instead of Celsius, or kg m⁻² instead of mm). These step changes are invisible to the short-window `temporal_<var>.png` plot, which is the reason this plot exists.
- [ ] **No step change in the std line (materialized forecast / ensemble datasets).** The secondary-axis std line reflects the spread across lead time / ensemble members at each timestep. An abrupt rise or drop that persists indicates a distribution change in the source (e.g. a change in precision, smoothing, or member generation). Analysis datasets and virtual stores (whose series is a single pinned slice) have no std line, so only the mean-line check applies there.
- [ ] **Whole plot matches expectations across time.** Drifts, ramps, or level shifts that don't correspond to a known seasonal cycle or source transition warrant investigation — cross-check against the source archive at the timestamps where the change appears.

### Availability (from `availability_heatmap.png`, `availability_<var>.png`, and `unavailable_timestamps.txt`)

- [ ] **Every variable is fully available, or the gaps are explained.** Any incomplete variable should have a reason: source data unavailable before a date for a specific variable, known source outage, ocean point for a land-only variable. Unexplained gaps are the bug.
- [ ] **Unavailable pattern is not structural.** Gaps concentrated at specific lead times, specific hours of day, or specific forecast cycles suggest a processing or indexing bug, not a random source outage. Use the heatmap and the first/last incomplete columns in the summary table to spot patterns shared across variables (e.g. a consistent first-incomplete date points to source coverage starting later; a variable available only after a date is typically a model-version addition — document it, don't retry it).
- [ ] **Any availability gap you intend to label as an upstream outage has been verified per the sampling tactic in [3d](#3d-dig-into-each-follow-up-item).** If every sampled file (and its sidecar index, if applicable) is present and intact, the gap is an ingestion bug, not an outage — keep it in `### For further review` and target the positions listed in `unavailable_timestamps.txt`.
- [ ] **First step of an analysis dataset is NaN for accumulated variables.** For analysis datasets, `step_type` ≠ `instant` variables (accumulation / average / max / min) are structurally NaN at the very first timestamp — there is no prior window to accumulate / average / extremize over. This is expected and not a bug.

Note on `step_type` ≠ `instant` variables (accumulation / average / max): the first lead time of each forecast is structurally NaN (there is no prior window to accumulate/average over). Both availability paths exclude it — the value scan drops that slice from the null counts, and the manifest probe never probes an accumulated variable at lead 0 — so "complete" for an accumulated variable means *no unexpected* gaps.

### Cross-dataset checks

- [ ] **Reference availability.** If the reference dataset doesn't cover your window, `validation_summary.md` will show `variable not available in reference dataset` — that is not a bug in the validation dataset, but spatial / temporal comparisons lose their signal. Consider a different reference or re-run on a time range the reference covers.
- [ ] **Ensemble member is plausible.** For ensemble datasets `validation_summary.md` records the randomly-selected member. Rerun once more to confirm that a different member also looks right.

## 5. Sharing and publishing the report

Once a run is reviewed, render it to a static HTML report, share it as one or more drafts for internal and external review, and finally publish the approved version. Two storage paths exist:

- **Draft** — every non-final upload goes here, timestamped, kept forever. Reachable only by its direct link and never surfaced to dataset users (the dynamical.org catalog links only the stable path), so uploading a draft does **not** make the report externally viewable — it's for sharing a run for review.
- **Stable** — the canonical, published report for a dataset, overwritten by each new publish. Embedded in dynamical.org and therefore seen by external users.

Both paths live in the `dataset-validation-reports` Cloudflare R2 bucket, served publicly at `https://dataset-validation-reports.dynamical.org`. Drafts and previously-published reports are archived forever — only the file at the stable path is overwritten.

### 5a. Render the HTML report

```bash
uv run src/scripts/validation/plots.py render-report <run-dir>
```

Reads `<run-dir>/validation_summary.md` and writes `<run-dir>/validation_report.html` next to it. Self-contained HTML (inline CSS/JS, no build step), images referenced at relative paths so the rendered file works locally and after upload.

The HTML mirrors the markdown 1:1 with two viewing affordances:

- **Left TOC** (slide-out hamburger on mobile) lists the top-level sections plus a Variables group with a checkbox per variable. "All / none" toggles let you compare a subset of variables side-by-side. Each variable section has `id="var-<name>"` so URLs can deep-link (e.g. `validation_report.html#var-temperature_2m`).
- **Images** are full-width on mobile and wrapped in a link to the underlying file so tap/click opens the full-resolution image.

`upload` re-renders before uploading, so this command is only needed for local-only previews.

### 5b. Upload drafts and publish the final

```bash
uv run src/scripts/validation/plots.py upload <run-dir>             # draft
uv run src/scripts/validation/plots.py upload <run-dir> --publish   # publish to stable + archive
```

`upload` re-renders the HTML, then uploads the entire run directory (`validation_summary.md`, all `*.png`, `unavailable_timestamps.txt` if present, `validation_report.html`) to R2.

Without `--publish`:

```
<dataset-id>/drafts/<version>_<YYYY-MM-DDTHH-MM>/
```

Use this to share a single run for review without committing to it. Drafts are kept forever; iterate by re-running `run-all` (a new timestamped run dir) and re-uploading. Re-upload a fresh draft after **every** change to the validation summary so the shared link always reflects the current review state. Drafts go to timestamped paths that are never overwritten, so uploading a draft is non-destructive and does not require confirmation.

`--publish` is the opposite: it overwrites the stable, website-linked report, so **never run it without explicit direction from the user** — a draft is always the right default.

With `--publish`:

```
<dataset-id>/latest/                                  # stable, overwritten each publish
<dataset-id>/published/<version>_<YYYY-MM-DDTHH-MM>/  # archive copy, kept forever
```

The stable path is what the dynamical-stac catalog links to. The archive copy preserves what `latest/` was before this publish; previously published reports are never lost.

Prints the public URL of `validation_report.html` on completion (e.g. `https://dataset-validation-reports.dynamical.org/<dataset-id>/latest/validation_report.html`).

### 5c. Draft → publish review cycle

The report moves through three audiences before it can be published. Each phase is just another `upload` (no `--publish` until the final step), but the `## Summary` block is rewritten between phases for the next audience.

**Phase 1 — Internal-review drafts.** End each internal-review pass with `upload <run-dir>` (no `--publish`). Audience: internal data reviewers (you and other repo contributors). The summary's `### For further review` section drives the conversation; internal jargon ("P1/P2", filenames, repo paths, ticket numbers) is fine here because every reader has the repo context. Iterate — investigate each item per [3d](#3d-dig-into-each-follow-up-item), update the summary, rerun `run-all` if new plots are needed, re-upload — until `### For further review` is empty (every item is either resolved or has been moved to `### Review notes`).

**Phase 2 — External-audience draft.** When `### For further review` is empty, rewrite the `## Summary` block for an external audience and upload one more draft (still no `--publish`). Audience: external dataset users who have never seen the run directory or our review process. Rewrite involves:

- Update the opening sentences to affirm the dataset is reviewed and ready for use, and to call out anything from `### Review notes` that needs special care.
- Drop the (now empty) `### For further review` subsection.
- Reword every remaining bullet for a public dataset consumer: spell out variable names, expand acronyms, avoid `P1`/`P2` (use the explicit lat/lon or describe the point), avoid ticket numbers, internal codenames, file paths, and process shorthand. Each bullet must make sense in isolation to someone with no repo context.

**Phase 3 — Publish.** Only after a human reviewer approves the Phase 2 draft, run `upload <run-dir> --publish` to write to the stable path. Do not run `--publish` while `### For further review` is non-empty or while the summary still reads as internal-audience prose — share another draft instead.

### 5d. Surfacing the published report on dynamical.org

Once published, the report is picked up automatically on the next deploy of the dynamical.org site — the site's build pulls the latest published report for each dataset and incorporates it into the catalog page for that data product. No per-dataset wiring is required; just redeploy dynamical.org to refresh the catalog with the new report.

### Configuration

`upload` (both for uploading drafts and for publishing) reads R2 credentials from these environment variables:

- `R2_VALIDATION_REPORTS_ENDPOINT_URL`
- `R2_VALIDATION_REPORTS_ACCESS_KEY_ID`
- `R2_VALIDATION_REPORTS_SECRET_ACCESS_KEY`

Scoped to the `dataset-validation-reports` bucket. Set them in the environment before running `upload`.
