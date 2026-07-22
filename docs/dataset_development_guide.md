# Dataset Development Guide

Take a data product from nothing (or an existing dataset that needs a new variable) all the way to published, by running each stage of the pipeline in its own sub-agent. This doc is written for the agent that coordinates that work.

You, the reader, are the **coordinator**. You do not do the stage work yourself — you sequence it, spawn a sub-agent per stage, and drive to completion.

## Two modes

- **New dataset** — full integration. The Implement stage follows [implementation_guide.md](implementation_guide.md).
- **Add a variable** to an existing dataset — the Implement and Backfill stages follow [add_new_variable.md](add_new_variable.md) and are smaller. Explore collapses to inspecting one recent source file for the variable, and Checkpoint A collapses to confirming the variable's name, level, and attrs.

The stages are otherwise the same; Validate and Publish are shared.

## Coordinator rules

- **Hold only thread state**: the mode, the current stage, the human's scope decisions, and the artifact locations (report path, PR, store URL, report URL). Push every heavy step into a per-stage sub-agent (the Agent tool) and act on its compact return — keep the sub-agent's raw output (plot images, file dumps, long logs) out of your own context.
- **One stage at a time, gated.** Advance only when the stage's done-check passes. On failure, loop within the stage (bounded) — do not skip ahead or escalate on the first miss.
- **Two human checkpoints are hard gates.** Stop, surface what's needed, and wait for an answer. Never cross a checkpoint on your own.
- **Long-running work** (a Kubernetes backfill, a ~1–2 h virtual `run-all`) — launch it detached with a monitor and don't block; see the long-runs note in [validation.md](validation.md) §1.
- **Multiple datasets in one run.** A run often produces several related datasets — forecast + analysis, early + late, virtual + materialized. Implement them all in a single shared Implement agent (they share source exploration, config, and utilities); Backfill, Validate, and Publish then run per dataset.
- **Drive to completion.** After each sub-agent returns and each checkpoint is answered, proceed to the next stage automatically until Publish is done or you are blocked on a human.

## Stages

### 1. Explore

- **Goal**: know exactly what source data exists, how it's structured, and how to access it.
- **Sub-agent**: follow [source_data_exploration_guide.md](source_data_exploration_guide.md) and produce its filled-in template. New-dataset mode does the full archive search; add-variable mode only inspects a recent source file for the variable ([add_new_variable.md](add_new_variable.md) §1a).
- **Output**: an exploration report (a markdown file — keep the path).
- **Done**: every claim in the report is verified against real source files, with gaps noted rather than guessed.

### ⛔ Checkpoint A — human scopes the dataset(s)

Present the exploration findings and settle the scope with the human. Always align on the exact **provider, model, and variant(s)** to produce — including whether this run creates several related datasets (see "Multiple datasets in one run" above). Beyond that, raise the non-obvious questions that surfaced while mapping the source data onto a datacube in our conventions — the decisions the exploration can't settle on its own (an awkwardly-structured or intermittently-available variable, an irregular level set, a coordinate that doesn't map cleanly, a choice between combining or splitting sources). Use judgement about what's worth asking rather than running a fixed checklist. In add-variable mode this collapses to confirming the variable's name, level, and attrs. Record what's decided — it drives every later stage.

### 2. Implement

- **Goal**: reviewed code that reads the source data and writes the dataset, with tests.
- **Sub-agent(s)**:
  - New dataset: one agent implements every variant in this run, following [implementation_guide.md](implementation_guide.md) (init → register → `TemplateConfig` → `RegionJob` → `DynamicalDataset` → integration test with snapshot values).
  - Add variable: follow [add_new_variable.md](add_new_variable.md) §1.
  - Then a **code-review** sub-agent focused on correctness, simplicity, and the future maintainer — drive to the simplest maintainable end state (the `/code-review` skill, or a general-purpose agent).
- **Output**: a PR (code + regenerated `templates/latest.zarr` + tests). A human reviews and merges it — the backfill runs from `main`.
- **Done**: `ruff format`, `ruff check`, `ty check`, and the dataset's tests are green, and the PR is merged to `main`.

### 3. Backfill

- **Goal**: a populated store.
- **Sub-agent**: follow [backfill.md](backfill.md). New dataset: create the bucket, then a `create-new-store` backfill. Add variable: an `overwrite-chunks-and-metadata` backfill filtered to the new variable.
- If the dataset already has an active operational update cronjob, do not suspend it — that would delay the production pipeline. Instead run the backfill between update fires so an update publish doesn't fail the backfill's finalize (see [backfill.md](backfill.md) and [parallel_processing.md](parallel_processing.md)).
- **Output**: the store URL, with data written.
- **Done**: the backfill job succeeded and the expected data is present.

### 4. Validate

- **Goal**: a reviewed, ready-to-publish dataset and a draft validation report.
- **Sub-agent**: follow [validation.md](validation.md) — run `run-all`, review every plot (a many-variable dataset uses the §3f batched sub-agent process), and investigate-and-verify every anomaly. Fix issues found, run a **targeted re-backfill** of only the affected variables/positions, and re-validate. Iterate until `### For further review` is empty, then rewrite the summary for an external audience and upload a draft (`upload`, no `--publish`).
- **Output**: the draft report URL.
- **Done**: draft uploaded, `### For further review` empty, summary reworded for external readers.

### ⛔ Checkpoint B — human approves the draft report

Share the draft report URL (including after your own fix-and-re-validate passes). The human reviews it. Never run `upload --publish` while `### For further review` is non-empty or unapproved. Once the user approves, publish the report to the stable path immediately (`upload --publish`; see [validation.md](validation.md) §5) — it becomes visible on dynamical.org when the STAC change in the next stage triggers a site deploy.

### 5. Publish to dynamical-stac

- The dataset catalog on dynamical.org is built from the STAC catalog (`https://stac.dynamical.org/catalog.json`), maintained in [`dynamical-org/dynamical-stac`](https://github.com/dynamical-org/dynamical-stac). It is the source of truth — you never edit the website; you update the STAC and the site reflects it on its next deploy (which also surfaces the validation report published at Checkpoint B). Regenerate the committed STAC output and open a PR to `main` in dynamical-stac:
  ```bash
  ./scripts/generate   # opens each Zarr store; picks up new datasets and variables
  git add stac/        # commit the regenerated collection.json
  ```
  A **new dataset** first needs an entry added to `src/catalog.py`; **adding a variable** to an existing dataset needs only the regenerate (the generator reads variables from the store).
- **Done**: the STAC PR is merged to `main`.

### 6. Publish to external catalogs

Placeholder — this process is in flux (Source Coop, Earthmover Marketplace, AWS Open Data, and others). Leave it to a human for now.
