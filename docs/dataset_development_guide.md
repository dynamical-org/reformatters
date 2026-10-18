# Dataset development guide

Coordinate each stage through a sub-agent, retaining scope decisions and artifact locations. Complete each stage before advancing; group related variants in one implementation, then backfill and validate each dataset. Continue until publication or a human decision is needed. Checkpoints are hard gates: stop and wait for the human.

## 1. Explore and agree scope

Follow [source_data_exploration_guide.md](source_data_exploration_guide.md). Verify the report against real files and record gaps. For a new variable, inspect a recent source file as described in [add_new_variable.md](add_new_variable.md).

**Checkpoint A:** obtain human agreement on provider, model, variants, variables, and unresolved mapping decisions. Usually include all available variables for a virtual dataset and a common subset for a materialized one. For an added variable, confirm its name, level, and attributes.

## 2. Implement

Follow [implementation_guide.md](implementation_guide.md) for a new dataset or [add_new_variable.md](add_new_variable.md) for an existing one. Use a separate code review. Regenerate templates and pass formatting, lint, type checks, and dataset tests. A human reviews and merges the PR before backfilling from `main`.

## 3. Backfill

Follow [backfill.md](backfill.md): create the bucket and store for a new dataset, or overwrite only the added variable. Confirm the job succeeds and expected data is present.

Keep existing operational updates running; schedule backfills between update fires to avoid finalization conflicts. New datasets ship with suspended CronJobs; after the first backfill, open a PR enabling updates. Run later targeted backfills between updates as well.

Launch long-running backfills and validation detached with a monitor; see [validation.md](validation.md).

## 4. Validate and approve

Follow [validation.md](validation.md): run the full `run-all`, review every plot, investigate anomalies, fix and re-backfill affected variables/positions, and repeat validation. For added variables, carry forward the published report's approved summary and update only relevant findings.

Upload an externally worded draft without `--publish` only after `### For further review` is empty.

**Checkpoint B:** share the draft URL and obtain human approval before `upload --publish`. Publish the approved report to its stable path immediately; the catalog deployment makes it visible on the website.

## 5. Publish

Update the public [dynamical-stac](https://github.com/dynamical-org/dynamical-stac) repository, whose [production catalog](https://stac.dynamical.org/catalog.json) is the source of truth. Add new datasets to `src/catalog.py`; existing datasets with added variables need only regeneration. Batch related datasets into one PR.

```bash
./scripts/generate
git add stac/
```

Complete publication when the STAC PR merges; its deployment updates the website and surfaces the published validation report. Leave publication to other external catalogs to a human.
