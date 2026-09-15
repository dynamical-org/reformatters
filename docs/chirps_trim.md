# Repair a CHIRPS store's trailing missing days

`src/scripts/trim_chirps.py` removes only the trailing times for which every
time-dependent data variable is entirely NaN. It scans the store, not the source
archive; zero precipitation counts as data, and interior missing days remain.
An entirely missing store is rejected rather than emptied.

Run one dataset at a time, supplying its Icechunk URI explicitly. The default is
a read-only dry run, using anonymous access for S3. For example:

```bash
uv run python -m scripts.trim_chirps \
  ucsb-chc-chirps-analysis-final \
  s3://dynamical-ucsb-chc-chirps/ucsb-chc-chirps-analysis-final/v0.1.0.icechunk

uv run python -m scripts.trim_chirps \
  ucsb-chc-chirps-analysis-preliminary \
  s3://dynamical-ucsb-chc-chirps/ucsb-chc-chirps-analysis-preliminary/v0.1.0.icechunk
```

After reviewing the reported extents, an authorized operator can add `--commit`
to each invocation. S3 writes use the AWS credential environment, without loading
Kubernetes secrets. A local Icechunk directory can replace the S3 URI for testing.
The script checks the stored dataset ID against the requested one.

Scanning proceeds backward in time, visiting every spatial chunk in the candidate
time block until the latest non-NaN day is established. It reads one logical chunk
at a time; CHIRPS chunks span 365 days, so inspecting a short tail can still require
reading a year's chunks across the grid. This is an exhaustive scan, not a few
sampled points.

The repair resizes every array carrying `time`, including coordinates, in one
Icechunk session and commits atomically to `main`. Arrays without `time`, retained
values, attributes, and encodings are preserved. Icechunk does not support a
separate consolidated-metadata write. Existing snapshots remain available; the
script does not expire snapshots or garbage collect storage. A second invocation
at the same extent creates no commit.

Run the repair between operational updates when possible. If `main` changes
during scanning or before commit, the script fails rather than rebasing the trim;
rerun the dry run against the new snapshot. An update already processing a
temporary branch may instead lose the race to the repair and fail its own
publication check. The script does not suspend crons or change their configuration.

## Why an operational update cannot repair the tail

A new-store backfill defaults its exclusive `append_dim_end` to now and publishes
that requested extent even when the source has not yet published its final days.
Backfills do not trim their template from processing results.

CHIRPS operational updates also build a template through now, but finalization
trims it to the latest successfully read source file. The common retraction guard
then rejects any result shorter than the existing store. An update therefore
cannot shrink an already-published missing tail. Neither an overwrite backfill
nor metadata refresh provides a shrink operation.

After repair, ordinary updates preserve the repaired extent when the last stored
day can still be read and there are no newer source files. They extend it when new
files are successfully read; a failed reread can still trigger the retraction guard. Missing future
files do not recreate the padded tail. The trim is based on processing success,
however, not a non-NaN scan: a successfully decoded all-NaN source file would count
as success. If such files occur, CHIRPS would need to exclude all-NaN days from its
successful results. This repair does not change that policy or the shared guard.

The 50G update memory request remains unchanged. Revisit it once peak memory can
be measured; the initial backfill's cluster metrics were unavailable.
