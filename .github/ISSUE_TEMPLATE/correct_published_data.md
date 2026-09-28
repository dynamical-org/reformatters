---
name: Correct published data in [dataset(s)]
about: Record a correction to already-published values, so downstream consumers know what to recompute
labels: ""
assignees: ""
---

Anyone who read the affected data before the first corrected snapshot has to recompute from it. Fill in every field. "All", "unknown" and "pending" are valid answers; blanks are not. The fix PR can merge before the backfill runs, so this issue stays open until the published-correction section is complete.

Owner (updates this issue after the backfill):

## Affected data

- Dataset ID(s), version(s) and store URL(s):
- Variable(s):
- `init_time` (or `time`) range, half-open, UTC: `[YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SS)`
- Other dimensions, if only part is affected (`ensemble_member`, `lead_time`, spatial), or "all":
- What the published values meant, what they should have meant, and by how much (e.g. "1000x too small: the source encodes `tp` in metres before this date"):
- Whether a consumer can rescale, or must recompute anything derived from these values:
- How it was established on real data:
- When the wrong values were being served (first and last publication, or snapshot range, if known):

## Fix

- [ ] PR:
- [ ] Merge commit:

## Published correction

- [ ] State: pending / partial / complete (if partial, the range actually corrected):
- [ ] Backfill operation and filters (workflow run or job name):
- [ ] Completed (UTC):
- [ ] Last snapshot with the wrong values, and first corrected snapshot, on `main` (Icechunk snapshot IDs and timestamps):
- [ ] Validated against real data (how):

## Downstream

- [ ] Internal consumers told what to recompute (e.g. a dynamical-org/meta issue naming the variables and range above for scorecard):
- [ ] User-facing note (dataset page or dynamical.org updates), if readers could have used the wrong values:
