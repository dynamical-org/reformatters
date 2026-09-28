---
name: Correct published data in [dataset(s)]
about: Record a correction to already-published values, so downstream consumers know what to recompute
labels: ""
assignees: ""
---

Anyone whose reads overlap the affected range while the wrong values were served must apply the remediation named below to the raw values and to anything derived from them. "Pending" and "unknown" are fine while this issue is open; blanks are not. The fix PR can merge before the backfill runs. Close this issue only when the whole affected range is corrected, validated and on `main` in every store, and each downstream item is done or marked not applicable with a reason.

Owner (updates this issue through the backfill and the notices):

## Affected data

- Dataset ID(s), version(s) and store URL(s):
- Variable(s):
- `init_time` (or `time`) range, half-open, UTC: `[YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SS)`
- Other dimensions, if only part is affected (`ensemble_member`, `lead_time`, spatial), or "all":
- What the published values meant, what they should have meant, and by how much (e.g. "1000x too small: the source encodes `tp` in metres before this date"):
- Remediation: can consumers rescale raw values, and must anything derived from them be recomputed?
- How it was established on real data:
- When the wrong values were being served (the `main` heads or times that first and last published them, if known):

## Fix

- [ ] PR:
- [ ] Merge commit:

## Published correction

One entry per store and per corrected range, if the backfill was split:

- [ ] State: pending / partial / complete (if partial, the range actually corrected):
- [ ] Backfill operation and filters (workflow run or job name):
- [ ] Last `main` head with the wrong values, and first `main` head with the correction (Icechunk snapshot IDs):
- [ ] When that head became `main` (UTC; a bounded interval with how it was observed, if not exact). This is not the snapshot's creation time:
- [ ] Validated against real data (how):

## Downstream

- [ ] Internal consumers told what to remediate (e.g. a dynamical-org/meta issue naming the variables, range and `main` transition above for scorecard):
- [ ] User-facing note (dataset page or dynamical.org updates), if readers could have used the wrong values:
