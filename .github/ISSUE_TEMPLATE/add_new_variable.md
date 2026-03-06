---
name: Add [variables] to [dataset]
about: Checklist for adding a new data variable to an existing dataset
labels: ""
assignees: ""
---

## Checklist

- [ ] PR: add variable to template config, regenerate zarr metadata, merge
- [ ] Run operational update to expand dataset variables
- [ ] Backfill new variable(s)
- [ ] Validate (run plots, inspect output)
- [ ] Re-run notebook and push
- [ ] Update external docs (dynamical.org)
