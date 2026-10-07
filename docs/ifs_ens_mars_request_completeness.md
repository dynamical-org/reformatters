# Proposed strict field counts for IFS ENS MARS requests

This is a proposal for [ecmwf-ifs-backfill](https://github.com/dynamical-org/ecmwf-ifs-backfill), based on commit `a3d6848f9dea783e9e02ab36cfdd439af06879ff`. It does not change a running request builder or authorize deployment. The source-repair work is tracked in [reformatters#446](https://github.com/dynamical-org/reformatters/issues/446).

## Proposed change

Remove `"expect": "any"` from `mars_staging.mars_request_body` and `mars_probe.make_mars_request_body`. Allow MARS to enforce its expected field count before transferring a result. ECMWF documents that it computes the expected count from the requested field identities unless `expect` overrides it, and fails an unsatisfied request ([MARS request syntax](https://confluence.ecmwf.int/pages/viewpage.action?pageId=87854654)). Retain the independent local framing, exact tuple-coverage and payload checks; a server success does not establish that every compressed payload decodes.

The existing split campaigns require an explicit compatibility exception. Their saved request bodies, identity records and provenance hashes include `expect=any`. `split_repair._base` also checks that exact specification. Preserve that legacy body when changing the shared builder, rather than rewriting saved plans or recomputing their hashes.

Proposed patch against the pinned source:

```diff
--- a/mars_staging.py
+++ b/mars_staging.py
@@
         "grid": "0.25/0.25",
-        "expect": "any",
         "target": "output",
--- a/mars_probe.py
+++ b/mars_probe.py
@@
         "grid": grid,
-        "expect": "any",
     }
--- a/split_repair.py
+++ b/split_repair.py
@@
-    body = mars_request_body(date, kind)
+    body = {**mars_request_body(date, kind), "expect": "any"}
```

The final line freezes the existing split format; it does not make split requests strict. New strict split campaigns would need a separately reviewed, versioned request format. The two held campaigns must not be silently migrated, and neither this proposal nor its compatibility exception authorizes another split retrieval.

## Blast radius

| Path in ecmwf-ifs-backfill | Proposed effect |
| --- | --- |
| `mars_staging.mars_request_body` | Future canonical ENS submissions for all five request types: `cf_sfc`, `pf_sfc_0`, `pf_sfc_1`, `cf_pl`, `pf_pl`. A request missing fields fails on the server instead of yielding a partial result for local rejection. |
| `mars_probe.make_mars_request_body` | Exploratory ENS subset retrievals also become strict. A mixed available/unavailable parameter group can fail as a whole; a failed probe no longer says which individual parameters exist. |
| `split_repair._base`, plan loading, splitting, assembly, `retrieve_split_leaf` | Preserve the old body and hashes explicitly. Existing leaves and provenance remain usable; no automatic submission or migration. |
| `transfer.validate_grib` / `validate_index`, canonical transfer and `reconcile.py` | Keep independent framing and exact coverage checks. These compare field identities, not the `expect` keyword. No relaxation. |

The direct dataset is the historical IFS ENS archive staged by this repository: surface fields for control and 50 perturbed members, and geopotential/temperature at 500/850/925 hPa. The normal builder covers 85 forecast steps per initialization. Its documented archive range is 2016-03-08 through the April 2024 open-data boundary.

The downstream reformatter is `ecmwf-ifs-ens-forecast-15-day-0-25-degree`: its pre-2024-04-01 source path reads staged GRIB/index pairs from Source Cooperative. It does not submit MARS requests. This proposal changes no reformatter, template, published metadata, values, Icechunk store or operational schedule. Open Data retrievals from April 2024 onward are outside this builder. The 46-day products use separate Open Data/archive paths and are outside these builders.

## Required implementation checks

Before adopting the patch in ecmwf-ifs-backfill:

- Test all five canonical request types and the probe builder: only `expect` disappears; every selection field remains equal to the pinned baseline.
- Load a saved legacy split plan and its provenance from the pinned version. Assert unchanged identity, request hashes, active leaves and exact coverage, including after a split or assembly.
- Keep invalid framing, truncated files, missing/duplicate tuples and payload-decode regression tests. A strict server count is an additional check, not their replacement.
- Add a sanitized real incomplete-request response to terminal-error classification tests. Establish that it becomes a request-level failure without penalizing an account or starting a resubmission loop. Do not invent an error signature or submit a diagnostic request merely to produce this fixture.
- Run the external repository's full suite and lint checks, then obtain review of the implementation and description. A real count-failure fixture has not yet been collected for this proposal. Implementation in ecmwf-ifs-backfill and deployment are separate follow-ups for Marsh.

Ordinary incomplete archives already fail the local exact-coverage gate. This proposal moves detection earlier; it does not promise previously incomplete dates will become available. ECMWF's reported cache repair and omission of `expect` are two different changes. A successful retry would not identify which change caused success or prove that `expect=any` caused corruption.

