# GFS virtual: the two source-verified metadata fixes

2026-09-06. Both items that PR
[#1018](https://github.com/dynamical-org/reformatters/pull/1018) had held back pending
source verification are now settled and folded into that same PR, as two commits on
`gfs-virtual-metadata-fixes`. No separate PR was opened: #1018 is an unreviewed draft
touching the same lines, so keeping them together avoids a merge-order dependency and a
second review cycle.

## Commits

| SHA | Fix |
| --- | --- |
| `61bf72b9` | State the verified UV-B wavelength band |
| `4eee3b42` | Say which storm motion the helicity values use |

## Fix 1 — UV-B band, 263-345 nm

`uv_b_downward_solar_flux_surface` and `clear_sky_uv_b_downward_solar_flux_surface` now
name the band:

> The UV-B portion of the downward shortwave flux at the surface, covering 263-345 nm.

The clear-sky one adds `, computed with clouds removed.` The disproved 280-315 nm was
not re-added.

The chain, verified against raw source rather than the summary: UPP relays the
diagnostic (`INITPOST_NETCDF.f` L2827-2836 reads `duvb_ave`/`cduvb_ave`);
`GFS_diagnostics.F90` L243-268 maps those to radiation slots 21/22;
`GFS_radiation_driver.F90` L2073-2076 fills them from `scmpsw%uvbfc`/`uvbf0`; RRTMG sets
`nuvb = 27` (`radsw_main.f` L346) and selects `ibd = nuvb - nblow + 1`; with `NBLOW=16`
that is table element 12, whose bounds in `radsw_param.f` are 29,000-38,000 cm-1.
10^7/38000 = 263.16 nm and 10^7/29000 = 344.83 nm, rounded to 263-345 nm.

## Fix 2 — helicity storm motion

`storm_relative_helicity_3000_0m` gained a comment it did not have:

> Uses a right-moving storm motion in both hemispheres, so southern hemisphere values are
> positive-mean rather than mirrored.

Stated as a fact about the data. No UPP version numbers and no mention of future UPP
versions, both of which would go stale in static template metadata.

Verified: `global-workflow release/gfs.v16.3.34` pins `upp_v8.3.0`, which dereferences to
commit `c5f3053`. At that commit `MISCLN.f` L168-184 sets `DEPTH(1)=3000.0` and emits
`HELI(:,:,1)` for this variable, and `CALHEL.f` L329-345 applies the Bunkers 7.5 m s-1
right-moving deviation unconditionally. Grepping that whole file for
`gdlat|latitude|hemisph|left` returns exactly one hit, a prose comment — no latitude is
read anywhere, so there is no hemisphere branch. The claim earlier dropped as contested
is correct for this archive.

## The thing to watch, later

UPP commit `12ab90c` ("Account for left-mover in SH in SRH", 2022-10-10) added a
`GDLAT`-based switch that picks the left-moving solution south of the equator. It is in
current `develop` (`CALHEL.f` L314-322) but not in any UPP version that produced data in
this archive. So the convention is a property of the source version, not a law of nature.
If a future GFS upgrade brings that UPP line in, southern-hemisphere helicity flips sign;
that boundary would belong in the validation report's Review notes, not in the comment.
Not a current problem.

Full findings with citations: `.upp_source_check.md` in this worktree.

## PR description

#1018's "Pending source verification" section was removed — with both fixes in the PR,
that wording would have been false. In its place is a "Source verification" section
citing the UPP files and lines and linking `.upp_source_check.md`. Item 7 was also
corrected, since it claimed the UV-B comments carry no replacement number, and a new item
9 covers the helicity comment. "Deliberately not included" is unchanged.

## Checks

`ruff format`, `ruff check --fix`, `ty check` clean. `pytest -m "not slow" -n 4`: 2395
passed. `common_template_config_subclasses_test.py` + `datasets_cf_compliance_test.py`:
425 passed. All four `mark.slow` modules under `tests/noaa/gfs/{analysis,forecast}_virtual/`
plus both `template_config_test.py` modules: 141 passed. No test asserted either comment
string, so no test needed updating. Both templates regenerated in each commit; the
template diffs are comment-only.
