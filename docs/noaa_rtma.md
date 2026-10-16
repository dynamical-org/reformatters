# NOAA RTMA virtual analysis

`noaa-rtma-analysis-virtual` contains hourly CONUS analysis fields from
2017-05-02 00:00 UTC onward. Source files are public NODD objects in
`noaa-rtma-pds`:

```text
rtma2p5.YYYYMMDD/rtma2p5.tHHz.2dvaranl_ndfd.grb2_wexp
```

The source exploration is recorded in
[reformatters #1214](https://github.com/dynamical-org/reformatters/issues/1214).
Source coverage contains gaps; a missing analysis remains missing rather than
being replaced by a first guess or another product.

## Layout and values

The native grid has 2345 columns and 1597 rows, on a spherical Lambert conformal
projection with radius 6,371,200 m, standard parallels and latitude of origin
25°N, central meridian 95°W, and spacing 2,539.703 m. The dataset uses pixel-center
`x`/`y` coordinates, descending `y`, and two-dimensional latitude/longitude.
Each virtual chunk references one complete GRIB map for one variable and hour.

The analysis includes surface geopotential height and pressure, 2 m temperature,
dew point and specific humidity, 10 m wind components, speed and direction,
wind gust, visibility, cloud ceiling and total cloud cover. Grid-relative wind
components describe velocity along the projected axes. Surface geopotential
height remains source-backed at each time. Finite visibility and ceiling values
are preserved; decoder agreement alone does not establish that a finite value
is a missing-data marker.

Precipitation is a separate source product. RTMA `pcp.184` uses the smaller
2145×1377 grid and is not part of this dataset. First guesses, analysis errors,
other domains and materialized output are also excluded.

## Operations

The update configuration uses one virtual writer with a recent-time catch-up
window. Its freshness validator is a reformatters operational check, reported
through the configured Sentry integration; it does not configure wxopticon or
an SLA ledger. New deployments keep the CronJob suspended until the store is
backfilled. See [backfill.md](backfill.md) and [validation.md](validation.md) for
the backfill, activation and publication workflow.
