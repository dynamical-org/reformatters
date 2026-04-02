## ECMWF IFS ENS Forecast Archive

Archives of ensemble weather forecast data from the [European Centre for Medium-Range Weather
Forecasts](https://www.ecmwf.int/) (ECMWF).

ECMWF's public open data bucket (`ecmwf-forecasts`) only provides data from April 2024 onward.
Earlier data is stored in ECMWF's [MARS](https://www.ecmwf.int/en/forecasts/access-archive-datasets)
(Meteorological Archival and Retrieval System), a tape-backed archive that requires careful request
optimization and authenticated API access.

This archive retrieves IFS ENS (Ensemble) forecast data from MARS and stages it as GRIB files in
S3, covering the period from 2016-03-08 (IFS cycle 41r2, when ENS resolution increased to ~18km)
through 2024-03-31 (when S3 open data begins). The staged GRIBs are being progressively filled in
as part of an ongoing backfill process — not all dates have all file types yet. The staged GRIBs
can then be processed by the [reformatters](https://github.com/dynamical-org/reformatters) into
cloud-optimized Icechunk Zarr format, available at [dynamical.org](https://dynamical.org).

The files are in GRIB Edition 1 format at 0.25 degree resolution. Each date directory may contain
up to five GRIB files:

| File | Description | Approx Size |
|---|---|---|
| `cf_sfc.grib` | Control forecast (member 0), surface variables, all steps | ~2.4 GB |
| `cf_pl.grib` | Control forecast (member 0), pressure level variables, all steps | ~1.1 GB |
| `pf_sfc_0.grib` | Perturbed forecast, surface variables, members 1–25, all steps | ~60 GB |
| `pf_sfc_1.grib` | Perturbed forecast, surface variables, members 26–50, all steps | ~60 GB |
| `pf_pl.grib` | Perturbed forecast, pressure level variables, members 1–50, all steps | ~53 GB |

Each GRIB file has a companion `.grib.idx` index file containing one JSON record per GRIB message,
with byte offsets and lengths enabling efficient partial reads without downloading entire files.

Surface variables: `sp` (surface pressure), `2t` (2m temperature), `10u` (10m u-wind), `10v`
(10m v-wind), `100u` (100m u-wind), `100v` (100m v-wind), `tp` (total precipitation), `strd`
(surface thermal radiation downwards), `ssrd` (surface solar radiation downwards), `msl` (mean sea
level pressure), `2d` (2m dewpoint temperature), `ptype` (precipitation type), `10fg` (10m wind
gust), `tcc` (total cloud cover).

Pressure level variables: `z` (geopotential, m²/s²) and `t` (temperature, °C) at 500, 850, and
925 hPa.

Forecast steps: 3-hourly for hours 0–144 and 6-hourly for hours 150–360 (85 steps total).
Ensemble members: 0 (control) + 1–50 (perturbed). All forecasts initialized at 00Z.

The directory structure in S3 is:

`{s3_prefix}/{YYYY-MM-DD}/{file_type}.grib[.idx]`

For example:
- `dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/2023-07-01/cf_sfc.grib`
- `dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/2023-07-01/cf_sfc.grib.idx`

### Backfill status

The archive is filled in progressively — not all dates have all five file types. `cf_pl` (control
forecast at pressure levels) is the most complete, spanning nearly the full date range. Surface
files and perturbed forecast files are added as the backfill process runs. Check what files are
present for a given date before processing.

The code used to retrieve the files from MARS is
[available on GitHub](https://github.com/dynamical-org/ecmwf-ifs-backfill). The compute to
retrieve the files is provided by [dynamical.org](https://dynamical.org/).

*Data license*: ECMWF data is available under the
[Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/) license, provided the
source is acknowledged.
