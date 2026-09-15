## DWD

This directory contains code for downloading and processing NWP data from Germany's National
Meteorological Service, the [Deutscher Wetterdienst](https://www.dwd.de/EN) (DWD).

## DWD's NWPs are unlike most other NWPs on Dynamical.org

Unlike most other NWPs on Dynamical.org, DWD currently only maintains a 24-hour rolling archive of
their NWPs. As such, the code in this directory performs two separate jobs:

1. Copies ICON-EU GRIB files from DWD's HTTPS server to a public Source Co-Op bucket, on both
   the regular lat/lon grid and DWD's native icosahedral grid.
2. Converts a subset of the regular lat/lon `.grib2.bz2` files in the Source Co-Op bucket to Zarr.

## Transferring files from DWD's HTTPS server

DWD publish a 24-hour rolling archive of their operational NWPs on their HTTPS server at
`https://opendata.dwd.de/weather`. The `archive-grib-files` cron job copies ICON-EU in two
phases, one after the other in the same pod, so the two copies never load DWD's server at once:

1. **Regular lat/lon.** `archive_gribs/copy_files_from_dwd.py` copies
   `/weather/nwp/icon-eu/grib/<HH>/<param>/<filename>.grib2.bz2` to
   `dwd-icon-grib/icon-eu/regular-lat-lon/2026-01-20T00/t_2m/<filename>.grib2.bz2`.
2. **Icosahedral.** `archive_gribs/copy_icosahedral_files_from_dwd.py` copies DWD's
   `/weather/nwp/v1/m/icon-eu/p/<PARAM>/[lvt1/<level type>/lv1/<level>/]r/<run>/s/<step>.grib2`
   to `dwd-icon-grib/icon-eu/icosahedral/2026-01-20T00/T_2M/PT000H00M.grib2`, keeping DWD's
   parameter and level directories (e.g. `.../2026-01-20T00/T/lvt1/100/lv1/85000/PT000H00M.grib2`).
   The `--icosahedral-level-types` option selects which level types are copied in addition to
   single-level parameters.

Both phases list the source and destination, then copy only the missing files with
`rclone copyurl`, so a run interrupted part way through is completed by the next one. The
icosahedral phase copies one run at a time, oldest first, and skips runs younger than 4 hours,
which may still be publishing. It starts no run within 30 minutes of the job's deadline; the job
fails instead, so the deadline only interrupts a run that is already copying.

This code uses [`rclone`](https://rclone.org) under the hood. `rclone copyurl --urls` needs a
newer `rclone` than Ubuntu's package; the Docker image copies it from `rclone/rclone:latest`, and
[rclone.org/install](https://rclone.org/install/) has other options.

### Testing HTTPS transfer locally

Note that the code is hard-coded to call `rclone` from `/usr/bin/rclone`. So, if you installed
`rclone` somewhere else, you'll first have to create a symbolic link, e.g.:

```sh
sudo ln -s /snap/bin/rclone /usr/bin/rclone
```

You can test locally like this:
`uv run main dwd-icon-eu-forecast-5-day archive-grib-files --dst-root-path=/local/path/regular-lat-lon --icosahedral-dst-root-path=/local/path/icosahedral`

To test uploading to a cloud bucket, the destination paths can start with an `rclone` remote, in
the form `--dst-root-path=remote:path`

For command line arguments that allow you to limit the number of files downloaded (e.g.
`--nwp-init-hours`, `--icosahedral-nwp-init-hours` and `--icosahedral-params`), see
`uv run main dwd-icon-eu-forecast-5-day archive-grib-files --help`
