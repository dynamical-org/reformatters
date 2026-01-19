## DWD

This directory contains code for downloading and processing NWP data from Germany's National
Meteorological Service, the [Deutscher Wetterdienst](https://www.dwd.de/EN) (DWD).

## DWD's NWPs are unlike most other NWPs on Dynamical.org

Unlike most other NWPs on Dynamical.org, DWD currently only maintains a 24-hour rolling archive of
their NWPs. As such, the code in this directory performs two separate jobs:

1. Copies the `.grib2.bz2` files from DWD's FTP server to a public Source Co-Op bucket.
2. Converts a subset of the `.grib2.bz2` files in the Source Co-Op bucket to Zarr.

## Transferring files from DWD's FTP server

DWD publish a 24-hour rolling archive of their operational NWPs on their FTP server at
`ftp://opendata.dwd.de/weather`. You can [browse DWD's files](https://opendata.dwd.de/weather) from
a web browser.

The code in `copy_files_from_dwd_ftp.py` copies files from DWD's FTP server, and transforms the
destination path to the form `2026-01-20T00Z/t2_m/filename.grib2.bz2`. This code uses
[`rclone`](https://rclone.org) under the hood. Install `rclone` on `Ubuntu` with `sudo snap install
rclone`.

### Testing FTP transfer locally 

The simple script in `run_copy_test.py` can be run locally with `uv run run_copy_test.py
/path/to/local/data/directory`. Optionally, for the purposes of testing, you can pass in
`--max-files n` to only transfer up to `n` files per NWP variable. See `run_copy_test.py --help` for
more options.
