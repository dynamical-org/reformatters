## ECCC

This directory contains code for downloading and processing NWP data from Environment
and Climate Change Canada (ECCC), starting with the High Resolution Deterministic
Prediction System (HRDPS) continental (2.5 km) domain.

## Archiving live HRDPS gribs

ECCC's MSC Datamart (`https://dd.weather.gc.ca`) only keeps a rolling ~30 day
window of GRIB2 files. `hrdps/archive_gribs/copy_files_from_eccc.py` copies files from
the Datamart to a public Source Co-Op bucket using [`rclone`](https://rclone.org),
preserving the Datamart's own `{date}/{init_hour}` directory structure.

### Testing locally

`rclone` must be on `PATH` at `/usr/bin/rclone`. Test with:

```sh
uv run main eccc-hrdps-forecast archive-grib-files --dst-root-path=/local/path
```

To test uploading to a cloud bucket, `--dst-root-path` can start with an `rclone`
remote, in the form `--dst-root-path=remote:path`.
