## ECCC

This directory contains code for downloading and processing NWP data from Environment
and Climate Change Canada (ECCC), starting with the High Resolution Deterministic
Prediction System (HRDPS) continental (2.5 km) domain.

## Archiving live HRDPS gribs

ECCC's MSC Datamart (`https://dd.weather.gc.ca`) only keeps a rolling ~30 day
window of GRIB2 files. `hrdps/archive_gribs/copy_files_from_eccc.py` copies files from
the Datamart to a public Source Co-Op bucket using [`rclone`](https://rclone.org),
preserving the Datamart's own `{date}/{init_hour}` directory structure.

### MSC Datamart request volume

ECCC's [MSC Open Data usage policy](https://eccc-msc.github.io/open-data/usage-policy/readme_en/)
asks users making at least 86,400 requests per day to contact them and directs systematic
retrieval away from directory listings.

The HRDPS archive contains about 20,000 files per run. Without `--http-no-head`,
rclone's HTTP backend sends one HEAD request per entry while traversing the Datamart's
directory listings. The archive cron examines eight date and initialization-hour pairs
per invocation and runs four times per day, producing about 640,000 HEAD requests in
addition to the day's roughly 80,000 file downloads. Reformatter updates add about
6,600 requests, for a total near 730,000 requests per day.

The archive enables `--http-no-head`, reducing traversal to roughly one GET per
directory, or about 1,600 enumeration requests per day. Downloads and reformatter
updates bring the resulting total to about 88,000 requests per day, so contacting MSC
is still required.

Without HEAD requests, rclone does not know source file sizes or modification times
while listing. `--ignore-existing` still unconditionally skips paths already present at
the destination, but rclone cannot use an age filter to avoid a source file that has
become visible while it is still being published. If such a file were copied, later
archive runs would not repair it automatically. The archive schedule normally starts
after a complete run is available, but this residual risk remains.

### Testing locally

`rclone` must be on `PATH` at `/usr/bin/rclone`. Test with:

```sh
uv run main eccc-hrdps-forecast archive-grib-files --dst-root-path=/local/path
```

To test uploading to a cloud bucket, `--dst-root-path` can start with an `rclone`
remote, in the form `--dst-root-path=remote:path`.
