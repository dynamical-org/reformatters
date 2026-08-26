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
directory. It also skips initializations less than six hours old; at the four scheduled
run times, this leaves 22 date and initialization-hour pairs to examine per day. The
result is about 1,100 enumeration requests per day. Downloads and reformatter updates
bring the total to about 88,000 requests per day, so contacting MSC is still required.

Without HEAD requests, rclone does not know source file sizes or modification times
while listing. New objects therefore have an epoch rclone modification time, while
objects archived before this change retain the Datamart time. Do not use rclone-visible
modification times across this transition for age or inventory decisions. A 32 MiB
streaming upload cutoff buffers each current GRIB before upload so its unknown source
size does not force an S3 multipart upload.

`--ignore-existing` still unconditionally skips paths already present at the
destination. Because an unknown source size also prevents rclone's post-transfer size
comparison, the archiver waits until an initialization is at least six hours old. The
scheduled cron therefore archives a run at about init+10h, more than six hours after
the observed init+3h49m p99 publication time. If an exceptionally late file were still
being published then, later archive runs would not repair it automatically; this
residual risk remains.

### Testing locally

`rclone` must be on `PATH` at `/usr/bin/rclone`. Test with:

```sh
uv run main eccc-hrdps-forecast archive-grib-files --dst-root-path=/local/path
```

To test uploading to a cloud bucket, `--dst-root-path` can start with an `rclone`
remote, in the form `--dst-root-path=remote:path`.
