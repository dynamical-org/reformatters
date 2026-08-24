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

## Reformatting HRDPS to Zarr

`hrdps/forecast` reformats 17 HRDPS continental variables into the
`eccc-hrdps-forecast` dataset. Backfills read the Source Co-Op archive, the only
source that reaches back beyond the Datamart's rolling ~30 day window. Operational
updates read the Datamart directly, which has each complete run about 4 hours after
its init time, hours before the archive job above mirrors it.

### MSC Datamart request limits

ECCC's [MSC Open Data usage policy](https://eccc-msc.github.io/open-data/usage-policy/readme_en/)
states no concurrency or bandwidth limit. It asks users of 86,400 requests per day or
more (about 1 request per second sustained) to contact them, requires a meaningful
HTTP `User-Agent` (`http_download_to_disk` sends one), and directs systematic
retrieval away from directory listings — we construct every URL from the init time,
lead time and variable, and never list.

Measured against `dd.weather.gc.ca` in August 2026, over ~850 requests at concurrency
4, 8, 16 and 32, in both ascending and descending order: no 429, no 503, no
connection resets, every request 200. Aggregate throughput plateaued at 8 concurrent
downloads (~20-45 MB/s) and did not improve above it, while per-request p50 latency
grew from 0.12s at 4 concurrent to 0.67s at 32. So the region job downloads 8 files at
a time: past that we add load without gaining throughput.

Each update reprocesses the newest init time and the one before it, 828 files per init
time (12 variables x 49 lead times, plus 5 that start at lead time 1), so about 1,656
files and 3 GB per update and 6,624 requests per day across the four updates. The
archive job dominates our footprint by comparison: it copies every field of every run,
about 20,000 files per run and 80,000 per day. The two together sit just under ECCC's
86,400 requests per day threshold for getting in touch with them, so weigh that before
adding runs, lowering the update schedule's period, or widening what the archive job
copies.
