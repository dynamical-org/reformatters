# GFS Virtual Icechunk Dataset Prototype

Prototype demonstrating a virtual icechunk dataset backed by NOAA GFS GRIB2 files on S3.
Data variable chunks are virtual references to byte ranges within remote GRIB files,
decoded at read time by GribberishCodec. Only coordinate arrays and icechunk metadata
are stored locally.

Related: [Issue #509](https://github.com/dynamical-org/reformatters/issues/509)

## Approach

1. Create a local icechunk repository with a virtual chunk container pointing at `s3://noaa-gfs-bdp-pds/`
2. Write zarr v3 metadata and coordinate arrays (real data) to the store
3. For each init_time/lead_time combination:
   - Download the GRIB `.idx` index file
   - Parse byte ranges for each variable using our existing `grib_message_byte_ranges_from_index`
   - Set virtual references via `store.set_virtual_ref()` pointing to the S3 GRIB file at the parsed byte offset/length
4. Commit each phase as an icechunk snapshot
5. To add a new init_time, `resize()` the coordinate and data arrays then set virtual refs for the new chunks

### Key design choices

- **Codec**: `GribberishCodec` (zarr v3 `ArrayBytesCodec`) decodes raw GRIB2 messages at read time
- **Chunk shape**: `(1, 1, 721, 1440)` — one chunk per GRIB message, matching the native grid
- **No shards**: Virtual references map 1:1 to GRIB messages, sharding is unnecessary
- **No compression**: GribberishCodec handles the full decode pipeline; no additional compressors or filters

### Dataset structure

- **Dimensions**: `init_time` x `lead_time` (7) x `latitude` (721) x `longitude` (1440)
- **Variables**: `temperature_2m`, `pressure_surface`, `wind_u_10m` (3 instant variables)
- **Init times**: 2026-03-10 00Z, 06Z, 12Z, 18Z
- **Lead times**: 0h through 6h (hourly)

## Prototype phases and logs

### Phase 1: Backfill

Initialize the dataset with 3 init times, each with all 7 lead times fully populated.

```
PHASE 1: Initialize and backfill
Zarr metadata written
Backfilling init_time=2026-03-10 00:00:00 (all lead times)
Backfilling init_time=2026-03-10 06:00:00 (all lead times)
Backfilling init_time=2026-03-10 12:00:00 (all lead times)
Phase 1 committed: 7MWMJRTPSQXA6T1F03A0
```

**Dataset state after Phase 1:**

```
Dimensions: {'init_time': 3, 'lead_time': 7, 'latitude': 721, 'longitude': 1440}
Init times: [2026-03-10 00:00, 2026-03-10 06:00, 2026-03-10 12:00]
Lead times: [0h, 1h, 2h, 3h, 4h, 5h, 6h]
```

| init_time | temperature_2m | wind_u_10m | pressure_surface |
|---|---|---|---|
| 2026-03-10 00:00 | 1.0000 (7267680/7267680) | 1.0000 | 1.0000 |
| 2026-03-10 06:00 | 1.0000 (7267680/7267680) | 1.0000 | 1.0000 |
| 2026-03-10 12:00 | 1.0000 (7267680/7267680) | 1.0000 | 1.0000 |

All 3 init times are fully populated across all 7 lead times.

### Phase 2: Add a new init time (partial)

Resize the dataset to add a 4th init time (2026-03-10 18:00Z). This grows the
`init_time` coordinate and all data variable arrays. Only 3 of 7 lead times
(0h-2h) are filled, simulating a forecast that is still being produced.

```
PHASE 2: Add new init time (partial)
Resized dataset: init_time dimension is now 4
Adding init_time=2026-03-10 18:00:00 (lead times 0 days 00:00:00-0 days 02:00:00)
Phase 2 committed: ANS558Z79ZY93X14BYR0
```

**Dataset state after Phase 2:**

```
Dimensions: {'init_time': 4, 'lead_time': 7, 'latitude': 721, 'longitude': 1440}
Init times: [2026-03-10 00:00, 2026-03-10 06:00, 2026-03-10 12:00, 2026-03-10 18:00]
```

| init_time | temperature_2m | wind_u_10m | pressure_surface |
|---|---|---|---|
| 2026-03-10 00:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 06:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 12:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 18:00 | 0.4286 (3114720/7267680) | 0.4286 | 0.4286 |

The 4th init time shows 3/7 = 0.4286 non-NaN fraction (lead times 0-2h populated, 3-6h missing).
Missing chunks return NaN fill values automatically — no pre-allocation needed.

### Phase 3: Fill missing lead times

Add lead times 3h and 4h to the 4th init time, simulating updates arriving as
the source agency produces more forecast hours.

```
PHASE 3: Fill missing lead times
Filling in lead times [3h, 4h] for init_time=2026-03-10 18:00:00
Phase 3 committed: 15YA6YR7RF2PY5SCJKQG
```

**Dataset state after Phase 3:**

| init_time | temperature_2m | wind_u_10m | pressure_surface |
|---|---|---|---|
| 2026-03-10 00:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 06:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 12:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 18:00 | 0.7143 (5191200/7267680) | 0.7143 | 0.7143 |

The 4th init time now has 5/7 = 0.7143 non-NaN fraction. Lead times 5h and 6h
remain unfilled, as they would be in a real scenario where the forecast run
hasn't produced those hours yet.

## Snapshot history

```
15YA6YR7RF2P  2026-03-14 18:21:18  Phase 3: Fill remaining lead times for 4th init time
ANS558Z79ZY9  2026-03-14 18:21:09  Phase 2: Add 4th init time with partial lead times
7MWMJRTPSQXA  2026-03-14 18:20:57  Phase 1: Initialize and backfill 3 init times
1CECHNKREP0F  2026-03-14 18:20:55  Repository initialized
```

## Repository size

```
On-disk repository size: 15.5 KB
```

The entire repository is 15.5 KB on disk. Data variable chunks are virtual references to
S3 GRIB files — no weather data is stored locally. The repository contains only:
- Icechunk metadata (snapshots, manifests, chunk references)
- Coordinate arrays (init_time, lead_time, latitude, longitude)

For comparison, the same data stored as float32 arrays would be:
`3 vars x 4 init_times x 7 lead_times x 721 lat x 1440 lon x 4 bytes = ~350 MB`

## Observations

1. **Write speed**: Setting virtual references is extremely fast (~1s per phase) since only byte offsets are recorded, not actual data.
2. **Read speed**: Reading data requires fetching GRIB bytes from S3 and decoding with gribberish at read time. Each chunk read fetches one GRIB message (~1MB compressed).
3. **Incremental updates**: Icechunk's transactional model naturally supports resizing to add new init times and filling in lead times as separate commits with full snapshot history.
4. **Resize for new init times**: Growing the dataset along `init_time` requires calling `resize()` on the coordinate array and all data variable arrays, then writing the new coordinate value. New chunks default to NaN fill values until virtual references are set.
5. **Index parsing**: Our existing `grib_message_byte_ranges_from_index` works directly — it parses `.idx` files to extract byte offsets per variable, which map 1:1 to virtual chunk references.
6. **Chunk alignment**: GFS GRIB messages are naturally one-per-file-per-variable, so the chunk shape `(1, 1, 721, 1440)` aligns perfectly with the virtual reference model.
