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

### Key design choices

- **Codec**: `GribberishCodec` (zarr v3 `ArrayBytesCodec`) decodes raw GRIB2 messages at read time
- **Chunk shape**: `(1, 1, 721, 1440)` — one chunk per GRIB message, matching the native grid
- **No shards**: Virtual references map 1:1 to GRIB messages, sharding is unnecessary
- **No compression**: GribberishCodec handles the full decode pipeline; no additional compressors or filters

### Dataset structure

- **Dimensions**: `init_time` (4) x `lead_time` (7) x `latitude` (721) x `longitude` (1440)
- **Variables**: `temperature_2m`, `pressure_surface`, `wind_u_10m` (3 instant variables)
- **Init times**: 2026-03-10 00Z, 06Z, 12Z, 18Z
- **Lead times**: 0h through 6h (hourly)

## Prototype phases and logs

### Phase 1: Initialize and backfill

Backfill 2 init times fully (all 7 lead times) and 1 init time partially (lead times 0-4h only).
The 4th init time slot exists in the array shape but has no virtual references yet.

```
PHASE 1: Initialize and backfill
Zarr metadata written
Backfilling init_time=2026-03-10 00:00:00 (all lead times)
Backfilling init_time=2026-03-10 06:00:00 (all lead times)
Backfilling init_time=2026-03-10 12:00:00 (partial: lead times 0-4h only)
Phase 1 committed: V2PJMJ40H0GZQ2F7KFSG
```

**Dataset state after Phase 1:**

```
Dimensions: {'init_time': 4, 'lead_time': 7, 'latitude': 721, 'longitude': 1440}
Init times: [2026-03-10 00:00, 2026-03-10 06:00, 2026-03-10 12:00, 2026-03-10 18:00]
Lead times: [0h, 1h, 2h, 3h, 4h, 5h, 6h]
Data variables: [temperature_2m, wind_u_10m, pressure_surface]
```

| init_time | temperature_2m | wind_u_10m | pressure_surface |
|---|---|---|---|
| 2026-03-10 00:00 | 1.0000 (7267680/7267680) | 1.0000 | 1.0000 |
| 2026-03-10 06:00 | 1.0000 (7267680/7267680) | 1.0000 | 1.0000 |
| 2026-03-10 12:00 | 0.7143 (5191200/7267680) | 0.7143 | 0.7143 |
| 2026-03-10 18:00 | 0.0000 (0/7267680) | 0.0000 | 0.0000 |

The 3rd init time shows 5/7 = 0.7143 non-NaN fraction (lead times 0-4h populated, 5-6h missing).
The 4th init time is entirely NaN (no virtual references set yet).

### Phase 2: Add a new init time

Add all 7 lead times for the 4th init time (2026-03-10 18:00Z).

```
PHASE 2: Add new init time
Adding init_time=2026-03-10 18:00:00 (all lead times)
Phase 2 committed: ZSPT9SW7K2W8YR5N09EG
```

**Dataset state after Phase 2:**

| init_time | temperature_2m | wind_u_10m | pressure_surface |
|---|---|---|---|
| 2026-03-10 00:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 06:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 12:00 | 0.7143 | 0.7143 | 0.7143 |
| 2026-03-10 18:00 | 1.0000 | 1.0000 | 1.0000 |

The 4th init time is now fully populated. The 3rd remains incomplete.

### Phase 3: Fill missing lead times

Add lead times 5h and 6h to the previously incomplete 3rd init time (2026-03-10 12:00Z),
simulating updates arriving as a forecast is produced by the source agency.

```
PHASE 3: Add lead times to incomplete init time
Filling in lead times [5h, 6h] for init_time=2026-03-10 12:00:00
Phase 3 committed: W75YVXJFNGWAQKX3TGG0
```

**Dataset state after Phase 3:**

| init_time | temperature_2m | wind_u_10m | pressure_surface |
|---|---|---|---|
| 2026-03-10 00:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 06:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 12:00 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03-10 18:00 | 1.0000 | 1.0000 | 1.0000 |

All init times and lead times are now fully populated.

## Snapshot history

```
W75YVXJFNGWA  2026-03-14 17:42:54  Phase 3: Fill missing lead times for 3rd init time
ZSPT9SW7K2W8  2026-03-14 17:42:43  Phase 2: Add 4th init time
V2PJMJ40H0GZ  2026-03-14 17:42:31  Phase 1: Initialize and backfill 3 init times (3rd partial)
1CECHNKREP0F  2026-03-14 17:42:29  Repository initialized
```

## Repository size

```
On-disk repository size: 15.4 KB
```

The entire repository is 15.4 KB on disk. Data variable chunks are virtual references to
S3 GRIB files — no weather data is stored locally. The repository contains only:
- Icechunk metadata (snapshots, manifests, chunk references)
- Coordinate arrays (init_time, lead_time, latitude, longitude)

For comparison, the same data stored as float32 arrays would be:
`3 vars x 4 init_times x 7 lead_times x 721 lat x 1440 lon x 4 bytes = ~350 MB`

## Observations

1. **Write speed**: Setting virtual references is extremely fast (~1s per phase) since only byte offsets are recorded, not actual data.
2. **Read speed**: Reading data requires fetching GRIB bytes from S3 and decoding with gribberish at read time. Each chunk read fetches one GRIB message (~1MB compressed).
3. **Incremental updates**: Icechunk's transactional model naturally supports adding new init times and filling in missing lead times as separate commits with snapshot history.
4. **Index parsing**: Our existing `grib_message_byte_ranges_from_index` works directly — it parses `.idx` files to extract byte offsets per variable, which map 1:1 to virtual chunk references.
5. **Chunk alignment**: GFS GRIB messages are naturally one-per-file-per-variable, so the chunk shape `(1, 1, 721, 1440)` aligns perfectly with the virtual reference model.
