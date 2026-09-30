# MARS raw-unit regression fixtures

These fixtures derive from ECMWF ENS control forecasts initialized at
2024-01-04 00:00 UTC, step 6 hours, staged in the public
[Source Cooperative archive](https://data.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/2024-01-04/).
`audit.json` records each source URL, inclusive byte range, index record, original
SHA-256, GRIB edition/packing, decoder versions, and full-field audit results.

Each `.grib.gz` preserves the source message's parameter/level headers and
721 × 1440 grid. To keep fixtures small, ecCodes decoded the original message,
retained nine samples at source rows 120/360/600 and columns 120/720/1200,
zeroed all other values, and repacked with `codes_set_values` followed by
`codes_get_message`. The result was gzip-compressed with `mtime=0`.
These are **derived sparse fields**, not unmodified weather fields.
`fixture_sha256` hashes the decompressed GRIB. GRIB quantization can shift the
background slightly away from zero; `background_value` and `samples` are from
an independent ecCodes decode of the repacked message. Sample columns include
GDAL's 720-column longitude rotation. The fixtures require no ecCodes package
or network access at test time.

The full original fields were compared cell-for-cell: with
`GRIB_NORMALIZE_UNITS=NO`, GDAL float64 values equaled ecCodes exactly for all
19 configured outputs. With `YES`, only pressure-level temperatures changed
numerically (−273.15); 2t/2d acquired Celsius labels while retaining Kelvin
values. The surface-temperature reproducer pins this failure mode.

| Outputs | Raw GDAL units | Conversion to output |
| --- | --- | --- |
| Surface and mean sea level pressure | Pa | None |
| 2 m temperature and dew point, 850/925 hPa temperature | K | Subtract 273.15 during each MARS read |
| 10 m U/V wind | m/s | None |
| 100 m U/V wind and 10 m gust | Undefined (`[-]`) | None; ecCodes identifies m/s |
| Total precipitation | m | Existing ×1000 and lead-time deaccumulation |
| Precipitation type | Category table | None |
| Downward long/short-wave radiation | W*s/m² | Existing lead-time deaccumulation |
| 500/850/925 hPa geopotential | m²/s² | Divide by 9.80665 during each MARS read |
| Total cloud cover | Fraction (`[-]`) | Existing ×100 |

MARS conversion uses float64 arithmetic before a single float32 cast. The tests
compare the entire result exactly to independently decoded fixture values
converted in that order. Scaling and offsets run per source coordinate because
a shard can cross the MARS/Open Data boundary. Open Data retains GDAL's default
decoding; its download/read test checks bitwise equality to a direct float32
GDAL read for every variable.

This audit covers one date, control member and lead. It does not establish
all-era source integrity, ensemble coverage, gust windows or cutover continuity;
those remain historical-backfill validation requirements.
