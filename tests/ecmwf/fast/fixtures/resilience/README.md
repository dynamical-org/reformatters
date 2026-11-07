# Native precipitation, radiation, and wind messages

These six files are verbatim single-message byte slices from ECMWF IFS and AIFS
control forecasts initialized at 2026-11-04T00:00Z, valid at 06:00Z, member 0.
Each model contributes total precipitation (`tp`), accumulated downward short-wave
radiation (`ssrd`), and the 10 m U wind component (`10u`). All retain the native
721 × 1440 grid and original packing. No messages are re-encoded, repacked, or
cropped. The six GRIB messages total 8,157,984 bytes.

| Model | Parameter | Bytes | Packing | Bits per value |
| --- | --- | ---: | --- | ---: |
| IFS | tp | 2,076,588 | GRIB1 simple | 16 |
| IFS | ssrd | 2,076,588 | GRIB1 simple | 16 |
| IFS | 10u | 1,557,468 | GRIB1 simple | 12 |
| AIFS | tp | 752,401 | GRIB2 CCSDS (template 5.42) | 16 |
| AIFS | ssrd | 885,529 | GRIB2 CCSDS (template 5.42) | 16 |
| AIFS | 10u | 809,410 | GRIB2 CCSDS (template 5.42) | 12 |

`metadata.json` records each original key in `s3://dynamical-ecmwf/`, its original
byte offset and length, the message SHA-256, units, packing scales, and decoded
snapshots. Source offsets locate messages in the original objects; each fixture
contains exactly one message at offset zero.

During acquisition, every cell decoded by Gribberish matched rasterio/GDAL exactly
with `GRIB_NORMALIZE_UNITS=NO`. The template's GribberishCodec and filter chain
also matched the complete GDAL field exactly after output unit conversion. IFS
precipitation is stored in metres and converted to kg m-2 by the template's
ScaleOffset filter (`scale=0.001`); AIFS precipitation already uses kg m-2.
Radiation uses J m-2 (equivalent to the output W s m-2), and wind uses m s-1.
Both accumulated fields contain genuine zero cells as well as nonzero values.
GDAL's unit labels for AIFS accumulated fields differ from the template and
Gribberish labels; metadata preserves them verbatim. The oracle comparison uses
unmodified numerical arrays and the template's explicit output conversion.

`grib_real_messages_test.py` requires no credentials or network access. It declares
one parameter and member per scanner inventory, checks the native message identity
and checksum, then decodes the complete field through the template's codec and
filters. It checks finite values, extrema, zero counts, and stable pixel snapshots
without repeating the GDAL oracle comparison in CI. Forecast lead comes from the
source coordinate and scanner rather than Gribberish time metadata.
