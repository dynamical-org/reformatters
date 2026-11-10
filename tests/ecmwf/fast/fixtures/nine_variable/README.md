# Native nine-variable test delivery messages

These eleven files are verbatim message slices from ECMWF test deliveries
initialized at 2026-11-10T00:00Z. They retain the complete 721 × 1440 grid.
`metadata.json` records the test-feed keys, offsets, lengths, SHA-256 checksums,
member numbers, product lengths, bit widths, units, output filters and independent
GDAL snapshots.

Six IFS control messages cover surface pressure (`sp`, 11 bits), 2 m dewpoint
(`2d`, 10 bits), precipitation and radiation (`tp` and `ssrd`, 14 bits at 6h),
and the two constant-zero 0h accumulations (zero-bit packing). Both new quantities
use surface level type 1, value 0. These GRIB1 messages have a 52-byte product
section. One 162-byte IFS member-1 zero-bit precipitation message separately pins
the 106-byte product section, ECMWF local definition 30, and member placement.

Four AIFS control messages cover `sp` (11 bits), `2d` (10 bits), `tp` at 6h
(14 bits) and `tp` at 0h (zero bits). All use GRIB2 CCSDS template 5.42 without
a bitmap. Pressure is instantaneous template 4.1 at the surface; dewpoint uses
template 4.1 at height 2 m; precipitation uses accumulated template 4.11.
The grid starts at longitude 180 and wraps to 179.75.

Source dewpoint values are Kelvin; the expected output subtracts 273.15.
IFS precipitation source values are metres; its expected output divides by 0.001.
AIFS precipitation already uses kg m-2. GDAL snapshots were captured with
`GRIB_NORMALIZE_UNITS=NO` so both editions retain Kelvin. Without that option,
GDAL converts GRIB2 temperatures to Celsius itself.

On acquisition, GDAL and gribberish agreed across every field and member of all
eight original files (cf and pf at 0h and 6h for both models). IFS comparisons
are exact; AIFS differences are within GDAL's float32 precision. The AIFS snapshot
absolute tolerance is one float32 ULP at the largest source magnitude; IFS uses
zero tolerance. An independent simple-packing unpacker also matched the IFS
control messages exactly.

Only messages needed to pin distinct new identities, packing widths and the
IFS perturbed product section are included. Earlier native-message fixtures
cover the previous delivery. `grib_real_messages_test.py` checks these bytes
through the scanner and configured Zarr codec/filter chain without network
access or credentials.
