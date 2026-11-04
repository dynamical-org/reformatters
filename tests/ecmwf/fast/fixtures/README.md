# ECMWF fast integration fixtures

Verbatim GRIB message slices from ECMWF deliveries initialized at
2026-11-04T00:00Z and 2026-11-04T06:00Z. The sixteen files cover both models,
control member 0 and perturbed member 1, and temperature at 2 m. IFS leads are 0
and 1 h; AIFS leads are 0 and 6 h.

The source messages retain their exact bytes and native 721 × 1440 grid. They are
selected by byte range and concatenated without re-encoding, cropping, repacking,
or compression wrappers. Total GRIB size is 17,454,373 bytes: 12,459,960 for IFS
and 4,994,413 for AIFS.

`metadata.json` records original source keys, selected source byte ranges,
identities, and numerical snapshots at latitude 0 / longitude 0. Snapshots include
the template's Kelvin-to-Celsius conversion. Each original file passed its full
seven-field control or 350-field perturbed inventory before selection. Each slice
then passed the same production scanner against its exact one-field, one-member
declared inventory.

`fixture_integration_test.py` reads these files exclusively through a local
Icechunk virtual container and `file://` references. Slice offsets refer to these
files and must never be combined with the original source URLs. The test runs the
ordinary backfill and update paths, appends the second initialization on update,
checks numerical snapshots and historical snapshot isolation, and preserves
native virtual chunk geometry. No credentials are needed.

The synthetic parser and scaling tests cover precipitation; synthetic parser tests
and separate authenticated source probes cover 360 h GRIB1 forecast-time encoding.
The fixtures concentrate on the two earliest steps.
`real_source_test.py` provides additional opt-in coverage of complete authenticated
files and all seven quantities. It requires `ECMWF_FAST_REAL_SOURCE_TEST=1` and
explicit source credentials in the caller's environment, forbids Kubernetes Secret
reads, and fails on missing or invalid configuration once enabled.
