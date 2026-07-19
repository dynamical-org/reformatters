from pathlib import Path


def grib_decimal_scale_factors(path: Path) -> list[int]:
    """Decimal scale factor D from each GRIB2 data representation section (section 5)
    in the file, in field order (GDAL/rasterio band order). A GRIB2 field's values are
    (R + X * 2^E) / 10^D; each field's binary scale factor E is asserted to be 0, so
    its values are exact multiples of 10^-D.
    """
    data = path.read_bytes()
    scale_factors: list[int] = []
    pos = 0
    while pos < len(data):
        assert data[pos : pos + 4] == b"GRIB", (
            f"Expected GRIB message start at byte {pos} in {path}"
        )
        message_end = pos + int.from_bytes(data[pos + 8 : pos + 16])
        pos += 16
        while pos < message_end and data[pos : pos + 4] != b"7777":
            section_length = int.from_bytes(data[pos : pos + 4])
            if data[pos + 4] == 5:
                binary_scale = _sign_and_magnitude_int(data[pos + 15 : pos + 17])
                assert binary_scale == 0, (
                    f"Binary scale factor {binary_scale} != 0 in {path}; "
                    "values are not multiples of 10^-D"
                )
                scale_factors.append(_sign_and_magnitude_int(data[pos + 17 : pos + 19]))
            pos += section_length
        pos = message_end

    assert scale_factors, f"No data representation sections found in {path}"
    return scale_factors


def _sign_and_magnitude_int(raw: bytes) -> int:
    """GRIB2 signed integers use a sign bit plus magnitude, not two's complement."""
    value = int.from_bytes(raw)
    sign_bit = 1 << (len(raw) * 8 - 1)
    return -(value & ~sign_bit) if value & sign_bit else value
