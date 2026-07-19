import struct
from pathlib import Path


def grib_decimal_scale_factors(path: Path) -> list[int]:
    """Decimal scale factor D from each GRIB2 data representation section (section 5)
    in the file, in field order (GDAL/rasterio band order). A GRIB2 field's values are
    (R + X * 2^E) / 10^D; each field is asserted to have binary scale factor E == 0 and
    an integer reference value R, which together guarantee its values are exact
    multiples of 10^-D.
    """
    # A GRIB2 file is a sequence of messages. Byte layout behind the offsets below
    # (0-based from the message or section start; multi-byte integers big-endian):
    #
    #   Section 0 (indicator, fixed 16 bytes):
    #     [0:4]    b"GRIB"
    #     [8:16]   total message length
    #   Sections 1-7 (sections 4-7 repeat per field within a message):
    #     [0:4]    section length
    #     [4]      section number
    #   Section 5 (data representation), following that 5-byte section header:
    #     [5:9]    number of data points
    #     [9:11]   data representation template number
    #     [11:15]  reference value R (IEEE float32)
    #     [15:17]  binary scale factor E (sign-and-magnitude int16)
    #     [17:19]  decimal scale factor D (sign-and-magnitude int16)
    #   End section: the literal bytes b"7777" terminate each message.
    #
    # R/E/D sit at the same octets in all common templates (5.0 simple, 5.2/5.3
    # complex, 5.40 JPEG2000, 5.41 PNG), so no branching on template is needed.
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
                (reference_value,) = struct.unpack(">f", data[pos + 11 : pos + 15])
                assert reference_value == round(reference_value), (
                    f"Non-integer reference value {reference_value} in {path}; "
                    "values are not multiples of 10^-D"
                )
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
