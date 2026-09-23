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
    # Some archived source files pad after a message with junk bytes (e.g. Iowa Mesonet
    # MRMS files from 2014-2015 append zero padding), so locate each message by its
    # magic bytes rather than assuming messages are contiguous; GDAL scans the same way.
    while (pos := data.find(b"GRIB", pos)) != -1:
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


# GRIB2 section 0: b"GRIB", 2 reserved bytes, discipline, edition, then the message's
# total length as a big endian u64.
GRIB_SECTION_0_BYTES = 16


def grib2_message_length(section_0: bytes) -> int | None:
    """The total length an edition 2 section 0 declares, or None if the bytes are not
    one."""
    if (
        len(section_0) < GRIB_SECTION_0_BYTES
        or section_0[:4] != b"GRIB"
        or section_0[7] != 2
    ):
        return None
    return int.from_bytes(section_0[8:GRIB_SECTION_0_BYTES])


def grib_message_offsets(path: Path) -> list[int] | None:
    """The start byte of every GRIB2 message in the file, or None if the file is not a
    non-empty sequence of complete edition 2 messages tiling it exactly. A file cut
    between messages still passes."""
    size = path.stat().st_size
    offset = 0
    offsets: list[int] = []
    with path.open("rb") as f:
        while offset < size:
            f.seek(offset)
            length = grib2_message_length(f.read(GRIB_SECTION_0_BYTES))
            if (
                length is None
                or length < GRIB_SECTION_0_BYTES
                or offset + length > size
            ):
                return None
            f.seek(offset + length - 4)
            if f.read(4) != b"7777":
                return None
            offsets.append(offset)
            offset += length
    return offsets or None
