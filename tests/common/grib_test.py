import struct
from pathlib import Path

import pytest

from reformatters.common.grib import grib_decimal_scale_factors


def _sign_and_magnitude_bytes(value: int) -> bytes:
    return ((0x8000 | -value) if value < 0 else value).to_bytes(2)


def _grib2_message(fields: list[tuple[int, int]]) -> bytes:
    """Minimal GRIB2 message with a (binary_scale, decimal_scale) section 5 per field."""
    body = (21).to_bytes(4) + bytes([1]) + bytes(16)  # section 1
    for binary_scale, decimal_scale in fields:
        body += (9).to_bytes(4) + bytes([4]) + bytes(4)  # section 4
        section5_payload = (
            (4).to_bytes(4)  # number of data points
            + (41).to_bytes(2)  # data representation template (PNG packing)
            + struct.pack(">f", -30.0)  # reference value R
            + _sign_and_magnitude_bytes(binary_scale)
            + _sign_and_magnitude_bytes(decimal_scale)
            + bytes([16, 0])  # bits per value, original field type
        )
        body += (5 + len(section5_payload)).to_bytes(4) + bytes([5]) + section5_payload
        body += (5).to_bytes(4) + bytes([7])  # section 7
    header = b"GRIB" + bytes(2) + bytes([0, 2]) + (16 + len(body) + 4).to_bytes(8)
    return header + body + b"7777"


def test_grib_decimal_scale_factors_single_field(tmp_path: Path) -> None:
    path = tmp_path / "single.grib2"
    path.write_bytes(_grib2_message([(0, 1)]))
    assert grib_decimal_scale_factors(path) == [1]


def test_grib_decimal_scale_factors_multiple_messages_and_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multi.grib2"
    path.write_bytes(_grib2_message([(0, 2), (0, 0)]) + _grib2_message([(0, -1)]))
    assert grib_decimal_scale_factors(path) == [2, 0, -1]


def test_grib_decimal_scale_factors_rejects_nonzero_binary_scale(
    tmp_path: Path,
) -> None:
    path = tmp_path / "binary_scaled.grib2"
    path.write_bytes(_grib2_message([(-2, 1)]))
    with pytest.raises(AssertionError, match="Binary scale factor"):
        grib_decimal_scale_factors(path)


def test_grib_decimal_scale_factors_rejects_non_grib(tmp_path: Path) -> None:
    path = tmp_path / "not_a.grib2"
    path.write_bytes(b"NOPE" + bytes(20))
    with pytest.raises(AssertionError, match="Expected GRIB message start"):
        grib_decimal_scale_factors(path)


def test_grib_decimal_scale_factors_real_negative_decimal_scale(
    tmp_path: Path,
) -> None:
    path = tmp_path / "neg_d.grib2"
    path.write_bytes(_grib2_message([(0, -3)]))
    assert grib_decimal_scale_factors(path) == [-3]
