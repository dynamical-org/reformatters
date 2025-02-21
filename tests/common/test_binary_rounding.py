import numpy as np
import pytest

from reformatters.common.binary_rounding import round_float32


def test_round_float32_negative_values() -> None:
    values = np.linspace(-1000, -1, num=1000, dtype=np.float32)
    rounded_values = round_float32(values, keep_mantissa_bits=9)
    # All within 2% of the original value
    assert np.allclose(values, rounded_values, rtol=0.02)
    # Near zero bias introduced by rounding
    assert np.isclose(np.mean(values), np.mean(rounded_values))


def test_round_float32_positive_values() -> None:
    values = np.linspace(1, 1000, num=1000, dtype=np.float32)
    rounded_values = round_float32(values, keep_mantissa_bits=9)
    assert np.allclose(values, rounded_values, rtol=0.02)
    assert np.isclose(np.mean(values), np.mean(rounded_values))


def test_round_float32_negative_to_positive_values() -> None:
    values = np.linspace(-1e7, 1e7, num=2000, dtype=np.float32)
    rounded_values = round_float32(values, keep_mantissa_bits=9)
    assert np.allclose(values, rounded_values, rtol=0.02)
    assert np.isclose(np.mean(values), np.mean(rounded_values), rtol=0.02, atol=0.5)


def test_round_float32_special_values() -> None:
    special_values = np.array([0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    rounded_values = round_float32(special_values, keep_mantissa_bits=2)
    assert np.array_equal(special_values, rounded_values, equal_nan=True)


def test_round_float32_overflow_to_inf() -> None:
    # np.finfo(np.float32).max is the largest representable float32 value
    max_float32 = np.array([np.finfo(np.float32).max], dtype=np.float32)
    # With only 3 mantissa bits kept, this should overflow to infinity
    rounded_values = round_float32(max_float32, keep_mantissa_bits=3)
    assert np.all(np.isinf(rounded_values))


def test_round_float32_overflow_to_neg_inf() -> None:
    # np.finfo(np.float32).min is the most negative representable float32 value
    min_float32 = np.array([np.finfo(np.float32).min], dtype=np.float32)
    # With only 3 mantissa bits kept, this should overflow to negative infinity
    rounded_values = round_float32(min_float32, keep_mantissa_bits=3)
    assert np.all(np.isinf(rounded_values))
    assert np.all(rounded_values < 0)


if __name__ == "__main__":
    pytest.main()
