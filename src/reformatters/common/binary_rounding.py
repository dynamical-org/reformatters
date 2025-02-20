from typing import Final

import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.uint32]

# IEEE 754 binary32 constants
MANTISSA_BITS: Final[int] = 23
MANTISSA_MASK: Final[np.uint32] = np.uint32((1 << MANTISSA_BITS) - 1)
EXPONENT_MASK: Final[np.uint32] = np.uint32(((1 << 8) - 1) << (32 - 8 - 1))
SIGN_MASK: Final[np.uint32] = np.uint32(1 << 31)


def round_float32(value: FloatArray, keep_mantissa_bits: int) -> FloatArray:
    """Round float32 values to keep a specific number of mantissa bits.

    Vectorized implementation that works on numpy float32 arrays.

    Args:
        value: The float32 array to round
        keep_mantissa_bits: Number of mantissa bits to keep (1-23)

    Returns:
        Rounded float32 array
    """
    if keep_mantissa_bits >= MANTISSA_BITS:
        return value

    # Early return for any NaN or Inf values
    if np.any(~np.isfinite(value)):
        return value.copy()

    bits: IntArray = value.view(np.uint32)

    # Number of trailing bits to drop
    drop_bits = MANTISSA_BITS - keep_mantissa_bits

    # Extract components
    sign: IntArray = bits & SIGN_MASK
    exponent: IntArray = bits & EXPONENT_MASK
    mantissa: IntArray = bits & MANTISSA_MASK

    # Get rounding bits
    round_bit: IntArray = bits & np.uint32(1 << drop_bits)
    half_bit: IntArray = bits & np.uint32(1 << (drop_bits - 1))
    sticky_mask = np.uint32((1 << (drop_bits - 1)) - 1)
    sticky_bits: IntArray = bits & sticky_mask

    # Vectorized rounding logic
    round_down_mask = half_bit == 0
    round_up_mask = (~round_down_mask) & (sticky_bits != 0)
    round_even_mask = (~round_down_mask) & (sticky_bits == 0)

    # Apply rounding rules
    keep_mask = ~np.uint32((1 << drop_bits) - 1)
    increment = np.uint32(1 << drop_bits)

    # Round down (clear lower bits)
    mantissa &= keep_mask

    # Round up where needed
    mantissa[round_up_mask] += increment

    # Round to even where needed (if round bit is 1)
    round_even_up_mask = round_even_mask & (round_bit != 0)
    mantissa[round_even_up_mask] += increment

    # Handle mantissa overflow
    overflow_mask = mantissa > MANTISSA_MASK
    if np.any(overflow_mask):
        # Increment exponent and wrap mantissa
        exponent[overflow_mask] += np.uint32(1 << MANTISSA_BITS)
        mantissa[overflow_mask] &= MANTISSA_MASK

        # Check for exponent overflow
        inf_mask = exponent >= EXPONENT_MASK
        if np.any(inf_mask):
            result = value.copy()
            result[inf_mask] = np.where(sign[inf_mask] == 0, np.inf, -np.inf)
            return result

    # Recombine components
    result_bits = sign | exponent | mantissa
    return result_bits.view(np.float32)
