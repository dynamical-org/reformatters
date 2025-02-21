from typing import Final

import numpy as np

from reformatters.common.types import ArrayFloat32

# IEEE 754 binary32 (float32) constants
MANTISSA_BITS: Final[int] = 23
MANTISSA_MASK: Final[np.uint32] = np.uint32((1 << MANTISSA_BITS) - 1)
EXPONENT_MASK: Final[np.uint32] = np.uint32(((1 << 8) - 1) << (32 - 8 - 1))
SIGN_MASK: Final[np.uint32] = np.uint32(1 << 31)


def round_float32_inplace(value: ArrayFloat32, keep_mantissa_bits: int) -> ArrayFloat32:
    """
    Round float32 values to keep a specific number of mantissa bits.
    This improves compression by creating more trailing zeros.

    Modifies the input array in place.

    Args:
        value: The float32 array to round
        keep_mantissa_bits: Number of mantissa bits to keep (1-23)

    Returns:
        Rounded float32 array
    """
    if keep_mantissa_bits < 0:
        raise ValueError("keep_mantissa_bits must be at least 0")
    if keep_mantissa_bits > MANTISSA_BITS:
        raise ValueError(f"keep_mantissa_bits must be less than {MANTISSA_BITS}")
    if keep_mantissa_bits == MANTISSA_BITS:
        return value

    bits: np.ndarray[tuple[int, ...], np.dtype[np.uint32]] = value.view(np.uint32)

    drop_bits = MANTISSA_BITS - keep_mantissa_bits

    mantissa = bits & MANTISSA_MASK

    round_bit = bits & np.uint32(1 << drop_bits)
    half_bit = bits & np.uint32(1 << (drop_bits - 1))
    sticky_bits = bits & np.uint32((1 << (drop_bits - 1)) - 1)

    # Apply rounding rules directly to mantissa
    keep_mask = ~np.uint32((1 << drop_bits) - 1)
    increment = np.uint32(1 << drop_bits)

    # clear lower bits
    mantissa &= keep_mask

    # Round up where needed
    round_up = (half_bit != 0) & (sticky_bits != 0)
    mantissa[round_up] += increment

    # Round to even where needed
    round_even = (half_bit != 0) & (sticky_bits == 0) & (round_bit != 0)
    mantissa[round_even] += increment

    # Handle mantissa overflow
    mantissa_overflow_mask = mantissa > MANTISSA_MASK
    if np.any(mantissa_overflow_mask):
        # Extract sign and exponent only where needed
        sign = bits & SIGN_MASK
        exponent = bits & EXPONENT_MASK

        # Increment exponent and wrap mantissa
        exponent[mantissa_overflow_mask] += np.uint32(1 << MANTISSA_BITS)
        mantissa[mantissa_overflow_mask] &= MANTISSA_MASK

        # Check for exponent overflow
        exponent_overflow_mask = exponent >= EXPONENT_MASK
        if np.any(exponent_overflow_mask):
            bits_float = bits.view(np.float32)
            bits_float[exponent_overflow_mask] = np.where(
                sign[exponent_overflow_mask] == 0, np.inf, -np.inf
            )

        # Recombine sign, exponent, and mantissa
        # bits = sign | exponent | mantissa
        bits = np.bitwise_or(sign, exponent, out=bits)
        bits = np.bitwise_or(bits, mantissa, out=bits)
    else:
        # No overflow - just update mantissa bits
        # bits = (bits & ~MANTISSA_MASK) | mantissa
        bits = np.bitwise_and(~MANTISSA_MASK, bits, out=bits)
        bits = np.bitwise_or(bits, mantissa, out=bits)

    return bits.view(np.float32)
