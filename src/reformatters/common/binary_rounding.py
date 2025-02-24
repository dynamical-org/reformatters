from typing import Final

import numpy as np
from numba import njit, prange  # type: ignore

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

    Modifies the input array in place. Does not make any large allocations.

    keep_mantissa_bits must be between 0 (only keep the exponent)
    and 23 (no rounding), lower values round more.
    """
    if value.dtype != np.float32:
        raise ValueError("value must be a float32 ndarray")
    if keep_mantissa_bits < 0:
        raise ValueError("keep_mantissa_bits must be at least 0")
    if keep_mantissa_bits > MANTISSA_BITS:
        raise ValueError(f"keep_mantissa_bits must be {MANTISSA_BITS} or less")

    if keep_mantissa_bits == MANTISSA_BITS:
        return value

    bits = value.view(np.uint32)
    bits = _round_float32_inplace_numba(
        bits, keep_mantissa_bits, MANTISSA_BITS, MANTISSA_MASK, EXPONENT_MASK, SIGN_MASK
    )
    return bits.view(np.float32)


@njit(parallel=True)  # type: ignore
def _round_float32_inplace_numba(
    bits: np.ndarray[tuple[int, ...], np.dtype[np.uint32]],
    keep_mantissa_bits: int,
    # The following arguments are constants but numba can't access globals
    mantissa_bits: int,
    mantissa_mask: np.uint32,
    exponent_mask: np.uint32,
    sign_mask: np.uint32,
) -> np.ndarray[tuple[int, ...], np.dtype[np.uint32]]:
    drop_bits = mantissa_bits - keep_mantissa_bits
    keep_mask = ~np.uint32((1 << drop_bits) - 1)
    increment = np.uint32(1 << drop_bits)

    flat_bits = bits.ravel()  # modify 1D view in place

    for i in prange(len(flat_bits)):
        mantissa = flat_bits[i] & mantissa_mask
        round_bit = flat_bits[i] & np.uint32(1 << drop_bits)
        half_bit = flat_bits[i] & np.uint32(1 << (drop_bits - 1))
        sticky_bits = flat_bits[i] & np.uint32((1 << (drop_bits - 1)) - 1)

        # clear lower bits
        mantissa &= keep_mask

        # Round up where needed
        if (half_bit != 0) and (sticky_bits != 0):
            mantissa += increment

        # Round to even where needed
        if (half_bit != 0) and (sticky_bits == 0) and (round_bit != 0):
            mantissa += increment

        # Handle mantissa overflow
        if mantissa > mantissa_mask:
            sign = flat_bits[i] & sign_mask
            exponent = flat_bits[i] & exponent_mask

            # Increment exponent and wrap mantissa
            exponent += np.uint32(1 << mantissa_bits)
            mantissa &= mantissa_mask

            # Handle exponent overflow
            if exponent >= exponent_mask:
                inf = np.inf if sign == 0 else -np.inf
                flat_bits[i] = np.float32(inf).view(np.uint32)
            else:
                # Recombine sign, exponent, and mantissa
                flat_bits[i] = sign | exponent | mantissa
        else:
            # No mantissa overflow - just update mantissa bits
            flat_bits[i] = (flat_bits[i] & ~mantissa_mask) | mantissa

    return bits  # Return modified input array
