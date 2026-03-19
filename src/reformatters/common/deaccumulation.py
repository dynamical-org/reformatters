import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange

from reformatters.common.types import Array1D, ArrayFloat32

# OK to add units to this list if you believe they are reasonable output units to deaccumulate to.
# We typically expect these to be per-second rates.
VALID_OUTPUT_UNITS_FOR_DEACCUMULATION = ["mm s-1", "m s-1", "kg m-2 s-1", "W m-2"]

# mm s-1, ~= 0.25 mm/h or "trace" precipitation
PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD = -7e-5
RADIATION_INVALID_BELOW_THRESHOLD = -50.0  # W/m^2 aka J/m^2/s


def deaccumulate_to_rates_inplace(
    data_array: xr.DataArray,
    *,
    dim: str,
    reset_frequency: pd.Timedelta,
    skip_step: Array1D[np.bool] | None = None,
    invalid_below_threshold_rate: float = PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
    expected_invalid_fraction: float = 0.0,
    expected_clamp_fraction: float = 0.05,
) -> xr.DataArray:
    """
    Convert accumulated values to per-second rates in place.

    Args:
        data_array: Array containing accumulated values to be converted to rates in place
        dim: Dimension over which to deaccumulate
        reset_frequency: Frequency at which the accumulation resets (eg 6 hours for GEFS and GFS)
        skip_step: Array of booleans indicating whether to skip the step. Values in skipped
            steps are left unchanged and the deaccumulation acts as if they are not present.
        invalid_below_threshold_rate: Threshold below which values are considered invalid
        expected_invalid_fraction: Fraction of values expected to be invalid (set to NaN)
            after deaccumulation. For example, MRMS data has ~6% no-data sentinel values
            that become invalid after deaccumulation. Only raises ValueError if the actual
            invalid fraction exceeds this expected amount.
        expected_clamp_fraction: Fraction of values expected to be clamped to 0 (small
            negative rates from lossy compression artifacts). Raises ValueError if actual
            clamped fraction exceeds this.
    """
    assert data_array.attrs["units"] in VALID_OUTPUT_UNITS_FOR_DEACCUMULATION, (
        "Output units must be a per-second rate"
    )

    # Support timedelta or datetime dimension values, converting either to seconds
    times = data_array[dim].values
    if np.issubdtype(times.dtype, np.datetime64):
        start_time = np.datetime64(pd.Timestamp(times[0]).floor(reset_frequency))  # type: ignore[arg-type]
        timedeltas = times - start_time
    else:
        assert np.issubdtype(times.dtype, np.timedelta64)
        timedeltas = times

    seconds = timedeltas.astype("timedelta64[s]").astype(np.int64)

    reset_after = (seconds % reset_frequency.total_seconds()) == 0

    if skip_step is None:
        skip_step = np.zeros(seconds.shape, dtype=np.bool)
    else:
        assert skip_step.shape == seconds.shape

    # make array 3D with shape (flattend_leading_dims, lead_time, flattend_trailing_dims)
    time_dim_index = data_array.dims.index(dim)
    values = data_array.values.reshape(
        np.prod(data_array.shape[:time_dim_index] or 1),
        data_array.shape[time_dim_index],
        np.prod(data_array.shape[time_dim_index + 1 :] or 1),
    )

    invalid_negative_count, clamped_count = _deaccumulate_to_rates_numba(
        values, seconds, reset_after, skip_step, invalid_below_threshold_rate
    )

    invalid_fraction = invalid_negative_count / values.size
    if invalid_fraction > expected_invalid_fraction:
        raise ValueError(
            f"Found {invalid_negative_count} values ({invalid_fraction:.1%}) below threshold, "
            f"expected at most {expected_invalid_fraction:.1%}"
        )
    clamped_fraction = clamped_count / values.size
    if clamped_fraction > expected_clamp_fraction:
        raise ValueError(
            f"Over {expected_clamp_fraction:.0%} ({clamped_count} total, {clamped_fraction:.1%}) values were clamped to 0"
        )

    return data_array


@njit(parallel=True)
def _deaccumulate_to_rates_numba(
    values: ArrayFloat32,
    seconds: Array1D[np.int64],
    reset_after: Array1D[np.bool],
    skip_step: Array1D[np.bool],
    invalid_below_threshold_rate: float,
) -> tuple[int, int]:
    """
    Convert accumulated values to per-second rates, mutating `values` in place.

    Accumulations should only go up. If they go down a tiny bit,
    most likely due to numerical precision issues, we clamp to 0.
    If they go down to a *rate* that is less than `invalid_below_threshold`,
    this sets the value to NaN.

    Returns (invalid_negative_count, clamped_count) for the caller to decide whether to raise.

    Parallel processing is done over the leading dimension of values.
    """
    assert values.ndim == 3
    assert seconds.ndim == 1
    assert reset_after.ndim == 1
    assert skip_step.ndim == 1
    assert values.shape[1] == seconds.size
    assert values.shape[1] == reset_after.size
    assert values.shape[1] == skip_step.size

    n_lead_times = values.shape[1]

    invalid_negative_count = 0
    clamped_count = 0

    for i in prange(values.shape[0]):  # ty: ignore[not-iterable]
        for j in range(values.shape[2]):
            sequence = values[i, :, j]
            previous_seconds = seconds[0]

            # If first step is a reset point, accumulation starts from 0.
            # Otherwise, use the actual value at first step as baseline for subsequent calculations.
            previous_accumulation = 0 if reset_after[0] else sequence[0]

            # Without any previous values to deaccumulate from we write nan into the first step
            sequence[0] = np.nan

            # Begin deaccumulating from the second step
            for t in range(1, n_lead_times):
                if skip_step[t]:
                    continue

                time_step = seconds[t] - previous_seconds
                previous_seconds = seconds[t]

                step_accumulation = sequence[t] - previous_accumulation
                # store previous accumulation before we overwrite sequence[t] with rate
                previous_accumulation = sequence[t]
                sequence[t] = step_accumulation / time_step

                if reset_after[t]:
                    previous_accumulation = 0

                # Accumulations should only go up
                # If they go down a tiny bit, clamp to 0
                # If they go down more, set to NaN
                if sequence[t] < 0:
                    if sequence[t] > invalid_below_threshold_rate:
                        clamped_count += 1
                        sequence[t] = 0
                    else:
                        invalid_negative_count += 1
                        sequence[t] = np.nan

    return invalid_negative_count, clamped_count
