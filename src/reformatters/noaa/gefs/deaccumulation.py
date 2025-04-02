from typing import Final

import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange  # type: ignore

from reformatters.common.types import Array1D, ArrayFloat32

SECONDS_PER_6_HOUR: Final[int] = 6 * 60 * 60


def deaccumulate_to_rates_inplace(
    data_array: xr.DataArray,
    *,
    dim: str,
    reset_frequency: pd.Timedelta,
) -> xr.DataArray:
    """
    Convert accumulated values to per-second rates in place.
    For NOAA GEFS data, accumulations are either over 3h or 6h periods.
    See https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf
    """
    assert data_array.attrs["units"].endswith("/s"), (
        "Output units must be a per-second rate"
    )

    # Support timedelta or datetime dimension values, converting either to seconds
    times = data_array[dim].values
    if np.issubdtype(times.dtype, np.datetime64):
        start_time = np.datetime64(pd.Timestamp(times[0]).floor(reset_frequency))  # type: ignore[arg-type]
        times = times - start_time

    # First step must be a resetting step so we know the start point of the accumulation is zero.
    assert times[0] % reset_frequency == pd.Timedelta(0)

    assert np.issubdtype(times.dtype, np.timedelta64)
    seconds = times.astype("timedelta64[s]").astype(np.int64)

    resets_after = (seconds % reset_frequency.total_seconds()) == 0
    assert resets_after[0]  # Again, first step must be a resetting step

    # make array 3D with shape (flattend_leading_dims, lead_time, flattend_trailing_dims)
    lead_time_dim_index = data_array.dims.index(dim)
    values = data_array.values.reshape(
        np.prod(data_array.shape[:lead_time_dim_index] or 1),
        data_array.shape[lead_time_dim_index],
        np.prod(data_array.shape[lead_time_dim_index + 1 :] or 1),
    )

    _deaccumulate_to_rates_numba(
        values,
        resets_after,
        seconds,
    )

    return data_array


@njit(parallel=True)  # type: ignore
def _deaccumulate_to_rates_numba(
    values: ArrayFloat32,
    resets_after: Array1D[np.bool],
    seconds: Array1D[np.int64],
    invalid_below_threshold_rate: float = -2e-5,
) -> None:
    """
    Convert GEFS 3 or 6 hour accumulated values to per-second rates in place.

    GEFS accumulations are over the last 6 hour period for 00, 06, 12, 18
    UTC hours or the last 3 hour period for 03, 09, 15, 21 UTC hours.

    Accumulations should only go up. If they go down a tiny bit,
    most likely due to numerical precision issues, we clamp to 0.
    If they go down to a *rate* that is less than `invalid_below_threshold`,
    this sets the value to NaN and raises an error.

    Parallel processing is done over the leading dimension of values.

    Args:
        values: Array to modify in place. Must be 3D with lead_time the *middle* dimension
        is_3h_accum: 1D array of booleans indicating if the accumulation is 3h or 6h
        seconds: 1D array of seconds since forecast start
        epsilon: Threshold below which values are considered invalid
    """
    assert values.ndim == 3
    assert seconds.ndim == 1
    assert resets_after.ndim == 1
    assert values.shape[1] == seconds.size
    assert values.shape[1] == resets_after.size

    n_lead_times = values.shape[1]

    negative_count = 0
    clamped_count = 0

    for i in prange(values.shape[0]):
        for j in range(values.shape[2]):
            sequence = values[i, :, j]

            current_reset_window_accumulation = 0

            # Skip first step, no accumulation yet
            for t in range(1, n_lead_times):
                time_step = seconds[t] - seconds[t - 1]

                if resets_after[t - 1]:
                    # First step after reset - simple division by time since reset
                    current_reset_window_accumulation = sequence[t]
                    sequence[t] /= time_step
                else:
                    # There was an accumulation before - calculate rate for just the last step
                    accumulation = sequence[t] - current_reset_window_accumulation
                    current_reset_window_accumulation = sequence[t]
                    sequence[t] = accumulation / time_step

                # TODO: Is this necessary because in the esets_after[t - 1] case
                # we update current_reset_window_accumulation to the first valuse
                # post reset?
                if resets_after[t]:
                    current_reset_window_accumulation = 0

                # Accumulations should only go up
                # If they go down a tiny bit, clamp to 0
                # If they go down more, set to NaN and then raise
                if sequence[t] < 0:
                    if sequence[t] > invalid_below_threshold_rate:
                        clamped_count += 1
                        sequence[t] = 0
                    else:
                        negative_count += 1
                        sequence[t] = np.nan

    if negative_count > 0:
        raise ValueError(f"Found {negative_count} values below threshold")
    if clamped_count / values.size > 0.05:
        raise ValueError(f"Over 5% ({clamped_count} total) values were clamped to 0")
