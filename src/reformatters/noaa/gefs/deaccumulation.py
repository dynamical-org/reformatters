from typing import Final

import numpy as np
import xarray as xr
from numba import njit, prange  # type: ignore

from reformatters.common.types import Array1D, ArrayFloat32

SECONDS_PER_6_HOUR: Final[int] = 6 * 60 * 60


def deaccumulate_to_rates_inplace(
    data_array: xr.DataArray, *, dim: str
) -> xr.DataArray:
    """
    Convert accumulated values to per-second rates in place.
    For NOAA GEFS data, accumulations are either over 3h or 6h periods.
    See https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf
    """
    assert data_array.attrs["units"].endswith("/s"), (
        "Output units must be a per-second rate"
    )

    lead_time_seconds = data_array.lead_time.dt.total_seconds().values
    supported_accum_durations = [SECONDS_PER_6_HOUR, SECONDS_PER_6_HOUR // 2]
    assert np.isin(np.diff(lead_time_seconds), supported_accum_durations).all()
    is_3h_accum = (lead_time_seconds % SECONDS_PER_6_HOUR) != 0

    # make array 3D with shape (flattend_leading_dims, lead_time, flattend_trailing_dims)
    lead_time_dim_index = data_array.dims.index(dim)
    values = data_array.values.reshape(
        np.prod(data_array.shape[:lead_time_dim_index] or 1),
        data_array.shape[lead_time_dim_index],
        np.prod(data_array.shape[lead_time_dim_index + 1 :] or 1),
    )

    _deaccumulate_to_rates_numba(
        values,
        is_3h_accum,
        lead_time_seconds,
    )

    return data_array


@njit(parallel=True)  # type: ignore
def _deaccumulate_to_rates_numba(
    values: ArrayFloat32,
    is_3h_accum: Array1D[np.bool],
    lead_time_seconds: Array1D[np.int64],
    invalid_below_threshold_rate: float = -1e-5,
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
        lead_time_seconds: 1D array of seconds since forecast start
        epsilon: Threshold below which values are considered invalid
    """
    assert values.ndim == 3
    assert lead_time_seconds.ndim == 1
    assert is_3h_accum.ndim == 1
    assert values.shape[1] == lead_time_seconds.size
    assert values.shape[1] == is_3h_accum.size

    n_lead_times = values.shape[1]

    negative_count = 0
    clamped_count = 0

    for i in prange(values.shape[0]):
        for j in range(values.shape[2]):
            sequence = values[i, :, j]

            # Skip first step, no accumulation yet
            for t in range(1, n_lead_times):
                time_step = lead_time_seconds[t] - lead_time_seconds[t - 1]

                if is_3h_accum[t]:
                    # 3h accumulation point - simple division
                    sequence[t] /= time_step
                else:
                    # 6h accumulation point

                    if t > 1 and is_3h_accum[t - 1]:
                        # There was a 3h accumulation before - calculate rate for just the last 3h
                        previous_time_step = (
                            lead_time_seconds[t - 1] - lead_time_seconds[t - 2]
                        )
                        previous_accumulation = sequence[t - 1] * previous_time_step
                        accumulation = sequence[t] - previous_accumulation
                        sequence[t] = accumulation / time_step
                    else:
                        # No 3h accumulation before - calculate rate for full 6h period
                        sequence[t] /= time_step

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
