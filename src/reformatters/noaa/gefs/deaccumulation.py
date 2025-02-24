from typing import Final

import numpy as np
import xarray as xr
from numba import njit, prange  # type: ignore

from reformatters.common.types import ArrayFloat32

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

    # Move lead_time to last dimension for simpler numba code
    lead_time_dim_index = data_array.dims.index(dim)
    values = np.moveaxis(data_array.values, lead_time_dim_index, -1)

    _deaccumulate_to_rates_numba(
        values,
        is_3h_accum,
        lead_time_seconds,
    )

    return data_array


@njit(parallel=True)  # type: ignore
def _deaccumulate_to_rates_numba(
    values: ArrayFloat32,
    is_3h_accum: np.ndarray[tuple[int], np.dtype[np.bool]],
    lead_time_seconds: np.ndarray[tuple[int], np.dtype[np.int64]],
    invalid_below_threshold: float = -1e-6,
) -> None:
    """Convert accumulated values to per-second rates in place.

    Args:
        values: Array to modify in place, with lead_time as the last dimension
        is_3h_accum: 1D array of booleans indicating if the accumulation is 3h or 6h
        lead_time_seconds: 1D array of seconds since forecast start
        epsilon: Threshold below which values are considered invalid
    """
    n_lead_times = values.shape[-1]

    # Flatten all dimensions except lead_time for parallel processing
    flat_values = values.reshape((-1, n_lead_times))

    negative_count = 0

    for i in prange(flat_values.shape[0]):
        sequence = flat_values[i]

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

            if sequence[t] < 0:
                if sequence[t] < invalid_below_threshold:
                    negative_count += 1
                    sequence[t] = np.nan
                else:
                    sequence[t] = 0

    if negative_count > 0:
        raise ValueError(
            f"Found {negative_count} values below threshold {invalid_below_threshold}"
        )
