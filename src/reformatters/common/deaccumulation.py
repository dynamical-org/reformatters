import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange  # type: ignore[import-untyped]

from reformatters.common.types import Array1D, ArrayFloat32


def deaccumulate_to_rates_inplace(
    data_array: xr.DataArray,
    *,
    dim: str,
    reset_frequency: pd.Timedelta,
    skip_step: Array1D[np.bool] | None = None,
) -> xr.DataArray:
    """
    Convert accumulated values to per-second rates in place.

    Args:
        data_array: Array containing accumulated values to be converted to rates in place
        dim: Dimension over which to deaccumulate
        reset_frequency: Frequency at which the accumulation resets (eg 6 hours for GEFS and GFS)
        skip_step: Array of booleans indicating whether to skip the step. Values in skipped
            steps are left unchanged and the deaccumulation acts as if they are not present.
    """
    assert data_array.attrs["units"].endswith("/s"), (
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

    # First step must be a resetting step so we know the start point of the accumulation is zero.
    assert timedeltas[0] % reset_frequency == pd.Timedelta(0)

    seconds = timedeltas.astype("timedelta64[s]").astype(np.int64)

    reset_after = (seconds % reset_frequency.total_seconds()) == 0
    assert reset_after[0]  # Again, first step must be a resetting step

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

    _deaccumulate_to_rates_numba(values, seconds, reset_after, skip_step)

    return data_array


@njit(parallel=True)  # type: ignore[misc]
def _deaccumulate_to_rates_numba(
    values: ArrayFloat32,
    seconds: Array1D[np.int64],
    reset_after: Array1D[np.bool],
    skip_step: Array1D[np.bool],
    invalid_below_threshold_rate: float = -2e-5,
) -> None:
    """
    Convert accumulated values to per-second rates, mutating `values` in place.

    Accumulations should only go up. If they go down a tiny bit,
    most likely due to numerical precision issues, we clamp to 0.
    If they go down to a *rate* that is less than `invalid_below_threshold`,
    this sets the value to NaN and raises an error.

    Parallel processing is done over the leading dimension of values.

    Args:
        values: Array to modify in place. Must be 3D with accumulation dimension as the *middle* dimension
        seconds: 1D array of seconds since forecast start or a reference time
        reset_after: 1D array of booleans where True indicates the accumulation resets after the current step
        skip_step: 1D array of booleans where True indicates the step should be skipped
        invalid_below_threshold_rate: Threshold below which values are considered invalid
    """
    assert values.ndim == 3
    assert seconds.ndim == 1
    assert reset_after.ndim == 1
    assert skip_step.ndim == 1
    assert values.shape[1] == seconds.size
    assert values.shape[1] == reset_after.size
    assert values.shape[1] == skip_step.size

    n_lead_times = values.shape[1]

    negative_count = 0
    clamped_count = 0

    for i in prange(values.shape[0]):
        for j in range(values.shape[2]):
            sequence = values[i, :, j]
            previous_seconds = seconds[0]
            previous_accumulation = 0

            # Skip first step, no accumulation yet
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
