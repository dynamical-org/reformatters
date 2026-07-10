from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange

from reformatters.common.logging import get_logger
from reformatters.common.types import Array1D, ArrayFloat32

log = get_logger(__name__)

# OK to add units to this list if you believe they are reasonable output units to deaccumulate to.
# We typically expect these to be per-second rates.
VALID_OUTPUT_UNITS_FOR_DEACCUMULATION = ["mm s-1", "m s-1", "kg m-2 s-1", "W m-2"]

# mm s-1, ~= 0.25 mm/h or "trace" precipitation
PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD = -7e-5
RADIATION_INVALID_BELOW_THRESHOLD = -50.0  # W/m^2 aka J/m^2/s

# How the input values along `dim` should be interpreted:
#   - "accumulated": values are cumulative totals (e.g. J m-2 since forecast start).
#     Step rate = (A_t - A_{t-1}) / dt.
#   - "running_mean": values are running-mean rates whose averaging window grows
#     from forecast start.
#     Step rate = A_t + (A_t - A_{t-1}) * t_{t-1} / dt. This rearrangement of
#     (A_t * t_t - A_{t-1} * t_{t-1}) / dt keeps every intermediate product small,
#     preserving float32 precision across long forecast horizons.
AccumulationType = Literal["accumulated", "running_mean"]

# Integer encoding of AccumulationType to keep the numba loop branch cheap.
_ACCUMULATION_TYPE_ACCUMULATED = 0
_ACCUMULATION_TYPE_RUNNING_MEAN = 1

# PROTOTYPE (issue #722): how to handle an implausibly-negative step drop rather than
# leaving it NaN. "none" is the current, default behavior. The other two are candidate
# repairs for corrupt source accumulation fields; see issue #722.
#   - "monotonic": repair the accumulated series to be non-decreasing (isotonic) before
#     deaccumulating. Fixes both the negative step and the paired positive spike, and
#     conserves the window total. Accumulated inputs only.
#   - "temporal": deaccumulate, then replace the corrupt step-pair's rates with a linear
#     interpolation from neighboring valid steps. Accumulated inputs only.
RepairMode = Literal["none", "monotonic", "temporal"]


def deaccumulate_to_rates_inplace(
    data_array: xr.DataArray,
    *,
    dim: str,
    reset_frequency: pd.Timedelta,
    skip_step: Array1D[np.bool] | None = None,
    invalid_below_threshold_rate: float = PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
    expected_invalid_fraction: float = 0.0,
    expected_clamp_fraction: float = 0.05,
    accumulation_type: AccumulationType = "accumulated",
    repair_implausible_drops: RepairMode = "none",
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
        accumulation_type: How to interpret the input values. "accumulated" (default) treats
            them as cumulative totals. "running_mean" treats them as running-mean rates whose
            averaging window grows from forecast start and converts them to per-step rates.
        repair_implausible_drops: PROTOTYPE (issue #722). "none" (default) leaves the current
            behavior unchanged. "monotonic" and "temporal" repair implausibly-negative step
            drops from corrupt source fields instead of leaving them NaN; both are only
            implemented for accumulation_type="accumulated" and for series without interior
            resets or skipped steps.
    """
    assert data_array.attrs["units"] in VALID_OUTPUT_UNITS_FOR_DEACCUMULATION, (
        "Output units must be a per-second rate"
    )

    if accumulation_type == "accumulated":
        accumulation_type_int = _ACCUMULATION_TYPE_ACCUMULATED
    elif accumulation_type == "running_mean":
        accumulation_type_int = _ACCUMULATION_TYPE_RUNNING_MEAN
    else:
        raise ValueError(
            f"Unknown accumulation_type {accumulation_type!r}; "
            "expected 'accumulated' or 'running_mean'."
        )

    # Support timedelta or datetime dimension values, converting either to seconds
    times = data_array[dim].values
    if np.issubdtype(times.dtype, np.datetime64):
        start_time = np.datetime64(pd.Timestamp(times[0]).floor(reset_frequency))  # ty: ignore[invalid-argument-type]
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

    temporal_mask = _prepare_repair(
        values,
        seconds,
        reset_after,
        skip_step,
        invalid_below_threshold_rate,
        accumulation_type,
        repair_implausible_drops,
    )

    invalid_negative_count, clamped_count = _deaccumulate_to_rates_numba(
        values,
        seconds,
        reset_after,
        skip_step,
        invalid_below_threshold_rate,
        accumulation_type_int,
    )

    if temporal_mask is not None:
        repaired = _temporal_repair(values, seconds, temporal_mask)
        if repaired:
            log.info(f"deaccumulation temporal repair: {repaired} step cells repaired")

    if repair_implausible_drops != "none":
        return data_array

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
    accumulation_type: int,
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

                step_accumulation = sequence[t] - previous_accumulation
                # store previous accumulation before we overwrite sequence[t] with rate
                previous_accumulation = sequence[t]

                if accumulation_type == _ACCUMULATION_TYPE_ACCUMULATED:
                    sequence[t] = step_accumulation / time_step
                elif accumulation_type == _ACCUMULATION_TYPE_RUNNING_MEAN:
                    sequence[t] = sequence[t] + (
                        (step_accumulation * previous_seconds) / time_step
                    )
                # Note: no `else` branch here even as a safety net. Numba disables
                # `prange` parallelisation if the loop has any additional exit points
                # (raise, assert, etc.) so we rely on the wrapper to reject unknown
                # accumulation_type values before entering this kernel.

                previous_seconds = seconds[t]

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


# --- PROTOTYPE repairs for issue #722 (not numba, opt-in, not performance tuned) ---

BoolArray3D = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


def _prepare_repair(
    values: ArrayFloat32,
    seconds: Array1D[np.int64],
    reset_after: Array1D[np.bool],
    skip_step: Array1D[np.bool],
    invalid_below_threshold_rate: float,
    accumulation_type: AccumulationType,
    repair_implausible_drops: RepairMode,
) -> BoolArray3D | None:
    """Apply the pre-deaccumulation repair step. Returns the temporal-repair mask (to be
    applied after deaccumulation) for the "temporal" mode, else None."""
    if repair_implausible_drops == "none":
        return None
    if accumulation_type != "accumulated":
        raise NotImplementedError(
            f"repair_implausible_drops={repair_implausible_drops!r} is only "
            "implemented for accumulation_type='accumulated'"
        )
    assert not skip_step.any(), "repair modes do not support skipped steps"
    assert not reset_after[1:].any(), (
        "repair modes only support a reset at the first step (e.g. precipitation)"
    )

    if repair_implausible_drops == "monotonic":
        repaired = _repair_accumulated_monotonic(
            values, seconds, invalid_below_threshold_rate
        )
        if repaired:
            log.info(
                f"deaccumulation monotonic repair: {repaired} pixel series repaired"
            )
        return None

    return _temporal_affected_mask(values, seconds, invalid_below_threshold_rate)


def _pava_nondecreasing(y: Array1D[np.floating]) -> Array1D[np.float64]:
    """Least-squares non-decreasing fit via pool-adjacent-violators. Input must be finite."""
    values: list[float] = []
    counts: list[int] = []
    for value in y:
        values.append(float(value))
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            v_hi, c_hi = values.pop(), counts.pop()
            v_lo, c_lo = values.pop(), counts.pop()
            values.append((v_lo * c_lo + v_hi * c_hi) / (c_lo + c_hi))
            counts.append(c_lo + c_hi)
    out = np.empty(y.shape[0], dtype=np.float64)
    offset = 0
    for value, count in zip(values, counts, strict=True):
        out[offset : offset + count] = value
        offset += count
    return out


def _implausible_drop_mask(
    values: ArrayFloat32,
    seconds: Array1D[np.int64],
    invalid_below_threshold_rate: float,
) -> BoolArray3D:
    """Per step t (t>=1), True where the deaccumulated rate would fall below the threshold."""
    dt = np.diff(seconds).astype(np.float64)
    rates = (values[:, 1:, :] - values[:, :-1, :]) / dt[None, :, None]
    drop = np.zeros(values.shape, dtype=np.bool_)
    drop[:, 1:, :] = rates < invalid_below_threshold_rate  # NaN comparisons are False
    return drop


def _repair_accumulated_monotonic(
    values: ArrayFloat32,
    seconds: Array1D[np.int64],
    invalid_below_threshold_rate: float,
) -> int:
    """Repair accumulated series with an implausible drop to be non-decreasing, in place.

    Only touches pixels whose deaccumulated rate would fall below the threshold; other
    pixels keep their exact current (compression-noise) behavior. Returns pixels repaired.
    """
    triggered = _implausible_drop_mask(
        values, seconds, invalid_below_threshold_rate
    ).any(axis=1)
    repaired = 0
    for i, j in zip(*np.nonzero(triggered), strict=True):
        sequence = values[i, :, j]
        finite = np.isfinite(sequence)
        first = int(np.argmax(finite))
        segment = sequence[first:]
        if not np.isfinite(segment).all():
            continue  # interior gaps: leave to the normal path
        sequence[first:] = _pava_nondecreasing(segment)
        repaired += 1
    return repaired


def _temporal_affected_mask(
    values: ArrayFloat32,
    seconds: Array1D[np.int64],
    invalid_below_threshold_rate: float,
) -> BoolArray3D:
    """Mark, per pixel, the step-pair corrupted by an implausible drop (the negative step
    and the paired positive spike/dip), computed from the raw accumulated values."""
    drop = _implausible_drop_mask(values, seconds, invalid_below_threshold_rate)
    mask = np.zeros(values.shape, dtype=np.bool_)
    n_lead = values.shape[1]
    for t in range(1, n_lead):
        drop_t = drop[:, t, :]
        if not drop_t.any():
            continue
        a_prev, a_cur = values[:, t - 1, :], values[:, t, :]
        if t - 1 >= 1:
            a_prev_prev = values[:, t - 2, :]
            is_spike = (
                drop_t
                & np.isfinite(a_prev_prev)
                & (a_prev > a_prev_prev)
                & (a_prev >= a_cur)
            )
        else:
            is_spike = np.zeros_like(drop_t)
        is_dip = drop_t & ~is_spike
        mask[:, t - 1, :] |= is_spike  # spike feeds the drop from the step before
        mask[:, t, :] |= drop_t
        if t + 1 < n_lead:
            mask[:, t + 1, :] |= is_dip  # a low dip corrupts the step after it too
    return mask


def _temporal_repair(
    values: ArrayFloat32,
    seconds: Array1D[np.int64],
    affected: BoolArray3D,
) -> int:
    """Replace affected step rates with a linear interpolation over lead time from the
    nearest valid (finite, unaffected) steps, in place. Returns cells repaired."""
    x = seconds.astype(np.float64)
    repaired = 0
    for i, j in zip(*np.nonzero(affected.any(axis=1)), strict=True):
        pixel_mask = affected[i, :, j]
        rates = values[i, :, j]
        donor = np.isfinite(rates) & ~pixel_mask
        donor[0] = False  # first step is the baseline NaN, never a donor
        if donor.sum() < 2:
            continue
        target = np.flatnonzero(pixel_mask)
        values[i, target, j] = np.interp(x[target], x[donor], rates[donor])
        repaired += int(pixel_mask.sum())
    return repaired
