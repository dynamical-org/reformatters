import numpy as np
import xarray as xr
from numba import njit, prange  # type: ignore

from reformatters.common.types import Array1D, ArrayFloat32


def linear_interpolate_1d_inplace(
    data_array: xr.DataArray, *, dim: str, where: Array1D[np.bool]
) -> xr.DataArray:
    """
    Only interpolates individual values surrounded by two valid values.
    `dim` must be dimension 0.
    """
    assert data_array.dims.index(dim) == 0
    assert data_array.ndim == 3
    assert where.shape[0] == data_array.shape[0]
    _linear_interpolate_zero_dim_1d_inplace_numba(data_array.values, where)
    return data_array


@njit(parallel=True)  # type: ignore
def _linear_interpolate_zero_dim_1d_inplace_numba(
    values: ArrayFloat32, where: Array1D[np.bool]
) -> None:
    # Interpolate along dim 0, parallel loop over dim 1, and loop over dim 2
    for i in prange(values.shape[1]):
        for j in range(values.shape[2]):
            seq = values[:, i, j]
            # Loop along interpolation dimension,
            # skipping first and last points where we can't interpolate
            for k in range(1, len(seq) - 1):
                if where[k]:
                    seq[k] = seq[k - 1] + (seq[k + 1] - seq[k - 1]) / 2
