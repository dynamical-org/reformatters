import numpy as np
import xarray as xr

from reformatters.common.interpolation import linear_interpolate_1d_inplace


def test_linear_interpolate_1d_inplace_every_other_point() -> None:
    # Create a 3D data array with linearly increasing values along first dimension
    z = np.arange(5)
    y = np.arange(3)
    x = np.arange(2)
    data = np.zeros((len(z), len(y), len(x)), dtype=np.float32)
    for i in range(len(y)):
        for j in range(len(x)):
            data[:, i, j] = np.linspace(i * 10 + j, (i + 1) * 10 + j, len(z))

    da = xr.DataArray(data, coords=[("z", z), ("y", y), ("x", x)])

    orig_da = da.copy(deep=True)

    da.values[1::2, :, :] = np.nan

    result = linear_interpolate_1d_inplace(
        da, dim="z", where=np.isnan(da.values[:, 0, 0])
    )

    np.testing.assert_allclose(result.values, orig_da.values)
    assert result is da


def test_linear_interpolate_1d_inplace_where_all_false() -> None:
    # Create a 3D data array with linearly increasing values along first dimension
    z = np.arange(5)
    y = np.arange(3)
    x = np.arange(2)
    data = np.zeros((len(z), len(y), len(x)), dtype=np.float32)
    for i in range(len(y)):
        for j in range(len(x)):
            data[:, i, j] = np.linspace(i * 10 + j, (i + 1) * 10 + j, len(z))

    da = xr.DataArray(data, coords=[("z", z), ("y", y), ("x", x)])

    orig_da = da.copy(deep=True)

    da.values[1::2, :, :] = np.nan

    values_copy = da.values.copy()

    result = linear_interpolate_1d_inplace(
        da, dim="z", where=np.zeros(da.shape[0], dtype=bool)
    )

    np.testing.assert_allclose(result.values, values_copy)
    assert not np.allclose(result.values, orig_da.values)
    assert result is da


def test_linear_interpolate_1d_inplace_nans_propagate() -> None:
    # Create a 3D data array with linearly increasing values along first dimension

    data = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    da = xr.DataArray(data, dims=["z", "y", "x"])

    da.values[0, :, :] = np.nan
    da.values[1::2, :, :] = np.nan
    da.values[-1, :, :] = np.nan

    result = linear_interpolate_1d_inplace(
        da, dim="z", where=np.isnan(da.values[:, 0, 0])
    )

    np.testing.assert_equal(
        result.values.ravel(),
        np.array([np.nan, np.nan, 2, 3, 4, np.nan], dtype=np.float32),
    )
    assert result is da
