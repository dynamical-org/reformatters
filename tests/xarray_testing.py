import xarray as xr


def assert_no_nulls(ds: xr.Dataset | xr.DataArray) -> None:
    no_nulls = (ds.isnull().sum() == 0).all()
    if isinstance(ds, xr.Dataset):
        no_nulls = no_nulls.to_array().all()
    assert no_nulls
