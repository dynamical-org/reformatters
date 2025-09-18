import xarray as xr


def assert_no_nulls(ds: xr.Dataset) -> None:
    assert (ds.isnull().sum() == 0).all().to_array().all()
