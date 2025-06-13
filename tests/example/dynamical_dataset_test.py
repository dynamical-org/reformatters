# from pathlib import Path

# import numpy as np
# import pandas as pd
# import pytest
# import xarray as xr

# from reformatters.example.dynamical_dataset import ExampleDataset

# pytestmark = pytest.mark.slow


# def test_reformat_local_and_operational_update(monkeypatch: pytest.MonkeyPatch) -> None:
#     dataset = ExampleDataset()

#     # Local backfill reformat
#     dataset.reformat_local(append_dim_end=pd.Timestamp("2000-01-02"))
#     ds = xr.open_zarr(dataset._final_store(), chunks=None)
#     assert ds.time.max() == pd.Timestamp("2000-01-01")

#     # Operational update
#     monkeypatch.setattr(
#         dataset.region_job_class,
#         "_update_append_dim_end",
#         lambda: pd.Timestamp("2000-01-03"),
#     )
#     monkeypatch.setattr(
#         dataset.region_job_class,
#         "_update_append_dim_start",
#         lambda existing_ds: pd.Timestamp(existing_ds.time.max().item()),
#     )

#     dataset.reformat_operational_update("test-reformat-operational-update")

#     # Check resulting dataset
#     updated_ds = xr.open_zarr(dataset._final_store(), chunks=None)

#     np.testing.assert_array_equal(
#         updated_ds.time, pd.date_range("1981-10-01", "1981-10-03")
#     )
#     subset_ds = updated_ds.sel(latitude=48.583335, longitude=-94, method="nearest")
#     np.testing.assert_array_equal(
#         subset_ds["your_variable"].values, [190.0, 163.0, 135.0]
#     )
