# from pathlib import Path

# import numpy as np
# import pandas as pd
# import pytest
# import xarray as xr

# from reformatters.common import validation
# from reformatters.dwd.icon_eu.forecast.dynamical_dataset import DwdIconEuForecastDataset

# @pytest.mark.slow
# def test_backfill_local_and_operational_update(monkeypatch: pytest.MonkeyPatch) -> None:
#     dataset = DwdIconEuForecastDataset()

#     # Local backfill reformat
#     dataset.backfill_local(append_dim_end=pd.Timestamp("2000-01-02"))
#     ds = xr.open_zarr(dataset.primary_store_factory.primary_store(), chunks=None)
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

#     dataset.update("test-update")

#     # Check resulting dataset
#     updated_ds = xr.open_zarr(dataset.primary_store_factory.primary_store(), chunks=None)

#     np.testing.assert_array_equal(
#         updated_ds.time, pd.date_range("1981-10-01", "1981-10-03")
#     )
#     subset_ds = updated_ds.sel(latitude=48.583335, longitude=-94, method="nearest")
#     np.testing.assert_array_equal(
#         subset_ds["your_variable"].values, [190.0, 163.0, 135.0]
#     )


# def test_operational_kubernetes_resources(
#     dataset: DwdIconEuForecastDataset,
# ) -> None:
#     cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")

#     assert len(cron_jobs) == 2
#     update_cron_job, validation_cron_job = cron_jobs
#     assert update_cron_job.name == f"{dataset.dataset_id}-operational-update"
#     assert validation_cron_job.name == f"{dataset.dataset_id}-validation"
#     assert update_cron_job.secret_names == [dataset.storage_config.k8s_secret_name]
#     assert validation_cron_job.secret_names == [dataset.storage_config.k8s_secret_name]


# def test_validators(dataset: DwdIconEuForecastDataset) -> None:
#     validators = tuple(dataset.validators())
#     assert len(validators) == 2
#     assert all(isinstance(v, validation.DataValidator) for v in validators)
