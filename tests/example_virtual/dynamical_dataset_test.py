# import numpy as np
# import pandas as pd
# import pytest
# import xarray as xr

# from reformatters.common import validation
# from reformatters.common.storage import DatasetFormat, StorageConfig
# from reformatters.example_virtual.dynamical_dataset import ExampleDataset


# @pytest.fixture
# def dataset(tmp_path: Path) -> ExampleDataset:
#     # A virtual dataset's primary store must be ICECHUNK, and icechunk_virtual_config
#     # (set on ExampleDataset) registers the source buckets the refs may point into.
#     return ExampleDataset(
#         primary_storage_config=StorageConfig(
#             base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
#         ),
#     )


# @pytest.mark.slow
# def test_backfill_local_and_operational_update(
#     monkeypatch: pytest.MonkeyPatch, dataset: ExampleDataset
# ) -> None:
#     # Backfill writes chunk references into the icechunk manifest (no bytes are copied);
#     # opening the store decodes them through the serializer.
#     dataset.backfill_local(append_dim_end=pd.Timestamp("2020-01-02"))
#     ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
#     assert ds.init_time.min() == pd.Timestamp("2020-01-01")

#     # Operational update: a single-writer polling job (processing_mode="update")
#     # appends references for newer init times onto main.
#     dataset.update("test-update")
#     updated_ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
#     assert updated_ds.init_time.max() > ds.init_time.max()

#     # See tests/common/virtual_region_job_test.py and virtual_multi_group_test.py for
#     # runnable end-to-end virtual write-loop tests (value round-trips, per-file commit
#     # atomicity, per-group append-dim growth).


# def test_operational_kubernetes_resources(dataset: ExampleDataset) -> None:
#     cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")
#     assert len(cron_jobs) == 2
#     update_cron_job, validation_cron_job = cron_jobs
#     assert update_cron_job.name == f"{dataset.dataset_id}-update"
#     assert validation_cron_job.name == f"{dataset.dataset_id}-validate"


# def test_validators_include_virtual_checks(dataset: ExampleDataset) -> None:
#     validators = tuple(dataset.validators())
#     # The two virtual-specific validators need manifest/store access, so they are
#     # VirtualDataValidator instances (not plain xarray validator functions).
#     assert any(
#         isinstance(v, validation.CheckVirtualManifestCompleteness) for v in validators
#     )
#     assert any(isinstance(v, validation.CheckVirtualDecodeHealth) for v in validators)
