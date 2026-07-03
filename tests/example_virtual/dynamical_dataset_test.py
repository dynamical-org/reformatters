# from pathlib import Path

# import numpy as np
# import pandas as pd
# import pytest
# import xarray as xr

# from reformatters.common import validation
# from reformatters.common.storage import DatasetFormat, StorageConfig
# from reformatters.example_virtual.dynamical_dataset import ExampleSpatialDynamicalDataset
# from tests.common.dynamical_dataset_test import assert_configured_validators


# @pytest.fixture
# def dataset(tmp_path: Path) -> ExampleSpatialDynamicalDataset:
#     # A virtual dataset's primary store must be ICECHUNK, and icechunk_virtual_config
#     # (set on ExampleSpatialDynamicalDataset) registers the source buckets the refs may point into.
#     return ExampleSpatialDynamicalDataset(
#         primary_storage_config=StorageConfig(
#             base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
#         ),
#     )


# @pytest.mark.slow
# def test_backfill_local_and_operational_update(
#     monkeypatch: pytest.MonkeyPatch, dataset: ExampleSpatialDynamicalDataset
# ) -> None:
#     # Keep the test small: trim the template to a couple lead times (and ensemble
#     # members, if any) so the backfill fetches few files. Do NOT shrink chunks - a
#     # virtual dataset's chunks are one-per-message and must keep that shape; trim with
#     # .isel/.sel only.
#     orig_get_template = dataset.template_config.get_template
#     monkeypatch.setattr(
#         type(dataset.template_config),
#         "get_template",
#         lambda self, end_time: orig_get_template(end_time).sel(
#             lead_time=slice("0h", "6h")
#         ),
#     )

#     # 1. Backfill a fixed, immutable slice of the archive. Backfill writes chunk
#     # references into the icechunk manifest (no bytes are copied); opening the store
#     # decodes them through the serializer.
#     dataset.backfill_local(
#         append_dim_end=pd.Timestamp("2020-01-01T06:00"),
#         filter_variable_names=["temperature_2m"],
#     )
#     ds = xr.open_zarr(
#         dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
#     )
#     assert ds.init_time.values[-1] == np.datetime64("2020-01-01T00:00")

#     # Assert decoded snapshot values at a specific point. These are the RAW source
#     # values (e.g. Kelvin, not the materialized dataset's degC); read them once from
#     # the real source and paste them in. Snapshot values catch silent regressions in
#     # the serializer, grid orientation, and coordinate alignment that shape checks miss.
#     point = ds.sel(init_time="2020-01-01T00:00", latitude=0, longitude=0)
#     np.testing.assert_allclose(
#         point["temperature_2m"].sel(lead_time=["3h", "6h"]).values,
#         [300.23886718750003, 300.437578125],  # placeholder: your source's values
#     )

#     # 2. Operational update: a single-writer polling job (processing_mode="update")
#     # appends references for newer init times onto main. The update window is anchored
#     # on pd.Timestamp.now(), so pin "now" into the next init's publication window.
#     monkeypatch.setattr(
#         pd.Timestamp,
#         "now",
#         classmethod(lambda *args, **kwargs: pd.Timestamp("2020-01-01T07:00")),
#     )
#     dataset.update("test-update")
#     updated_ds = xr.open_zarr(
#         dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
#     )
#     assert updated_ds.init_time.values[-1] == np.datetime64("2020-01-01T06:00")
#     updated_point = updated_ds.sel(
#         init_time="2020-01-01T06:00", lead_time="3h", latitude=0, longitude=0
#     )
#     np.testing.assert_allclose(
#         updated_point["temperature_2m"].values, 300.7870703125  # placeholder
#     )

#     # Confirm every configured validator is wired up and runnable.
#     assert_configured_validators(dataset)

#     # See tests/common/virtual_region_job_test.py and virtual_multi_group_test.py for
#     # runnable end-to-end virtual write-loop tests (value round-trips, per-file commit
#     # atomicity, per-group append-dim growth).


# def test_operational_kubernetes_resources(dataset: ExampleSpatialDynamicalDataset) -> None:
#     cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")
#     assert len(cron_jobs) == 2
#     update_cron_job, validation_cron_job = cron_jobs
#     assert update_cron_job.name == f"{dataset.dataset_id}-update"
#     assert validation_cron_job.name == f"{dataset.dataset_id}-validate"


# def test_validators_include_virtual_checks(dataset: ExampleSpatialDynamicalDataset) -> None:
#     validators = tuple(dataset.validators())
#     # The two virtual-specific validators need manifest/store access, so they are
#     # VirtualDataValidator instances (not plain xarray validator functions).
#     assert any(
#         isinstance(v, validation.CheckVirtualManifestCompleteness) for v in validators
#     )
#     assert any(isinstance(v, validation.CheckVirtualDecodeHealth) for v in validators)
