import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from reformatters.common import validation
from reformatters.eccc.hrdps.analysis.dynamical_dataset import EcccHrdpsAnalysisDataset
from tests.chunk_utils import shrink_chunks_and_shards
from tests.common.dynamical_dataset_test import (
    NOOP_STORAGE_CONFIG,
    assert_configured_validators,
)
from tests.xarray_testing import assert_no_nulls


@pytest.fixture
def dataset() -> EcccHrdpsAnalysisDataset:
    return _make_dataset()


def _make_dataset() -> EcccHrdpsAnalysisDataset:
    return EcccHrdpsAnalysisDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


@pytest.mark.slow
def test_backfill_local_and_operational_update(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset()

    filter_variable_names = [
        "temperature_2m",  # instantaneous
        "precipitation_surface",  # accumulation we deaccumulate
    ]

    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: shrink_chunks_and_shards(orig_get_template(end_time)),
    )

    time_start = pd.Timestamp("2026-07-09T00:00")
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2026-07-09T02:00"),
        filter_variable_names=filter_variable_names,
    )

    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert_array_equal(
        backfill_ds["time"], pd.date_range(time_start, "2026-07-09T01:00", freq="1h")
    )

    space_subset_ds = backfill_ds.isel(y=slice(0, 10), x=slice(0, 10))
    assert_no_nulls(space_subset_ds[["temperature_2m"]])
    # The first time step's precipitation comes from the run initialized 6 hours
    # before the archive begins, so it is unavailable and NaN.
    assert space_subset_ds["precipitation_surface"].isel(time=0).isnull().all()
    assert_no_nulls(
        space_subset_ds[["precipitation_surface"]].isel(time=slice(1, None))
    )

    # Point in the western Canadian mountains with rain; values match the
    # forecast dataset's 2026-07-09T00Z lead times 0-1h at the same point.
    point_ds = backfill_ds.isel(y=697, x=306)
    assert_array_equal(
        point_ds["temperature_2m"].values,
        np.array([1.078125, 1.0234375], dtype=np.float32),
    )
    np.testing.assert_allclose(
        point_ds["precipitation_surface"].values,
        np.array([np.nan, 0.00074768], dtype=np.float32),
        rtol=1e-4,
    )

    # Operational update
    dataset = _make_dataset()
    append_dim_end = pd.Timestamp("2026-07-09T04:30")
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: append_dim_end)
    )
    orig_get_jobs = dataset.region_job_class.get_jobs
    monkeypatch.setattr(
        dataset.region_job_class,
        "get_jobs",
        lambda *args, **kwargs: orig_get_jobs(
            *args, **{**kwargs, "filter_variable_names": filter_variable_names}
        ),
    )

    dataset.update("test-update")

    updated_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    # update_template_with_results trims the final (accumulated-variables-only) hour
    assert_array_equal(
        updated_ds["time"], pd.date_range(time_start, "2026-07-09T03:00", freq="1h")
    )

    space_subset_ds = updated_ds.isel(y=slice(0, 10), x=slice(0, 10))
    assert_no_nulls(space_subset_ds[["temperature_2m"]])
    assert_no_nulls(
        space_subset_ds[["precipitation_surface"]].isel(time=slice(1, None))
    )

    point_ds = updated_ds.isel(y=697, x=306)
    assert_array_equal(
        point_ds["temperature_2m"].values,
        np.array([1.078125, 1.0234375, 0.984375, 0.921875], dtype=np.float32),
    )
    np.testing.assert_allclose(
        point_ds["precipitation_surface"].values,
        np.array([np.nan, 0.00074768, 0.00044918, 0.00070381], dtype=np.float32),
        rtol=1e-4,
    )

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(dataset: EcccHrdpsAnalysisDataset) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.suspend is True  # until the initial backfill is complete
    assert len(update_cron_job.secret_names) > 0

    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert validation_cron_job.suspend is True
    assert len(validation_cron_job.secret_names) > 0


def test_validators(dataset: EcccHrdpsAnalysisDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
