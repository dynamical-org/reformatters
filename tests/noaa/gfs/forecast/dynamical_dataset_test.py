from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import MonkeyPatch

from reformatters.common import validation
from reformatters.noaa.gfs.forecast import NoaaGfsForecastDataset
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    dataset = NoaaGfsForecastDataset(primary_storage_config=NOOP_STORAGE_CONFIG)

    init_time_start = dataset.template_config.append_dim_start
    init_time_end = init_time_start + timedelta(hours=12)

    # Trim to first 12 hours of lead time dimension to speed up test
    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).sel(
            lead_time=slice("0h", "12h")
        ),
    )

    filter_variable_names = [
        "temperature_2m",  # instantaneous
        "precipitation_surface",  # accumulation we deaccumulate
        "maximum_temperature_2m",  # max over window
        "minimum_temperature_2m",  # min over window
        "categorical_freezing_rain_surface",  # average over window
    ]

    # 1. Backfill archive
    dataset.backfill_local(
        append_dim_end=init_time_end, filter_variable_names=filter_variable_names
    )
    original_ds = xr.open_zarr(
        dataset.store_factory.primary_store(),
        decode_timedelta=True,
        chunks=None,
    )

    space_subset_ds = original_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))

    # These variables are present at all lead times
    assert (
        (space_subset_ds[["temperature_2m"]].isnull().mean() == 0)
        .all()
        .to_array()
        .all()
    )

    # These variables are not present at hour 0
    assert (
        (
            space_subset_ds[
                [
                    "precipitation_surface",
                    "maximum_temperature_2m",
                    "minimum_temperature_2m",
                    "categorical_freezing_rain_surface",
                ]
            ]
            .sel(lead_time=slice("1h", None))
            .isnull()
            .mean()
            == 0
        )
        .all()
        .to_array()
        .all()
    )

    point_ds = original_ds.sel(
        latitude=0,
        longitude=0,
        init_time=init_time_start,
        lead_time="2h",
    )

    assert point_ds["temperature_2m"] == 27.875
    assert point_ds["maximum_temperature_2m"] == 28.125
    assert point_ds["minimum_temperature_2m"] == 27.875
    assert point_ds["precipitation_surface"] == 1.7404556e-05
    assert point_ds["categorical_freezing_rain_surface"] == 0.0
    np.testing.assert_array_equal(
        original_ds.init_time.values,
        np.array(
            [
                "2021-05-01T00:00:00.000000000",
                "2021-05-01T06:00:00.000000000",
            ],
            dtype="datetime64[ns]",
        ),
    )

    # 2. Operational update
    # Set "now" to just past the 12 UTC init time so we add a third init_time step
    dataset = NoaaGfsForecastDataset(primary_storage_config=NOOP_STORAGE_CONFIG)
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2021-05-01T14:00")),
    )
    # Dataset updates always update all variables. For the test we hook into get_jobs to limit vars.
    orig_get_jobs = dataset.region_job_class.get_jobs
    monkeypatch.setattr(
        dataset.region_job_class,
        "get_jobs",
        lambda *args, **kwargs: orig_get_jobs(
            *args, **{**kwargs, "filter_variable_names": filter_variable_names}
        ),
    )

    dataset.update("test-update-job-name")

    updated_ds = xr.open_zarr(
        dataset.store_factory.primary_store(),
        decode_timedelta=True,
        chunks=None,
    )

    np.testing.assert_array_equal(
        updated_ds.init_time.values,
        np.array(
            [
                "2021-05-01T00:00:00.000000000",
                "2021-05-01T06:00:00.000000000",
                "2021-05-01T12:00:00.000000000",
            ],
            dtype="datetime64[ns]",
        ),
    )
    point_ds2 = updated_ds.sel(
        latitude=0,
        longitude=0,
        init_time=init_time_end,
        lead_time="3h",
    )

    assert point_ds2["temperature_2m"] == 28.375
    assert point_ds2["maximum_temperature_2m"] == 28.5
    assert point_ds2["minimum_temperature_2m"] == 28.25
    assert point_ds2["precipitation_surface"] == 0.000347137451171875
    assert point_ds2["categorical_freezing_rain_surface"] == 0.0


def test_operational_kubernetes_resources() -> None:
    dataset = NoaaGfsForecastDataset(primary_storage_config=NOOP_STORAGE_CONFIG)
    cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs
    assert update_cron_job.name == f"{dataset.dataset_id}-operational-update"
    assert validation_cron_job.name == f"{dataset.dataset_id}-validation"
    assert update_cron_job.secret_names == dataset.store_factory.k8s_secret_names()
    assert validation_cron_job.secret_names == dataset.store_factory.k8s_secret_names()


def test_validators() -> None:
    dataset = NoaaGfsForecastDataset(primary_storage_config=NOOP_STORAGE_CONFIG)
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
