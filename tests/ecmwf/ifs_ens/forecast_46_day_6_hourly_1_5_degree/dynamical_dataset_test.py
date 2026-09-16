from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.types import DatetimeLike
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import (
    MATERIALIZED_PRODUCT_ECDS_VARIABLES,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_6_hourly_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_region_job import (
    EcmwfIfsEns46DayRegionJob,
)
from tests.chunk_utils import shrink_chunks_and_shards
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset:
    return EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


def test_dataset_reuses_the_shared_region_job(
    dataset: EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
) -> None:
    assert dataset.region_job_class is EcmwfIfsEns46DayRegionJob


def test_dataset_variables_are_archived_at_native_resolution(
    dataset: EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
) -> None:
    assert {
        data_var.internal_attrs.ecds_variable
        for data_var in dataset.template_config.data_vars
    } == set(MATERIALIZED_PRODUCT_ECDS_VARIABLES["6-hourly"])


def test_operational_cron_jobs_are_suspended_and_do_not_collide_with_daily(
    dataset: EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
) -> None:
    update, validate = dataset.operational_kubernetes_resources("test-image-tag")

    assert update.name == "ecmwf-ifs-ens-46-day-6-hourly-update"
    assert validate.name == "ecmwf-ifs-ens-46-day-6-hourly-validate"
    assert update.schedule == "0 10 * * *"
    assert validate.schedule == "20 10 * * *"
    assert update.suspend is True
    assert validate.suspend is True


def test_validators_cover_current_data_and_recent_nans(
    dataset: EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
) -> None:
    current, recent_nans = dataset.validators()

    assert isinstance(current, validation.CheckCurrentData)
    assert isinstance(recent_nans, validation.CheckRecentNans)


# (init_time, lead_time, ensemble_member) at 10.5N 60W for the 2026-08-10 and
# 2026-08-11 initializations, lead times 0, 6 and 12 hours, ensemble members 0 and 1.
EXPECTED_POINT_VALUES = {
    # Winds are instantaneous, so lead zero holds values.
    "wind_u_10m": np.array(
        [
            [[-6.0, -6.3125], [-5.75, -5.4375], [-5.25, -5.375]],
            [[-4.0, -4.5], [-5.3125, -5.25], [-4.4375, -5.75]],
        ],
        dtype=np.float32,
    ),
    "wind_v_10m": np.array(
        [
            [[0.1015625, 0.5], [-0.625, -0.32421875], [-0.14453125, -1.234375]],
            [[-0.328125, -0.28125], [-0.6171875, -1.203125], [2.3125, 2.5]],
        ],
        dtype=np.float32,
    ),
    # Extremes are native 6 hour windows whose first window ends at lead 6.
    "maximum_temperature_2m": np.array(
        [
            [[np.nan, np.nan], [29.125, 28.75], [28.5, 28.5]],
            [[np.nan, np.nan], [28.75, 28.625], [28.5, 28.25]],
        ],
        dtype=np.float32,
    ),
    "minimum_temperature_2m": np.array(
        [
            [[np.nan, np.nan], [28.5, 28.5], [28.25, 28.125]],
            [[np.nan, np.nan], [28.5, 27.75], [28.125, 27.625]],
        ],
        dtype=np.float32,
    ),
    # Precipitation deaccumulates the running total to a rate over the previous 6 hours.
    "precipitation_surface": np.array(
        [
            [[np.nan, np.nan], [9.033829e-08, 0.0], [9.033829e-08, 1.8067658e-07]],
            [
                [np.nan, np.nan],
                [4.9769878e-06, 6.6041946e-05],
                [1.9013882e-05, 4.4465065e-05],
            ],
        ],
        dtype=np.float32,
    ),
}


def assert_point_values(ds: xr.Dataset, init_times: list[np.datetime64]) -> None:
    point = ds.sel(init_time=init_times, latitude=10.5, longitude=-60)
    for name, expected in EXPECTED_POINT_VALUES.items():
        np.testing.assert_allclose(
            point[name].values, expected[: len(init_times)], rtol=1e-6
        )


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch,
    dataset: EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
) -> None:
    orig_get_template = dataset.template_config.get_template

    def small_template(self: object, end_time: DatetimeLike) -> xr.DataTree:
        tree = orig_get_template(end_time).sel(
            lead_time=slice("0h", "12h"), ensemble_member=slice(0, 1)
        )
        # Leave init_time at production geometry: one shard per init, so
        # filter_start scopes the run to a single initialization.
        return shrink_chunks_and_shards(
            tree, dims=("lead_time", "ensemble_member", "latitude", "longitude")
        )

    monkeypatch.setattr(type(dataset.template_config), "get_template", small_template)
    first_init = np.datetime64("2026-08-10T00:00:00")
    second_init = np.datetime64("2026-08-11T00:00:00")

    dataset.backfill_local(
        append_dim_end=pd.Timestamp(second_init),
        filter_start=pd.Timestamp(first_init),
    )
    store = dataset.store_factory.primary_store()
    backfilled = xr.open_zarr(store, chunks=None)
    assert second_init not in backfilled.init_time.values
    np.testing.assert_array_equal(
        backfilled.lead_time.values, pd.to_timedelta(["0h", "6h", "12h"])
    )
    assert_point_values(backfilled, [first_init])

    # The update reprocesses the latest existing initialization and appends the next.
    monkeypatch.setattr(
        pd.Timestamp, "now", Mock(return_value=pd.Timestamp("2026-08-12T00:00"))
    )
    dataset.update("test-update")
    assert_point_values(xr.open_zarr(store, chunks=None), [first_init, second_init])
