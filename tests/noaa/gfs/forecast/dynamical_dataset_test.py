from datetime import timedelta
from pathlib import Path

import pytest
import xarray as xr
from pytest import MonkeyPatch

from reformatters.common import validation, zarr
from reformatters.noaa.gfs.forecast import NoaaGfsForecastDataset
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG

pytestmark = pytest.mark.slow


@pytest.fixture
def dataset() -> NoaaGfsForecastDataset:
    return NoaaGfsForecastDataset(storage_config=NOOP_STORAGE_CONFIG)


def test_reformat_local(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    dataset = NoaaGfsForecastDataset(storage_config=NOOP_STORAGE_CONFIG)
    init_time_start = NoaaGfsForecastTemplateConfig().append_dim_start
    init_time_end = init_time_start + timedelta(hours=12)

    monkeypatch.setattr(zarr, "_LOCAL_ZARR_STORE_BASE_PATH", tmp_path)
    orig_get_template = NoaaGfsForecastTemplateConfig.get_template
    monkeypatch.setattr(
        NoaaGfsForecastTemplateConfig,
        "get_template",
        lambda self, end_time: orig_get_template(self, end_time).sel(
            lead_time=slice("0h", "12h")
        ),
    )

    # 1. Backfill archive
    dataset.reformat_local(
        append_dim_end=init_time_end,
        filter_variable_names=[
            "temperature_2m",  # instantaneous
            "precipitation_surface",  # accumulation we deaccumulate
            "precipitable_water_atmosphere",  # average over window
            "maximum_temperature_2m",  # max over window
            "minimum_temperature_2m",  # min over window
        ],
    )
    original_ds = xr.open_zarr(
        dataset._final_store(), decode_timedelta=True, chunks=None
    )

    space_subset_ds = original_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))

    # These variables are present at all lead times
    assert (
        (
            space_subset_ds[["temperature_2m", "precipitable_water_atmosphere"]]
            .isnull()
            .mean()
            == 0
        )
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
    assert point_ds["precipitable_water_atmosphere"] == 56.5


def test_operational_kubernetes_resources_and_validators(
    dataset: NoaaGfsForecastDataset,
) -> None:
    image_tag = "test-image"
    jobs = dataset.operational_kubernetes_resources(image_tag)

    assert len(jobs) == 2
    update_job, validation_job = jobs
    assert update_job.name == f"{dataset.dataset_id}-operational-update"
    assert validation_job.name == f"{dataset.dataset_id}-validation"


def test_validators(dataset: NoaaGfsForecastDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
