from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.forecast_48_hour_virtual_fast.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualFastDataset,
)

# Same heavy-rain cell and init as the full 48-hour virtual snapshot test: the fast
# variant reads the same source messages, so decoded values must match exactly.
_Y, _X = 635, 1062
_INIT = "2024-06-01T00:00"

_FILTER_VARS = ["temperature_2m", "wind_u_10m", "total_precipitation_surface"]


def make_dataset(tmp_path: Path) -> NoaaHrrrForecast48HourVirtualFastDataset:
    return NoaaHrrrForecast48HourVirtualFastDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> NoaaHrrrForecast48HourVirtualFastDataset:
    return make_dataset(tmp_path)


def test_polls_only_sfc_files(
    dataset: NoaaHrrrForecast48HourVirtualFastDataset,
) -> None:
    # Every variable is single-level, so a lead is visible once wrfsfc lands,
    # never waiting on wrfprs/wrfnat.
    template_ds = dataset.template_config.get_template(pd.Timestamp("2018-07-14T00:00"))
    job = dataset.region_job_class(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=dataset.template_config.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode="backfill",
    )
    region_ds = template_ds.to_dataset().isel(init_time=slice(0, 1))
    coords = job.generate_source_file_coords(region_ds, job.data_vars)
    assert len(coords) > 0
    assert {c.file_type for c in coords} == {"sfc"}


def test_operational_resources_carry_fast_dataset_id(
    dataset: NoaaHrrrForecast48HourVirtualFastDataset,
) -> None:
    # Everything else about the cron jobs is inherited from the full 48-hour dataset.
    assert dataset.dataset_id == "noaa-hrrr-forecast-48-hour-virtual-fast"
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )
    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"


@pytest.mark.slow
def test_backfill_local_matches_full_dataset_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path)

    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).isel(lead_time=[0, 6]),
    )

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-06-01T01:00"),
        filter_start=pd.Timestamp(_INIT),
        filter_variable_names=_FILTER_VARS,
    )

    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert ds.init_time.values[-1] == np.datetime64(_INIT)

    cell = ds.isel(y=_Y, x=_X).sel(init_time=_INIT)
    f6 = cell.sel(lead_time=pd.Timedelta("6h"))
    # Values match tests/noaa/hrrr/forecast_48_hour_virtual's snapshots exactly.
    np.testing.assert_allclose(f6["temperature_2m"].values, 20.752984619140648)
    np.testing.assert_allclose(f6["wind_u_10m"].values, -7.187067031860352)
    np.testing.assert_allclose(f6["total_precipitation_surface"].values, 98.101)

    f0 = cell.sel(lead_time=pd.Timedelta("0h"))
    assert np.isnan(f0["total_precipitation_surface"].values)
    assert not np.isnan(f0["temperature_2m"].values)
