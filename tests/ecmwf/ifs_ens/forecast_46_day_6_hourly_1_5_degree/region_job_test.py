from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.iterating import item
from reformatters.ecmwf.ifs_ens.forecast_46_day_6_hourly_1_5_degree.template_config import (
    EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_config_models import (
    EcmwfIfsEns46DayDataVar,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_region_job import (
    EcmwfIfsEns46DayRegionJob,
)

CONFIG = EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig()


def data_var(name: str) -> EcmwfIfsEns46DayDataVar:
    return item(data_var for data_var in CONFIG.data_vars if data_var.name == name)


def region_job(
    selected_data_var: EcmwfIfsEns46DayDataVar, tmp_path: Path
) -> EcmwfIfsEns46DayRegionJob:
    return EcmwfIfsEns46DayRegionJob(
        tmp_store=tmp_path / "tmp.zarr",
        template_ds=CONFIG.get_template(CONFIG.append_dim_start),
        data_vars=[selected_data_var],
        append_dim=CONFIG.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )


def test_native_extreme_reads_one_message_per_six_hour_step(tmp_path: Path) -> None:
    maximum = data_var("maximum_temperature_2m")
    job = region_job(maximum, tmp_path)
    processing_region_ds, _ = job._get_region_datasets()

    coords = job.generate_source_file_coords(
        processing_region_ds.isel(lead_time=slice(0, 3), ensemble_member=[0]),
        [maximum],
    )

    assert [coord.lead_time for coord in coords] == list(pd.to_timedelta(["6h", "12h"]))
    assert all(coord.source_lead_times == (coord.lead_time,) for coord in coords)


def test_precipitation_deaccumulates_each_six_hour_interval(tmp_path: Path) -> None:
    precipitation = data_var("precipitation_surface")
    values = np.array([0.0, 21_600.0, 64_800.0], dtype=np.float32)
    array = xr.DataArray(
        values,
        dims=("lead_time",),
        coords={"lead_time": pd.to_timedelta(["0h", "6h", "12h"])},
        attrs=precipitation.attrs.model_dump(exclude_none=True),
    )

    region_job(precipitation, tmp_path).apply_data_transformations(array, precipitation)

    np.testing.assert_allclose(array.values, [np.nan, 1.0, 2.0], equal_nan=True)


def test_variables_resolve_to_the_two_native_archive_blob_groups() -> None:
    groups = EcmwfIfsEns46DayRegionJob.source_file_var_groups(CONFIG.data_vars)

    assert {frozenset(data_var.name for data_var in group) for group in groups} == {
        frozenset({"precipitation_surface", "wind_u_10m", "wind_v_10m"}),
        frozenset({"maximum_temperature_2m", "minimum_temperature_2m"}),
    }
