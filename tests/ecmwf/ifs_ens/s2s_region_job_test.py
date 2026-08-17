"""Read and transform real ECMWF S2S messages of the 2026-08-10T00Z initialization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.iterating import item
from reformatters.ecmwf.archive_gribs.request_shards import DAILY_LEAD_TIMES
from reformatters.ecmwf.ifs_ens.forecast_46_day_1_5_degree.template_config import (
    EcmwfIfsEnsForecast46Day15DegreeTemplateConfig,
)
from reformatters.ecmwf.ifs_ens.s2s_config_models import EcmwfS2sDataVar
from reformatters.ecmwf.ifs_ens.s2s_region_job import (
    EcmwfS2sRegionJob,
    EcmwfS2sSourceFileCoord,
    selections_by_variable,
)
from tests.ecmwf.s2s_fixtures import blob_record, extract_messages

INIT_TIME = pd.Timestamp("2026-08-10T00:00")
DAILY_CONFIG = EcmwfIfsEnsForecast46Day15DegreeTemplateConfig()


def daily_var(name: str) -> EcmwfS2sDataVar:
    return item(v for v in DAILY_CONFIG.data_vars if v.name == name)


def region_job(data_var: EcmwfS2sDataVar, tmp_path: Path) -> EcmwfS2sRegionJob:
    return EcmwfS2sRegionJob(
        tmp_store=tmp_path / "tmp.zarr",
        template_ds=DAILY_CONFIG.get_template(DAILY_CONFIG.append_dim_start),
        data_vars=[data_var],
        append_dim=DAILY_CONFIG.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )


def source_file_coord(
    data_var: EcmwfS2sDataVar,
    lead_hours: int,
    downloaded_path: Path,
    levels: tuple[str | None, ...] = (),
) -> EcmwfS2sSourceFileCoord:
    ecds_variable = data_var.internal_attrs.ecds_variable
    return EcmwfS2sSourceFileCoord(
        init_time=INIT_TIME,
        lead_time=pd.Timedelta(hours=lead_hours),
        ensemble_member=0,
        ecds_variable=ecds_variable,
        levels=levels,
        selection=selections_by_variable([data_var])[
            (ecds_variable, "control_forecast")
        ],
        downloaded_path=downloaded_path,
    )


def read_one(
    name: str,
    lead_hours: int,
    tmp_path: Path,
    level: str = "",
    levels: tuple[str | None, ...] = (),
) -> np.ndarray:  # type: ignore[type-arg]
    data_var = daily_var(name)
    ecds_variable = data_var.internal_attrs.ecds_variable
    path = extract_messages(
        tmp_path / "extract.grib2", blob_record(ecds_variable, level, lead_hours)
    )
    coord = source_file_coord(data_var, lead_hours, path, levels=levels)
    return region_job(data_var, tmp_path).read_data(coord, data_var)


def test_reads_a_surface_message_in_raw_grib_units(tmp_path: Path) -> None:
    """GDAL normalizes some temperatures to Celsius; the read must stay in Kelvin."""
    values = read_one("average_temperature_2m", 24, tmp_path)

    assert values.shape == (121, 240)
    assert values.dtype == np.float32
    assert values[60, 120] == pytest.approx(296.5889, abs=1e-3)
    assert np.nanmin(values) == pytest.approx(204.5693, abs=1e-3)


def test_reads_a_pressure_level_message_into_its_output_level(tmp_path: Path) -> None:
    levels: tuple[str | None, ...] = tuple(
        "500_hpa" if level == 500 else None
        for level in DAILY_CONFIG.dimension_coordinates()["pressure_level"]
    )
    data_var = item(
        v for v in DAILY_CONFIG.data_vars if v.path == "pressure_level/temperature"
    )
    path = extract_messages(
        tmp_path / "extract.grib2", blob_record("temperature", "500_hpa", 24)
    )
    coord = source_file_coord(data_var, 24, path, levels=levels)

    values = region_job(data_var, tmp_path).read_data(coord, data_var)

    assert values.shape == (121, 240, len(levels))
    level_index = levels.index("500_hpa")
    assert values[60, 120, level_index] == pytest.approx(270.1207, abs=1e-3)
    assert np.isnan(values[:, :, levels.index(None)]).all()


def test_masks_the_land_only_sentinel(tmp_path: Path) -> None:
    values = read_one("soil_moisture_0_20cm", 24, tmp_path)

    assert np.isnan(values[60, 120])
    assert not (values == 9999.0).any()
    # The sea mask covers 66% of the 1.5 degree grid.
    assert int(np.isnan(values).sum()) == 19220


def test_rejects_a_message_of_the_wrong_variable(tmp_path: Path) -> None:
    data_var = daily_var("average_temperature_2m")
    path = extract_messages(
        tmp_path / "extract.grib2", blob_record("total_cloud_cover", "", 24)
    )
    coord = source_file_coord(data_var, 24, path)

    with pytest.raises(AssertionError, match="element="):
        region_job(data_var, tmp_path).read_data(coord, data_var)


def data_array(
    data_var: EcmwfS2sDataVar,
    values: np.ndarray,
    lead_times: pd.TimedeltaIndex,  # type: ignore[type-arg]
) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("lead_time", "latitude", "longitude"),
        coords={"lead_time": lead_times},
        attrs=data_var.attrs.model_dump(exclude_none=True),
    )


def test_converts_kelvin_to_celsius(tmp_path: Path) -> None:
    data_var = daily_var("average_temperature_2m")
    values = read_one("average_temperature_2m", 24, tmp_path)[np.newaxis]
    array = data_array(data_var, values, pd.to_timedelta([pd.Timedelta("24h")]))

    region_job(data_var, tmp_path).apply_data_transformations(array, data_var)

    # 296.5889 K is 23.4389 C, rounded to 23.5 by keep_mantissa_bits=7.
    assert array.values[0, 60, 120] == pytest.approx(23.5, abs=1e-4)


def test_clamps_cloud_cover_to_a_percentage(tmp_path: Path) -> None:
    data_var = daily_var("total_cloud_cover_atmosphere")
    values = read_one("total_cloud_cover_atmosphere", 24, tmp_path)[np.newaxis]
    values[0, 0, 0] = 100.01
    array = data_array(data_var, values, pd.to_timedelta([pd.Timedelta("24h")]))

    region_job(data_var, tmp_path).apply_data_transformations(array, data_var)

    assert array.values[0, 0, 0] == 100.0
    assert array.values[0, 60, 120] == pytest.approx(74.0, abs=1e-4)


def test_clamps_soil_moisture_to_non_negative(tmp_path: Path) -> None:
    data_var = daily_var("soil_moisture_0_20cm")
    values = read_one("soil_moisture_0_20cm", 24, tmp_path)[np.newaxis]
    values[0, 0, 0] = -6.8e-10
    array = data_array(data_var, values, pd.to_timedelta([pd.Timedelta("24h")]))

    region_job(data_var, tmp_path).apply_data_transformations(array, data_var)

    assert array.values[0, 0, 0] == 0.0


def test_deaccumulates_precipitation_to_a_rate(tmp_path: Path) -> None:
    data_var = daily_var("precipitation_convective_surface")
    lead_hours = (0, 24, 48)
    paths = [
        extract_messages(
            tmp_path / f"extract_{hours}.grib2",
            blob_record("convective_precipitation", "", hours),
        )
        for hours in lead_hours
    ]
    job = region_job(data_var, tmp_path)
    accumulated = np.stack(
        [
            job.read_data(source_file_coord(data_var, hours, path), data_var)
            for hours, path in zip(lead_hours, paths, strict=True)
        ]
    )
    array = data_array(
        data_var, accumulated.copy(), pd.to_timedelta([f"{h}h" for h in lead_hours])
    )
    wettest = np.unravel_index(np.argmax(accumulated[2]), accumulated[2].shape)

    job.apply_data_transformations(array, data_var)

    twenty_four_hours = 24 * 3600
    assert array.values[1][wettest] == pytest.approx(
        accumulated[1][wettest] / twenty_four_hours, rel=5e-3
    )
    assert array.values[2][wettest] == pytest.approx(
        (accumulated[2][wettest] - accumulated[1][wettest]) / twenty_four_hours,
        rel=5e-3,
    )


def test_a_24_hour_mean_variable_has_no_lead_time_zero_coord() -> None:
    """A mean over the previous 24 hours cannot exist at the initialization time."""
    mean_var = daily_var("average_temperature_2m")
    point_var = daily_var("pressure_surface")

    assert not mean_var.has_hour_0_values()
    assert point_var.has_hour_0_values()
    assert DAILY_LEAD_TIMES[0] == "0"
