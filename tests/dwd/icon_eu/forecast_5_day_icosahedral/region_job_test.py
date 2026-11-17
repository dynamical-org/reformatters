from pathlib import Path
from unittest.mock import Mock

import eccodes
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from reformatters.common.pydantic import replace
from reformatters.common.types import Array1D
from reformatters.dwd.icon_eu import icon_eu_region_job as shared_module
from reformatters.dwd.icon_eu.forecast_5_day_icosahedral import icosahedral_grid
from reformatters.dwd.icon_eu.forecast_5_day_icosahedral.region_job import (
    DwdIconEuForecast5DayIcosahedralRegionJob,
    DwdIconEuForecast5DayIcosahedralSourceFileCoord,
)
from reformatters.dwd.icon_eu.forecast_5_day_icosahedral.template_config import (
    DwdIconEuForecast5DayIcosahedralTemplateConfig,
)


def write_grib(
    path: Path,
    values: Array1D[np.float64],
    grid_uuid: str = "832c1507fbc73eb384139890094d2c60",
    grid_template: int = 101,
) -> None:
    handle = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        eccodes.codes_set(handle, "gridDefinitionTemplateNumber", grid_template)
        if grid_template == 101:
            eccodes.codes_set(handle, "uuidOfHGrid", grid_uuid)
        eccodes.codes_set(handle, "numberOfDataPoints", len(values))
        eccodes.codes_set(handle, "packingType", "grid_ccsds")
        eccodes.codes_set(handle, "bitsPerValue", 24)
        eccodes.codes_set_values(handle, values)
        with path.open("wb") as file:
            eccodes.codes_write(handle, file)
    finally:
        eccodes.codes_release(handle)


@pytest.fixture
def region_job(tmp_path: Path) -> DwdIconEuForecast5DayIcosahedralRegionJob:
    config = DwdIconEuForecast5DayIcosahedralTemplateConfig()
    return DwdIconEuForecast5DayIcosahedralRegionJob(
        tmp_store=tmp_path,
        template_ds=config.get_template("2026-10-07T06"),
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


@pytest.mark.parametrize(
    ("name", "source_values", "expected"),
    [
        ("temperature_2m", [273.15, 274.15, 272.15, 280.15], [0, 1, -1, 7]),
        ("dew_point_temperature_2m", [273.15, 274.15, 272.15, 280.15], [0, 1, -1, 7]),
        ("wind_u_10m", [3, -1, 0, 8], [3, -1, 0, 8]),
    ],
)
def test_download_decode_and_transform_preserves_native_cell_values(
    region_job: DwdIconEuForecast5DayIcosahedralRegionJob,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    source_values: list[float],
    expected: list[float],
) -> None:
    monkeypatch.setattr(icosahedral_grid, "CELL_COUNT", 4)
    path = tmp_path / "field.grib2"
    write_grib(path, np.array(source_values, dtype=np.float64))
    download = Mock(return_value=path)
    monkeypatch.setattr(shared_module, "http_download_to_disk", download)
    var = next(v for v in region_job.data_vars if v.name == name)
    coord = DwdIconEuForecast5DayIcosahedralSourceFileCoord(
        init_time=pd.Timestamp("2026-11-16T00"),
        lead_time=pd.Timedelta("12h"),
        data_var=var,
    )
    downloaded = replace(coord, downloaded_path=region_job.download_file(coord))
    values = region_job.read_data(downloaded, var)
    data = xr.DataArray(values, dims=("cell",))
    region_job.apply_data_transformations(data, var)
    assert_allclose(data.values, expected, atol=0.001, rtol=0)
    download.assert_called_once_with(coord.get_url(), region_job.dataset_id)


def test_archive_miss_falls_back_to_plain_grib_on_dwd(
    region_job: DwdIconEuForecast5DayIcosahedralRegionJob,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    var = next(v for v in region_job.data_vars if v.name == "temperature_2m")
    coord = DwdIconEuForecast5DayIcosahedralSourceFileCoord(
        init_time=pd.Timestamp("2026-11-16T06"),
        lead_time=pd.Timedelta("81h"),
        data_var=var,
    )
    path = tmp_path / "fallback.grib2"
    download = Mock(side_effect=[FileNotFoundError(), path])
    monkeypatch.setattr(shared_module, "http_download_to_disk", download)
    assert region_job.download_file(coord) == path
    assert [call.args[0] for call in download.call_args_list] == [
        "https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/dynamical/dwd-icon-grib/icon-eu/icosahedral/2026-11-16T06/T_2M/PT081H00M.grib2",
        "https://opendata.dwd.de/weather/nwp/v1/m/icon-eu/p/T_2M/r/2026-11-16T06:00/s/PT081H00M.grib2",
    ]


@pytest.mark.parametrize(
    ("uuid", "count", "grid_template"),
    [
        ("ffffffffffffffffffffffffffffffff", 4, 101),
        ("832c1507fbc73eb384139890094d2c60", 5, 101),
        ("832c1507fbc73eb384139890094d2c60", 4, 0),
    ],
)
def test_read_rejects_wrong_grid_before_values_can_be_mislabelled(
    region_job: DwdIconEuForecast5DayIcosahedralRegionJob,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uuid: str,
    count: int,
    grid_template: int,
) -> None:
    monkeypatch.setattr(icosahedral_grid, "CELL_COUNT", 4)
    path = tmp_path / "wrong-grid.grib2"
    write_grib(path, np.arange(count, dtype=np.float64), uuid, grid_template)
    var = region_job.data_vars[0]
    coord = DwdIconEuForecast5DayIcosahedralSourceFileCoord(
        init_time=pd.Timestamp("2026-11-16T00"),
        lead_time=pd.Timedelta("12h"),
        data_var=var,
        downloaded_path=path,
    )
    with pytest.raises(AssertionError):
        region_job.read_data(coord, var)


def test_source_selection_ignores_extra_hourly_files_after_hour_78(
    region_job: DwdIconEuForecast5DayIcosahedralRegionJob,
) -> None:
    var = next(v for v in region_job.data_vars if v.name == "wind_gust_10m")
    coords = region_job.generate_source_file_coords(
        region_job.template_ds.to_dataset(), [var]
    )
    assert_array_equal(
        [coord.lead_time for coord in coords],
        pd.to_timedelta([*range(79), *range(81, 121, 3)], unit="h"),
    )
    assert coords[0].get_url().endswith("/VMAX_10M/PT000H00M.grib2")
    assert coords[-1].get_url().endswith("/VMAX_10M/PT120H00M.grib2")
