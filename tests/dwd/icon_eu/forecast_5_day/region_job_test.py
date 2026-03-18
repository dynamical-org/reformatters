from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils
from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.common.pydantic import replace
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.dwd.icon_eu.forecast_5_day.region_job import (
    DwdIconEuForecast5DayRegionJob,
    DwdIconEuForecast5DaySourceFileCoord,
)
from reformatters.dwd.icon_eu.forecast_5_day.template_config import (
    DwdIconEuDataVar,
    DwdIconEuForecast5DayTemplateConfig,
    DwdIconEuInternalAttrs,
)

BASE_FILENAME = (
    "icon-eu_europe_regular-lat-lon_single-level_2000010100_000_T_2M.grib2.bz2"
)
SOURCE_CO_OP_URL = (
    "https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
    "dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/2000-01-01T00/t_2m/"
    + BASE_FILENAME
)


@pytest.fixture
def t_2m_data_var() -> DwdIconEuDataVar:
    return DwdIconEuDataVar(
        name="temperature_2m",
        encoding=Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=(1, 100, 100),
            shards=(1, 100, 100),
        ),
        attrs=DataVarAttrs(
            short_name="2t",
            long_name="2 metre temperature",
            units="degree_Celsius",
            step_type="instant",
            standard_name="air_temperature",
        ),
        internal_attrs=DwdIconEuInternalAttrs(
            variable_name_in_filename="t_2m",
            keep_mantissa_bits=7,
        ),
    )


@pytest.fixture
def source_file_coord(
    t_2m_data_var: DwdIconEuDataVar,
) -> DwdIconEuForecast5DaySourceFileCoord:
    return DwdIconEuForecast5DaySourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta(0),
        data_var=t_2m_data_var,
    )


@pytest.fixture
def region_job() -> DwdIconEuForecast5DayRegionJob:
    template_config = DwdIconEuForecast5DayTemplateConfig()
    template_ds = template_config.get_template(
        end_time=template_config.append_dim_start + template_config.append_dim_frequency
    )
    return DwdIconEuForecast5DayRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )


def test_source_file_coord_get_url(
    source_file_coord: DwdIconEuForecast5DaySourceFileCoord,
) -> None:
    assert source_file_coord.get_url() == SOURCE_CO_OP_URL


def test_source_file_coord_get_fallback_url(
    source_file_coord: DwdIconEuForecast5DaySourceFileCoord,
) -> None:
    dir_name = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/"
    expected = dir_name + BASE_FILENAME
    assert source_file_coord.get_fallback_url() == expected


def test_source_file_coord_get_variable_name_in_filename(
    source_file_coord: DwdIconEuForecast5DaySourceFileCoord,
) -> None:
    assert source_file_coord.variable_name_in_filename == "t_2m"


def test_source_file_coord_out_loc(
    source_file_coord: DwdIconEuForecast5DaySourceFileCoord,
) -> None:
    out_loc = source_file_coord.out_loc()
    assert out_loc == {
        "init_time": pd.Timestamp("2000-01-01T00:00"),
        "lead_time": pd.Timedelta(0),
    }


def test_region_job_source_groups() -> None:
    template_config = DwdIconEuForecast5DayTemplateConfig()
    groups = DwdIconEuForecast5DayRegionJob.source_groups(template_config.data_vars)
    # Each variable gets its own group (one var per GRIB file)
    assert len(groups) == len(template_config.data_vars)
    for group in groups:
        assert len(group) == 1


def test_region_job_generate_source_file_coords(
    region_job: DwdIconEuForecast5DayRegionJob,
) -> None:
    template_config = DwdIconEuForecast5DayTemplateConfig()
    processing_region_ds, _ = region_job._get_region_datasets()

    source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    # 1 init_time x 93 lead times
    assert len(source_file_coords) == 93


def test_region_job_download_file(
    region_job: DwdIconEuForecast5DayRegionJob,
    source_file_coord: DwdIconEuForecast5DaySourceFileCoord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_to_disk_mock = Mock()
    monkeypatch.setattr(
        "reformatters.dwd.icon_eu.forecast_5_day.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(source_file_coord)

    url, dataset_id = download_to_disk_mock.call_args[0]
    assert url == SOURCE_CO_OP_URL
    assert dataset_id == "dwd-icon-eu-forecast-5-day"


def test_region_job_download_file_fallback(
    region_job: DwdIconEuForecast5DayRegionJob,
    source_file_coord: DwdIconEuForecast5DaySourceFileCoord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def mock_download(url: str, dataset_id: str) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise FileNotFoundError("not found")
        return Path("/fake/fallback.grib2.bz2")

    monkeypatch.setattr(
        "reformatters.dwd.icon_eu.forecast_5_day.region_job.http_download_to_disk",
        mock_download,
    )

    result = region_job.download_file(source_file_coord)
    assert result == Path("/fake/fallback.grib2.bz2")
    assert call_count == 2


def test_region_job_apply_data_transformations_deaccumulation(
    region_job: DwdIconEuForecast5DayRegionJob,
    t_2m_data_var: DwdIconEuDataVar,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_var = replace(
        t_2m_data_var,
        internal_attrs=replace(
            t_2m_data_var.internal_attrs,
            deaccumulate_to_rate=True,
            window_reset_frequency=pd.Timedelta(hours=1),
        ),
    )
    times = pd.date_range("2000-01-01", periods=3, freq="1h")
    data = np.array([0, 1, 2], dtype=np.float32)
    data_array = xr.DataArray(data, coords={"lead_time": times}, dims=["lead_time"])

    mock_deaccum = Mock()
    monkeypatch.setattr(
        "reformatters.dwd.icon_eu.forecast_5_day.region_job.deaccumulate_to_rates_inplace",
        mock_deaccum,
    )

    region_job.apply_data_transformations(data_array, data_var)
    mock_deaccum.assert_called_once()


def test_region_job_apply_data_transformations_scale_factor(
    region_job: DwdIconEuForecast5DayRegionJob,
    t_2m_data_var: DwdIconEuDataVar,
) -> None:
    data_var = replace(
        t_2m_data_var,
        internal_attrs=replace(
            t_2m_data_var.internal_attrs,
            scale_factor=0.001,
        ),
    )
    data = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    data_array = xr.DataArray(data.copy(), dims=["x"])

    region_job.apply_data_transformations(data_array, data_var)
    np.testing.assert_allclose(data_array.values, [1.0, 2.0, 3.0])


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template_config = DwdIconEuForecast5DayTemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "prod"),
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset-dwd",
        template_config_version="test-version",
    )

    now = pd.Timestamp("2026-05-02T15:48")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now))

    existing_ds_end_time = pd.Timestamp("2026-05-01T00:01")
    existing_ds = template_config.get_template(end_time=existing_ds_end_time)
    template_utils.write_metadata(existing_ds, store_factory)

    jobs, template_ds = DwdIconEuForecast5DayRegionJob.operational_update_jobs(
        primary_store=store_factory.primary_store(),
        tmp_store=tmp_path / "tmp.zarr",
        get_template_fn=template_config.get_template,
        append_dim=template_config.append_dim,
        all_data_vars=template_config.data_vars,
        reformat_job_name="test_job",
    )

    assert template_ds.init_time.max() == pd.Timestamp("2026-05-02T12:00")
    assert len(jobs) == 7  # 2026-05-01T00 through 2026-05-02T12
    for job in jobs:
        assert isinstance(job, DwdIconEuForecast5DayRegionJob)
        assert job.data_vars == template_config.data_vars


@pytest.mark.slow
def test_download_and_read_all_variables() -> None:
    """Download a real ICON-EU GRIB file and read all template variables."""
    template_config = DwdIconEuForecast5DayTemplateConfig()
    init_time = pd.Timestamp("2026-03-01T00:00")

    region_job = DwdIconEuForecast5DayRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_config.get_template(init_time + pd.Timedelta(days=1)),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    lead_time = pd.Timedelta(hours=1)

    for data_var in template_config.data_vars:
        coord = DwdIconEuForecast5DaySourceFileCoord(
            init_time=init_time,
            lead_time=lead_time,
            data_var=data_var,
        )
        coord = replace(coord, downloaded_path=region_job.download_file(coord))

        data = region_job.read_data(coord, data_var)
        assert data.shape == (657, 1377), (
            f"Unexpected shape for {data_var.name}: {data.shape}"
        )
        assert np.all(np.isfinite(data)), f"Non-finite values for {data_var.name}"
