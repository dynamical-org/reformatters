from io import BytesIO
from pathlib import Path
from typing import Final
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils
from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.common.pydantic import replace
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.dwd.icon_eu.forecast.region_job import (
    DwdIconEuForecastRegionJob,
    DwdIconEuForecastSourceFileCoord,
)
from reformatters.dwd.icon_eu.forecast.template_config import (
    DwdIconEuDataVar,
    DwdIconEuForecastTemplateConfig,
    DwdIconEuInternalAttrs,
)

BASE_FILENAME: Final[str] = (
    "icon-eu_europe_regular-lat-lon_single-level_2000010100_000_T_2M.grib2.bz2"
)
SOURCE_CO_OP_URL: Final[str] = (
    "https://source.coop/dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/2000-01-01T00Z/t_2m/"
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
            comment="Temperature at 2m above ground, averaged over all tiles of a grid point.",
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
) -> DwdIconEuForecastSourceFileCoord:
    return DwdIconEuForecastSourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta(0),
        data_var=t_2m_data_var,
    )


@pytest.fixture
def region_job() -> DwdIconEuForecastRegionJob:
    template_config = DwdIconEuForecastTemplateConfig()
    template_ds = template_config.get_template(
        end_time=template_config.append_dim_start + template_config.append_dim_frequency
    )
    # use `model_construct` to skip pydantic validation so we can pass mock stores
    return DwdIconEuForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )


def test_source_file_coord_get_url(
    source_file_coord: DwdIconEuForecastSourceFileCoord,
) -> None:
    assert source_file_coord.get_url() == SOURCE_CO_OP_URL


def test_source_file_coord_get_fallback_url(
    source_file_coord: DwdIconEuForecastSourceFileCoord,
) -> None:
    dir_name = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/"
    expected = dir_name + BASE_FILENAME
    assert source_file_coord.get_fallback_url() == expected


def test_source_file_coord_get_variable_name_in_filename(
    source_file_coord: DwdIconEuForecastSourceFileCoord,
) -> None:
    assert source_file_coord.variable_name_in_filename == "t_2m"


def test_region_job_generate_source_file_coords(
    region_job: DwdIconEuForecastRegionJob,
) -> None:
    template_config = DwdIconEuForecastTemplateConfig()
    processing_region_ds, _ = region_job._get_region_datasets()

    source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    # 1 init_time x 1 variables x 93 time steps
    assert len(source_file_coords) == 93


def test_region_job_download_file(
    region_job: DwdIconEuForecastRegionJob,
    source_file_coord: DwdIconEuForecastSourceFileCoord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_to_disk_mock = Mock()
    monkeypatch.setattr(
        "reformatters.dwd.icon_eu.forecast.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(source_file_coord)

    url, dataset_id = download_to_disk_mock.call_args[0]

    assert url == SOURCE_CO_OP_URL
    assert dataset_id == "dwd-icon-eu-forecast"


def test_region_job_read_data(
    region_job: DwdIconEuForecastRegionJob,
    source_file_coord: DwdIconEuForecastSourceFileCoord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file_coord = replace(
        source_file_coord, downloaded_path=Path("/fake/path.grib2.bz2")
    )

    # Mock bz2.open to return a BytesIO with some dummy grib data
    dummy_grib_data = b"dummy grib data"
    monkeypatch.setattr("bz2.open", Mock(return_value=BytesIO(dummy_grib_data)))

    # Mock rasterio/MemoryFile
    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 1
    test_data = np.ones((100, 100), dtype=np.float32)
    rasterio_reader.read = Mock(return_value=test_data)

    mock_memory_file = Mock()
    mock_memory_file.__enter__ = Mock(return_value=mock_memory_file)
    mock_memory_file.__exit__ = Mock(return_value=False)
    mock_memory_file.open = Mock(return_value=rasterio_reader)

    monkeypatch.setattr(
        "reformatters.dwd.icon_eu.forecast.region_job.MemoryFile",
        Mock(return_value=mock_memory_file),
    )

    result = region_job.read_data(source_file_coord, source_file_coord.data_var)

    # Verify the result
    assert np.array_equal(result, test_data)
    assert result.shape == (100, 100)
    assert result.dtype == np.float32

    rasterio_reader.read.assert_called_once_with(indexes=1, out_dtype=np.float32)


def test_region_job_apply_data_transformations(
    region_job: DwdIconEuForecastRegionJob,
    t_2m_data_var: DwdIconEuDataVar,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Test deaccumulation
    t_2m_data_var = replace(
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
        "reformatters.dwd.icon_eu.forecast.region_job.deaccumulate_to_rates_inplace",
        mock_deaccum,
    )

    region_job.apply_data_transformations(data_array, t_2m_data_var)
    mock_deaccum.assert_called_once()


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template_config = DwdIconEuForecastTemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "prod"),
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset-dwd",
        template_config_version="test-version",
    )

    # Set current time
    now = pd.Timestamp("2026-02-02T15:48")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now))

    # Set existing data max time
    existing_ds = template_config.get_template(pd.Timestamp("2026-02-01T00:01"))
    template_utils.write_metadata(existing_ds, store_factory)

    # Actually write the coordinate data so max() works
    existing_ds[["init_time"]].to_zarr(
        store_factory.primary_store(writable=True), mode="a"
    )

    jobs, template_ds = DwdIconEuForecastRegionJob.operational_update_jobs(
        primary_store=store_factory.primary_store(),
        tmp_store=tmp_path / "tmp.zarr",
        get_template_fn=template_config.get_template,
        append_dim=template_config.append_dim,
        all_data_vars=template_config.data_vars,
        reformat_job_name="test_job",
    )

    assert template_ds.init_time.max() >= now.floor("12h")
    assert len(jobs) > 0
    for job in jobs:
        assert isinstance(job, DwdIconEuForecastRegionJob)
