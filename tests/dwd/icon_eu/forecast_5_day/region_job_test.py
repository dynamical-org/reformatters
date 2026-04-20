import bz2
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from obstore.exceptions import GenericError, PermissionDeniedError
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from reformatters.common import template_utils
from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.common.pydantic import replace
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.dwd.icon_eu.forecast_5_day import region_job as region_job_module
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
    reset_freq = pd.Timedelta(hours=1)
    data_var = replace(
        t_2m_data_var,
        internal_attrs=replace(
            t_2m_data_var.internal_attrs,
            deaccumulate_to_rate=True,
            window_reset_frequency=reset_freq,
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
    mock_deaccum.assert_called_once_with(
        data_array,
        dim="lead_time",
        reset_frequency=reset_freq,
        accumulation_type="accumulated",
    )


def test_region_job_apply_data_transformations_deaccumulation_optional_kwargs(
    region_job: DwdIconEuForecast5DayRegionJob,
    t_2m_data_var: DwdIconEuDataVar,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_freq = pd.Timedelta.max
    data_var = replace(
        t_2m_data_var,
        internal_attrs=replace(
            t_2m_data_var.internal_attrs,
            deaccumulate_to_rate=True,
            window_reset_frequency=reset_freq,
            deaccumulation_invalid_below_threshold_rate=-50.0,
            deaccumulation_expected_clamp_fraction=0.25,
            deaccumulation_type="running_mean",
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
    mock_deaccum.assert_called_once_with(
        data_array,
        dim="lead_time",
        reset_frequency=reset_freq,
        accumulation_type="running_mean",
        invalid_below_threshold_rate=-50.0,
        expected_clamp_fraction=0.25,
    )


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


def test_region_job_download_file_both_fail(
    region_job: DwdIconEuForecast5DayRegionJob,
    source_file_coord: DwdIconEuForecast5DaySourceFileCoord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mock_download(url: str, dataset_id: str) -> Path:
        raise FileNotFoundError("not found")

    monkeypatch.setattr(
        "reformatters.dwd.icon_eu.forecast_5_day.region_job.http_download_to_disk",
        mock_download,
    )

    with pytest.raises(FileNotFoundError):
        region_job.download_file(source_file_coord)


def test_region_job_download_file_fallback_on_generic_error(
    region_job: DwdIconEuForecast5DayRegionJob,
    source_file_coord: DwdIconEuForecast5DaySourceFileCoord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def mock_download(url: str, dataset_id: str) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise GenericError("generic obstore error")
        return Path("/fake/fallback.grib2.bz2")

    monkeypatch.setattr(
        "reformatters.dwd.icon_eu.forecast_5_day.region_job.http_download_to_disk",
        mock_download,
    )

    result = region_job.download_file(source_file_coord)
    assert result == Path("/fake/fallback.grib2.bz2")
    assert call_count == 2


def test_region_job_download_file_fallback_on_permission_denied(
    region_job: DwdIconEuForecast5DayRegionJob,
    source_file_coord: DwdIconEuForecast5DaySourceFileCoord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def mock_download(url: str, dataset_id: str) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise PermissionDeniedError("permission denied")
        return Path("/fake/fallback.grib2.bz2")

    monkeypatch.setattr(
        "reformatters.dwd.icon_eu.forecast_5_day.region_job.http_download_to_disk",
        mock_download,
    )

    result = region_job.download_file(source_file_coord)
    assert result == Path("/fake/fallback.grib2.bz2")
    assert call_count == 2


def test_region_job_read_data(
    region_job: DwdIconEuForecast5DayRegionJob,
    t_2m_data_var: DwdIconEuDataVar,
    tmp_path: Path,
) -> None:
    height, width = 657, 1377
    # from_bounds(west, south, east, north, ...) produces a north-up transform
    # (row 0 at north, matching the ICON-EU GRIB row order and our descending
    # template latitude coord).
    transform = from_bounds(-23.5, 29.5, 62.5, 70.5, width, height)
    # Row-varying values so the north/south orientation is detectable.
    data = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, np.newaxis], (height, width)
    ).copy()

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        tiff_bytes = memfile.read()

    bz2_path = tmp_path / "test.grib2.bz2"
    bz2_path.write_bytes(bz2.compress(tiff_bytes))

    coord = DwdIconEuForecast5DaySourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta(0),
        data_var=t_2m_data_var,
    )
    coord = replace(coord, downloaded_path=bz2_path)

    result = region_job.read_data(coord, t_2m_data_var)
    assert result.shape == (height, width)
    assert result.dtype == np.float32
    # Source rows run north->south (row 0 = north). Our template latitude is
    # also descending (lat[0] = 70.5), so read_data passes the array through
    # unchanged: result[0] is the northern source row, result[-1] the southern.
    np.testing.assert_array_equal(result, data)
    assert result[0, 0] == 0
    assert result[-1, 0] == height - 1


def test_region_job_read_data_multi_band_raises(
    region_job: DwdIconEuForecast5DayRegionJob,
    t_2m_data_var: DwdIconEuDataVar,
    tmp_path: Path,
) -> None:
    height, width = 10, 10
    transform = from_bounds(0, 0, 1, 1, width, height)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=2,  # Two bands — should trigger assertion
            dtype="float32",
            transform=transform,
        ) as dst:
            dst.write(np.ones((2, height, width), dtype=np.float32))
        tiff_bytes = memfile.read()

    bz2_path = tmp_path / "test_multi.grib2.bz2"
    bz2_path.write_bytes(bz2.compress(tiff_bytes))

    coord = DwdIconEuForecast5DaySourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta(0),
        data_var=t_2m_data_var,
    )
    coord = replace(coord, downloaded_path=bz2_path)

    with pytest.raises(AssertionError, match="Expected exactly 1 element"):
        region_job.read_data(coord, t_2m_data_var)


def _download_and_read_one(
    region_job: DwdIconEuForecast5DayRegionJob,
    init_time: pd.Timestamp,
    lead_time: pd.Timedelta,
    data_var: DwdIconEuDataVar,
) -> None:
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


def _parallel_download_and_read_all(
    region_job: DwdIconEuForecast5DayRegionJob,
    init_time: pd.Timestamp,
    lead_time: pd.Timedelta,
    data_vars: Sequence[DwdIconEuDataVar],
) -> None:
    with ThreadPoolExecutor(max_workers=6) as executor:
        # list() forces the map to drain so exceptions from any variable surface here.
        list(
            executor.map(
                lambda v: _download_and_read_one(region_job, init_time, lead_time, v),
                data_vars,
            )
        )


@pytest.mark.slow
def test_download_from_dwd_and_read_all_variables() -> None:
    """Download a real ICON-EU GRIB file from DWD and read all template variables."""
    template_config = DwdIconEuForecast5DayTemplateConfig()
    # DWD only keeps a ~24h rolling window on their HTTPS server. Pick the most recent
    # init that should be complete (ICON-EU is fully available ~4h after the 00/06/12/18 UTC run).
    init_time = (pd.Timestamp.now() - pd.Timedelta(hours=5)).floor("6h")

    region_job = DwdIconEuForecast5DayRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_config.get_template(init_time + pd.Timedelta(days=1)),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    _parallel_download_and_read_all(
        region_job,
        init_time,
        pd.Timedelta(hours=1),
        template_config.data_vars,
    )


@pytest.mark.slow
def test_download_from_dynamical_source_coop_archive_and_read_all_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify every template variable is available from the Source Co-Op archive.

    We fail loudly on any DWD fallback so the test can only pass if all required files
    are genuinely present in the archive. Uses a fixed init_time far enough in the past
    that DWD's ~24h rolling window would not help.
    """
    real_http_download_to_disk = region_job_module.http_download_to_disk

    def source_coop_only(url: str, dataset_id: str) -> Path:
        assert "opendata.dwd.de" not in url, (
            f"Unexpected DWD fallback for url that should be in the archive: {url}"
        )
        return real_http_download_to_disk(url, dataset_id)

    monkeypatch.setattr(region_job_module, "http_download_to_disk", source_coop_only)

    template_config = DwdIconEuForecast5DayTemplateConfig()
    init_time = pd.Timestamp("2026-04-01T00:00")

    region_job = DwdIconEuForecast5DayRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_config.get_template(init_time + pd.Timedelta(days=1)),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    _parallel_download_and_read_all(
        region_job,
        init_time,
        pd.Timedelta(hours=1),
        template_config.data_vars,
    )
