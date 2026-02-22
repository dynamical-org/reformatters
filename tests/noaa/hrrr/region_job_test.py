from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.noaa.hrrr.forecast_48_hour.region_job import (
    NoaaHrrrForecast48HourRegionJob,
    NoaaHrrrForecast48HourSourceFileCoord,
)
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.hrrr.region_job import (
    NoaaHrrrRegionJob,
    NoaaHrrrSourceFileCoord,
)
from reformatters.noaa.hrrr.template_config import NoaaHrrrCommonTemplateConfig
from reformatters.noaa.noaa_utils import has_hour_0_values


@pytest.fixture
def template_config() -> NoaaHrrrCommonTemplateConfig:
    # Use the Forecast 48 Hour template because we need
    # a concrete implementation to get data_vars
    return NoaaHrrrForecast48HourTemplateConfig()


def test_source_file_coord_get_url(
    template_config: NoaaHrrrCommonTemplateConfig,
) -> None:
    """Test URL generation for HRRR source file coordinates."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )
    expected = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t00z.wrfsfcf00.grib2"
    assert coord.get_url() == expected


def test_source_file_coord_get_url_different_lead_time(
    template_config: NoaaHrrrCommonTemplateConfig,
) -> None:
    """Test URL generation for different lead times."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T12:00"),
        lead_time=pd.Timedelta(hours=24),
        domain="conus",
        file_type="prs",
        data_vars=template_config.data_vars,
    )
    expected = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t12z.wrfprsf24.grib2"
    assert coord.get_url() == expected


def test_source_file_coord_get_idx_url(
    template_config: NoaaHrrrCommonTemplateConfig,
) -> None:
    """Test index URL generation."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T06:00"),
        lead_time=pd.Timedelta(hours=6),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )
    expected = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t06z.wrfsfcf06.grib2.idx"
    assert coord.get_idx_url() == expected


def test_source_file_coord_invalid_lead_time(
    template_config: NoaaHrrrCommonTemplateConfig,
) -> None:
    """Test that invalid lead times raise appropriate errors."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(minutes=30),  # 0.5 hours - not a whole hour
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:1],
    )

    with pytest.raises(AssertionError):
        coord.get_url()


def test_region_job_source_groups(
    template_config: NoaaHrrrCommonTemplateConfig,
) -> None:
    """Test that data variables are grouped by file type."""
    # Test source grouping with available sfc variables
    groups = NoaaHrrrRegionJob.source_groups(template_config.data_vars)

    # Two groups expected: those with hour 0 values and those without
    assert len(groups) == 2
    for group in groups:
        assert len({v.internal_attrs.hrrr_file_type for v in group}) == 1
        assert len({has_hour_0_values(v) for v in group}) == 1


def test_region_job_source_groups_multiple_file_types(
    template_config: NoaaHrrrCommonTemplateConfig,
) -> None:
    """Test source grouping with variables from different file types."""
    data_vars = template_config.data_vars

    # Mix variables from different file types (if available)
    mixed_vars = []
    file_types_seen = set()
    for var in data_vars:
        if var.internal_attrs.hrrr_file_type not in file_types_seen:
            mixed_vars.append(var)
            file_types_seen.add(var.internal_attrs.hrrr_file_type)
        if len(mixed_vars) >= 2:  # Get at least 2 different file types
            break

    if len(mixed_vars) > 1:
        groups = NoaaHrrrRegionJob.source_groups(mixed_vars)
        # Should have separate groups for different file types
        assert len(groups) >= 1

        # Each group should contain only variables from the same file type
        for group in groups:
            file_types_in_group = {v.internal_attrs.hrrr_file_type for v in group}
            assert len(file_types_in_group) == 1


def test_region_job_download_file(
    template_config: NoaaHrrrCommonTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HRRR file download with mocked network calls."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:3],
    )

    region_job = NoaaHrrrRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    mock_download_path = Path("/fake/downloaded/hrrr_file.grib2")
    mock_http_download_to_disk = Mock(return_value=mock_download_path)
    monkeypatch.setattr(
        NoaaHrrrRegionJob,
        "dataset_id",
        "test-dataset-hrrr",
    )

    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.grib_message_byte_ranges_from_index",
        Mock(return_value=([0, 200], [100, 350])),
    )
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.http_download_to_disk",
        mock_http_download_to_disk,
    )

    result = region_job.download_file(coord)

    assert result == mock_download_path

    assert mock_http_download_to_disk.call_count == 2
    assert mock_http_download_to_disk.call_args_list[0].args == (
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t00z.wrfsfcf00.grib2.idx",
        "test-dataset-hrrr",
    )
    assert mock_http_download_to_disk.call_args_list[0].kwargs == {}

    assert mock_http_download_to_disk.call_args_list[1].args == (
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t00z.wrfsfcf00.grib2",
        "test-dataset-hrrr",
    )
    assert mock_http_download_to_disk.call_args_list[1].kwargs == {
        "byte_ranges": ([0, 200], [100, 350]),
        "local_path_suffix": "-24863b9b",
    }


def test_region_job_read_data(
    template_config: NoaaHrrrCommonTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HRRR data reading with mocked file operations."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:1],
        downloaded_path=Path("fake/path/to/downloaded/file.grib2"),
    )

    region_job = NoaaHrrrRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 1
    rasterio_reader.descriptions = ['0[-] EATM="Entire Atmosphere"']
    rasterio_reader.tags = Mock(return_value={"GRIB_ELEMENT": "REFC"})
    test_data = np.ones((1059, 1799), dtype=np.float32) * 42.0
    rasterio_reader.read = Mock(return_value=test_data)
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(coord, template_config.data_vars[0])

    # Verify the result
    assert np.array_equal(result, test_data)
    assert result.shape == (1059, 1799)  # HRRR CONUS grid dimensions (y, x)
    assert result.dtype == np.float32

    rasterio_reader.read.assert_called_once_with(1, out_dtype=np.float32)


def test_region_job_read_data_no_matching_bands(
    template_config: NoaaHrrrCommonTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that read_data raises an error when no matching bands are found."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:1],
        downloaded_path=Path("fake/path/to/downloaded/file.grib2"),
    )

    region_job = NoaaHrrrRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 1
    rasterio_reader.descriptions = ["Wrong description"]
    rasterio_reader.tags = Mock(return_value={"GRIB_ELEMENT": "WRONG"})
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    with pytest.raises(
        AssertionError, match="Expected exactly 1 matching band, found 0"
    ):
        region_job.read_data(coord, template_config.data_vars[0])


def test_region_job_read_data_multiple_matching_bands(
    template_config: NoaaHrrrCommonTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that read_data raises an error when multiple matching bands are found."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:1],
        downloaded_path=Path("fake/path/to/downloaded/file.grib2"),
    )

    region_job = NoaaHrrrRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 2
    rasterio_reader.descriptions = [
        '0[-] EATM="Entire Atmosphere"',
        '0[-] EATM="Entire Atmosphere"',
    ]
    rasterio_reader.tags = Mock(return_value={"GRIB_ELEMENT": "REFC"})
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    with pytest.raises(
        AssertionError, match="Expected exactly 1 matching band, found 2"
    ):
        region_job.read_data(coord, template_config.data_vars[0])


def test_apply_data_transformations_binary_rounding(
    template_config: NoaaHrrrCommonTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that binary rounding is called when keep_mantissa_bits is set."""
    region_job = NoaaHrrrRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    # Create a data var with binary rounding enabled
    data_var = replace(
        template_config.data_vars[0],
        internal_attrs=replace(
            template_config.data_vars[0].internal_attrs,
            keep_mantissa_bits=10,
            deaccumulate_to_rate=False,
        ),
    )

    test_data = np.array([1.23456789, 2.34567890, 3.45678901], dtype=np.float32)
    data_array = xr.DataArray(test_data.copy(), dims=["x"])

    mock_round = Mock()
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.round_float32_inplace",
        mock_round,
    )

    region_job.apply_data_transformations(data_array, data_var)

    # Verify rounding was called with correct arguments
    mock_round.assert_called_once_with(data_array.values, 10)


def test_apply_data_transformations_deaccumulation(
    template_config: NoaaHrrrCommonTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that deaccumulation is called when deaccumulate_to_rate is True."""
    region_job = NoaaHrrrRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    # Create a data var with deaccumulation enabled
    data_var = replace(
        template_config.data_vars[0],
        internal_attrs=replace(
            template_config.data_vars[0].internal_attrs,
            deaccumulate_to_rate=True,
            window_reset_frequency=pd.Timedelta(hours=1),
            keep_mantissa_bits="no-rounding",
        ),
    )

    times = pd.date_range("2024-01-01", periods=5, freq="1h")
    test_data = np.array([0.0, 3.6, 7.2, 10.8, 14.4], dtype=np.float32)
    data_array = xr.DataArray(
        test_data, dims=["time"], coords={"time": times}, attrs={"units": "mm s-1"}
    )

    mock_deaccumulate = Mock()
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.deaccumulate_to_rates_inplace",
        mock_deaccumulate,
    )

    region_job.apply_data_transformations(data_array, data_var)

    # Verify deaccumulation was called with correct arguments
    mock_deaccumulate.assert_called_once()
    call_args = mock_deaccumulate.call_args
    assert call_args.kwargs["dim"] == "time"
    assert call_args.kwargs["reset_frequency"] == pd.Timedelta(hours=1)


def test_update_append_dim_end() -> None:
    """Test that _update_append_dim_end returns current time."""
    before = pd.Timestamp.now()
    result = NoaaHrrrRegionJob._update_append_dim_end()
    after = pd.Timestamp.now()

    assert before <= result <= after


def test_update_append_dim_start() -> None:
    """Test that _update_append_dim_start returns max time from existing data."""
    times = pd.date_range("2024-01-01", periods=10, freq="1h")
    time_coord = xr.DataArray(times, dims=["time"])

    result = NoaaHrrrRegionJob._update_append_dim_start(time_coord)

    assert result == pd.Timestamp("2024-01-01 09:00:00")


def test_source_file_coord_get_url_nomads(
    template_config: NoaaHrrrCommonTemplateConfig,
) -> None:
    """Test that get_url(nomads=True) returns a NOMADS URL with the same path as S3."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T12:00"),
        lead_time=pd.Timedelta(hours=6),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )
    assert (
        coord.get_url()
        == "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t12z.wrfsfcf06.grib2"
    )
    assert (
        coord.get_url(nomads=False)
        == "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t12z.wrfsfcf06.grib2"
    )
    assert (
        coord.get_url(nomads=True)
        == "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.20240229/conus/hrrr.t12z.wrfsfcf06.grib2"
    )


def test_source_file_coord_get_idx_url_nomads(
    template_config: NoaaHrrrCommonTemplateConfig,
) -> None:
    """Test that get_idx_url(nomads=True) returns the NOMADS index URL."""
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T06:00"),
        lead_time=pd.Timedelta(hours=3),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )
    assert (
        coord.get_idx_url(nomads=True)
        == "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.20240229/conus/hrrr.t06z.wrfsfcf03.grib2.idx"
    )
    assert (
        coord.get_idx_url(nomads=False)
        == "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t06z.wrfsfcf03.grib2.idx"
    )


def _make_hrrr_region_job(
    template_config: NoaaHrrrCommonTemplateConfig,
) -> NoaaHrrrRegionJob:
    return NoaaHrrrRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )


def test_download_file_uses_s3_for_old_init_time(
    template_config: NoaaHrrrCommonTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that download_file uses S3 directly for init times older than the NOMADS threshold."""
    fixed_now = pd.Timestamp("2025-01-15T12:00")
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: fixed_now)
    )
    region_job = _make_hrrr_region_job(template_config)
    monkeypatch.setattr(NoaaHrrrRegionJob, "dataset_id", "test-dataset-hrrr")

    coord = NoaaHrrrSourceFileCoord(
        init_time=fixed_now - pd.Timedelta(hours=24),  # old: > 18h threshold
        lead_time=pd.Timedelta(hours=2),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:1],
    )

    mock_download = Mock(return_value=Mock())
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.http_download_to_disk", mock_download
    )
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.grib_message_byte_ranges_from_index",
        Mock(return_value=([0], [100])),
    )

    region_job.download_file(coord)

    urls_called = [call.args[0] for call in mock_download.call_args_list]
    assert all(
        url.startswith("https://noaa-hrrr-bdp-pds.s3.amazonaws.com")
        for url in urls_called
    )


def test_download_file_tries_nomads_first_for_recent_init_time(
    template_config: NoaaHrrrCommonTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that download_file tries NOMADS first for init times within 18h."""
    fixed_now = pd.Timestamp("2025-01-15T12:00")
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: fixed_now)
    )
    region_job = _make_hrrr_region_job(template_config)
    monkeypatch.setattr(NoaaHrrrRegionJob, "dataset_id", "test-dataset-hrrr")

    coord = NoaaHrrrSourceFileCoord(
        init_time=fixed_now - pd.Timedelta(hours=6),  # recent: < 18h threshold
        lead_time=pd.Timedelta(hours=2),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:1],
    )

    mock_data_path = Mock()
    mock_download = Mock(return_value=mock_data_path)
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.http_download_to_disk", mock_download
    )
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.grib_message_byte_ranges_from_index",
        Mock(return_value=([0], [100])),
    )

    result = region_job.download_file(coord)

    assert result == mock_data_path
    # NOMADS succeeds, so both idx and data calls go to NOMADS
    urls_called = [call.args[0] for call in mock_download.call_args_list]
    assert all(url.startswith("https://nomads.ncep.noaa.gov") for url in urls_called)


def test_download_file_falls_back_to_s3_when_nomads_fails(
    template_config: NoaaHrrrCommonTemplateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that download_file falls back to S3 when NOMADS raises FileNotFoundError."""
    fixed_now = pd.Timestamp("2025-01-15T12:00")
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: fixed_now)
    )
    region_job = _make_hrrr_region_job(template_config)
    monkeypatch.setattr(NoaaHrrrRegionJob, "dataset_id", "test-dataset-hrrr")

    coord = NoaaHrrrSourceFileCoord(
        init_time=fixed_now - pd.Timedelta(hours=6),  # recent
        lead_time=pd.Timedelta(hours=2),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:1],
    )

    mock_s3_path = Mock()

    def mock_download_side_effect(url: str, dataset_id: str, **kwargs: object) -> Mock:
        if "nomads" in url:
            raise FileNotFoundError(url)
        return mock_s3_path

    mock_download = Mock(side_effect=mock_download_side_effect)
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.http_download_to_disk", mock_download
    )
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.region_job.grib_message_byte_ranges_from_index",
        Mock(return_value=([0], [100])),
    )

    result = region_job.download_file(coord)

    assert result == mock_s3_path
    urls_called = [call.args[0] for call in mock_download.call_args_list]
    # NOMADS is tried first (idx fails), then S3 is used (idx + data)
    assert urls_called[0].startswith("https://nomads.ncep.noaa.gov")
    assert all(
        url.startswith("https://noaa-hrrr-bdp-pds.s3.amazonaws.com")
        for url in urls_called[1:]
    )


@pytest.mark.slow
def test_download_file_from_nomads_hrrr() -> None:
    """Download a recent HRRR init time from NOMADS and read all template variables."""
    template_config = NoaaHrrrForecast48HourTemplateConfig()
    # 6h-old init time is within the 18h NOMADS window and typically complete
    init_time = (pd.Timestamp.now() - pd.Timedelta(hours=6)).floor("h")

    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_config.get_template(pd.Timestamp.now()),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    # lead_time=2h: all vars (instant and accumulated) are present in the f02 GRIB file
    lead_time = pd.Timedelta(hours=2)
    for group in NoaaHrrrForecast48HourRegionJob.source_groups(
        template_config.data_vars
    ):
        file_type = group[0].internal_attrs.hrrr_file_type
        coord = NoaaHrrrForecast48HourSourceFileCoord(
            init_time=init_time,
            lead_time=lead_time,
            domain="conus",
            file_type=file_type,
            data_vars=group,
        )
        coord = replace(coord, downloaded_path=region_job.download_file(coord))

        for data_var in group:
            data = region_job.read_data(coord, data_var)
            assert np.all(np.isfinite(data)), f"Non-finite values for {data_var.name}"
