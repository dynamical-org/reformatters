from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.hrrr.region_job import (
    NoaaHrrrRegionJob,
    NoaaHrrrSourceFileCoord,
)
from reformatters.noaa.hrrr.template_config import NoaaHrrrTemplateConfig
from reformatters.noaa.noaa_utils import has_hour_0_values


@pytest.fixture
def template_config() -> NoaaHrrrTemplateConfig:
    # Use the Forecast 48 Hour template because we need
    # a concrete implementation to get data_vars
    return NoaaHrrrForecast48HourTemplateConfig()


def test_source_file_coord_get_url(
    template_config: NoaaHrrrTemplateConfig,
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
    template_config: NoaaHrrrTemplateConfig,
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
    template_config: NoaaHrrrTemplateConfig,
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
    template_config: NoaaHrrrTemplateConfig,
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
    template_config: NoaaHrrrTemplateConfig,
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
    template_config: NoaaHrrrTemplateConfig,
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
    template_config: NoaaHrrrTemplateConfig,
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
    template_config: NoaaHrrrTemplateConfig,
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
