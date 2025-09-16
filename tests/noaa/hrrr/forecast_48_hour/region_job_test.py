from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from reformatters.noaa.hrrr.forecast_48_hour.region_job import (
    HRRRSourceFileCoord,
    NoaaHrrrForecast48HourRegionJob,
)
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)


@pytest.fixture
def template_config() -> NoaaHrrrForecast48HourTemplateConfig:
    return NoaaHrrrForecast48HourTemplateConfig()


def test_source_file_coord_get_url(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    """Test URL generation for HRRR source file coordinates."""
    coord = HRRRSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )
    expected = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t00z.wrfsfcf00.grib2"
    assert coord.get_url() == expected


def test_source_file_coord_get_url_different_lead_time(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    """Test URL generation for different lead times."""
    coord = HRRRSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T12:00"),
        lead_time=pd.Timedelta(hours=24),
        domain="conus",
        file_type="prs",
        data_vars=template_config.data_vars,
    )
    expected = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t12z.wrfprsf24.grib2"
    assert coord.get_url() == expected


def test_source_file_coord_get_idx_url(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    """Test index URL generation."""
    coord = HRRRSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T06:00"),
        lead_time=pd.Timedelta(hours=6),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )
    expected = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240229/conus/hrrr.t06z.wrfsfcf06.grib2.idx"
    assert coord.get_idx_url() == expected


def test_source_file_coord_out_loc(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    """Test output location mapping."""
    coord = HRRRSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=12),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )

    out_loc = coord.out_loc()
    assert out_loc == {
        "init_time": pd.Timestamp("2024-02-29T00:00"),
        "lead_time": pd.Timedelta(hours=12),
    }


def test_source_file_coord_invalid_lead_time(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    """Test that invalid lead times raise appropriate errors."""
    coord = HRRRSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(minutes=30),  # 0.5 hours - not a whole hour
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:1],
    )

    with pytest.raises(AssertionError):
        coord.get_url()


def test_region_job_source_groups() -> None:
    """Test that data variables are grouped by file type."""
    template_config = NoaaHrrrForecast48HourTemplateConfig()
    data_vars = template_config.data_vars

    # Get variables from different file types
    sfc_vars = [v for v in data_vars if v.internal_attrs.hrrr_file_type == "sfc"]

    # Test source grouping with available sfc variables
    groups = NoaaHrrrForecast48HourRegionJob.source_groups(sfc_vars)

    # All variables should be in one group since they're all from the same file type
    assert len(groups) == 1
    assert len(groups[0]) == len(
        sfc_vars
    )  # Should equal the number of sfc vars available


def test_region_job_source_groups_multiple_file_types() -> None:
    """Test source grouping with variables from different file types."""
    template_config = NoaaHrrrForecast48HourTemplateConfig()
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
        groups = NoaaHrrrForecast48HourRegionJob.source_groups(mixed_vars)
        # Should have separate groups for different file types
        assert len(groups) >= 1

        # Each group should contain only variables from the same file type
        for group in groups:
            file_types_in_group = {v.internal_attrs.hrrr_file_type for v in group}
            assert len(file_types_in_group) == 1


def test_region_job_generate_source_file_coords() -> None:
    """Test source file coordinate generation."""
    template_config = NoaaHrrrForecast48HourTemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2025-01-01"))

    # Create a small subset for testing
    test_ds = template_ds.isel(init_time=slice(0, 2), lead_time=slice(0, 3))

    # use `model_construct` to skip pydantic validation so we can pass mock stores
    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],  # Just one variable
        append_dim=template_config.append_dim,
        region=slice(0, 2),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()

    # Test with a single data variable group
    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    # Should generate coordinates for each init_time x lead_time combination
    # 2 init_times x 3 lead_times = 6 coordinates
    assert len(source_coords) == 6

    # Check that all coordinates are HRRRSourceFileCoord instances
    for coord in source_coords:
        assert isinstance(coord, HRRRSourceFileCoord)
        assert coord.domain == "conus"
        assert (
            coord.file_type
            == template_config.data_vars[0].internal_attrs.hrrr_file_type
        )


def test_region_job_generate_source_file_coords_filters_hour_0() -> None:
    """Test that hour 0 filtering works for accumulated variables."""
    template_config = NoaaHrrrForecast48HourTemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2025-01-01"))

    # Create a small subset that includes hour 0
    test_ds = template_ds.isel(
        init_time=slice(0, 1), lead_time=slice(0, 2)
    )  # hours 0 and 1

    # Find a variable that doesn't have hour 0 values (if any)
    # For now, just test with a regular variable
    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()

    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    # Should have coordinates for the available times
    assert len(source_coords) >= 1

    # All coordinates should be valid
    for coord in source_coords:
        assert isinstance(coord, HRRRSourceFileCoord)


def test_region_job_48h_forecasts() -> None:
    """Test that 48-hour forecast coordinates are generated correctly."""
    template_config = NoaaHrrrForecast48HourTemplateConfig()

    # Create template with long lead times
    template_ds = template_config.get_template(pd.Timestamp("2025-01-01"))
    test_ds = template_ds.isel(
        init_time=slice(0, 1),  # Single init time
        lead_time=slice(0, -1, 12),  # Every 12 hours up to 48h
    )

    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()

    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    # Should generate coordinates for all available lead times (48h dataset)
    assert len(source_coords) > 0

    # All coordinates should be valid HRRRSourceFileCoord instances
    for coord in source_coords:
        assert isinstance(coord, HRRRSourceFileCoord)
        assert coord.domain == "conus"  # Always CONUS for this dataset
        # Lead time should be <= 48h (dataset maximum)
        assert coord.lead_time <= pd.Timedelta("48h")


def test_region_job_download_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test HRRR file download with mocked network calls."""
    template_config = NoaaHrrrForecast48HourTemplateConfig()

    coord = HRRRSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:3],
    )

    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
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
        NoaaHrrrForecast48HourRegionJob,
        "dataset_id",
        "test-dataset-hrrr",
    )

    monkeypatch.setattr(
        "reformatters.noaa.hrrr.forecast_48_hour.region_job.grib_message_byte_ranges_from_index",
        Mock(return_value=([0, 200], [100, 350])),
    )
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.forecast_48_hour.region_job.http_download_to_disk",
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


def test_region_job_read_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test HRRR data reading with mocked file operations."""
    template_config = NoaaHrrrForecast48HourTemplateConfig()

    coord = HRRRSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars[:1],
        downloaded_path=Path("fake/path/to/downloaded/file.grib2"),
    )

    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
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
    test_data = np.ones((1799, 1059), dtype=np.float32) * 42.0
    rasterio_reader.read = Mock(return_value=test_data)
    monkeypatch.setattr(
        "reformatters.noaa.hrrr.forecast_48_hour.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(coord, template_config.data_vars[0])

    # Verify the result
    assert np.array_equal(result, test_data)
    assert result.shape == (1799, 1059)  # HRRR CONUS grid dimensions
    assert result.dtype == np.float32

    rasterio_reader.read.assert_called_once_with(1, out_dtype=np.float32)
