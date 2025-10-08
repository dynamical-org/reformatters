from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pandas as pd
import pytest

from reformatters.contrib.nasa.smap.level3_36km_v9.region_job import (
    NasaSmapLevel336KmV9RegionJob,
    NasaSmapLevel336KmV9SourceFileCoord,
)
from reformatters.contrib.nasa.smap.level3_36km_v9.template_config import (
    NasaSmapLevel336KmV9TemplateConfig,
)


def test_source_file_coord_get_url() -> None:
    coord = NasaSmapLevel336KmV9SourceFileCoord(time=pd.Timestamp("2025-09-30"))
    expected_url = (
        "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected"
        "/SMAP/SPL3SMP/009/2025/09/SMAP_L3_SM_P_20250930_R19240_001.h5"
    )
    assert coord.get_url() == expected_url


def test_source_file_coord_out_loc() -> None:
    coord = NasaSmapLevel336KmV9SourceFileCoord(time=pd.Timestamp("2025-09-30"))
    assert coord.out_loc() == {"time": pd.Timestamp("2025-09-30")}


def test_region_job_generate_source_file_coords(tmp_path: Path) -> None:
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2000-01-23"))

    region_job = NasaSmapLevel336KmV9RegionJob(
        tmp_store=tmp_path,
        template_ds=template_ds,
        data_vars=template_config.data_vars[:2],
        append_dim=template_config.append_dim,
        region=slice(0, 10),
        reformat_job_name="test",
    )

    processing_region_ds, _output_region_ds = region_job._get_region_datasets()

    source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    assert len(source_file_coords) == len(processing_region_ds["time"])
    for i, coord in enumerate(source_file_coords):
        assert coord.time == processing_region_ds["time"].values[i]


def test_download_file_success(tmp_path: Path) -> None:
    """Test successful file download."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2015-04-01"))

    region_job = NasaSmapLevel336KmV9RegionJob(
        tmp_store=tmp_path,
        template_ds=template_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NasaSmapLevel336KmV9SourceFileCoord(time=pd.Timestamp("2015-04-01"))

    # Mock the session and response
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.iter_content = Mock(return_value=[b"test", b"data"])

    mock_session = Mock()
    mock_session.get = Mock(return_value=mock_response)

    with patch(
        "reformatters.contrib.nasa.smap.level3_36km_v9.region_job.get_authenticated_session",
        return_value=mock_session,
    ):
        result = region_job.download_file(coord)

        # Verify the download happened
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert coord.get_url() in str(call_args)

        # Verify file was written
        assert result.exists()
        assert result.read_bytes() == b"testdata"


def test_download_file_retries_on_failure(tmp_path: Path) -> None:
    """Test that download_file retries on failure."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2015-04-01"))

    region_job = NasaSmapLevel336KmV9RegionJob(
        tmp_store=tmp_path,
        template_ds=template_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NasaSmapLevel336KmV9SourceFileCoord(time=pd.Timestamp("2015-04-01"))

    # Mock session that fails once then succeeds
    mock_response_fail = Mock()
    mock_response_fail.raise_for_status = Mock(side_effect=Exception("Network error"))

    mock_response_success = Mock()
    mock_response_success.raise_for_status = Mock()
    mock_response_success.iter_content = Mock(return_value=[b"success"])

    mock_session = Mock()
    mock_session.get = Mock(side_effect=[mock_response_fail, mock_response_success])

    with patch(
        "reformatters.contrib.nasa.smap.level3_36km_v9.region_job.get_authenticated_session",
        return_value=mock_session,
    ):
        result = region_job.download_file(coord)

        # Should have retried 3 times total
        assert mock_session.get.call_count == 3
        assert result.exists()
        assert result.read_bytes() == b"success"


def _create_mock_smap_hdf5(path: Path, y_size: int = 406, x_size: int = 964) -> None:
    """Create a mock SMAP HDF5 file with realistic structure."""
    with h5py.File(path, "w") as f:
        # Create AM data group
        am_group = f.create_group("Soil_Moisture_Retrieval_Data_AM")
        am_data = np.random.rand(y_size, x_size).astype(np.float32) * 0.5
        # Add some fill values
        am_data[0:10, 0:10] = -9999.0
        am_group.create_dataset("soil_moisture", data=am_data)

        # Create PM data group
        pm_group = f.create_group("Soil_Moisture_Retrieval_Data_PM")
        pm_data = np.random.rand(y_size, x_size).astype(np.float32) * 0.5
        # Add some fill values
        pm_data[20:30, 20:30] = -9999.0
        pm_group.create_dataset("soil_moisture_pm", data=pm_data)


def test_read_data_am(tmp_path: Path) -> None:
    """Test reading AM soil moisture data."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2015-04-01"))

    region_job = NasaSmapLevel336KmV9RegionJob(
        tmp_store=tmp_path,
        template_ds=template_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    # Create mock HDF5 file
    mock_file = tmp_path / "test_smap.h5"
    _create_mock_smap_hdf5(mock_file)

    coord = NasaSmapLevel336KmV9SourceFileCoord(time=pd.Timestamp("2015-04-01"))
    coord.downloaded_path = mock_file

    # Get the AM data variable
    am_var = template_config.data_vars[0]
    assert am_var.name == "soil_moisture_am"

    result = region_job.read_data(coord, am_var)

    # Check shape
    assert result.shape == (406, 964)
    assert result.dtype == np.float32

    # Check that fill values were converted to NaN
    assert np.isnan(result[0:10, 0:10]).all()

    # Check that valid data is in expected range
    valid_data = result[~np.isnan(result)]
    assert valid_data.min() >= 0.0
    assert valid_data.max() <= 0.5


def test_read_data_pm(tmp_path: Path) -> None:
    """Test reading PM soil moisture data."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2015-04-01"))

    region_job = NasaSmapLevel336KmV9RegionJob(
        tmp_store=tmp_path,
        template_ds=template_ds,
        data_vars=template_config.data_vars[:2],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    # Create mock HDF5 file
    mock_file = tmp_path / "test_smap.h5"
    _create_mock_smap_hdf5(mock_file)

    coord = NasaSmapLevel336KmV9SourceFileCoord(time=pd.Timestamp("2015-04-01"))
    coord.downloaded_path = mock_file

    # Get the PM data variable
    pm_var = template_config.data_vars[1]
    assert pm_var.name == "soil_moisture_pm"

    result = region_job.read_data(coord, pm_var)

    # Check shape
    assert result.shape == (406, 964)
    assert result.dtype == np.float32

    # Check that fill values were converted to NaN
    assert np.isnan(result[20:30, 20:30]).all()

    # Check that valid data is in expected range
    valid_data = result[~np.isnan(result)]
    assert valid_data.min() >= 0.0
    assert valid_data.max() <= 0.5


def test_read_data_requires_downloaded_path(tmp_path: Path) -> None:
    """Test that read_data raises if file hasn't been downloaded."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2015-04-01"))

    region_job = NasaSmapLevel336KmV9RegionJob(
        tmp_store=tmp_path,
        template_ds=template_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NasaSmapLevel336KmV9SourceFileCoord(time=pd.Timestamp("2015-04-01"))
    # Don't set downloaded_path

    am_var = template_config.data_vars[0]

    with pytest.raises(AssertionError, match="File must be downloaded first"):
        region_job.read_data(coord, am_var)
