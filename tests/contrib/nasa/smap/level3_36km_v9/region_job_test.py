from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from reformatters.common.types import ArrayFloat32
from reformatters.contrib.nasa.smap.level3_36km_v9.region_job import (
    NasaSmapLevel336KmV9RegionJob,
    NasaSmapLevel336KmV9SourceFileCoord,
)
from reformatters.contrib.nasa.smap.level3_36km_v9.template_config import (
    NasaSmapLevel336KmV9TemplateConfig,
)


@pytest.fixture
def mock_smap_am_data() -> ArrayFloat32:
    """Create mock AM soil moisture data with fill values."""
    data = np.random.rand(406, 964).astype(np.float32) * 0.5
    data[0:10, 0:10] = -9999.0
    return data


@pytest.fixture
def mock_smap_pm_data() -> ArrayFloat32:
    """Create mock PM soil moisture data with fill values."""
    data = np.random.rand(406, 964).astype(np.float32) * 0.5
    data[20:30, 20:30] = -9999.0
    return data


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

        # Should have retried 2 times total (1 failure + 1 success)
        assert mock_session.get.call_count == 2
        assert result.exists()
        assert result.read_bytes() == b"success"


def test_read_data_am(tmp_path: Path, mock_smap_am_data: ArrayFloat32) -> None:
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

    coord = NasaSmapLevel336KmV9SourceFileCoord(
        time=pd.Timestamp("2015-04-01"), downloaded_path=tmp_path / "fake.h5"
    )

    # Get the AM data variable
    am_var = template_config.data_vars[0]
    assert am_var.name == "soil_moisture_am"

    # Patch rasterio.open to return mock data
    with patch(
        "reformatters.contrib.nasa.smap.level3_36km_v9.region_job.rasterio.open"
    ) as mock_open:
        # Create mock rasterio dataset that returns the mock data
        mock_dataset = Mock()
        mock_dataset.read.return_value = mock_smap_am_data
        mock_open.return_value.__enter__.return_value = mock_dataset
        mock_open.return_value.__exit__.return_value = None

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


def test_read_data_pm(tmp_path: Path, mock_smap_pm_data: ArrayFloat32) -> None:
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

    coord = NasaSmapLevel336KmV9SourceFileCoord(
        time=pd.Timestamp("2015-04-01"), downloaded_path=tmp_path / "fake.h5"
    )

    # Get the PM data variable
    pm_var = template_config.data_vars[1]
    assert pm_var.name == "soil_moisture_pm"

    # Patch rasterio.open to return mock data
    with patch(
        "reformatters.contrib.nasa.smap.level3_36km_v9.region_job.rasterio.open"
    ) as mock_open:
        # Create mock rasterio dataset that returns the mock data
        mock_dataset = Mock()
        mock_dataset.read.return_value = mock_smap_pm_data
        mock_open.return_value.__enter__.return_value = mock_dataset
        mock_open.return_value.__exit__.return_value = None

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


def test_operational_update_jobs(tmp_path: Path) -> None:
    """Test that operational_update_jobs creates correct jobs for updating dataset."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()

    # Create a mock existing dataset with data up to 2025-09-28
    existing_end = pd.Timestamp("2025-09-28")
    existing_ds = template_config.get_template(existing_end)

    # Mock the primary store to return our existing dataset
    mock_store = Mock()

    with patch("xarray.open_zarr", return_value=existing_ds):
        with patch("pandas.Timestamp.now", return_value=pd.Timestamp("2025-09-30")):
            jobs, template_ds = NasaSmapLevel336KmV9RegionJob.operational_update_jobs(
                primary_store=mock_store,
                tmp_store=tmp_path,
                get_template_fn=template_config.get_template,
                append_dim=template_config.append_dim,
                all_data_vars=template_config.data_vars,
                reformat_job_name="test-update",
            )

    # Should create jobs for the new time steps (2025-09-29 and 2025-09-30)
    assert len(jobs) > 0

    # Template should extend to current time
    assert template_ds["time"].max() >= pd.Timestamp("2025-09-30")

    # Verify jobs have correct configuration
    for job in jobs:
        assert job.tmp_store == tmp_path
        assert job.append_dim == template_config.append_dim
        assert job.reformat_job_name == "test-update"
        assert len(job.data_vars) > 0
