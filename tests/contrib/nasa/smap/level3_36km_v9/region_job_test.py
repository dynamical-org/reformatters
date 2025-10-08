from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

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
