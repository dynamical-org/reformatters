import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.common.pydantic import replace
from reformatters.common.region_job import SourceFileStatus
from reformatters.common.retry import retry
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
    StoreFactory,
    get_local_tmp_store,
)
from reformatters.noaa.gefs.analysis.region_job import (
    GefsAnalysisRegionJob,
    GefsAnalysisSourceFileCoord,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFSDataVar,
    GEFSInternalAttrs,
)


@pytest.fixture
def template_ds() -> xr.Dataset:
    """Create a template dataset for testing."""
    num_time = 24  # 1 day of 3-hourly data
    return xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                data=np.ones((num_time, 10, 15), dtype=np.float32),
                dims=["time", "latitude", "longitude"],
                encoding={
                    "dtype": "float32",
                    "chunks": (num_time // 2, 10, 15),
                    "shards": (num_time, 10, 15),
                },
            ),
            "precipitation_surface": xr.Variable(
                data=np.ones((num_time, 10, 15), dtype=np.float32),
                dims=["time", "latitude", "longitude"],
                encoding={
                    "dtype": "float32",
                    "chunks": (num_time // 2, 10, 15),
                    "shards": (num_time, 10, 15),
                },
            ),
        },
        coords={
            "time": pd.date_range("2000-01-01T00:00", freq="3h", periods=num_time),
            "latitude": np.linspace(-90, 90, 10),
            "longitude": np.linspace(-180, 179, 15),
        },
        attrs={"dataset_id": "noaa-gefs-analysis"},
    )


@pytest.fixture
def example_data_vars() -> list[GEFSDataVar]:
    """Create example GEFS data variables for testing."""
    encoding = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=(12, 10, 15),
        shards=(24, 10, 15),
    )

    return [
        GEFSDataVar(
            name="temperature_2m",
            encoding=encoding,
            attrs=DataVarAttrs(
                long_name="2 metre temperature",
                short_name="t2m",
                units="C",
                step_type="instant",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="TMP",
                grib_description='2[m] HTGL="Specified height level above ground"',
                grib_index_level="2 m above ground",
                gefs_file_type="s+a",
                index_position=10,
                keep_mantissa_bits=10,
            ),
        ),
        GEFSDataVar(
            name="precipitation_surface",
            encoding=encoding,
            attrs=DataVarAttrs(
                long_name="Total precipitation",
                short_name="tp",
                units="mm",
                step_type="accum",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="APCP",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+a",
                index_position=15,
                keep_mantissa_bits=10,
                window_reset_frequency=pd.Timedelta("6h"),
            ),
        ),
    ]


def test_max_vars_per_backfill_job() -> None:
    """Test max_vars_per_backfill_job is correctly set."""
    assert GefsAnalysisRegionJob.max_vars_per_backfill_job == 1


def test_get_processing_region(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test processing region includes proper buffer."""
    tmp_store = get_local_tmp_store()

    job = GefsAnalysisRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=example_data_vars[:1],  # Single variable
        append_dim="time",
        region=slice(4, 16),  # Middle region
        reformat_job_name="test-job",
    )

    processing_region = job.get_processing_region()

    # Should have 2-step buffer on each side
    assert processing_region.start == 2  # max(0, 4-2) = 2
    assert processing_region.stop == 18  # min(24, 16+2) = 18


def test_get_processing_region_at_boundaries(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test processing region handles boundaries correctly."""
    tmp_store = get_local_tmp_store()

    # Test at start of dataset
    job_start = GefsAnalysisRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=example_data_vars[:1],
        append_dim="time",
        region=slice(0, 8),  # Start region
        reformat_job_name="test-job",
    )
    processing_region = job_start.get_processing_region()
    assert processing_region.start == 0  # Can't go below 0
    assert processing_region.stop == 10  # 8 + 2

    # Test at end of dataset
    job_end = GefsAnalysisRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=example_data_vars[:1],
        append_dim="time",
        region=slice(16, 24),  # End region
        reformat_job_name="test-job",
    )
    processing_region = job_end.get_processing_region()
    assert processing_region.start == 14  # 16 - 2
    assert processing_region.stop == 26  # dataset length + chunk size


def test_source_groups(example_data_vars: list[GEFSDataVar]) -> None:
    """Test variable grouping by file type and hour 0 values."""
    encoding = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=(12, 10, 15),
        shards=(24, 10, 15),
    )

    # Add more variables to test grouping
    all_vars = [
        *example_data_vars,
        GEFSDataVar(
            name="wind_u_10m",
            encoding=encoding,
            attrs=DataVarAttrs(
                long_name="10 metre U wind component",
                short_name="u10",
                units="m s-1",
                step_type="instant",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="UGRD",
                grib_description='10[m] HTGL="Specified height level above ground"',
                grib_index_level="10 m above ground",
                gefs_file_type="s+a",
                index_position=12,
                keep_mantissa_bits=10,
            ),
        ),
        GEFSDataVar(
            name="geopotential_height_500",
            encoding=encoding,
            attrs=DataVarAttrs(
                long_name="Geopotential height",
                short_name="gh",
                units="gpm",
                step_type="instant",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="HGT",
                grib_description='500[mb] ISOBARIC="Isobaric surface"',
                grib_index_level="500 mb",
                gefs_file_type="a",
                index_position=8,
                keep_mantissa_bits=10,
            ),
        ),
    ]

    groups = GefsAnalysisRegionJob.source_groups(all_vars)

    # We're grouping everything together since max_vars_per_download_group is 1
    assert len(groups) == 1


def test_generate_source_file_coords_ensemble(
    template_ds: xr.Dataset,
) -> None:
    """Test source file coordinate generation for ensemble data."""
    tmp_store = get_local_tmp_store()

    # Use a variable that has ensemble data (control member only for analysis)
    var = GEFSDataVar(
        name="temperature_2m",
        encoding=Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=(12, 10, 15),
            shards=(24, 10, 15),
        ),
        attrs=DataVarAttrs(
            long_name="2 metre temperature",
            short_name="t2m",
            units="C",
            step_type="instant",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="TMP",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            gefs_file_type="s+a",
            index_position=10,
            keep_mantissa_bits=10,
        ),
    )

    job = GefsAnalysisRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=[var],
        append_dim="time",
        region=slice(0, 8),
        reformat_job_name="test-job",
    )

    # Create processing region dataset
    processing_region_ds = template_ds.isel(time=slice(0, 10))  # With buffer

    coords = job.generate_source_file_coords(processing_region_ds, [var])

    # Should generate coordinates for control member (ensemble_member=0) only
    assert len(coords) > 0
    for coord in coords:
        assert isinstance(coord, GefsAnalysisSourceFileCoord)
        assert coord.ensemble_member == 0  # Control member for analysis


def test_source_file_coord_url_generation(example_data_vars: list[GEFSDataVar]) -> None:
    """Test URL generation for source file coordinates."""
    coord = GefsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2021-01-01T00:00"),  # Use current archive period
        lead_time=pd.Timedelta("6h"),
        data_vars=example_data_vars[:1],
    )

    url = coord.get_url()

    assert "gefs.20210101" in url
    assert "00/atmos/pgrb2sp25" in url
    assert "gec00" in url  # Control member
    assert "f006" in url  # 6-hour forecast


def test_source_file_coord_out(
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test output location mapping for analysis coordinates."""
    coord = GefsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2020-01-01T00:00"),
        lead_time=pd.Timedelta("6h"),
        data_vars=example_data_vars,
    )

    out_loc = coord.out_loc()

    # Should map init_time + lead_time to time coordinate
    assert out_loc == {
        "time": pd.Timestamp("2020-01-01T06:00"),
    }


def test_source_file_coord_append_dim_coord(
    example_data_vars: list[GEFSDataVar],
) -> None:
    coord = GefsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2021-01-01T00:00"),  # Use current archive period
        lead_time=pd.Timedelta("6h"),
        data_vars=example_data_vars[:1],
    )
    assert coord.append_dim_coord == pd.Timestamp("2021-01-01T06:00")


def test_download_file(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test download file method."""
    tmp_store = get_local_tmp_store()
    data_vars = example_data_vars[:1]

    job = GefsAnalysisRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="time",
        region=slice(0, 8),
        reformat_job_name="test-job",
    )

    coord = GefsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2020-01-01T00:00"),
        lead_time=pd.Timedelta("6h"),
        data_vars=data_vars,
    )

    # Mock the http_download_to_disk function to avoid actual network calls
    mock_download = Mock()
    mock_index_path = Mock()
    mock_index_path.read_text.return_value = "ignored"
    mock_data_path = Mock()

    # Configure the mock to return different paths for index and data files
    def mock_download_side_effect(url: str, dataset_id: str, **kwargs: object) -> Mock:
        if url.endswith(".idx"):
            return mock_index_path
        else:
            return mock_data_path

    mock_download.side_effect = mock_download_side_effect
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.http_download_to_disk",
        mock_download,
    )

    # Mock grib_message_byte_ranges_from_index to return some byte ranges
    mock_grib_message_byte_ranges_from_index = Mock(
        return_value=([123456, 234567], [234566, 345678])
    )
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.grib_message_byte_ranges_from_index",
        mock_grib_message_byte_ranges_from_index,
    )

    result = job.download_file(coord)

    # Verify the result
    assert result == mock_data_path

    # Verify http_download_to_disk was called correctly
    assert mock_download.call_count == 2

    # First call should be for the index file
    first_call = mock_download.call_args_list[0]
    assert first_call[0][0].endswith(".idx")
    assert first_call[0][1] == "noaa-gefs-analysis"

    # Second call should be for the data file with byte ranges
    second_call = mock_download.call_args_list[1]
    assert not second_call[0][0].endswith(".idx")
    assert second_call[0][1] == "noaa-gefs-analysis"
    assert second_call[1]["byte_ranges"] == ([123456, 234567], [234566, 345678])
    assert "local_path_suffix" in second_call[1]


@patch("reformatters.noaa.gefs.analysis.region_job.read_data")
def test_read_data(
    mock_read_data: MagicMock,
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test read data method."""
    mock_data = np.ones((10, 15), dtype=np.float32)
    mock_read_data.return_value = mock_data

    tmp_store = get_local_tmp_store()

    job = GefsAnalysisRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=example_data_vars[:1],
        append_dim="time",
        region=slice(0, 8),
        reformat_job_name="test-job",
    )

    coord = GefsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2020-01-01T00:00"),
        lead_time=pd.Timedelta("6h"),
        data_vars=example_data_vars,
    )
    # Set a mock downloaded path since read_data expects this
    coord = replace(coord, downloaded_path=Path("/mock/path/to/file.grib2"))

    data_var = example_data_vars[0]
    result = job.read_data(coord, data_var)

    # The method should handle the coordinate properly and attempt to read data
    # The read_data function is already mocked, so this should work
    assert result.dtype == np.float32


def test_apply_data_transformations(template_ds: xr.Dataset) -> None:
    """Test data transformation application."""
    # Create test data with known values
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    var_with_rounding = GEFSDataVar(
        name="test_var",
        encoding=Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=(4,),
            shards=(4,),
        ),
        attrs=DataVarAttrs(
            long_name="Test variable",
            short_name="test",
            units="test",
            step_type="instant",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="TEST",
            grib_description="Test variable",
            grib_index_level="surface",
            gefs_file_type="s+a",
            index_position=1,
            keep_mantissa_bits=10,
        ),
    )

    # Create a DataArray with proper time coordinates as expected by the method
    time_coords = pd.date_range("2000-01-01T00:00", freq="3h", periods=4)
    data_array = xr.DataArray(data, dims=["time"], coords={"time": time_coords})

    # Need to create a job instance to call apply_data_transformations
    tmp_store = get_local_tmp_store()
    job = GefsAnalysisRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=[var_with_rounding],
        append_dim="time",
        region=slice(0, 4),
        reformat_job_name="test-job",
    )

    # Apply transformations (this is an instance method)
    job.apply_data_transformations(data_array, var_with_rounding)

    # Data should be modified in place (exact changes depend on rounding)
    assert data_array.dtype == np.float32
    # The binary rounding should have been applied


@patch("xarray.open_zarr")
def test_operational_update_jobs(
    mock_open_zarr: MagicMock,
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test operational_update_jobs method."""
    # Create mock existing dataset
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-gefs-analysis",
        template_config_version="test-version",
    )
    existing_time = pd.date_range("2000-01-01T00:00", freq="3h", periods=10)
    existing_ds = xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                data=np.ones((10, 5, 10), dtype=np.float32),
                dims=["time", "latitude", "longitude"],
            )
        },
        coords={
            "time": existing_time,
            "latitude": np.linspace(-90, 90, 5),
            "longitude": np.linspace(-180, 179, 10),
        },
    )
    mock_open_zarr.return_value = existing_ds

    # Mock get_template_fn
    def mock_get_template_fn(
        end_time: pd.Timestamp | np.datetime64 | datetime | str,
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "temperature_2m": xr.Variable(
                    data=np.ones((20, 5, 10), dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                )
            },
            coords={
                "time": pd.date_range("2000-01-01T00:00", freq="3h", periods=20),
                "latitude": np.linspace(-90, 90, 5),
                "longitude": np.linspace(-180, 179, 10),
            },
        )

    # Mock get_jobs method
    with patch.object(GefsAnalysisRegionJob, "get_jobs") as mock_get_jobs:
        mock_jobs = [MagicMock() for _ in range(2)]  # Mock job instances
        mock_get_jobs.return_value = mock_jobs

        # Call operational_update_jobs
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_store_path = Path(tmp_dir)
            jobs, template_ds = GefsAnalysisRegionJob.operational_update_jobs(
                primary_store=store_factory.primary_store(),
                tmp_store=tmp_store_path,
                get_template_fn=mock_get_template_fn,
                append_dim="time",
                all_data_vars=example_data_vars,
                reformat_job_name="test-update",
            )

            # Verify results
            assert jobs == mock_jobs
            assert template_ds is not None
            assert len(template_ds.time) == 20

            # Verify get_jobs was called with correct parameters
            mock_get_jobs.assert_called_once()
            call_args = mock_get_jobs.call_args
            assert call_args.kwargs["kind"] == "operational-update"
            assert call_args.kwargs["tmp_store"] == tmp_store_path
            assert call_args.kwargs["append_dim"] == "time"
            assert call_args.kwargs["all_data_vars"] == example_data_vars
            assert call_args.kwargs["reformat_job_name"] == "test-update"
            assert call_args.kwargs["filter_start"] == existing_time.max()

            # Verify existing dataset was opened correctly
            mock_open_zarr.assert_called_once_with(
                store_factory.primary_store(), decode_timedelta=True, chunks=None
            )


def test_update_template_with_results(
    template_ds: xr.Dataset, example_data_vars: list[GEFSDataVar]
) -> None:
    data_vars = example_data_vars[:1]
    job = GefsAnalysisRegionJob(
        tmp_store=get_local_tmp_store(),
        template_ds=template_ds,
        data_vars=example_data_vars,
        append_dim="time",
        region=slice(0, 8),
        reformat_job_name="test-job",
    )
    coord = GefsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2000-01-03T21:00"),
        lead_time=pd.Timedelta("6h"),
        data_vars=data_vars,
        status=SourceFileStatus.Succeeded,
    )
    process_results = {
        "temperature_2m": [coord],
    }
    updated_template = job.update_template_with_results(process_results)
    assert updated_template.time.max() == pd.Timestamp("2000-01-03T18:00")


def test_download_file_fallback(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test download_file falls back to alternative source when primary fails for recent data."""
    tmp_store = get_local_tmp_store()
    data_vars = example_data_vars[:1]

    job = GefsAnalysisRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test-job",
    )

    # Use a recent init_time (within 4 days) to trigger fallback behavior
    recent_init_time = pd.Timestamp.now() - pd.Timedelta(days=1)
    coord = GefsAnalysisSourceFileCoord(
        init_time=recent_init_time,
        lead_time=pd.Timedelta("6h"),
        data_vars=data_vars,
    )

    mock_index_path = Mock()
    mock_index_path.read_text.return_value = "ignored"
    mock_data_path = Mock()

    primary_index_url = coord.get_index_url()
    fallback_url = coord.get_fallback_url()
    fallback_index_url = coord.get_index_url(fallback=True)

    call_count = 0

    def mock_download_side_effect(url: str, dataset_id: str, **kwargs: object) -> Mock:
        nonlocal call_count
        call_count += 1
        # Primary source fails with FileNotFoundError
        if url == primary_index_url:
            raise FileNotFoundError(f"Primary index not found: {url}")
        # Fallback source succeeds
        if url == fallback_index_url:
            return mock_index_path
        if url == fallback_url:
            return mock_data_path
        raise AssertionError(f"Unexpected URL: {url}")

    mock_download = Mock(side_effect=mock_download_side_effect)
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.http_download_to_disk",
        mock_download,
    )

    original_retry = retry
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.retry",
        lambda func, max_attempts=1: original_retry(func, max_attempts=max_attempts),
    )

    mock_grib_message_byte_ranges_from_index = Mock(
        return_value=([123456, 234567], [234566, 345678])
    )
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.grib_message_byte_ranges_from_index",
        mock_grib_message_byte_ranges_from_index,
    )

    result = job.download_file(coord)

    assert result == mock_data_path

    # Should have called: primary index (failed) + fallback index + fallback data
    assert call_count == 3

    # Verify fallback URLs were used
    calls = mock_download.call_args_list
    assert calls[0][0][0] == primary_index_url  # First tried primary
    assert calls[1][0][0] == fallback_index_url  # Then fallback index
    assert calls[2][0][0] == fallback_url  # Then fallback data


def test_download_file_no_fallback_for_old_data(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test download_file does not fallback for old data (>4 days)."""
    tmp_store = get_local_tmp_store()
    data_vars = example_data_vars[:1]

    job = GefsAnalysisRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test-job",
    )

    # Use an old init_time (>4 days ago) - should NOT trigger fallback
    old_init_time = pd.Timestamp("2000-01-01T00:00")
    coord = GefsAnalysisSourceFileCoord(
        init_time=old_init_time,
        lead_time=pd.Timedelta("6h"),
        data_vars=data_vars,
    )

    original_retry = retry
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.retry",
        lambda func, max_attempts=1: original_retry(func, max_attempts=max_attempts),
    )

    def mock_download_side_effect(url: str, dataset_id: str, **kwargs: object) -> Mock:
        raise FileNotFoundError(f"Not found: {url}")

    mock_download = Mock(side_effect=mock_download_side_effect)
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.http_download_to_disk",
        mock_download,
    )

    # Should raise FileNotFoundError without attempting fallback
    with pytest.raises(FileNotFoundError):
        job.download_file(coord)

    # Should have only tried primary source
    # The retry logic will try the primary source multiple times before giving up
    for call in mock_download.call_args_list:
        url = call[0][0]
        # All calls should be to primary source (AWS S3), not fallback (NOMADS)
        assert urlparse(url).netloc == "noaa-gefs-retrospective.s3.amazonaws.com"
