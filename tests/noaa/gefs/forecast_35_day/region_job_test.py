import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from obstore.exceptions import PermissionDeniedError

from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.common.pydantic import replace
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
    StoreFactory,
    get_local_tmp_store,
)
from reformatters.noaa.gefs.forecast_35_day.region_job import (
    GefsForecast35DayRegionJob,
    GefsForecast35DaySourceFileCoord,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFSDataVar,
    GefsEnsembleSourceFileCoord,
    GEFSInternalAttrs,
)


@pytest.fixture
def template_ds() -> xr.Dataset:
    """Create a template dataset for testing."""
    num_init_time = 2
    num_lead_time = 8
    num_ensemble = 4
    return xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                data=np.ones(
                    (num_init_time, num_lead_time, num_ensemble, 10, 15),
                    dtype=np.float32,
                ),
                dims=[
                    "init_time",
                    "lead_time",
                    "ensemble_member",
                    "latitude",
                    "longitude",
                ],
                encoding={
                    "dtype": "float32",
                    "chunks": (1, num_lead_time, num_ensemble, 10, 15),
                    "shards": (1, num_lead_time, num_ensemble, 10, 15),
                },
            ),
            "precipitation_surface": xr.Variable(
                data=np.ones(
                    (num_init_time, num_lead_time, num_ensemble, 10, 15),
                    dtype=np.float32,
                ),
                dims=[
                    "init_time",
                    "lead_time",
                    "ensemble_member",
                    "latitude",
                    "longitude",
                ],
                encoding={
                    "dtype": "float32",
                    "chunks": (1, num_lead_time, num_ensemble, 10, 15),
                    "shards": (1, num_lead_time, num_ensemble, 10, 15),
                },
            ),
        },
        coords={
            "init_time": pd.date_range(
                "2000-01-01T00:00", freq="6h", periods=num_init_time
            ),
            "lead_time": pd.timedelta_range("0h", freq="3h", periods=num_lead_time),
            "ensemble_member": np.arange(num_ensemble),
            "latitude": np.linspace(-90, 90, 10),
            "longitude": np.linspace(-180, 179, 15),
        },
        attrs={"dataset_id": "noaa-gefs-forecast-35-day"},
    )


@pytest.fixture
def example_data_vars() -> list[GEFSDataVar]:
    """Create example GEFS data variables for testing."""
    encoding = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=(1, 8, 4, 10, 15),
        shards=(1, 8, 4, 10, 15),
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
    assert GefsForecast35DayRegionJob.max_vars_per_backfill_job == 3


def test_get_processing_region(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test processing region includes proper buffer."""
    tmp_store = get_local_tmp_store()

    job = GefsForecast35DayRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=example_data_vars[:1],  # Single variable
        append_dim="init_time",
        region=slice(1, 2),  # Single init time
        reformat_job_name="test-job",
    )

    processing_region = job.get_processing_region()

    # Should return the same region as no buffering needed for init_time
    assert processing_region == slice(1, 2)


def test_source_groups(example_data_vars: list[GEFSDataVar]) -> None:
    """Test source groups based on GEFS file type and ensemble statistic."""
    groups = GefsForecast35DayRegionJob.source_groups(example_data_vars)

    # Both variables have the same file type, but we expect two groups due to the zero hour values
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert groups[0][0].name == "temperature_2m"
    assert groups[1][0].name == "precipitation_surface"


def test_generate_source_file_coords_ensemble(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test generation of source file coordinates for ensemble data."""
    tmp_store = get_local_tmp_store()

    job = GefsForecast35DayRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=example_data_vars[:1],  # Single variable
        append_dim="init_time",
        region=slice(0, 1),  # Single init time
        reformat_job_name="test-job",
    )

    processing_region_ds = template_ds.isel(init_time=slice(0, 1))
    coords = list(
        job.generate_source_file_coords(processing_region_ds, example_data_vars[:1])
    )

    # Should generate coordinates for each ensemble member and lead time
    # 1 init_time * 8 lead_times * 4 ensemble_members = 32 coords
    assert len(coords) == 32

    # All should be ensemble source file coords
    assert all(isinstance(c, GefsEnsembleSourceFileCoord) for c in coords)


def test_source_file_coord_url_generation(example_data_vars: list[GEFSDataVar]) -> None:
    """Test source file coordinate URL generation."""
    coord = GefsEnsembleSourceFileCoord(
        init_time=pd.Timestamp("2021-01-01T00:00"),  # Use current archive period
        lead_time=pd.Timedelta("3h"),
        ensemble_member=1,
        data_vars=example_data_vars[:1],
    )

    url = coord.get_url()

    assert "gefs.20210101/00" in url
    assert "gep01.t00z" in url
    assert "pgrb2s.0p25.f003" in url


def test_source_file_coord_fallback_url(example_data_vars: list[GEFSDataVar]) -> None:
    """Test fallback URL generation for source file coordinates."""
    coord = GefsEnsembleSourceFileCoord(
        init_time=pd.Timestamp("2021-01-01T00:00"),  # Use current archive period
        lead_time=pd.Timedelta("3h"),
        ensemble_member=1,
        data_vars=example_data_vars[:1],
    )

    primary_url = coord.get_url()
    fallback_url = coord.get_fallback_url()

    assert (
        primary_url
        == "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20210101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003"
    )
    assert (
        fallback_url
        == "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.20210101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003"
    )


def test_source_file_coord_out_loc_forecast(
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test source file coordinate output location for forecast data."""
    coord = GefsEnsembleSourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=1,
        data_vars=example_data_vars,
    )

    out_loc = coord.out_loc()

    # Should map to proper forecast array indices
    assert "init_time" in out_loc
    assert "lead_time" in out_loc
    assert "ensemble_member" in out_loc
    assert out_loc["ensemble_member"] == 1


def test_download_file(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test download_file method."""
    tmp_store = get_local_tmp_store()
    data_vars = example_data_vars[:1]

    job = GefsForecast35DayRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test-job",
    )

    coord = GefsForecast35DaySourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=1,
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
    assert first_call[0][1] == "noaa-gefs-forecast-35-day"

    # Second call should be for the data file with byte ranges
    second_call = mock_download.call_args_list[1]
    assert not second_call[0][0].endswith(".idx")
    assert second_call[0][1] == "noaa-gefs-forecast-35-day"
    assert second_call[1]["byte_ranges"] == ([123456, 234567], [234566, 345678])
    assert "local_path_suffix" in second_call[1]


def test_download_file_fallback(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test download_file falls back to alternative source when primary fails for recent data."""
    tmp_store = get_local_tmp_store()
    data_vars = example_data_vars[:1]

    job = GefsForecast35DayRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test-job",
    )

    # Use a recent init_time (within 4 days) to trigger fallback behavior
    recent_init_time = pd.Timestamp.now() - pd.Timedelta(days=1)
    coord = GefsForecast35DaySourceFileCoord(
        init_time=recent_init_time,
        lead_time=pd.Timedelta("3h"),
        ensemble_member=1,
        data_vars=data_vars,
    )

    mock_index_path = Mock()
    mock_index_path.read_text.return_value = "ignored"
    mock_data_path = Mock()

    fallback_url = coord.get_fallback_url()
    fallback_index_url = coord.get_index_url(fallback=True)

    call_count = 0

    # Primary source (http_download_to_disk) fails with FileNotFoundError
    def mock_primary_download(url: str, dataset_id: str, **kwargs: object) -> Mock:
        nonlocal call_count
        call_count += 1
        raise FileNotFoundError(f"Primary index not found: {url}")

    # Fallback source (httpx_download_to_disk for NOMADS) succeeds
    def mock_fallback_download(url: str, dataset_id: str, **kwargs: object) -> Mock:
        nonlocal call_count
        call_count += 1
        if url == fallback_index_url:
            return mock_index_path
        if url == fallback_url:
            return mock_data_path
        raise AssertionError(f"Unexpected fallback URL: {url}")

    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.http_download_to_disk",
        Mock(side_effect=mock_primary_download),
    )
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.httpx_download_to_disk",
        Mock(side_effect=mock_fallback_download),
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


def test_download_file_no_fallback_for_old_data(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test download_file does not fallback for old data (>4 days)."""
    tmp_store = get_local_tmp_store()
    data_vars = example_data_vars[:1]

    job = GefsForecast35DayRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test-job",
    )

    # Use an old init_time (>4 days ago) - should NOT trigger fallback
    old_init_time = pd.Timestamp("2025-01-01T00:00")
    coord = GefsForecast35DaySourceFileCoord(
        init_time=old_init_time,
        lead_time=pd.Timedelta("3h"),
        ensemble_member=1,
        data_vars=data_vars,
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
    for call in mock_download.call_args_list:
        url = call[0][0]
        # All calls should be to primary source (AWS S3), not fallback (NOMADS)
        assert urlparse(url).netloc == "noaa-gefs-pds.s3.amazonaws.com"


@patch("reformatters.noaa.gefs.forecast_35_day.region_job.read_data")
def test_read_data(
    mock_read_data: MagicMock,
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test read_data method."""
    mock_data = np.ones((10, 15), dtype=np.float32)
    mock_read_data.return_value = mock_data

    tmp_store = get_local_tmp_store()

    job = GefsForecast35DayRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=example_data_vars[:1],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test-job",
    )

    coord = GefsForecast35DaySourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=1,
        data_vars=example_data_vars,
    )
    # Set a mock downloaded path since read_data expects this
    coord = replace(coord, downloaded_path=Path("/mock/path/to/file.grib2"))

    result = job.read_data(coord, example_data_vars[0])

    # The read_data function is already mocked, so this should work
    assert result.dtype == np.float32


def test_apply_data_transformations(template_ds: xr.Dataset) -> None:
    """Test data transformation application."""
    # Create test data with known values
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((1, 1, 1, 4))

    var_with_rounding = GEFSDataVar(
        name="test_var",
        encoding=Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=(1, 1, 1, 4),
            shards=(1, 1, 1, 4),
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

    # Create a DataArray with proper forecast dimensions
    init_time_coords = pd.date_range("2000-01-01T00:00", freq="6h", periods=1)
    lead_time_coords = pd.timedelta_range("0h", freq="3h", periods=1)
    ensemble_coords = [0]
    data_array = xr.DataArray(
        data,
        dims=["init_time", "lead_time", "ensemble_member", "test"],
        coords={
            "init_time": init_time_coords,
            "lead_time": lead_time_coords,
            "ensemble_member": ensemble_coords,
        },
    )

    # Need to create a job instance to call apply_data_transformations
    tmp_store = get_local_tmp_store()
    job = GefsForecast35DayRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=[var_with_rounding],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test-job",
    )

    # Apply transformations (this is an instance method)
    job.apply_data_transformations(data_array, var_with_rounding)


@patch("xarray.open_zarr")
def test_operational_update_jobs(
    mock_open_zarr: MagicMock,
    example_data_vars: list[GEFSDataVar],
) -> None:
    """Test operational_update_jobs method."""
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-gefs-forecast-35-day",
        template_config_version="test-version",
    )
    # Create mock existing dataset
    existing_init_time = pd.date_range("2000-01-01T00:00", freq="24h", periods=5)
    existing_ds = xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                data=np.ones((5, 8, 4, 5, 10), dtype=np.float32),
                dims=[
                    "init_time",
                    "lead_time",
                    "ensemble_member",
                    "latitude",
                    "longitude",
                ],
            )
        },
        coords={
            "init_time": existing_init_time,
            "lead_time": pd.timedelta_range("0h", freq="3h", periods=8),
            "ensemble_member": np.arange(4),
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
                    data=np.ones((10, 8, 4, 5, 10), dtype=np.float32),
                    dims=[
                        "init_time",
                        "lead_time",
                        "ensemble_member",
                        "latitude",
                        "longitude",
                    ],
                )
            },
            coords={
                "init_time": pd.date_range("2000-01-01T00:00", freq="24h", periods=10),
                "lead_time": pd.timedelta_range("0h", freq="3h", periods=8),
                "ensemble_member": np.arange(4),
                "latitude": np.linspace(-90, 90, 5),
                "longitude": np.linspace(-180, 179, 10),
            },
        )

    # Mock get_jobs method
    with patch.object(GefsForecast35DayRegionJob, "get_jobs") as mock_get_jobs:
        mock_jobs = [MagicMock() for _ in range(3)]  # Mock job instances
        mock_get_jobs.return_value = mock_jobs

        # Call operational_update_jobs
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_store_path = Path(tmp_dir)
            jobs, template_ds = GefsForecast35DayRegionJob.operational_update_jobs(
                primary_store=store_factory.primary_store(),
                tmp_store=tmp_store_path,
                get_template_fn=mock_get_template_fn,
                append_dim="init_time",
                all_data_vars=example_data_vars,
                reformat_job_name="test-update",
            )

            # Verify results
            assert jobs == mock_jobs
            assert template_ds is not None
            assert len(template_ds.init_time) == 10

            # Verify get_jobs was called with correct parameters
            mock_get_jobs.assert_called_once()
            call_args = mock_get_jobs.call_args
            assert call_args.kwargs["kind"] == "operational-update"
            assert call_args.kwargs["tmp_store"] == tmp_store_path
            assert call_args.kwargs["append_dim"] == "init_time"
            assert call_args.kwargs["all_data_vars"] == example_data_vars
            assert call_args.kwargs["reformat_job_name"] == "test-update"
            assert call_args.kwargs["filter_start"] == existing_init_time.max()

            # Verify existing dataset was opened correctly
            mock_open_zarr.assert_called_once_with(
                store_factory.primary_store(), decode_timedelta=True, chunks=None
            )


def test_download_file_fallback_permission_denied_converts_to_file_not_found(
    template_ds: xr.Dataset,
    example_data_vars: list[GEFSDataVar],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that PermissionDeniedError from fallback is converted to FileNotFoundError."""
    tmp_store = get_local_tmp_store()
    data_vars = example_data_vars[:1]

    job = GefsForecast35DayRegionJob(
        tmp_store=tmp_store,
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test-job",
    )

    # Use a recent init_time (within 4 days) to trigger fallback behavior
    recent_init_time = pd.Timestamp.now() - pd.Timedelta(days=1)
    coord = GefsForecast35DaySourceFileCoord(
        init_time=recent_init_time,
        lead_time=pd.Timedelta("3h"),
        ensemble_member=1,
        data_vars=data_vars,
    )

    # Primary source fails with FileNotFoundError
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.http_download_to_disk",
        Mock(side_effect=FileNotFoundError("Primary index not found")),
    )
    # Fallback source (NOMADS via httpx) fails with PermissionDeniedError
    monkeypatch.setattr(
        "reformatters.noaa.gefs.utils.httpx_download_to_disk",
        Mock(side_effect=PermissionDeniedError("Permission denied")),
    )

    # Should raise FileNotFoundError (not PermissionDeniedError)
    with pytest.raises(FileNotFoundError) as exc_info:
        job.download_file(coord)

    # Verify it's a FileNotFoundError with PermissionDeniedError as cause
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, PermissionDeniedError)
