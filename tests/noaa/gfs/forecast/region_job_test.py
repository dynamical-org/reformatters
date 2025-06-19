from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.noaa.gfs.forecast.region_job import (
    NoaaGfsForecastRegionJob,
    NoaaGfsForecastSourceFileCoord,
)
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig


def test_source_file_coord_get_url() -> None:
    coord = NoaaGfsForecastSourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta(hours=0),
        data_vars=NoaaGfsForecastTemplateConfig().data_vars,
    )
    expected = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20000101/00/atmos/gfs.t00z.pgrb2.0p25.f000"
    assert coord.get_url() == expected


def test_region_job_generete_source_file_coords() -> None:
    template_config = NoaaGfsForecastTemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2025-01-01"))

    # use `model_construct` to skip pydantic validation so we can pass mock stores
    region_job = NoaaGfsForecastRegionJob.model_construct(
        final_store=Mock(),
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars[:3],
        append_dim=template_config.append_dim,
        region=slice(0, 10),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()

    source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, region_job.data_vars
    )

    assert isinstance(source_file_coords, list)
    # 10 init_times * 209 lead_times
    assert len(source_file_coords) == 10 * 209
    assert len(source_file_coords) == (
        region_job.region.stop - region_job.region.start
    ) * len(template_config.dimension_coordinates()["lead_time"])

    init_times = {coord.init_time for coord in source_file_coords}
    lead_times = {coord.lead_time for coord in source_file_coords}
    assert len(init_times) == 10
    assert len(lead_times) == 209
    assert set(pd.to_datetime(processing_region_ds["init_time"].values)) == init_times
    assert set(pd.to_timedelta(processing_region_ds["lead_time"].values)) == lead_times

    for coord in source_file_coords:
        assert coord.data_vars == region_job.data_vars
        assert len(coord.data_vars) == 3


def test_region_job_download_file(monkeypatch: pytest.MonkeyPatch) -> None:
    template_config = NoaaGfsForecastTemplateConfig()

    # Create a region job with mock stores
    region_job = NoaaGfsForecastRegionJob.model_construct(
        final_store=Mock(),
        tmp_store=Mock(),
        template_ds=template_config.get_template(pd.Timestamp("2025-01-01")),
        data_vars=template_config.data_vars[:2],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NoaaGfsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01T00:00"),
        lead_time=pd.Timedelta(hours=6),
        data_vars=region_job.data_vars,
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
        "reformatters.noaa.gfs.forecast.region_job.http_download_to_disk",
        mock_download,
    )

    # Mock parse_grib_index to return some byte ranges
    mock_parse = Mock(return_value=([123456, 234567], [234566, 345678]))
    monkeypatch.setattr(
        "reformatters.noaa.gfs.forecast.region_job.parse_grib_index",
        mock_parse,
    )

    result = region_job.download_file(coord)

    # Verify the result
    assert result == mock_data_path

    # Verify http_download_to_disk was called correctly
    assert mock_download.call_count == 2

    # First call should be for the index file
    first_call = mock_download.call_args_list[0]
    assert first_call[0][0].endswith(".idx")
    assert first_call[0][1] == "noaa-gfs-forecast"

    # Second call should be for the data file with byte ranges
    second_call = mock_download.call_args_list[1]
    assert not second_call[0][0].endswith(".idx")
    assert second_call[0][1] == "noaa-gfs-forecast"
    assert second_call[1]["byte_ranges"] == ([123456, 234567], [234566, 345678])
    assert "local_path_suffix" in second_call[1]


def test_operational_update_jobs(monkeypatch):
    template_config = NoaaGfsForecastTemplateConfig()
    # monkeypatch Timestamp.now
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda cls: pd.Timestamp("2025-01-01T12:34"))
    )
    # monkeypatch _get_append_dim_start
    start = pd.Timestamp("2025-01-01T06:00")
    monkeypatch.setattr(
        NoaaGfsForecastRegionJob,
        "_get_append_dim_start",
        classmethod(lambda cls, ds, append_dim: start),
    )
    # monkeypatch xr.open_zarr to return a ds
    final_store = Mock()
    existing_ds = template_config.get_template(start)
    monkeypatch.setattr(
        "reformatters.noaa.gfs.forecast.region_job.xr.open_zarr",
        lambda store: existing_ds,
    )
    jobs, template_ds = NoaaGfsForecastRegionJob.operational_update_jobs(
        final_store=final_store,
        tmp_store=Mock(),
        get_template_fn=template_config.get_template,
        append_dim=template_config.append_dim,
        all_data_vars=template_config.data_vars,
        reformat_job_name="test_job",
    )
    # template_ds should use end time from mocked now
    expected_template_ds = template_config.get_template(
        pd.Timestamp("2025-01-01T12:34")
    )
    import xarray as xr

    xr.testing.assert_equal(template_ds, expected_template_ds)
    # jobs should be RegionJob instances covering all data_vars
    assert jobs
    for job in jobs:
        assert isinstance(job, NoaaGfsForecastRegionJob)
        assert job.data_vars == template_config.data_vars
