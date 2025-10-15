import pandas as pd

from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job import (
    EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord,
)


def test_source_file_coord_get_url() -> None:
    coord = EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta("0h"),
        data_var_group=[],
        ensemble_member=0,
    )
    assert (
        coord.get_url()
        == "https://ecmwf-forecasts.s3.us-east-1.amazonaws.com/20250101/00z/0p25/enfo/20250101000000-0h-enfo-ef.grib2"
    )


# def test_region_job_generete_source_file_coords() -> None:
#     template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
#     template_ds = template_config.get_template(pd.Timestamp("2000-01-23"))

#     region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob(
#         tmp_store=Mock(),
#         template_ds=template_ds,
#         data_vars=[Mock(), Mock()],
#         append_dim=template_config.append_dim,
#         region=slice(0, 10),
#         reformat_job_name="test",
#     )

#     processing_region_ds, output_region_ds = region_job._get_region_datasets()

#     source_file_coords = region_job.generate_source_file_coords(
#         processing_region_ds, [Mock()]
#     )

#     assert len(source_file_coords) == ...
#     assert ...


# def test_region_job_download_file(monkeypatch: pytest.MonkeyPatch) -> None:
#     template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()

#     # Create a region job with mock stores
#     region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct(
#         tmp_store=Mock(),
#         template_ds=template_config.get_template(pd.Timestamp("2025-01-01")),
#         data_vars=template_config.data_vars[:2],
#         append_dim=template_config.append_dim,
#         region=slice(0, 1),
#         reformat_job_name="test",
#     )

#     coord = EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord(
#         init_time=pd.Timestamp("2025-01-01T00:00"),
#         lead_time=pd.Timedelta(hours=6),
#         data_vars=region_job.data_vars,
#     )

#     # Mock the http_download_to_disk function to avoid actual network calls
#     mock_download = Mock()
#     mock_index_path = Mock()
#     mock_index_path.read_text.return_value = "ignored"
#     mock_data_path = Mock()

#     # Configure the mock to return different paths for index and data files
#     def mock_download_side_effect(url: str, dataset_id: str, **kwargs: object) -> Mock:
#         if url.endswith(".idx"):
#             return mock_index_path
#         else:
#             return mock_data_path

#     mock_download.side_effect = mock_download_side_effect
#     monkeypatch.setattr(
#         "reformatters.noaa.gfs.forecast.region_job.http_download_to_disk",
#         mock_download,
#     )

#     # Mock parse_grib_index to return some byte ranges
#     mock_parse = Mock(return_value=([123456, 234567], [234566, 345678]))
#     monkeypatch.setattr(
#         "reformatters.noaa.gfs.forecast.region_job.grib_message_byte_ranges_from_index",
#         mock_parse,
#     )

#     result = region_job.download_file(coord)

#     # Verify the result
#     assert result == mock_data_path

#     # Verify http_download_to_disk was called correctly
#     assert mock_download.call_count == 2

#     # First call should be for the index file
#     first_call = mock_download.call_args_list[0]
#     assert first_call[0][0].endswith(".idx")
#     assert first_call[0][1] == "noaa-gfs-forecast"

#     # Second call should be for the data file with byte ranges
#     second_call = mock_download.call_args_list[1]
#     assert not second_call[0][0].endswith(".idx")
#     assert second_call[0][1] == "noaa-gfs-forecast"
#     assert second_call[1]["byte_ranges"] == ([123456, 234567], [234566, 345678])
#     assert "local_path_suffix" in second_call[1]
