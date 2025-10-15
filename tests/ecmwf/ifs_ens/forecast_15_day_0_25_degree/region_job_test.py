# from unittest.mock import Mock

from io import StringIO
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job import (
    # EcmwfIfsEnsForecast15Day025DegreeRegionJob,
    EcmwfIfsEnsForecast15Day025DegreeRegionJob,
    EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)

# from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
#     EcmwfIfsEnsDataVar,
#     EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
# )


def test_source_file_coord_get_url() -> None:
    coord = EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta("0h"),
        data_var_group=[],
        ensemble_member=0,
    )
    assert (
        coord.get_url()
        == "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/20250101/00z/ifs/0p25/enfo/20250101000000-0h-enfo-ef.grib2"
    )

    coord = EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord(
        init_time=pd.Timestamp("2024-02-28"),
        lead_time=pd.Timedelta("0h"),
        data_var_group=[],
        ensemble_member=0,
    )
    assert (
        coord.get_url()
        == "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/20240228/00z/0p25/enfo/20240228000000-0h-enfo-ef.grib2"
    )


def test_region_job_generate_source_file_coords() -> None:
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2024-02-03"))

    region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    processing_region_ds, _ = region_job._get_region_datasets()
    source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars
    )

    # Since our region is just a single init time (slice(0, 1)), we should get 51 * 85 * 1 = 4335 source file coords
    assert len(source_file_coords) == 51 * 85 * 1


def test_region_job_download_file(monkeypatch: pytest.MonkeyPatch) -> None:
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2024-02-02"))

    region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    source_file_coord = EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord(
        init_time=pd.Timestamp("2024-02-01"),
        lead_time=pd.Timedelta("3h"),
        data_var_group=[
            var for var in template_config.data_vars if var.name == "temperature_2m"
        ],
        ensemble_member=0,
    )

    example_grib_index = """
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "cf", "stream": "enfo", "step": "3", "levtype": "sfc", "param": "2t", "_offset": 0, "_length": 665525}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "cf", "stream": "enfo", "step": "3", "levtype": "pl", "levelist": "850", "param": "2t", "_offset": 665525, "_length": 100000}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "cf", "stream": "enfo", "step": "3", "levtype": "sfc", "param": "10u", "_offset": 765525, "_length": 888917}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "1", "param": "2t", "_offset": 1554442, "_length": 664922}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "pl", "levelist": "850", "number": "1", "param": "2t", "_offset": 2219364, "_length": 664716}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "1", "param": "10u", "_offset": 2884080, "_length": 889546}
"""
    mock_index_df = pd.read_json(StringIO(example_grib_index), lines=True)
    monkeypatch.setattr(
        "pandas.read_json",
        lambda path, **kwargs: mock_index_df,
    )

    download_to_disk_mock = Mock()
    monkeypatch.setattr(
        "reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(source_file_coord)
    url, dataset_id = download_to_disk_mock.call_args[0]
    kwargs = download_to_disk_mock.call_args[1]
    byte_ranges = kwargs["byte_ranges"]
    local_path_suffix = kwargs["local_path_suffix"]

    assert (
        url
        == "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/20240201/00z/0p25/enfo/20240201000000-3h-enfo-ef.grib2"
    )
    assert dataset_id == "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
    assert (
        local_path_suffix == "-4f434771"
    )  # result of calling digest on the byte ranges
    assert byte_ranges == ([0], [665525])
