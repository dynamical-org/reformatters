from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.common.iterating import item
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job import (
    EcmwfIfsEnsForecast15Day025DegreeRegionJob,
    EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
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


def test_region_job_source_groups() -> None:
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    groups = EcmwfIfsEnsForecast15Day025DegreeRegionJob.source_groups(
        template_config.data_vars
    )
    assert len(groups) == 2
    assert len(groups[0]) == 10

    # categorical_precipitation_type_surface is grouped separately
    # since it is the only one with a date_available value
    assert item(groups[1]).name == "categorical_precipitation_type_surface"


def test_region_job_generate_source_file_coords() -> None:
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2024-11-15"))

    region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds.isel(
            init_time=slice(-3, None)
        ),  # Slice so dataset represents the last 3 init times (Nov 12, 13, 14)
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 3),
        reformat_job_name="test",
    )
    processing_region_ds, _ = region_job._get_region_datasets()
    groups = EcmwfIfsEnsForecast15Day025DegreeRegionJob.source_groups(
        template_config.data_vars
    )
    # We are grouping by date_available, so we should get 2 groups
    # One for categorical_precipitation_type_surface (which is the only one with a date_available val)
    # and one for the rest
    group_0_source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, groups[0]
    )
    # Since our region is three init times (slice(0, 3)), we should get 51 * 85 * 3 = 13005 source file coords
    assert len(group_0_source_file_coords) == 51 * 85 * 3

    group_1_source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, groups[1]
    )
    assert len(group_1_source_file_coords[0].data_var_group) == 1
    assert (
        item(group_1_source_file_coords[0].data_var_group).name
        == "categorical_precipitation_type_surface"
    )


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
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "cf", "stream": "enfo", "step": "3", "levtype": "sfc", "param": "10u", "_offset": 3773626, "_length": 665525}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "cf", "stream": "enfo", "step": "3", "levtype": "sfc", "param": "10u", "_offset": 665525, "_length": 888917}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "1", "param": "2t", "_offset": 1554442, "_length": 664922}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "2", "param": "2t", "_offset": 2219364, "_length": 664716}
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


def test_region_job_read_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test grib data reading with mocked file operations."""
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2024-04-02"))

    region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    t2m_var = item(
        var for var in template_config.data_vars if var.name == "temperature_2m"
    )
    source_file_coord = EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord(
        init_time=pd.Timestamp("2024-04-01"),
        lead_time=pd.Timedelta("3h"),
        data_var_group=[t2m_var],
        ensemble_member=0,
        downloaded_path=Path("fake/path/to/downloaded/file.grib2"),
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 1
    rasterio_reader.descriptions = ['2[m] HTGL="Specified height level above ground"']
    rasterio_reader.tags = Mock(
        return_value={"GRIB_ELEMENT": "TMP", "GRIB_COMMENT": "Temperature [C]"}
    )
    test_data = np.ones((721, 1440), dtype=np.float32)
    rasterio_reader.read = Mock(return_value=test_data)
    monkeypatch.setattr(
        "reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(source_file_coord, t2m_var)

    # Verify the result
    assert np.array_equal(result, test_data)
    assert result.shape == (721, 1440)
    assert result.dtype == np.float32

    rasterio_reader.read.assert_called_once_with(1, out_dtype=np.float32)


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset-ecmwf",
        template_config_version="test-version",
    )

    # Set the append_dim_end for the update
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2024-05-02T15:48")),
    )
    # Set the append_dim_start for the update
    # Use a template_ds as a lightweight way to create a mock dataset with a known max append dim coordinate
    existing_ds = template_config.get_template(
        pd.Timestamp("2024-05-01T00:01")  # 00z will be max existing init time
    )
    template_utils.write_metadata(existing_ds, store_factory)

    jobs, template_ds = (
        EcmwfIfsEnsForecast15Day025DegreeRegionJob.operational_update_jobs(
            primary_store=store_factory.primary_store(),
            tmp_store=tmp_path / "tmp_ds.zarr",
            get_template_fn=template_config.get_template,
            append_dim=template_config.append_dim,
            all_data_vars=template_config.data_vars,
            reformat_job_name="test_job",
        )
    )

    assert template_ds.init_time.max() == pd.Timestamp("2024-05-02T00:00")
    assert (
        len(jobs) == 2
    )  # We reprocess the last forecast for March 1st and also process for March 2nd
    for job in jobs:
        assert isinstance(job, EcmwfIfsEnsForecast15Day025DegreeRegionJob)
        assert job.data_vars == template_config.data_vars
