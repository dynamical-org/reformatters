from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.common.iterating import item
from reformatters.common.pydantic import replace
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job import (
    EcmwfIfsEnsForecast15Day025DegreeRegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    MarsSourceFileCoord,
    OpenDataSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)


def test_region_job_source_groups() -> None:
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    groups = EcmwfIfsEnsForecast15Day025DegreeRegionJob.source_groups(
        template_config.data_vars
    )
    assert len(groups) == 4
    # Main group: vars with no date_available (available since dataset start), all with hour 0 values
    assert len(groups[0]) == 16
    # categorical_precipitation_type_surface is instant (has hour 0) and available from 2024-11-13
    assert item(groups[1]).name == "categorical_precipitation_type_surface"
    # wind_gust_10m is max-window (no hour 0) and available from 2024-11-13
    assert item(groups[2]).name == "wind_gust_10m"
    # total_cloud_cover_atmosphere is available from 2025-11-21
    assert item(groups[3]).name == "total_cloud_cover_atmosphere"


def test_region_job_generate_source_file_coords_open_data() -> None:
    """Open data init times (>= cutover) produce one OpenDataSourceFileCoord per lead_time."""
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2024-11-15"))

    region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct(
        tmp_store=Mock(),
        # Slice so dataset represents the last 3 init times (Nov 12, 13, 14), 2 ensemble members, and 12 lead times
        template_ds=template_ds.isel(
            init_time=slice(-3, None),
            ensemble_member=slice(0, 2),
            lead_time=slice(0, 12),
        ),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 3),
        reformat_job_name="test",
    )
    processing_region_ds, _ = region_job._get_region_datasets()
    groups = EcmwfIfsEnsForecast15Day025DegreeRegionJob.source_groups(
        template_config.data_vars
    )
    # We are grouping by date_available and has_hour_0_values, so we should get 4 groups
    group_0_source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, groups[0]
    )
    # Since our region is three init times (slice(0, 3)), 2 ensemble members, and 12 lead times,
    # we should get 3 * 2 * 12 = 72 source file coords
    assert len(group_0_source_file_coords) == 3 * 2 * 12
    assert all(
        isinstance(c, OpenDataSourceFileCoord) for c in group_0_source_file_coords
    )

    group_1_source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, groups[1]
    )
    # group 1 has categorical_precipitation_type_surface (instant, has hour 0) available from
    # 2024-11-13. Nov 12 is skipped, so 2 init times x 2 members x 12 lead times = 48.
    assert len(group_1_source_file_coords) == 2 * 2 * 12
    assert item(group_1_source_file_coords[0].data_var_group).name == (
        "categorical_precipitation_type_surface"
    )

    group_2_source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, groups[2]
    )
    # group 2 has wind_gust_10m (max-window, no hour 0) available from 2024-11-13.
    # Nov 12 is skipped, and lead_time=0h is excluded, so 2 * 2 * 11 = 44.
    assert len(group_2_source_file_coords) == 2 * 2 * 11
    assert item(group_2_source_file_coords[0].data_var_group).name == "wind_gust_10m"


def test_region_job_generate_source_file_coords_mars() -> None:
    """MARS init times (< cutover) produce MarsSourceFileCoords, one per lead_time like open data."""
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    # Override append_dim_start to allow pre-cutover dates
    object.__setattr__(template_config, "append_dim_start", pd.Timestamp("2024-01-01"))
    template_ds = template_config.get_template(pd.Timestamp("2024-01-03"))

    region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds.isel(
            init_time=slice(0, 2),
            ensemble_member=slice(0, 2),
            lead_time=slice(0, 5),
        ),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 2),
        reformat_job_name="test",
    )
    processing_region_ds, _ = region_job._get_region_datasets()

    # Use a single-var group (as process() does with max_vars_per_download_group=1)
    t2m = item(v for v in template_config.data_vars if v.name == "temperature_2m")
    coords = region_job.generate_source_file_coords(processing_region_ds, [t2m])
    # 2 init_times x 5 lead_times x 2 members = 20 coords
    assert len(coords) == 2 * 5 * 2
    assert all(isinstance(c, MarsSourceFileCoord) for c in coords)
    mars_coord = coords[0]
    assert isinstance(mars_coord, MarsSourceFileCoord)
    assert mars_coord.request_type == "cf_sfc"


def test_region_job_download_file_open_data(monkeypatch: pytest.MonkeyPatch) -> None:
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
    source_file_coord = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2024-04-01"),
        lead_time=pd.Timedelta("3h"),
        data_var_group=[
            var for var in template_config.data_vars if var.name == "temperature_2m"
        ],
        ensemble_member=0,
    )

    example_grib_index = """
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levelist": "500", "levtype": "pl", "number": "2", "param": "gh", "_offset": 674936844, "_length": 393429}
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "type": "cf", "stream": "enfo", "step": "3", "levtype": "sfc", "param": "2t", "_offset": 0, "_length": 665525}
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "type": "cf", "stream": "enfo", "step": "3", "levtype": "sfc", "param": "10u", "_offset": 3773626, "_length": 665525}
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "type": "cf", "stream": "enfo", "step": "3", "levtype": "sfc", "param": "10u", "_offset": 665525, "_length": 888917}
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "1", "param": "2t", "_offset": 1554442, "_length": 664922}
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "2", "param": "2t", "_offset": 2219364, "_length": 664716}
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "1", "param": "10u", "_offset": 2884080, "_length": 889546}
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
        == "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/20240401/00z/ifs/0p25/enfo/20240401000000-3h-enfo-ef.grib2"
    )
    assert dataset_id == "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
    assert (
        local_path_suffix == "-4f434771"
    )  # result of calling digest on the byte ranges
    assert byte_ranges == ([0], [665525])


def test_region_job_read_data_open_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test grib data reading for open data with mocked file operations."""
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
    source_file_coord = OpenDataSourceFileCoord(
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

    assert np.array_equal(result, test_data)
    assert result.shape == (721, 1440)
    assert result.dtype == np.float32
    rasterio_reader.read.assert_called_once_with(1, out_dtype=np.float32)


def test_region_job_read_data_mars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test grib data reading for MARS with unit-only comment validation."""
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
    source_file_coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=0,
        data_var_group=[t2m_var],
        request_type="cf_sfc",
        downloaded_path=Path("fake/path/to/downloaded/file.grib"),
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 1
    # MARS uses different descriptive text but same unit
    rasterio_reader.tags = Mock(
        return_value={"GRIB_COMMENT": "2 metre temperature [C]"}
    )
    test_data = np.ones((721, 1440), dtype=np.float32)
    rasterio_reader.read = Mock(return_value=test_data)
    monkeypatch.setattr(
        "reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(source_file_coord, t2m_var)

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
    )  # We reprocess the last forecast for May 1st and also process for May 2nd
    for job in jobs:
        assert isinstance(job, EcmwfIfsEnsForecast15Day025DegreeRegionJob)
        assert job.data_vars == template_config.data_vars


@pytest.mark.slow
def test_download_file_from_ecmwf_open_data() -> None:
    """Download a recent ECMWF IFS ENS init time and read all template variables at lead_times where they are present."""
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    # Use a recent date so the test catches format changes in the current ECMWF data
    init_time = (pd.Timestamp.now() - pd.Timedelta(days=5)).floor("D")

    full_template = template_config.get_template(init_time + pd.Timedelta(days=1))
    region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=full_template,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    # Test over lead_times [0h, 3h, 96h] to catch bugs where variables are missing from
    # the index at certain lead times. In particular:
    # - 0h: max-window variables (e.g. 10fg/wind_gust) are absent
    # - 96h: ECMWF uses explicitly-windowed param names (e.g. "10fg3" instead of "10fg")
    test_ds = full_template.sel(
        init_time=slice(init_time, None),
        lead_time=[pd.Timedelta("0h"), pd.Timedelta("3h"), pd.Timedelta("96h")],
        ensemble_member=[0],
    )

    def check_data_var(data_var: EcmwfDataVar) -> None:
        for source_coord in region_job.generate_source_file_coords(test_ds, [data_var]):
            downloaded_coord = replace(
                source_coord, downloaded_path=region_job.download_file(source_coord)
            )
            data = region_job.read_data(downloaded_coord, data_var)
            assert np.all(np.isfinite(data)), (
                f"Non-finite values for {data_var.name} at lead_time={source_coord.lead_time}"
            )

    all_data_vars = [
        data_var
        for group in EcmwfIfsEnsForecast15Day025DegreeRegionJob.source_groups(
            template_config.data_vars
        )
        for data_var in group
    ]
    with ThreadPoolExecutor() as executor:
        list(executor.map(check_data_var, all_data_vars))


@pytest.mark.slow
def test_download_and_read_mars_data() -> None:
    """Download MARS GRIB data from source.coop and read all template variables."""
    template_config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()

    test_init_time = pd.Timestamp("2016-03-08")
    test_member = 0
    test_lead_time = pd.Timedelta("3h")

    region_job = EcmwfIfsEnsForecast15Day025DegreeRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_config.get_template(test_init_time + pd.Timedelta(days=1)),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    def check_data_var(data_var: EcmwfDataVar) -> None:
        request_type = MarsSourceFileCoord.get_request_type(
            data_var.internal_attrs.grib_index_level_type, test_member
        )
        coord = MarsSourceFileCoord(
            init_time=test_init_time,
            lead_time=test_lead_time,
            ensemble_member=test_member,
            data_var_group=[data_var],
            request_type=request_type,
        ).resolve_data_vars()
        downloaded_coord = replace(
            coord, downloaded_path=region_job.download_file(coord)
        )
        result = region_job.read_data(downloaded_coord, data_var)

        assert result.shape == (721, 1440), (
            f"{data_var.name}: expected shape (721, 1440), got {result.shape}"
        )
        assert np.all(np.isfinite(result)), f"{data_var.name}: has non-finite values"

    with ThreadPoolExecutor() as executor:
        list(executor.map(check_data_var, template_config.data_vars))
