from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.common.iterating import item
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.ecmwf.aifs_deterministic.forecast.region_job import (
    AIFS_SINGLE_PATH_CHANGE_DATE,
    EcmwfAifsForecastRegionJob,
    EcmwfAifsForecastSourceFileCoord,
)
from reformatters.ecmwf.aifs_deterministic.forecast.template_config import (
    EcmwfAifsForecastTemplateConfig,
)


def test_source_file_coord_url_before_path_change() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    coord = EcmwfAifsForecastSourceFileCoord(
        init_time=pd.Timestamp("2024-07-01T12:00"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=list(config.data_vars[:1]),
    )
    url = coord.get_url()
    assert "/aifs/0p25/oper/" in url
    assert "20240701120000-6h-oper-fc.grib2" in url

    idx_url = coord.get_index_url()
    assert idx_url.endswith(".index")


def test_source_file_coord_url_after_path_change() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    coord = EcmwfAifsForecastSourceFileCoord(
        init_time=AIFS_SINGLE_PATH_CHANGE_DATE,
        lead_time=pd.Timedelta("12h"),
        data_var_group=list(config.data_vars[:1]),
    )
    url = coord.get_url()
    assert "/aifs-single/0p25/oper/" in url
    assert "20250226000000-12h-oper-fc.grib2" in url


def test_source_file_coord_out_loc() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    init_time = pd.Timestamp("2024-04-01T00:00")
    lead_time = pd.Timedelta("6h")
    coord = EcmwfAifsForecastSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        data_var_group=list(config.data_vars[:1]),
    )
    out = coord.out_loc()
    assert out["init_time"] == init_time
    assert out["lead_time"] == lead_time


def test_source_groups() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    groups = EcmwfAifsForecastRegionJob.source_groups(config.data_vars)
    assert len(groups) == 2

    group_without_date = [
        g for g in groups if g[0].internal_attrs.date_available is None
    ]
    group_with_date = [
        g for g in groups if g[0].internal_attrs.date_available is not None
    ]
    assert len(group_without_date) == 1
    assert len(group_with_date) == 1
    assert len(group_without_date[0]) > 0
    assert len(group_with_date[0]) > 0


def test_generate_source_file_coords() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2024-04-03"))

    region_job = EcmwfAifsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds.isel(
            init_time=slice(0, 4),
            lead_time=slice(0, 3),
        ),
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 4),
        reformat_job_name="test",
    )
    processing_region_ds, _ = region_job._get_region_datasets()
    groups = EcmwfAifsForecastRegionJob.source_groups(config.data_vars)

    coords = region_job.generate_source_file_coords(processing_region_ds, groups[0])
    # 4 init_times x 3 lead_times = 12
    assert len(coords) == 4 * 3


def test_download_file(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EcmwfAifsForecastTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2024-04-02"))

    region_job = EcmwfAifsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    source_file_coord = EcmwfAifsForecastSourceFileCoord(
        init_time=pd.Timestamp("2024-04-01"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=[t2m_var],
    )

    # Deterministic AIFS index has no "number" or "type" fields
    example_grib_index = """
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "stream": "oper", "step": "6", "levtype": "sfc", "param": "2t", "_offset": 0, "_length": 665525}
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "stream": "oper", "step": "6", "levtype": "sfc", "param": "10u", "_offset": 665525, "_length": 700000}
"""
    mock_index_df = pd.read_json(StringIO(example_grib_index), lines=True)
    monkeypatch.setattr(
        "pandas.read_json",
        lambda path, **kwargs: mock_index_df,
    )

    download_to_disk_mock = Mock()
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_deterministic.forecast.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(source_file_coord)

    # First call is for the index file, second for the GRIB data
    assert download_to_disk_mock.call_count == 2
    url, _dataset_id = download_to_disk_mock.call_args_list[1][0]
    kwargs = download_to_disk_mock.call_args_list[1][1]

    assert "20240401000000-6h-oper-fc.grib2" in url
    assert kwargs["byte_ranges"] == ([0], [665525])


def test_read_data(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EcmwfAifsForecastTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2024-04-02"))

    region_job = EcmwfAifsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    coord = EcmwfAifsForecastSourceFileCoord(
        init_time=pd.Timestamp("2024-04-01"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=[t2m_var],
        downloaded_path=Path("fake/path.grib2"),
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 1
    rasterio_reader.descriptions = ['2[m] HTGL="Specified height level above ground"']
    rasterio_reader.tags = Mock(return_value={"GRIB_COMMENT": "Temperature [C]"})
    test_data = np.ones((721, 1440), dtype=np.float32)
    rasterio_reader.read = Mock(return_value=test_data)
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_deterministic.forecast.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(coord, t2m_var)

    assert np.array_equal(result, test_data)
    assert result.dtype == np.float32
    rasterio_reader.read.assert_called_once_with(1, out_dtype=np.float32)


def test_read_data_multi_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """When max_vars_per_download_group > 1, the GRIB has multiple bands."""
    config = EcmwfAifsForecastTemplateConfig()
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    d2m_var = item(v for v in config.data_vars if v.name == "dew_point_temperature_2m")
    coord = EcmwfAifsForecastSourceFileCoord(
        init_time=pd.Timestamp("2024-04-01"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=[t2m_var, d2m_var],
        downloaded_path=Path("fake/path.grib2"),
    )

    region_job = EcmwfAifsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 2
    rasterio_reader.descriptions = [
        '2[m] HTGL="Specified height level above ground"',
        '2[m] HTGL="Specified height level above ground"',
    ]

    def mock_tags(band_i: int) -> dict[str, str]:
        if band_i == 1:
            return {"GRIB_COMMENT": "Temperature [C]"}
        return {"GRIB_COMMENT": "Dew point temperature [C]"}

    rasterio_reader.tags = mock_tags
    test_data = np.ones((721, 1440), dtype=np.float32)
    rasterio_reader.read = Mock(return_value=test_data)
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_deterministic.forecast.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(coord, d2m_var)
    assert np.array_equal(result, test_data)
    # Should read band 2 (dew point), not band 1 (temperature)
    rasterio_reader.read.assert_called_once_with(2, out_dtype=np.float32)


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = EcmwfAifsForecastTemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset-ecmwf-aifs",
        template_config_version="test-version",
    )

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2024-04-02T12:00")),
    )
    existing_ds = config.get_template(pd.Timestamp("2024-04-01T06:01"))
    template_utils.write_metadata(existing_ds, store_factory)

    jobs, template_ds = EcmwfAifsForecastRegionJob.operational_update_jobs(
        primary_store=store_factory.primary_store(),
        tmp_store=tmp_path / "tmp_ds.zarr",
        get_template_fn=config.get_template,
        append_dim=config.append_dim,
        all_data_vars=config.data_vars,
        reformat_job_name="test_job",
    )

    assert template_ds.init_time.max() == pd.Timestamp("2024-04-02T06:00")
    assert len(jobs) > 0
    for job in jobs:
        assert isinstance(job, EcmwfAifsForecastRegionJob)
