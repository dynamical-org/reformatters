from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils
from reformatters.common.iterating import item
from reformatters.common.pydantic import replace
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.types import ArrayFloat32
from reformatters.ecmwf.aifs_single.forecast.region_job import (
    AIFS_SINGLE_PATH_CHANGE_DATE,
    EcmwfAifsSingleForecastRegionJob,
    EcmwfAifsSingleForecastSourceFileCoord,
)
from reformatters.ecmwf.aifs_single.forecast.template_config import (
    EcmwfAifsSingleForecastTemplateConfig,
)
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar


def test_source_file_coord_url_before_path_change() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    coord = EcmwfAifsSingleForecastSourceFileCoord(
        init_time=pd.Timestamp("2024-07-01T12:00"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=list(config.data_vars[:1]),
    )
    assert coord.get_url() == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20240701/12z/aifs/0p25/oper/20240701120000-6h-oper-fc.grib2"
    )
    assert coord.get_index_url() == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20240701/12z/aifs/0p25/oper/20240701120000-6h-oper-fc.index"
    )


def test_source_file_coord_url_after_path_change() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    coord = EcmwfAifsSingleForecastSourceFileCoord(
        init_time=AIFS_SINGLE_PATH_CHANGE_DATE,
        lead_time=pd.Timedelta("12h"),
        data_var_group=list(config.data_vars[:1]),
    )
    assert coord.get_url() == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20250226/00z/aifs-single/0p25/oper/20250226000000-12h-oper-fc.grib2"
    )
    assert coord.get_index_url() == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20250226/00z/aifs-single/0p25/oper/20250226000000-12h-oper-fc.index"
    )
    assert coord.get_url("gcs") == (
        "https://storage.googleapis.com/ecmwf-open-data"
        "/20250226/00z/aifs-single/0p25/oper/20250226000000-12h-oper-fc.grib2"
    )
    assert coord.get_index_url("gcs") == (
        "https://storage.googleapis.com/ecmwf-open-data"
        "/20250226/00z/aifs-single/0p25/oper/20250226000000-12h-oper-fc.index"
    )


def test_source_file_coord_url_00z_init() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    coord = EcmwfAifsSingleForecastSourceFileCoord(
        init_time=pd.Timestamp("2024-10-15T00:00"),
        lead_time=pd.Timedelta("24h"),
        data_var_group=list(config.data_vars[:1]),
    )
    assert coord.get_url() == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20241015/00z/aifs/0p25/oper/20241015000000-24h-oper-fc.grib2"
    )


def test_source_file_coord_url_day_before_path_change() -> None:
    """The day before the path change should still use 'aifs'."""
    config = EcmwfAifsSingleForecastTemplateConfig()
    coord = EcmwfAifsSingleForecastSourceFileCoord(
        init_time=AIFS_SINGLE_PATH_CHANGE_DATE - pd.Timedelta("12h"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=list(config.data_vars[:1]),
    )
    assert coord.get_url() == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20250225/12z/aifs/0p25/oper/20250225120000-6h-oper-fc.grib2"
    )


def test_source_file_coord_out_loc() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    init_time = pd.Timestamp("2024-04-01T00:00")
    lead_time = pd.Timedelta("6h")
    coord = EcmwfAifsSingleForecastSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        data_var_group=list(config.data_vars[:1]),
    )
    out = coord.out_loc()
    assert out["init_time"] == init_time
    assert out["lead_time"] == lead_time


def test_source_groups() -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    groups = EcmwfAifsSingleForecastRegionJob.source_groups(config.data_vars)
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
    config = EcmwfAifsSingleForecastTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2024-04-03"))

    region_job = EcmwfAifsSingleForecastRegionJob.model_construct(
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
    processing_region_ds, output_region_ds = region_job._get_region_datasets()
    assert processing_region_ds.equals(output_region_ds)

    groups = EcmwfAifsSingleForecastRegionJob.source_groups(config.data_vars)

    coords = region_job.generate_source_file_coords(processing_region_ds, groups[0])
    # 4 init_times x 3 lead_times = 12
    assert len(coords) == 4 * 3

    for coord in coords:
        assert isinstance(coord, EcmwfAifsSingleForecastSourceFileCoord)


_EXAMPLE_GRIB_INDEX = """
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "stream": "oper", "step": "6", "levtype": "sfc", "param": "2t", "_offset": 0, "_length": 665525}
{"domain": "g", "date": "20240401", "time": "0000", "expver": "0001", "class": "od", "stream": "oper", "step": "6", "levtype": "sfc", "param": "10u", "_offset": 665525, "_length": 700000}
"""


def _make_download_test_region_job() -> EcmwfAifsSingleForecastRegionJob:
    config = EcmwfAifsSingleForecastTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2024-04-02"))
    return EcmwfAifsSingleForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )


def _t2m_coord() -> EcmwfAifsSingleForecastSourceFileCoord:
    config = EcmwfAifsSingleForecastTemplateConfig()
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    return EcmwfAifsSingleForecastSourceFileCoord(
        init_time=pd.Timestamp("2024-04-01"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=[t2m_var],
    )


def test_download_file_prefers_gcs(monkeypatch: pytest.MonkeyPatch) -> None:
    region_job = _make_download_test_region_job()
    mock_index_df = pd.read_json(StringIO(_EXAMPLE_GRIB_INDEX), lines=True)
    monkeypatch.setattr("pandas.read_json", lambda path, **kwargs: mock_index_df)

    download_to_disk_mock = Mock()
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_single.forecast.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(_t2m_coord())

    # First call is the index file, second the GRIB data; both on GCS.
    assert download_to_disk_mock.call_count == 2
    idx_url = download_to_disk_mock.call_args_list[0][0][0]
    data_url, _dataset_id = download_to_disk_mock.call_args_list[1][0]
    kwargs = download_to_disk_mock.call_args_list[1][1]

    assert "storage.googleapis.com/ecmwf-open-data" in idx_url
    assert "storage.googleapis.com/ecmwf-open-data" in data_url
    assert "20240401000000-6h-oper-fc.grib2" in data_url
    assert kwargs["byte_ranges"] == ([0], [665525])


def test_download_file_falls_back_to_s3_on_gcs_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region_job = _make_download_test_region_job()
    mock_index_df = pd.read_json(StringIO(_EXAMPLE_GRIB_INDEX), lines=True)
    monkeypatch.setattr("pandas.read_json", lambda path, **kwargs: mock_index_df)

    def fake_download(url: str, *_args: object, **_kwargs: object) -> Mock:
        if "storage.googleapis.com" in url:
            raise FileNotFoundError(url)
        return Mock()

    download_to_disk_mock = Mock(side_effect=fake_download)
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_single.forecast.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(_t2m_coord())

    urls = [call[0][0] for call in download_to_disk_mock.call_args_list]
    assert "storage.googleapis.com" in urls[0]
    assert all("ecmwf-forecasts.s3" in u for u in urls[1:])
    assert any("20240401000000-6h-oper-fc.grib2" in u for u in urls)


def test_download_file_raises_when_all_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region_job = _make_download_test_region_job()
    mock_index_df = pd.read_json(StringIO(_EXAMPLE_GRIB_INDEX), lines=True)
    monkeypatch.setattr("pandas.read_json", lambda path, **kwargs: mock_index_df)

    def always_404(url: str, *_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError(url)

    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_single.forecast.region_job.http_download_to_disk",
        Mock(side_effect=always_404),
    )

    with pytest.raises(FileNotFoundError):
        region_job.download_file(_t2m_coord())


def test_read_data(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2024-04-02"))

    region_job = EcmwfAifsSingleForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    coord = EcmwfAifsSingleForecastSourceFileCoord(
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
        "reformatters.ecmwf.aifs_single.forecast.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(coord, t2m_var)

    assert np.array_equal(result, test_data)
    assert result.dtype == np.float32
    rasterio_reader.read.assert_called_once_with(1, out_dtype=np.float32)


def test_read_data_multi_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """When max_vars_per_download_group > 1, the GRIB has multiple bands."""
    config = EcmwfAifsSingleForecastTemplateConfig()
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    d2m_var = item(v for v in config.data_vars if v.name == "dew_point_temperature_2m")
    coord = EcmwfAifsSingleForecastSourceFileCoord(
        init_time=pd.Timestamp("2024-04-01"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=[t2m_var, d2m_var],
        downloaded_path=Path("fake/path.grib2"),
    )

    region_job = EcmwfAifsSingleForecastRegionJob.model_construct(
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
        "reformatters.ecmwf.aifs_single.forecast.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(coord, d2m_var)
    assert np.array_equal(result, test_data)
    # Should read band 2 (dew point), not band 1 (temperature)
    rasterio_reader.read.assert_called_once_with(2, out_dtype=np.float32)


def test_apply_data_transformations_scale_factor() -> None:
    """scale_factor is applied in-place, converting geopotential (m²/s²) to height (m)."""
    config = EcmwfAifsSingleForecastTemplateConfig()
    region_job = EcmwfAifsSingleForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    gh500_var = item(
        v for v in config.data_vars if v.name == "geopotential_height_500hpa"
    )
    assert gh500_var.internal_attrs.scale_factor is not None

    raw_z = 49000.0  # m²/s² (geopotential at ~500 hPa)
    data = np.full((1, 2, 721, 1440), raw_z, dtype=np.float32)
    data_array = xr.DataArray(
        data,
        dims=("init_time", "lead_time", "latitude", "longitude"),
    )

    region_job.apply_data_transformations(data_array, gh500_var)

    expected_gh = raw_z * gh500_var.internal_attrs.scale_factor
    # float32 precision limits accuracy to ~1m at typical geopotential heights
    np.testing.assert_allclose(data_array.values, expected_gh, atol=1.0)


def test_apply_data_transformations_no_scale_factor() -> None:
    """Variables without scale_factor are not modified."""
    config = EcmwfAifsSingleForecastTemplateConfig()
    region_job = EcmwfAifsSingleForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    assert t2m_var.internal_attrs.scale_factor is None

    raw_value = 25.0
    data = np.full((1, 2, 721, 1440), raw_value, dtype=np.float32)
    data_array = xr.DataArray(
        data,
        dims=("init_time", "lead_time", "latitude", "longitude"),
    )

    region_job.apply_data_transformations(data_array, t2m_var)

    np.testing.assert_array_equal(data_array.values, raw_value)


def test_read_data_alt_precip_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """tp variable can be read using alt GRIB metadata (table v34+)."""
    config = EcmwfAifsSingleForecastTemplateConfig()
    region_job = EcmwfAifsSingleForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    precip_var = item(v for v in config.data_vars if v.name == "precipitation_surface")
    coord = EcmwfAifsSingleForecastSourceFileCoord(
        # v34+ alt metadata is the current "aifs-single" format, so use a current-era init.
        init_time=AIFS_SINGLE_PATH_CHANGE_DATE,
        lead_time=pd.Timedelta("6h"),
        data_var_group=[precip_var],
        downloaded_path=Path("fake/path.grib2"),
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 1
    # Use the v34+ alt description instead of the early form
    rasterio_reader.descriptions = ['0[-] SFC="Ground or water surface"']
    rasterio_reader.tags = Mock(
        return_value={"GRIB_COMMENT": "Total precipitation rate [kg/(m^2*s)]"}
    )
    test_data = np.ones((721, 1440), dtype=np.float32) * 0.001
    rasterio_reader.read = Mock(return_value=test_data)
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_single.forecast.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(coord, precip_var)
    assert np.array_equal(result, test_data)
    rasterio_reader.read.assert_called_once_with(1, out_dtype=np.float32)


def _precip_read_data_mock(
    monkeypatch: pytest.MonkeyPatch, init_time: pd.Timestamp
) -> ArrayFloat32:
    """Read precipitation_surface for the given init_time, returning the raw-scaled array."""
    config = EcmwfAifsSingleForecastTemplateConfig()
    region_job = EcmwfAifsSingleForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    precip_var = item(v for v in config.data_vars if v.name == "precipitation_surface")
    coord = EcmwfAifsSingleForecastSourceFileCoord(
        init_time=init_time,
        lead_time=pd.Timedelta("6h"),
        data_var_group=[precip_var],
        downloaded_path=Path("fake/path.grib2"),
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 1
    rasterio_reader.descriptions = ['2[-] SFC="Ground or water surface"']
    rasterio_reader.tags = Mock(
        return_value={"GRIB_COMMENT": "(prodType 0, cat 1, subcat 193) [-]"}
    )
    rasterio_reader.read = Mock(
        return_value=np.full((721, 1440), 0.001, dtype=np.float32)
    )
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_single.forecast.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )
    return region_job.read_data(coord, precip_var)


def test_read_data_legacy_precip_scaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy 'aifs' precip is stored in metres and scaled x1000 to mm (kg m-2)."""
    result = _precip_read_data_mock(
        monkeypatch, AIFS_SINGLE_PATH_CHANGE_DATE - pd.Timedelta("6h")
    )
    np.testing.assert_array_equal(result, 1.0)  # 0.001 m * 1000


def test_read_data_current_precip_not_scaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Current 'aifs-single' precip is already in mm (kg m-2) and is not rescaled."""
    result = _precip_read_data_mock(monkeypatch, AIFS_SINGLE_PATH_CHANGE_DATE)
    np.testing.assert_allclose(result, 0.001, rtol=1e-6)


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = EcmwfAifsSingleForecastTemplateConfig()
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

    jobs, template_ds = EcmwfAifsSingleForecastRegionJob.operational_update_jobs(
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
        assert isinstance(job, EcmwfAifsSingleForecastRegionJob)
        assert job.data_vars == config.data_vars


@pytest.mark.slow
def test_download_file_from_ecmwf_open_data() -> None:
    """Download a single recent ECMWF AIFS Single grib file and read all template variables.

    Scoped to a single (init_time, lead_time) so only one grib file is fetched.
    """
    template_config = EcmwfAifsSingleForecastTemplateConfig()
    init_time = (pd.Timestamp.now() - pd.Timedelta(days=5)).floor("D")

    full_template = template_config.get_template(init_time + pd.Timedelta(days=1))
    region_job = EcmwfAifsSingleForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=full_template,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    test_ds = full_template.sel(
        init_time=[init_time],
        lead_time=[pd.Timedelta("6h")],
    )

    def check_data_var(data_var: EcmwfDataVar) -> None:
        for source_coord in region_job.generate_source_file_coords(test_ds, [data_var]):
            downloaded_coord = replace(
                source_coord, downloaded_path=region_job.download_file(source_coord)
            )
            data = region_job.read_data(downloaded_coord, data_var)
            assert data.shape == (721, 1440), (
                f"{data_var.name}: expected shape (721, 1440), got {data.shape}"
            )
            assert np.all(np.isfinite(data)), (
                f"Non-finite values for {data_var.name} at lead_time={source_coord.lead_time}"
            )

    all_data_vars = [
        data_var
        for group in EcmwfAifsSingleForecastRegionJob.source_groups(
            template_config.data_vars
        )
        for data_var in group
    ]
    with ThreadPoolExecutor() as executor:
        list(executor.map(check_data_var, all_data_vars))
