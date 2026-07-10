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
from reformatters.ecmwf.aifs_ens.forecast.region_job import (
    EcmwfAifsEnsForecastRegionJob,
    EcmwfAifsEnsForecastSourceFileCoord,
)
from reformatters.ecmwf.aifs_ens.forecast.template_config import (
    EcmwfAifsEnsForecastTemplateConfig,
)
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar


def test_source_file_coord_url_cf() -> None:
    config = EcmwfAifsEnsForecastTemplateConfig()
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-07-02T12:00"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=0,
        data_var_group=list(config.data_vars[:1]),
    )
    assert coord.file_type == "cf"
    assert coord.get_url("s3") == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20250702/12z/aifs-ens/0p25/enfo/20250702120000-6h-enfo-cf.grib2"
    )
    assert coord.get_index_url("s3") == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20250702/12z/aifs-ens/0p25/enfo/20250702120000-6h-enfo-cf.index"
    )
    assert coord.get_url("gcs") == (
        "https://storage.googleapis.com/ecmwf-open-data"
        "/20250702/12z/aifs-ens/0p25/enfo/20250702120000-6h-enfo-cf.grib2"
    )
    assert coord.get_index_url("gcs") == (
        "https://storage.googleapis.com/ecmwf-open-data"
        "/20250702/12z/aifs-ens/0p25/enfo/20250702120000-6h-enfo-cf.index"
    )


def test_source_file_coord_url_pf() -> None:
    config = EcmwfAifsEnsForecastTemplateConfig()
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-07-02T00:00"),
        lead_time=pd.Timedelta("12h"),
        ensemble_member=23,
        data_var_group=list(config.data_vars[:1]),
    )
    assert coord.file_type == "pf"
    assert coord.get_url("s3") == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20250702/00z/aifs-ens/0p25/enfo/20250702000000-12h-enfo-pf.grib2"
    )
    assert coord.get_index_url("s3") == (
        "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        "/20250702/00z/aifs-ens/0p25/enfo/20250702000000-12h-enfo-pf.index"
    )
    assert coord.get_url("gcs") == (
        "https://storage.googleapis.com/ecmwf-open-data"
        "/20250702/00z/aifs-ens/0p25/enfo/20250702000000-12h-enfo-pf.grib2"
    )
    assert coord.get_index_url("gcs") == (
        "https://storage.googleapis.com/ecmwf-open-data"
        "/20250702/00z/aifs-ens/0p25/enfo/20250702000000-12h-enfo-pf.index"
    )


def test_source_file_coord_out_loc() -> None:
    config = EcmwfAifsEnsForecastTemplateConfig()
    init_time = pd.Timestamp("2025-07-02T00:00")
    lead_time = pd.Timedelta("6h")
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        ensemble_member=5,
        data_var_group=list(config.data_vars[:1]),
    )
    out = coord.out_loc()
    assert out["init_time"] == init_time
    assert out["lead_time"] == lead_time
    assert out["ensemble_member"] == 5


def test_source_groups() -> None:
    config = EcmwfAifsEnsForecastTemplateConfig()
    groups = EcmwfAifsEnsForecastRegionJob.source_file_var_groups(config.data_vars)
    # All vars are available from append_dim_start, so they all share date_available=None.
    assert len(groups) == 1
    assert len(groups[0]) == len(config.data_vars)


def test_generate_source_file_coords() -> None:
    config = EcmwfAifsEnsForecastTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2025-07-02T18:00"))

    region_job = EcmwfAifsEnsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds.isel(
            init_time=slice(0, 2),
            lead_time=slice(0, 3),
            ensemble_member=slice(0, 4),
        ),
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 2),
        reformat_job_name="test",
    )
    processing_region_ds, _ = region_job._get_region_datasets()

    groups = EcmwfAifsEnsForecastRegionJob.source_file_var_groups(config.data_vars)
    coords = region_job.generate_source_file_coords(processing_region_ds, groups[0])
    # 2 init_times x 3 lead_times x 4 ensemble_members = 24
    assert len(coords) == 2 * 3 * 4

    cf_coords = [c for c in coords if c.file_type == "cf"]
    pf_coords = [c for c in coords if c.file_type == "pf"]
    # 1 cf member (0) x 2 inits x 3 leads = 6
    assert len(cf_coords) == 2 * 3
    # 3 pf members (1-3) x 2 inits x 3 leads = 18
    assert len(pf_coords) == 2 * 3 * 3

    for coord in coords:
        assert isinstance(coord, EcmwfAifsEnsForecastSourceFileCoord)


_CF_INDEX = """
{"domain": "g", "date": "20250702", "time": "0000", "expver": "0001", "class": "ai", "type": "cf", "stream": "enfo", "step": "6", "levtype": "sfc", "param": "2t", "model": "aifs-ens", "_offset": 0, "_length": 665525}
{"domain": "g", "date": "20250702", "time": "0000", "expver": "0001", "class": "ai", "type": "cf", "stream": "enfo", "step": "6", "levtype": "sfc", "param": "10u", "_offset": 665525, "_length": 700000}
"""

_PF_INDEX = """
{"domain": "g", "date": "20250702", "time": "0000", "expver": "0001", "class": "ai", "type": "pf", "stream": "enfo", "step": "6", "levtype": "sfc", "number": "1", "param": "2t", "model": "aifs-ens", "_offset": 0, "_length": 665525}
{"domain": "g", "date": "20250702", "time": "0000", "expver": "0001", "class": "ai", "type": "pf", "stream": "enfo", "step": "6", "levtype": "sfc", "number": "2", "param": "2t", "model": "aifs-ens", "_offset": 665525, "_length": 670000}
{"domain": "g", "date": "20250702", "time": "0000", "expver": "0001", "class": "ai", "type": "pf", "stream": "enfo", "step": "6", "levtype": "sfc", "number": "3", "param": "2t", "model": "aifs-ens", "_offset": 1335525, "_length": 660000}
"""


def _make_region_job() -> EcmwfAifsEnsForecastRegionJob:
    config = EcmwfAifsEnsForecastTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2025-07-02T06:00"))
    return EcmwfAifsEnsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )


def _patch_pd_now(monkeypatch: pytest.MonkeyPatch, now: pd.Timestamp) -> None:
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now))


def _patch_index(monkeypatch: pytest.MonkeyPatch, index_text: str) -> None:
    mock_index_df = pd.read_json(StringIO(index_text), lines=True)
    monkeypatch.setattr("pandas.read_json", lambda path, **kwargs: mock_index_df)


def test_download_file_cf(monkeypatch: pytest.MonkeyPatch) -> None:
    region_job = _make_region_job()
    config = EcmwfAifsEnsForecastTemplateConfig()
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-07-02T00:00"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=0,
        data_var_group=[t2m_var],
    )
    # Recent: 12h after init_time, S3 should be tried first.
    _patch_pd_now(monkeypatch, pd.Timestamp("2025-07-02T12:00"))
    _patch_index(monkeypatch, _CF_INDEX)

    download_to_disk_mock = Mock()
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_ens.forecast.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(coord)

    assert download_to_disk_mock.call_count == 2
    idx_url = download_to_disk_mock.call_args_list[0][0][0]
    data_url, _dataset_id = download_to_disk_mock.call_args_list[1][0]
    kwargs = download_to_disk_mock.call_args_list[1][1]

    assert "ecmwf-forecasts.s3" in idx_url
    assert "ecmwf-forecasts.s3" in data_url
    assert "20250702000000-6h-enfo-cf.grib2" in data_url
    assert kwargs["byte_ranges"] == ([0], [665525])


def test_download_file_pf(monkeypatch: pytest.MonkeyPatch) -> None:
    region_job = _make_region_job()
    config = EcmwfAifsEnsForecastTemplateConfig()
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-07-02T00:00"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=2,
        data_var_group=[t2m_var],
    )
    # Recent: 12h after init_time, S3 should be tried first.
    _patch_pd_now(monkeypatch, pd.Timestamp("2025-07-02T12:00"))
    _patch_index(monkeypatch, _PF_INDEX)

    download_to_disk_mock = Mock()
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_ens.forecast.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(coord)

    assert download_to_disk_mock.call_count == 2
    data_url, _dataset_id = download_to_disk_mock.call_args_list[1][0]
    kwargs = download_to_disk_mock.call_args_list[1][1]

    assert "ecmwf-forecasts.s3" in data_url
    assert "20250702000000-6h-enfo-pf.grib2" in data_url
    # member=2 selects offsets 665525..1335525
    assert kwargs["byte_ranges"] == ([665525], [665525 + 670000])


def test_download_file_old_init_prefers_gcs(monkeypatch: pytest.MonkeyPatch) -> None:
    region_job = _make_region_job()
    config = EcmwfAifsEnsForecastTemplateConfig()
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-07-02T00:00"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=0,
        data_var_group=[t2m_var],
    )
    # Old: more than 24h after init_time, GCS should be tried first.
    _patch_pd_now(monkeypatch, pd.Timestamp("2025-07-04T00:00"))
    _patch_index(monkeypatch, _CF_INDEX)

    download_to_disk_mock = Mock()
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_ens.forecast.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(coord)

    assert download_to_disk_mock.call_count == 2
    idx_url = download_to_disk_mock.call_args_list[0][0][0]
    data_url = download_to_disk_mock.call_args_list[1][0][0]
    assert "storage.googleapis.com/ecmwf-open-data" in idx_url
    assert "storage.googleapis.com/ecmwf-open-data" in data_url


def test_download_file_falls_back_to_gcs_on_s3_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region_job = _make_region_job()
    config = EcmwfAifsEnsForecastTemplateConfig()
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-07-02T00:00"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=0,
        data_var_group=[t2m_var],
    )
    # Recent: S3 is primary, GCS is fallback.
    _patch_pd_now(monkeypatch, pd.Timestamp("2025-07-02T12:00"))
    _patch_index(monkeypatch, _CF_INDEX)

    download_to_disk_mock = Mock()

    def fake_download(url: str, *_args: object, **_kwargs: object) -> Mock:
        if "ecmwf-forecasts.s3" in url:
            raise FileNotFoundError(url)
        return Mock()

    download_to_disk_mock.side_effect = fake_download
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_ens.forecast.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    region_job.download_file(coord)

    # First attempt: S3 index (fails). Second attempt: GCS index + GCS data.
    urls = [call[0][0] for call in download_to_disk_mock.call_args_list]
    assert "ecmwf-forecasts.s3" in urls[0]
    assert all("storage.googleapis.com" in u for u in urls[1:])
    assert any("20250702000000-6h-enfo-cf.grib2" in u for u in urls)


def test_download_file_raises_when_all_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region_job = _make_region_job()
    config = EcmwfAifsEnsForecastTemplateConfig()
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-07-02T00:00"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=0,
        data_var_group=[t2m_var],
    )
    _patch_pd_now(monkeypatch, pd.Timestamp("2025-07-02T12:00"))
    _patch_index(monkeypatch, _CF_INDEX)

    def always_404(url: str, *_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError(url)

    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_ens.forecast.region_job.http_download_to_disk",
        Mock(side_effect=always_404),
    )

    with pytest.raises(FileNotFoundError):
        region_job.download_file(coord)


def test_read_data(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EcmwfAifsEnsForecastTemplateConfig()
    region_job = EcmwfAifsEnsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    t2m_var = item(v for v in config.data_vars if v.name == "temperature_2m")
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-07-02T00:00"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=1,
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
        "reformatters.ecmwf.aifs_ens.forecast.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(coord, t2m_var)

    assert np.array_equal(result, test_data)
    assert result.dtype == np.float32
    rasterio_reader.read.assert_called_once_with(1, out_dtype=np.float32)


def test_read_data_alt_precip_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """tp variable can be read using alt GRIB metadata (table v34+)."""
    config = EcmwfAifsEnsForecastTemplateConfig()
    region_job = EcmwfAifsEnsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=config.data_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    precip_var = item(v for v in config.data_vars if v.name == "precipitation_surface")
    coord = EcmwfAifsEnsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-07-02T00:00"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=0,
        data_var_group=[precip_var],
        downloaded_path=Path("fake/path.grib2"),
    )

    rasterio_reader = Mock()
    rasterio_reader.__enter__ = Mock(return_value=rasterio_reader)
    rasterio_reader.__exit__ = Mock(return_value=False)
    rasterio_reader.count = 1
    rasterio_reader.descriptions = ['0[-] SFC="Ground or water surface"']
    rasterio_reader.tags = Mock(
        return_value={"GRIB_COMMENT": "Total precipitation rate [kg/(m^2*s)]"}
    )
    test_data = np.ones((721, 1440), dtype=np.float32) * 0.001
    rasterio_reader.read = Mock(return_value=test_data)
    monkeypatch.setattr(
        "reformatters.ecmwf.aifs_ens.forecast.region_job.rasterio.open",
        Mock(return_value=rasterio_reader),
    )

    result = region_job.read_data(coord, precip_var)
    assert np.array_equal(result, test_data)
    rasterio_reader.read.assert_called_once_with(1, out_dtype=np.float32)


def test_apply_data_transformations_scale_factor() -> None:
    config = EcmwfAifsEnsForecastTemplateConfig()
    region_job = EcmwfAifsEnsForecastRegionJob.model_construct(
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

    raw_z = 49000.0
    data = np.full((1, 2, 1, 721, 1440), raw_z, dtype=np.float32)
    data_array = xr.DataArray(
        data,
        dims=("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
    )

    region_job.apply_data_transformations(data_array, gh500_var)

    expected_gh = raw_z * gh500_var.internal_attrs.scale_factor
    np.testing.assert_allclose(data_array.values, expected_gh, atol=1.0)


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = EcmwfAifsEnsForecastTemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset-ecmwf-aifs-ens",
        template_config_version="test-version",
    )

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2025-07-02T18:00")),
    )
    existing_ds = config.get_template(pd.Timestamp("2025-07-02T06:01"))
    template_utils.write_metadata(existing_ds, store_factory)

    jobs, template_ds = EcmwfAifsEnsForecastRegionJob.operational_update_jobs(
        primary_store=store_factory.primary_store(),
        tmp_store=tmp_path / "tmp_ds.zarr",
        get_template_fn=config.get_template,
        append_dim=config.append_dim,
        all_data_vars=config.data_vars,
        reformat_job_name="test_job",
    )

    assert template_ds.init_time.max() == pd.Timestamp("2025-07-02T12:00")
    # 2 regions (06z reprocess + new 12z) x ceil(17 vars / max_vars_per_job=4) = 10
    n_regions = 2
    n_jobs_per_region = -(-len(config.data_vars) // 4)
    assert len(jobs) == n_regions * n_jobs_per_region
    for job in jobs:
        assert isinstance(job, EcmwfAifsEnsForecastRegionJob)
    all_job_vars = {v.name for job in jobs for v in job.data_vars}
    expected_vars = {v.name for v in config.data_vars}
    assert all_job_vars == expected_vars


@pytest.mark.slow
def test_download_file_from_ecmwf_open_data() -> None:
    """Download a single recent ECMWF AIFS-ENS pf file and read all template variables.

    Scoped to a single (init_time, lead_time, ensemble_member) so only one grib file
    is fetched. The cf and pf URL/index paths are covered separately by the
    test_download_file_cf / test_download_file_pf unit tests.
    """
    template_config = EcmwfAifsEnsForecastTemplateConfig()
    init_time = (pd.Timestamp.now() - pd.Timedelta(days=5)).floor("D")

    full_template = template_config.get_template(init_time + pd.Timedelta(days=1))
    region_job = EcmwfAifsEnsForecastRegionJob.model_construct(
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
        ensemble_member=[1],  # first pf (perturbed) member
    ).to_dataset()

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
                f"Non-finite values for {data_var.name} at "
                f"lead_time={source_coord.lead_time}, "
                f"ensemble_member={source_coord.ensemble_member}"
            )

    all_data_vars = [
        data_var
        for group in EcmwfAifsEnsForecastRegionJob.source_file_var_groups(
            template_config.data_vars
        )
        for data_var in group
    ]
    with ThreadPoolExecutor() as executor:
        list(executor.map(check_data_var, all_data_vars))
