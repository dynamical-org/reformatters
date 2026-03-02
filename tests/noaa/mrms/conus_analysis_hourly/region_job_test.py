from pathlib import Path
from typing import Self
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.common.pydantic import replace
from reformatters.common.region_job import SourceFileStatus
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.noaa.mrms.conus_analysis_hourly.region_job import (
    NoaaMrmsRegionJob,
    NoaaMrmsSourceFileCoord,
)
from reformatters.noaa.mrms.conus_analysis_hourly.template_config import (
    NoaaMrmsConusAnalysisHourlyTemplateConfig,
)


@pytest.fixture
def template_config() -> NoaaMrmsConusAnalysisHourlyTemplateConfig:
    return NoaaMrmsConusAnalysisHourlyTemplateConfig()


def test_source_file_coord_out_loc() -> None:
    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T12:00"),
        product="MultiSensor_QPE_01H_Pass2",
        level="00.00",
        fallback_products=(),
    )
    assert coord.out_loc() == {"time": pd.Timestamp("2024-01-15T12:00")}


def test_source_file_coord_get_url_s3() -> None:
    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T12:00"),
        product="MultiSensor_QPE_01H_Pass2",
        level="00.00",
        fallback_products=(),
    )
    url = coord.get_url(source="s3")
    assert url == (
        "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/"
        "MultiSensor_QPE_01H_Pass2_00.00/20240115/"
        "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20240115-120000.grib2.gz"
    )


def test_source_file_coord_get_url_iowa() -> None:
    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2019-06-15T12:00"),
        product="GaugeCorr_QPE_01H",
        level="00.00",
        fallback_products=(),
    )
    url = coord.get_url(source="iowa")
    assert url == (
        "https://mtarchive.geol.iastate.edu/2019/06/15/mrms/ncep/"
        "GaugeCorr_QPE_01H/"
        "GaugeCorr_QPE_01H_00.00_20190615-120000.grib2.gz"
    )


def test_source_file_coord_get_url_ncep() -> None:
    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T23:00"),
        product="MultiSensor_QPE_01H_Pass2",
        level="00.00",
        fallback_products=(),
    )
    url = coord.get_url(source="ncep")
    assert url == (
        "https://mrms.ncep.noaa.gov/2D/"
        "MultiSensor_QPE_01H_Pass2/"
        "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20240115-230000.grib2.gz"
    )


def test_source_groups(
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
) -> None:
    groups = NoaaMrmsRegionJob.source_groups(template_config.data_vars)
    # Each MRMS variable is its own group (one file per variable)
    assert len(groups) == len(template_config.data_vars)
    for group in groups:
        assert len(group) == 1


def test_generate_source_file_coords_post_v12(
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
) -> None:
    template_ds = template_config.get_template(pd.Timestamp("2024-01-15T03:00"))
    test_ds = template_ds.sel(
        time=slice(pd.Timestamp("2024-01-15T00:00"), pd.Timestamp("2024-01-15T02:00"))
    )

    precip_var = next(
        v for v in template_config.data_vars if v.name == "precipitation_surface"
    )
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=[precip_var],
        append_dim=template_config.append_dim,
        region=slice(0, 3),
        reformat_job_name="test",
    )

    processing_region_ds, _ = region_job._get_region_datasets()
    coords = region_job.generate_source_file_coords(processing_region_ds, [precip_var])

    assert len(coords) == 3
    assert all(c.product == "MultiSensor_QPE_01H_Pass2" for c in coords)
    assert all(
        c.fallback_products == ("MultiSensor_QPE_01H_Pass1", "RadarOnly_QPE_01H")
        for c in coords
    )


def test_generate_source_file_coords_pre_v12(
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
) -> None:
    template_ds = template_config.get_template(pd.Timestamp("2019-06-15T03:00"))
    test_ds = template_ds.sel(
        time=slice(pd.Timestamp("2019-06-15T00:00"), pd.Timestamp("2019-06-15T02:00"))
    )

    precip_var = next(
        v for v in template_config.data_vars if v.name == "precipitation_surface"
    )
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=[precip_var],
        append_dim=template_config.append_dim,
        region=slice(0, 3),
        reformat_job_name="test",
    )

    processing_region_ds, _ = region_job._get_region_datasets()
    coords = region_job.generate_source_file_coords(processing_region_ds, [precip_var])

    assert len(coords) == 3
    # Pre-v12 should use GaugeCorr_QPE_01H
    assert all(c.product == "GaugeCorr_QPE_01H" for c in coords)
    assert all(c.fallback_products == ("RadarOnly_QPE_01H",) for c in coords)


def test_generate_source_file_coords_pass_1_pre_v12_skipped(
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
) -> None:
    template_ds = template_config.get_template(pd.Timestamp("2019-06-15T03:00"))
    test_ds = template_ds.sel(
        time=slice(pd.Timestamp("2019-06-15T00:00"), pd.Timestamp("2019-06-15T02:00"))
    )

    pass_1_var = next(
        v for v in template_config.data_vars if v.name == "precipitation_pass_1_surface"
    )
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=[pass_1_var],
        append_dim=template_config.append_dim,
        region=slice(0, 3),
        reformat_job_name="test",
    )

    processing_region_ds, _ = region_job._get_region_datasets()
    coords = region_job.generate_source_file_coords(processing_region_ds, [pass_1_var])

    # Pass 1 not available pre-v12, should have no coords
    assert len(coords) == 0


def test_download_file_uses_fallback_products(monkeypatch: pytest.MonkeyPatch) -> None:
    region_job = NoaaMrmsRegionJob.model_construct(
        template_ds=Mock(),
        data_vars=[],
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )

    attempts: list[tuple[str, str]] = []

    def fake_download(coord: NoaaMrmsSourceFileCoord, source: str) -> Path:
        attempts.append((coord.product, source))
        if coord.product == "RadarOnly_QPE_01H" and source == "s3":
            return Path("fake.grib2")
        raise FileNotFoundError(coord.product)

    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args: pd.Timestamp("2024-01-16T12:00"))
    )
    monkeypatch.setattr(region_job, "_download_from_source", fake_download)

    path = region_job.download_file(
        NoaaMrmsSourceFileCoord(
            time=pd.Timestamp("2024-01-15T12:00"),
            product="MultiSensor_QPE_01H_Pass2",
            level="00.00",
            fallback_products=("MultiSensor_QPE_01H_Pass1", "RadarOnly_QPE_01H"),
        )
    )

    assert path == Path("fake.grib2")
    assert attempts == [
        ("MultiSensor_QPE_01H_Pass2", "s3"),
        ("MultiSensor_QPE_01H_Pass1", "s3"),
        ("RadarOnly_QPE_01H", "s3"),
    ]


def test_read_data_pre_v12_two_band_selects_discipline_209(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region_job = NoaaMrmsRegionJob.model_construct(
        template_ds=Mock(),
        data_vars=[],
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )

    data = np.ones((2, 2), dtype=np.float32)

    class FakeReader:
        count = 2

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def tags(self, band: int) -> dict[str, str]:
            return {
                1: {"GRIB_DISCIPLINE": "0(Meteorological)"},
                2: {"GRIB_DISCIPLINE": "209(Local)"},
            }[band]

        def read(self, band: int, out_dtype: np.dtype[np.float32]) -> np.ndarray:
            assert out_dtype == np.float32
            assert band == 2
            return data

    monkeypatch.setattr("rasterio.open", lambda *_args, **_kwargs: FakeReader())

    result = region_job.read_data(
        NoaaMrmsSourceFileCoord(
            time=pd.Timestamp("2019-06-15T12:00"),
            product="GaugeCorr_QPE_01H",
            level="00.00",
            fallback_products=(),
            downloaded_path=Path("fake.grib2"),
        ),
        Mock(),
    )

    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize(
    ("region", "expected_processing_region"),
    [
        (slice(0, 100), slice(0, 100)),  # At start: no buffer possible
        (slice(1, 100), slice(0, 100)),  # At index 1: buffer clips to 0
        (slice(2, 100), slice(1, 100)),  # At index 2+: full buffer of 1
        (slice(10, 100), slice(9, 100)),  # Mid-dataset: full buffer of 1
    ],
)
def test_get_processing_region(
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
    region: slice,
    expected_processing_region: slice,
) -> None:
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=region,
        reformat_job_name="test",
    )

    assert region_job.get_processing_region() == expected_processing_region


def test_processing_region_buffered(
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
) -> None:
    template_ds = template_config.get_template(pd.Timestamp("2024-01-15T05:00"))
    test_ds = template_ds.isel(time=slice(0, 5))

    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(2, 5),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()
    assert len(processing_region_ds.time) == len(output_region_ds.time) + 1
    assert processing_region_ds.time[0] == output_region_ds.time[0] - pd.Timedelta("1h")


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template_config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-mrms",
        template_config_version="test-version",
    )

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2024-01-15T06:34")),
    )
    existing_ds = template_config.get_template(pd.Timestamp("2024-01-15T05:01"))
    template_utils.write_metadata(existing_ds, store_factory)

    jobs, template_ds = NoaaMrmsRegionJob.operational_update_jobs(
        primary_store=store_factory.primary_store(),
        tmp_store=tmp_path / "tmp_ds.zarr",
        get_template_fn=template_config.get_template,
        append_dim=template_config.append_dim,
        all_data_vars=template_config.data_vars,
        reformat_job_name="test_job",
    )

    assert template_ds.time.max() == pd.Timestamp("2024-01-15T06:00")
    assert len(jobs) >= 1
    for job in jobs:
        assert isinstance(job, NoaaMrmsRegionJob)


def test_update_template_with_results(
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
) -> None:
    template_ds = template_config.get_template(pd.Timestamp("2024-01-15T05:00"))

    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 5),
        reformat_job_name="test",
    )

    last_time = template_ds.time.values[-1]
    mock_coord = Mock()
    mock_coord.status = SourceFileStatus.Succeeded
    mock_coord.out_loc.return_value = {"time": last_time}

    process_results = {template_config.data_vars[0].name: [mock_coord]}
    result_ds = region_job.update_template_with_results(process_results)

    assert len(result_ds.time) == len(template_ds.time)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("time", "expected_product"),
    [
        (pd.Timestamp("2024-01-15T12:00"), "MultiSensor_QPE_01H_Pass2"),
        (pd.Timestamp("2019-06-15T12:00"), "GaugeCorr_QPE_01H"),
    ],
    ids=["post-v12-s3", "pre-v12-iowa"],
)
def test_download_and_read_precipitation(
    time: pd.Timestamp,
    expected_product: str,
    tmp_path: Path,
) -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    precip_var = next(v for v in config.data_vars if v.name == "precipitation_surface")

    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-mrms-conus-analysis-hourly"}
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=[precip_var],
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NoaaMrmsSourceFileCoord(
        time=time,
        product=expected_product,
        level=precip_var.internal_attrs.mrms_level,
        fallback_products=precip_var.internal_attrs.mrms_fallback_products,
    )

    downloaded_path = region_job.download_file(coord)
    updated_coord = replace(coord, downloaded_path=downloaded_path)

    data = region_job.read_data(updated_coord, precip_var)
    assert data.shape == (3500, 7000)
    assert np.all(np.isfinite(data))


@pytest.mark.slow
def test_download_and_read_radar_only(tmp_path: Path) -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    radar_var = next(
        v for v in config.data_vars if v.name == "precipitation_radar_only_surface"
    )

    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-mrms-conus-analysis-hourly"}
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=[radar_var],
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T12:00"),
        product="RadarOnly_QPE_01H",
        level=radar_var.internal_attrs.mrms_level,
        fallback_products=radar_var.internal_attrs.mrms_fallback_products,
    )

    downloaded_path = region_job.download_file(coord)
    updated_coord = replace(coord, downloaded_path=downloaded_path)

    data = region_job.read_data(updated_coord, radar_var)
    assert data.shape == (3500, 7000)
    assert np.all(np.isfinite(data))


@pytest.mark.slow
def test_download_and_read_precip_flag(tmp_path: Path) -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    ptype_var = next(
        v
        for v in config.data_vars
        if v.name == "categorical_precipitation_type_surface"
    )

    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-mrms-conus-analysis-hourly"}
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=[ptype_var],
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T12:00"),
        product="PrecipFlag",
        level=ptype_var.internal_attrs.mrms_level,
        fallback_products=ptype_var.internal_attrs.mrms_fallback_products,
    )

    downloaded_path = region_job.download_file(coord)
    updated_coord = replace(coord, downloaded_path=downloaded_path)

    data = region_job.read_data(updated_coord, ptype_var)
    assert data.shape == (3500, 7000)
    assert np.all(np.isfinite(data))
