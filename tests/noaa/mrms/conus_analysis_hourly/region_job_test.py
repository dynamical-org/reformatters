from pathlib import Path
from typing import Self
from unittest.mock import Mock

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from zarr.codecs import BloscCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer
from zarr.storage import LocalStore

from reformatters.common import template_utils
from reformatters.common.pydantic import replace
from reformatters.common.region_job import SourceFileResult, SourceFileStatus
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.types import ArrayFloat32
from reformatters.common.zarr import copy_zarr_metadata
from reformatters.noaa.mrms.conus_analysis_hourly.region_job import (
    NoaaMrmsRegionJob,
    NoaaMrmsSourceFileCoord,
)
from reformatters.noaa.mrms.conus_analysis_hourly.template_config import (
    NoaaMrmsConusAnalysisHourlyTemplateConfig,
    NoaaMrmsDataVar,
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
        data_var_name="precipitation_pass_2_surface",
    )
    assert coord.out_loc() == {"time": pd.Timestamp("2024-01-15T12:00")}


def test_source_file_coord_get_url_s3() -> None:
    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T12:00"),
        product="MultiSensor_QPE_01H_Pass2",
        level="00.00",
        fallback_products=(),
        data_var_name="precipitation_pass_2_surface",
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
        data_var_name="precipitation_surface",
    )
    url = coord.get_url(source="iowa")
    assert url == (
        "https://mtarchive.geol.iastate.edu/2019/06/15/mrms/ncep/"
        "GaugeCorr_QPE_01H/"
        "GaugeCorr_QPE_01H_00.00_20190615-120000.grib2.gz"
    )


@pytest.mark.parametrize(
    ("product", "directory"),
    [
        ("MultiSensor_QPE_01H_Pass2", "MultiSensor_QPE_01H_Pass2"),
        ("FLASH_QPE_FFGMAX", "FLASH/QPE_FFGMAX"),
    ],
)
def test_source_file_coord_get_url_ncep(product: str, directory: str) -> None:
    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T23:00"),
        product=product,
        level="00.00",
        fallback_products=(),
        data_var_name="precipitation_pass_2_surface",
    )
    url = coord.get_url(source="ncep")
    assert url == (
        "https://mrms.ncep.noaa.gov/2D/"
        f"{directory}/"
        f"MRMS_{product}_00.00_20240115-230000.grib2.gz"
    )


def test_source_groups(
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
) -> None:
    groups = NoaaMrmsRegionJob.source_file_var_groups(template_config.data_vars)
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
            data_var_name="precipitation_surface",
        )
    )

    assert path == Path("fake.grib2")
    assert attempts == [
        ("MultiSensor_QPE_01H_Pass2", "s3"),
        ("MultiSensor_QPE_01H_Pass1", "s3"),
        ("RadarOnly_QPE_01H", "s3"),
    ]


def test_download_from_source_uses_var_name_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-mrms-conus-analysis-hourly"}
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=[],
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )

    captured: dict[str, str] = {}
    fake_gz = tmp_path / "fake.grib2.gz_precipitation_pass_2_surface"
    fake_gz.touch()

    def fake_http_download(
        url: str, dataset_id: str, local_path_suffix: str = ""
    ) -> Path:
        captured["local_path_suffix"] = local_path_suffix
        return fake_gz

    def fake_decompress(gz_path: Path, local_path_suffix: str = "") -> Path:
        return tmp_path / "fake.grib2_precipitation_pass_2_surface"

    monkeypatch.setattr(
        "reformatters.noaa.mrms.conus_analysis_hourly.region_job.http_download_to_disk",
        fake_http_download,
    )
    monkeypatch.setattr(
        "reformatters.noaa.mrms.conus_analysis_hourly.region_job._decompress_gzip",
        fake_decompress,
    )

    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T12:00"),
        product="MultiSensor_QPE_01H_Pass2",
        level="00.00",
        fallback_products=(),
        data_var_name="precipitation_pass_2_surface",
    )
    region_job._download_from_source(coord, source="s3")

    assert captured["local_path_suffix"] == "_precipitation_pass_2_surface"


def test_read_data_pre_v12_two_band_selects_discipline_209(
    monkeypatch: pytest.MonkeyPatch,
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
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
    monkeypatch.setattr(
        "reformatters.noaa.mrms.conus_analysis_hourly.region_job.grib_decimal_scale_factors",
        lambda _path: [1, 1],
    )

    precipitation_surface = next(
        v for v in template_config.data_vars if v.name == "precipitation_surface"
    )
    result = region_job.read_data(
        NoaaMrmsSourceFileCoord(
            time=pd.Timestamp("2019-06-15T12:00"),
            product="GaugeCorr_QPE_01H",
            level="00.00",
            fallback_products=(),
            data_var_name="precipitation_surface",
            downloaded_path=Path("fake.grib2"),
        ),
        precipitation_surface,
    )

    np.testing.assert_array_equal(result, data)


def test_read_data_quantizes_to_grib_decimal_precision(
    monkeypatch: pytest.MonkeyPatch,
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
) -> None:
    region_job = NoaaMrmsRegionJob.model_construct(
        template_ds=Mock(),
        data_vars=[],
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )

    # Values as GDAL's float32 unpacking produces them on arm64 (FMA contraction):
    # a packed zero decodes to 4.4703484e-8 and 0.2 mm to 0.2 + ~5e-8.
    data = np.array(
        [[4.4703484e-08, 0.20000004768371582], [-3.0, 12.3]], dtype=np.float32
    )

    class FakeReader:
        count = 1

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, band: int, out_dtype: np.dtype[np.float32]) -> np.ndarray:
            assert out_dtype == np.float32
            assert band == 1
            return data

    monkeypatch.setattr("rasterio.open", lambda *_args, **_kwargs: FakeReader())
    monkeypatch.setattr(
        "reformatters.noaa.mrms.conus_analysis_hourly.region_job.grib_decimal_scale_factors",
        lambda _path: [1],
    )

    result = region_job.read_data(
        NoaaMrmsSourceFileCoord(
            time=pd.Timestamp("2024-01-15T12:00"),
            product="MultiSensor_QPE_01H_Pass2",
            level="00.00",
            fallback_products=(),
            data_var_name="precipitation_surface",
            downloaded_path=Path("fake.grib2"),
        ),
        next(v for v in template_config.data_vars if v.name == "precipitation_surface"),
    )

    assert result.dtype == np.float32
    expected = np.array([[0.0, 0.2], [-3.0, 12.3]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


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
        assert job.existing_time_end == pd.Timestamp("2024-01-15T05:00")


@pytest.fixture
def preservation_template(
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig,
) -> tuple[xr.DataTree, list[NoaaMrmsDataVar]]:
    names = ["flash_qpe_ffg_max_surface", "precipitation_surface"]
    ds = (
        template_config.get_template(pd.Timestamp("2026-10-07T20:00"))
        .sel(time=slice("2026-10-07T14:00", "2026-10-07T19:00"))
        .isel(latitude=slice(0, 2), longitude=slice(0, 2))
        .to_dataset()[names]
    )
    data_vars = []
    for name in names:
        data_var = next(v for v in template_config.data_vars if v.name == name)
        encoding = replace(data_var.encoding, chunks=(4, 2, 2), shards=(8, 2, 2))
        ds[name].encoding.update(chunks=encoding.chunks, shards=encoding.shards)
        data_vars.append(replace(data_var, encoding=encoding))
    return xr.DataTree.from_dict({"/": ds}), data_vars


@pytest.fixture
def preservation_repo(
    preservation_template: tuple[xr.DataTree, list[NoaaMrmsDataVar]],
    tmp_path: Path,
) -> icechunk.Repository:
    template_ds, data_vars = preservation_template
    existing_ds = template_ds.isel(time=slice(0, 4)).to_dataset().compute()
    old_flash = np.full((4, 2, 2), 165, dtype=np.float32)
    old_flash[:, 0, 0] = np.nan
    old_precip = np.full((4, 2, 2), 0.125, dtype=np.float32)
    existing_ds[data_vars[0].name].data = old_flash
    existing_ds[data_vars[1].name].data = old_precip
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo"))
    )
    initial_session = repo.writable_session("main")
    with template_utils.ignore_consolidated_metadata_spec_warning():
        existing_ds.to_zarr(initial_session.store, consolidated=True)
    initial_snapshot = initial_session.commit("Published data")
    repo.create_branch("update", initial_snapshot)
    return repo


@pytest.mark.parametrize("operational", [True, False], ids=["update", "backfill"])
@pytest.mark.parametrize("all_downloads_fail", [True, False], ids=["outage", "mixed"])
def test_preserve_published_data_on_download_failure(
    preservation_template: tuple[xr.DataTree, list[NoaaMrmsDataVar]],
    preservation_repo: icechunk.Repository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operational: bool,
    all_downloads_fail: bool,
) -> None:
    template_ds, data_vars = preservation_template
    repo = preservation_repo
    existing_ds = xr.open_zarr(repo.readonly_session("main").store, chunks=None)
    existing_time_end = pd.Timestamp(existing_ds.time.values[-1])

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2026-10-07T20:00")),
    )
    if operational:
        jobs, _ = NoaaMrmsRegionJob.operational_update_jobs(
            primary_store=repo.readonly_session("main").store,
            tmp_store=tmp_path / "tmp.zarr",
            get_template_fn=lambda end: template_ds,
            append_dim="time",
            all_data_vars=data_vars,
            reformat_job_name="test-update",
        )
    else:
        jobs = NoaaMrmsRegionJob.get_jobs(
            tmp_store=tmp_path / "tmp.zarr",
            template_ds=template_ds,
            append_dim="time",
            all_data_vars=data_vars,
            reformat_job_name="test-backfill",
        )
    assert len(jobs) == 1
    job = jobs[0]
    assert isinstance(job, NoaaMrmsRegionJob)
    assert job.existing_time_end == (existing_time_end if operational else None)

    template_utils.write_metadata(template_ds, job.tmp_store)
    expansion_session = repo.writable_session("update")
    copy_zarr_metadata(template_ds, job.tmp_store, expansion_session.store)
    expansion_session.commit("Expanded job-branch metadata")
    update_session = repo.writable_session("update")

    def download_file(self: NoaaMrmsRegionJob, coord: NoaaMrmsSourceFileCoord) -> Path:
        if all_downloads_fail or coord.time.hour not in (16, 17):
            raise FileNotFoundError(coord.get_url())
        return tmp_path / "unused-source"

    def read_data(
        self: NoaaMrmsRegionJob,
        coord: NoaaMrmsSourceFileCoord,
        data_var: NoaaMrmsDataVar,
    ) -> ArrayFloat32:
        if coord.time.hour == 17:
            return np.full((2, 2), np.nan, dtype=np.float32)
        value = 7200 if data_var.name == "precipitation_surface" else 30
        return np.full((2, 2), value, dtype=np.float32)

    monkeypatch.setattr(NoaaMrmsRegionJob, "download_file", download_file)
    monkeypatch.setattr(NoaaMrmsRegionJob, "read_data", read_data)
    results = job.process(primary_store=update_session.store, replica_stores=[])
    update_session.commit("Replayed region")
    updated_ds = xr.open_zarr(repo.readonly_session("update").store, chunks=None)

    for data_var in data_vars:
        expected = np.full((6, 2, 2), np.nan, dtype=np.float32)
        if operational:
            expected[:4] = existing_ds[data_var.name].values
        if not all_downloads_fail:
            expected[2] = 2 if data_var.name == "precipitation_surface" else 30
            expected[3] = np.nan
        np.testing.assert_array_equal(updated_ds[data_var.name].values, expected)
        coords = results[data_var.path]
        assert all(c.status == SourceFileStatus.Succeeded for c in coords)
        expected_hours = [14, 15, 16, 17] if operational else []
        if not operational and not all_downloads_fail:
            expected_hours = [16, 17]
        assert [c.time.hour for c in coords] == expected_hours

    if operational:
        process_results = {
            name: [
                SourceFileResult(status=c.status, out_loc=c.out_loc(), url=c.get_url())
                for c in coords
            ]
            for name, coords in results.items()
        }
        trimmed = job.update_template_with_results(process_results)
        np.testing.assert_array_equal(trimmed.time.values, existing_ds.time.values)

    published = xr.open_zarr(repo.readonly_session("main").store, chunks=None)
    xr.testing.assert_equal(published, existing_ds)


def test_preservation_batches_missing_times_and_uses_coordinate_intersection(
    preservation_template: tuple[xr.DataTree, list[NoaaMrmsDataVar]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_ds, data_vars = preservation_template
    data_var = data_vars[0]
    times = pd.date_range("2026-10-07T14:00", periods=50, freq="h")
    ds = template_ds.to_dataset()[[data_var.name]].reindex(time=times)
    ds[data_var.name].encoding.update(chunks=(648, 1, 1), shards=(648, 1, 1))
    template_ds = xr.DataTree.from_dict({"/": ds})
    existing_ds = ds.isel(time=[i for i in range(50) if i != 1]).compute()
    spatial_values = np.arange(4, dtype=np.float32).reshape(2, 2) + 165
    existing_ds[data_var.name].data[:] = spatial_values
    store = tmp_path / "existing.zarr"
    with template_utils.ignore_consolidated_metadata_spec_warning():
        existing_ds.to_zarr(store, consolidated=True)
    primary_store = LocalStore(store)
    job = NoaaMrmsRegionJob(
        tmp_store=tmp_path / "tmp.zarr",
        template_ds=template_ds,
        data_vars=[data_var],
        append_dim="time",
        region=slice(1, 50),
        reformat_job_name="test-update",
        existing_time_end=times[44],
    )
    data_array = template_ds.to_dataset()[data_var.name].compute()
    coords = [
        replace(coord, status=SourceFileStatus.DownloadFailed)
        for coord in job.generate_source_file_coords(
            template_ds.to_dataset(), [data_var]
        )
    ]
    decoded_time_chunks = []
    original_decode = BloscCodec._decode_single

    async def decode_single(
        self: BloscCodec, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        if chunk_spec.shape == (648, 1, 1):
            decoded_time_chunks.append(chunk_spec.shape)
        return await original_decode(self, chunk_bytes, chunk_spec)

    monkeypatch.setattr(BloscCodec, "_decode_single", decode_single)
    results = job.prepare_data_array_for_write(
        data_array, data_var, coords, primary_store
    )

    assert len(decoded_time_chunks) == 4
    reused_times = [c.time for c in results if c.status == SourceFileStatus.Succeeded]
    assert reused_times == list(times[2:45])
    assert len(reused_times) == 43
    expected = np.full((50, 2, 2), np.nan, dtype=np.float32)
    expected[2:45] = spatial_values
    np.testing.assert_array_equal(data_array.values, expected)


def test_preservation_does_not_restore_read_failures(
    preservation_template: tuple[xr.DataTree, list[NoaaMrmsDataVar]],
    tmp_path: Path,
) -> None:
    template_ds, data_vars = preservation_template
    data_var = data_vars[0]
    job = NoaaMrmsRegionJob(
        tmp_store=tmp_path,
        template_ds=template_ds,
        data_vars=[data_var],
        append_dim="time",
        region=slice(0, 6),
        reformat_job_name="test-update",
        existing_time_end=pd.Timestamp("2026-10-07T19:00"),
    )
    data_array = template_ds.to_dataset()[data_var.name].compute()
    coords = [
        replace(coord, status=SourceFileStatus.ReadFailed)
        for coord in job.generate_source_file_coords(
            template_ds.to_dataset(), [data_var]
        )
    ]
    primary_store = Mock()
    results = job.prepare_data_array_for_write(
        data_array, data_var, coords, primary_store
    )
    assert np.isnan(data_array.values).all()
    assert all(c.status == SourceFileStatus.ReadFailed for c in results)
    assert primary_store.mock_calls == []


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
    result = SourceFileResult(
        status=SourceFileStatus.Succeeded,
        out_loc={"time": pd.Timestamp(last_time)},
        url="https://test/",
    )

    process_results = {template_config.data_vars[0].name: [result]}
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
        data_var_name=precip_var.name,
    )

    downloaded_path = region_job.download_file(coord)
    updated_coord = replace(coord, downloaded_path=downloaded_path)

    data = region_job.read_data(updated_coord, precip_var)
    assert data.shape == (3500, 7000)
    assert np.all(np.isfinite(data))


@pytest.mark.slow
def test_download_and_read_pass_1(tmp_path: Path) -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    pass1_var = next(
        v for v in config.data_vars if v.name == "precipitation_pass_1_surface"
    )

    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-mrms-conus-analysis-hourly"}
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=[pass1_var],
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T12:00"),
        product=pass1_var.internal_attrs.mrms_product,
        level=pass1_var.internal_attrs.mrms_level,
        fallback_products=pass1_var.internal_attrs.mrms_fallback_products,
        data_var_name=pass1_var.name,
    )

    downloaded_path = region_job.download_file(coord)
    updated_coord = replace(coord, downloaded_path=downloaded_path)

    data = region_job.read_data(updated_coord, pass1_var)
    assert data.shape == (3500, 7000)
    assert np.all(np.isfinite(data))


@pytest.mark.slow
def test_download_and_read_pass_2(tmp_path: Path) -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    pass2_var = next(
        v for v in config.data_vars if v.name == "precipitation_pass_2_surface"
    )

    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-mrms-conus-analysis-hourly"}
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=[pass2_var],
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T12:00"),
        product=pass2_var.internal_attrs.mrms_product,
        level=pass2_var.internal_attrs.mrms_level,
        fallback_products=pass2_var.internal_attrs.mrms_fallback_products,
        data_var_name=pass2_var.name,
    )

    downloaded_path = region_job.download_file(coord)
    updated_coord = replace(coord, downloaded_path=downloaded_path)

    data = region_job.read_data(updated_coord, pass2_var)
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
        data_var_name=radar_var.name,
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
        data_var_name=ptype_var.name,
    )

    downloaded_path = region_job.download_file(coord)
    updated_coord = replace(coord, downloaded_path=downloaded_path)

    data = region_job.read_data(updated_coord, ptype_var)
    assert data.shape == (3500, 7000)
    assert np.all(np.isfinite(data))


@pytest.mark.slow
def test_download_and_read_flash_qpe_ffg_max(tmp_path: Path) -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    flash_var = next(
        v for v in config.data_vars if v.name == "flash_qpe_ffg_max_surface"
    )

    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-mrms-conus-analysis-hourly"}
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=[flash_var],
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NoaaMrmsSourceFileCoord(
        time=pd.Timestamp("2024-01-15T12:00"),
        product="FLASH_QPE_FFGMAX",
        level=flash_var.internal_attrs.mrms_level,
        fallback_products=flash_var.internal_attrs.mrms_fallback_products,
        data_var_name=flash_var.name,
    )

    downloaded_path = region_job.download_file(coord)
    updated_coord = replace(coord, downloaded_path=downloaded_path)

    data = region_job.read_data(updated_coord, flash_var)
    assert data.shape == (3500, 7000)
    # Source uses -999 as nodata sentinel, read_data replaces with NaN.
    # ~64% of pixels are outside radar/FFG coverage.
    nan_fraction = np.isnan(data).sum() / data.size
    assert 0.55 < nan_fraction < 0.75
    assert np.all(np.isfinite(data[~np.isnan(data)]))
