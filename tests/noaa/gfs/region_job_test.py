from collections.abc import Mapping
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from obstore.exceptions import GenericError

from reformatters.common import template_utils
from reformatters.common.pydantic import replace
from reformatters.common.region_job import CoordinateValueOrRange
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.types import Dim
from reformatters.noaa.gfs.forecast.region_job import (
    NoaaGfsForecastRegionJob,
    NoaaGfsForecastSourceFileCoord,
)
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig
from reformatters.noaa.gfs.region_job import (
    NoaaGfsCommonRegionJob,
    NoaaGfsSourceFileCoord,
)
from reformatters.noaa.noaa_utils import has_hour_0_values


class ConcreteSourceFileCoord(NoaaGfsSourceFileCoord):
    """Concrete implementation of NoaaGfsSourceFileCoord for testing."""

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


def test_source_file_coord_get_url() -> None:
    """Test that NoaaGfsSourceFileCoord.get_url() generates correct URLs."""
    coord = ConcreteSourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta(hours=0),
        data_vars=NoaaGfsForecastTemplateConfig().data_vars,
    )
    expected = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20000101/00/atmos/gfs.t00z.pgrb2.0p25.f000"
    assert coord.get_url() == expected


def test_source_file_coord_get_url_with_lead_time() -> None:
    """Test URL generation with different lead times."""
    coord = ConcreteSourceFileCoord(
        init_time=pd.Timestamp("2025-01-15T12:00"),
        lead_time=pd.Timedelta(hours=120),
        data_vars=NoaaGfsForecastTemplateConfig().data_vars[:1],
    )
    expected = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20250115/12/atmos/gfs.t12z.pgrb2.0p25.f120"
    assert coord.get_url() == expected


def test_source_file_coord_out_loc_not_implemented() -> None:
    """Test that base NoaaGfsSourceFileCoord.out_loc raises NotImplementedError."""
    coord = NoaaGfsSourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta(hours=0),
        data_vars=NoaaGfsForecastTemplateConfig().data_vars,
    )
    with pytest.raises(NotImplementedError):
        coord.out_loc()


def test_common_region_job_source_groups() -> None:
    """Test that source_groups correctly groups variables by has_hour_0_values."""
    template_config = NoaaGfsForecastTemplateConfig()
    data_vars = template_config.data_vars

    groups = NoaaGfsCommonRegionJob.source_groups(data_vars)

    # Should have 2 groups: instant variables (hour 0) and non-instant variables (no hour 0)
    assert len(groups) == 2

    # All variables in each group should have the same has_hour_0_values status
    for group in groups:
        group_has_hour_0 = {has_hour_0_values(v) for v in group}
        assert len(group_has_hour_0) == 1


def test_common_region_job_download_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test download_file method of common region job."""
    template_config = NoaaGfsForecastTemplateConfig()

    # Create a region job with mock stores
    region_job = NoaaGfsForecastRegionJob.model_construct(
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

    # Mock the http_download_to_disk function
    mock_download = Mock()
    mock_index_path = Mock()
    mock_index_path.read_text.return_value = "ignored"
    mock_data_path = Mock()

    def mock_download_side_effect(url: str, dataset_id: str, **kwargs: object) -> Mock:
        if url.endswith(".idx"):
            return mock_index_path
        else:
            return mock_data_path

    mock_download.side_effect = mock_download_side_effect
    monkeypatch.setattr(
        "reformatters.noaa.gfs.region_job.http_download_to_disk",
        mock_download,
    )

    # Mock parse_grib_index to return some byte ranges
    mock_parse = Mock(return_value=([123456, 234567], [234566, 345678]))
    monkeypatch.setattr(
        "reformatters.noaa.gfs.region_job.grib_message_byte_ranges_from_index",
        mock_parse,
    )

    # Call the common download_file method directly on the parent class
    # to test the common implementation (not the forecast-specific override)
    result = NoaaGfsCommonRegionJob.download_file(region_job, coord)

    # Verify the result
    assert result == mock_data_path

    # Verify http_download_to_disk was called correctly
    assert mock_download.call_count == 2

    # First call should be for the index file
    first_call = mock_download.call_args_list[0]
    assert first_call[0][0].endswith(".idx")

    # Second call should be for the data file with byte ranges
    second_call = mock_download.call_args_list[1]
    assert not second_call[0][0].endswith(".idx")
    assert second_call[1]["byte_ranges"] == ([123456, 234567], [234566, 345678])


def test_common_region_job_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test operational_update_jobs class method."""
    template_config = NoaaGfsForecastTemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset-common",
        template_config_version="test-version",
    )

    # Set the append_dim_end for the update
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2022-01-01T12:34")),
    )
    # Set the append_dim_start for the update
    existing_ds = template_config.get_template(
        pd.Timestamp("2022-01-01T06:01")  # 06 will be max existing init time
    )
    template_utils.write_metadata(existing_ds, store_factory)

    # Test through the forecast subclass
    jobs, template_ds = NoaaGfsForecastRegionJob.operational_update_jobs(
        primary_store=store_factory.primary_store(),
        tmp_store=tmp_path / "tmp_ds.zarr",
        get_template_fn=template_config.get_template,
        append_dim=template_config.append_dim,
        all_data_vars=template_config.data_vars,
        reformat_job_name="test_job",
    )

    assert template_ds.init_time.max() == pd.Timestamp("2022-01-01T12:00")

    assert len(jobs) == 2  # 06 and 12 UTC init times
    for job in jobs:
        assert isinstance(job, NoaaGfsForecastRegionJob)
        assert job.data_vars == template_config.data_vars


def test_source_file_coord_get_url_nomads() -> None:
    """Test that get_url(source="nomads") returns a NOMADS URL with the same path as S3."""
    coord = ConcreteSourceFileCoord(
        init_time=pd.Timestamp("2025-06-15T12:00"),
        lead_time=pd.Timedelta(hours=24),
        data_vars=NoaaGfsForecastTemplateConfig().data_vars[:1],
    )
    assert (
        coord.get_url()
        == "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20250615/12/atmos/gfs.t12z.pgrb2.0p25.f024"
    )
    assert (
        coord.get_url(source="s3")
        == "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20250615/12/atmos/gfs.t12z.pgrb2.0p25.f024"
    )
    assert (
        coord.get_url(source="nomads")
        == "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20250615/12/atmos/gfs.t12z.pgrb2.0p25.f024"
    )


def test_source_file_coord_get_idx_url_nomads() -> None:
    """Test that get_idx_url(source="nomads") returns the NOMADS index URL."""
    coord = ConcreteSourceFileCoord(
        init_time=pd.Timestamp("2025-06-15T06:00"),
        lead_time=pd.Timedelta(hours=3),
        data_vars=NoaaGfsForecastTemplateConfig().data_vars[:1],
    )
    assert (
        coord.get_idx_url(source="nomads")
        == "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20250615/06/atmos/gfs.t06z.pgrb2.0p25.f003.idx"
    )
    assert (
        coord.get_idx_url(source="s3")
        == "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20250615/06/atmos/gfs.t06z.pgrb2.0p25.f003.idx"
    )


def _make_gfs_region_job(
    template_config: NoaaGfsForecastTemplateConfig,
    now: pd.Timestamp,
) -> NoaaGfsForecastRegionJob:
    return NoaaGfsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_config.get_template(now),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )


def test_get_download_source(monkeypatch: pytest.MonkeyPatch) -> None:
    template_config = NoaaGfsForecastTemplateConfig()
    fixed_now = pd.Timestamp("2025-01-15T12:00")
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: fixed_now)
    )
    region_job = _make_gfs_region_job(template_config, fixed_now)

    assert region_job.get_download_source(fixed_now - pd.Timedelta(hours=6)) == "nomads"
    assert region_job.get_download_source(fixed_now - pd.Timedelta(hours=24)) == "s3"


def test_download_file_uses_nomads_for_recent_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_config = NoaaGfsForecastTemplateConfig()
    region_job = _make_gfs_region_job(template_config, pd.Timestamp("2025-01-15T12:00"))
    coord = NoaaGfsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta(hours=6),
        data_vars=template_config.data_vars[:1],
    )
    monkeypatch.setattr(
        NoaaGfsForecastRegionJob, "get_download_source", lambda self, t: "nomads"
    )
    mock_download = Mock(return_value=Mock())
    monkeypatch.setattr(
        "reformatters.noaa.gfs.region_job.http_download_to_disk", mock_download
    )
    monkeypatch.setattr(
        "reformatters.noaa.gfs.region_job.grib_message_byte_ranges_from_index",
        Mock(return_value=([0], [100])),
    )

    region_job.download_file(coord)

    urls_called = [call.args[0] for call in mock_download.call_args_list]
    assert all(url.startswith("https://nomads.ncep.noaa.gov") for url in urls_called)


def test_download_file_falls_back_to_s3_on_generic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that download_file falls back to S3 when NOMADS raises GenericError (bare redirect)."""
    template_config = NoaaGfsForecastTemplateConfig()
    region_job = _make_gfs_region_job(template_config, pd.Timestamp("2025-01-15T12:00"))
    coord = NoaaGfsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta(hours=6),
        data_vars=template_config.data_vars[:1],
    )
    monkeypatch.setattr(
        NoaaGfsForecastRegionJob, "get_download_source", lambda self, t: "nomads"
    )
    # Disable retry sleep for fast tests
    monkeypatch.setattr("reformatters.common.retry.time.sleep", lambda _: None)

    s3_result = Mock()
    nomads_call_count = 0

    def mock_download_from_source(
        self: NoaaGfsCommonRegionJob,
        coord: NoaaGfsForecastSourceFileCoord,
        source: str,
    ) -> Path:
        nonlocal nomads_call_count
        if source == "nomads":
            nomads_call_count += 1
            raise GenericError("Received redirect without LOCATION")
        return s3_result

    monkeypatch.setattr(
        NoaaGfsCommonRegionJob, "_download_from_source", mock_download_from_source
    )

    result = region_job.download_file(coord)

    assert result == s3_result
    # Should have retried NOMADS 4 times before falling back
    assert nomads_call_count == 4


def test_download_file_falls_back_to_s3_on_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that download_file falls back to S3 when NOMADS raises FileNotFoundError."""
    template_config = NoaaGfsForecastTemplateConfig()
    region_job = _make_gfs_region_job(template_config, pd.Timestamp("2025-01-15T12:00"))
    coord = NoaaGfsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta(hours=6),
        data_vars=template_config.data_vars[:1],
    )
    monkeypatch.setattr(
        NoaaGfsForecastRegionJob, "get_download_source", lambda self, t: "nomads"
    )

    s3_result = Mock()

    def mock_download_from_source(
        self: NoaaGfsCommonRegionJob,
        coord: NoaaGfsForecastSourceFileCoord,
        source: str,
    ) -> Path:
        if source == "nomads":
            raise FileNotFoundError("Not found on NOMADS")
        return s3_result

    monkeypatch.setattr(
        NoaaGfsCommonRegionJob, "_download_from_source", mock_download_from_source
    )

    result = region_job.download_file(coord)
    assert result == s3_result


def test_download_file_skips_nomads_for_old_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that download_file goes directly to S3 for old data."""
    template_config = NoaaGfsForecastTemplateConfig()
    region_job = _make_gfs_region_job(template_config, pd.Timestamp("2025-01-15T12:00"))
    coord = NoaaGfsForecastSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta(hours=6),
        data_vars=template_config.data_vars[:1],
    )
    monkeypatch.setattr(
        NoaaGfsForecastRegionJob, "get_download_source", lambda self, t: "s3"
    )

    s3_result = Mock()

    def mock_download_from_source(
        self: NoaaGfsCommonRegionJob,
        coord: NoaaGfsForecastSourceFileCoord,
        source: str,
    ) -> Path:
        assert source == "s3"
        return s3_result

    monkeypatch.setattr(
        NoaaGfsCommonRegionJob, "_download_from_source", mock_download_from_source
    )

    result = region_job.download_file(coord)
    assert result == s3_result


@pytest.mark.slow
def test_download_file_from_nomads_gfs() -> None:
    """Download a recent GFS init time from NOMADS and read all template variables."""
    template_config = NoaaGfsForecastTemplateConfig()
    # 6h-old init time is within the 18h NOMADS window and typically complete
    init_time = (pd.Timestamp.now() - pd.Timedelta(hours=6)).floor("6h")

    region_job = NoaaGfsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_config.get_template(pd.Timestamp.now()),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    # lead_time=6h: all vars (instant and accumulated) are present in the f006 GRIB file
    lead_time = pd.Timedelta(hours=6)
    for group in NoaaGfsForecastRegionJob.source_groups(template_config.data_vars):
        coord = NoaaGfsForecastSourceFileCoord(
            init_time=init_time,
            lead_time=lead_time,
            data_vars=group,
        )
        coord = replace(coord, downloaded_path=region_job.download_file(coord))

        for data_var in group:
            data = region_job.read_data(coord, data_var)
            assert np.all(np.isfinite(data)), f"Non-finite values for {data_var.name}"
