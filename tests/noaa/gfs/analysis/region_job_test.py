from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.common.region_job import SourceFileStatus
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.noaa.gfs.analysis.region_job import (
    NOAA_GFS_INIT_FREQUENCY,
    NoaaGfsAnalysisRegionJob,
    NoaaGfsAnalysisSourceFileCoord,
)
from reformatters.noaa.gfs.analysis.template_config import (
    NoaaGfsAnalysisTemplateConfig,
)
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig


@pytest.fixture
def template_config() -> NoaaGfsAnalysisTemplateConfig:
    return NoaaGfsAnalysisTemplateConfig()


def test_noaa_gfs_init_frequency_equals_forecast_append_dim_frequency() -> None:
    """Test that NOAA_GFS_INIT_FREQUENCY equals NoaaGfsForecastTemplateConfig.append_dim_frequency."""
    assert (
        NOAA_GFS_INIT_FREQUENCY == NoaaGfsForecastTemplateConfig().append_dim_frequency
    )


def test_source_file_coord_out_loc(
    template_config: NoaaGfsAnalysisTemplateConfig,
) -> None:
    """Test output location mapping."""
    coord = NoaaGfsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        data_vars=template_config.data_vars,
    )

    out_loc = coord.out_loc()
    assert out_loc == {
        "time": pd.Timestamp("2024-02-29T00:00"),
    }


def test_source_file_coord_out_loc_with_lead_time(
    template_config: NoaaGfsAnalysisTemplateConfig,
) -> None:
    """Test output location mapping with lead time."""
    coord = NoaaGfsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=3),
        data_vars=template_config.data_vars,
    )

    out_loc = coord.out_loc()
    assert out_loc == {
        "time": pd.Timestamp("2024-02-29T03:00"),
    }


def test_source_file_coord_get_url() -> None:
    """Test URL generation."""
    template_config = NoaaGfsAnalysisTemplateConfig()
    coord = NoaaGfsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01T06:00"),
        lead_time=pd.Timedelta(hours=3),
        data_vars=template_config.data_vars[:1],
    )
    expected = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20250101/06/atmos/gfs.t06z.pgrb2.0p25.f003"
    assert coord.get_url() == expected


@pytest.mark.parametrize(
    ("region", "expected_processing_region"),
    [
        (slice(0, 100), slice(0, 100)),  # At start: no buffer possible, clips to 0
        (slice(1, 100), slice(0, 100)),  # At index 1: buffer clips to 0
        (slice(2, 100), slice(1, 100)),  # At index 2+: full buffer of 1
        (slice(10, 100), slice(9, 100)),  # Mid-dataset: full buffer of 1
    ],
)
def test_get_processing_region(
    template_config: NoaaGfsAnalysisTemplateConfig,
    region: slice,
    expected_processing_region: slice,
) -> None:
    """Test that get_processing_region buffers by 1, clamped to 0 at dataset start."""
    region_job = NoaaGfsAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=region,
        reformat_job_name="test",
    )

    assert region_job.get_processing_region() == expected_processing_region


def test_region_job_processing_region_buffered(
    template_config: NoaaGfsAnalysisTemplateConfig,
) -> None:
    """Test that processing region is buffered by 1 step for deaccumulation (not at dataset start)."""
    template_ds = template_config.get_template(pd.Timestamp("2021-05-01T10:00"))

    test_ds = template_ds.isel(time=slice(0, 10))

    # Region starting at index 5 (not the start of the dataset)
    region_job = NoaaGfsAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(5, 10),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()
    # Processing region is buffered by 1 for deaccumulation
    assert len(processing_region_ds.time) == len(output_region_ds.time) + 1
    assert processing_region_ds.time[0] == output_region_ds.time[0] - pd.Timedelta("1h")


def test_region_job_generate_source_file_coords_hour_0_vars(
    template_config: NoaaGfsAnalysisTemplateConfig,
) -> None:
    """Test that instant (hour 0) variables use lead_time matching time % 6h."""
    template_ds = template_config.get_template(pd.Timestamp("2021-05-01T06:00"))

    test_ds = template_ds.isel(time=slice(0, 6))

    instant_vars = [
        v for v in template_config.data_vars if v.attrs.step_type == "instant"
    ][:1]
    assert len(instant_vars) == 1

    region_job = NoaaGfsAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=instant_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 6),
        reformat_job_name="test",
    )

    processing_region_ds, _output_region_ds = region_job._get_region_datasets()

    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, instant_vars
    )

    assert len(source_coords) == 6

    for coord in source_coords:
        assert isinstance(coord, NoaaGfsAnalysisSourceFileCoord)

    # For instant variables: init_times = times.floor("6h")
    # times: 00, 01, 02, 03, 04, 05
    # init_times: 00, 00, 00, 00, 00, 00
    # lead_times: 0h, 1h, 2h, 3h, 4h, 5h
    expected = [
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("0h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("1h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("2h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("3h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("4h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("5h")),
    ]

    for coord, (expected_init, expected_lead) in zip(
        source_coords, expected, strict=True
    ):
        assert coord.init_time == expected_init
        assert coord.lead_time == expected_lead


def test_region_job_generate_source_file_coords_non_hour_0_vars(
    template_config: NoaaGfsAnalysisTemplateConfig,
) -> None:
    """Test that non-instant (non-hour 0) variables use shifted init times."""
    template_ds = template_config.get_template(pd.Timestamp("2021-05-01T06:00"))

    test_ds = template_ds.isel(time=slice(0, 6))

    # Get avg/accumulated variables that don't have hour 0 values
    avg_vars = [v for v in template_config.data_vars if v.attrs.step_type == "avg"][:1]
    assert len(avg_vars) == 1

    region_job = NoaaGfsAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=avg_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 6),
        reformat_job_name="test",
    )

    processing_region_ds, _output_region_ds = region_job._get_region_datasets()

    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, avg_vars
    )

    assert len(source_coords) == 6

    for coord in source_coords:
        assert isinstance(coord, NoaaGfsAnalysisSourceFileCoord)

    # For non-instant variables: init_times = (times - 1h).floor("6h")
    # times: 00, 01, 02, 03, 04, 05
    # times - 1h: 23 (prev day), 00, 01, 02, 03, 04
    # init_times: 18 (prev day), 00, 00, 00, 00, 00
    # lead_times: 6h, 1h, 2h, 3h, 4h, 5h
    expected = [
        (pd.Timestamp("2021-04-30T18:00"), pd.Timedelta("6h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("1h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("2h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("3h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("4h")),
        (pd.Timestamp("2021-05-01T00:00"), pd.Timedelta("5h")),
    ]

    for coord, (expected_init, expected_lead) in zip(
        source_coords, expected, strict=True
    ):
        assert coord.init_time == expected_init
        assert coord.lead_time == expected_lead
        expected_time = coord.init_time + coord.lead_time
        assert coord.out_loc()["time"] == expected_time


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    template_config = NoaaGfsAnalysisTemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset-A",
        template_config_version="test-version",
    )

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2021-05-01T10:34")),
    )
    existing_ds = template_config.get_template(pd.Timestamp("2021-05-01T06:01"))
    template_utils.write_metadata(existing_ds, store_factory)

    jobs, template_ds = NoaaGfsAnalysisRegionJob.operational_update_jobs(
        primary_store=store_factory.primary_store(),
        tmp_store=tmp_path / "tmp_ds.zarr",
        get_template_fn=template_config.get_template,
        append_dim=template_config.append_dim,
        all_data_vars=template_config.data_vars,
        reformat_job_name="test_job",
    )

    assert template_ds.time.max() == pd.Timestamp("2021-05-01T10:00")

    assert len(jobs) == 1
    for job in jobs:
        assert isinstance(job, NoaaGfsAnalysisRegionJob)
        assert job.data_vars == template_config.data_vars


def test_update_template_with_results(
    template_config: NoaaGfsAnalysisTemplateConfig,
) -> None:
    """Test that update_template_with_results removes the last hour of data."""
    template_ds = template_config.get_template(pd.Timestamp("2021-05-01T06:00"))

    region_job = NoaaGfsAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 6),
        reformat_job_name="test",
    )

    # Create mock process_results that simulates successful processing up to the last time
    last_time = template_ds.time.values[-1]
    mock_coord = Mock()
    mock_coord.status = SourceFileStatus.Succeeded
    mock_coord.out_loc.return_value = {"time": last_time}

    process_results = {template_config.data_vars[0].name: [mock_coord]}

    result_ds = region_job.update_template_with_results(process_results)

    # Result should have one fewer time step than the template
    assert len(result_ds.time) == len(template_ds.time) - 1
    assert result_ds.time[-1] == template_ds.time.values[-2]
    assert result_ds.time[0] == template_ds.time.values[0]
