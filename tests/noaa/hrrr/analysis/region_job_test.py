from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.common.pydantic import replace
from reformatters.common.region_job import SourceFileStatus
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.noaa.hrrr.analysis.region_job import (
    NoaaHrrrAnalysisRegionJob,
    NoaaHrrrAnalysisSourceFileCoord,
)
from reformatters.noaa.hrrr.analysis.template_config import (
    NoaaHrrrAnalysisTemplateConfig,
)
from reformatters.noaa.noaa_utils import has_hour_0_values


@pytest.fixture
def template_config() -> NoaaHrrrAnalysisTemplateConfig:
    return NoaaHrrrAnalysisTemplateConfig()


def test_source_file_coord_out_loc(
    template_config: NoaaHrrrAnalysisTemplateConfig,
) -> None:
    """Test output location mapping."""
    coord = NoaaHrrrAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=0),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )

    out_loc = coord.out_loc()
    assert out_loc == {
        "time": pd.Timestamp("2024-02-29T00:00"),
    }


def test_source_file_coord_out_loc_with_lead_time(
    template_config: NoaaHrrrAnalysisTemplateConfig,
) -> None:
    """Test output location mapping with lead time."""
    coord = NoaaHrrrAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=1),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )

    out_loc = coord.out_loc()
    assert out_loc == {
        "time": pd.Timestamp("2024-02-29T01:00"),
    }


def test_region_job_generate_source_file_coords(
    template_config: NoaaHrrrAnalysisTemplateConfig,
) -> None:
    """Test source file coordinate generation."""
    template_ds = template_config.get_template(pd.Timestamp("2014-10-01T03:00"))

    test_ds = template_ds.isel(time=slice(0, 3))

    region_job = NoaaHrrrAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 3),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()
    # Processing region is buffered by 1 for deaccumulation, but not at the very start of the dataset
    assert len(processing_region_ds.time) == len(output_region_ds.time)

    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    assert len(source_coords) == 3

    for coord in source_coords:
        assert isinstance(coord, NoaaHrrrAnalysisSourceFileCoord)
        assert coord.domain == "conus"
        assert (
            coord.file_type
            == template_config.data_vars[0].internal_attrs.hrrr_file_type
        )


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
    template_config: NoaaHrrrAnalysisTemplateConfig,
    region: slice,
    expected_processing_region: slice,
) -> None:
    """Test that get_processing_region buffers by 1, clamped to 0 at dataset start."""
    region_job = NoaaHrrrAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=region,
        reformat_job_name="test",
    )

    assert region_job.get_processing_region() == expected_processing_region


def test_region_job_processing_region_buffered(
    template_config: NoaaHrrrAnalysisTemplateConfig,
) -> None:
    """Test that processing region is buffered by 1 step for deaccumulation (not at dataset start)."""
    template_ds = template_config.get_template(pd.Timestamp("2014-10-01T05:00"))

    test_ds = template_ds.isel(time=slice(0, 5))

    # Region starting at index 2 (not the start of the dataset)
    region_job = NoaaHrrrAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(2, 5),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()
    # Processing region is buffered by 1 for deaccumulation
    assert len(processing_region_ds.time) == len(output_region_ds.time) + 1
    assert processing_region_ds.time[0] == output_region_ds.time[0] - pd.Timedelta("1h")


def test_region_job_generate_source_file_coords_hour_0(
    template_config: NoaaHrrrAnalysisTemplateConfig,
) -> None:
    """Test that hour 0 variables use lead_time=0."""
    template_ds = template_config.get_template(pd.Timestamp("2014-10-01T04:00"))

    test_ds = template_ds.isel(time=slice(0, 4))

    instant_vars = [
        v for v in template_config.data_vars if v.attrs.step_type == "instant"
    ][:1]
    assert len(instant_vars) == 1

    # Use region starting at index 2 to test buffering behavior
    region_job = NoaaHrrrAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=instant_vars,
        append_dim=template_config.append_dim,
        region=slice(2, 4),
        reformat_job_name="test",
    )

    processing_region_ds, _output_region_ds = region_job._get_region_datasets()

    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, instant_vars
    )

    # 3 source coords: 1 buffer + 2 output
    assert len(source_coords) == 3

    expected_init_times = pd.date_range(
        "2014-10-01T01:00", "2014-10-01T03:00", freq="1h"
    )
    for coord, expected_init_time in zip(
        source_coords, expected_init_times, strict=True
    ):
        assert isinstance(coord, NoaaHrrrAnalysisSourceFileCoord)
        assert coord.init_time == expected_init_time
        assert coord.lead_time == pd.Timedelta("0h")


def test_region_job_generate_source_file_coords_hour_1(
    template_config: NoaaHrrrAnalysisTemplateConfig,
) -> None:
    """Test that non-hour 0 variables use lead_time=1."""
    template_ds = template_config.get_template(pd.Timestamp("2014-10-01T04:00"))

    test_ds = template_ds.isel(time=slice(0, 4))

    avg_vars = [v for v in template_config.data_vars if v.attrs.step_type == "avg"][:1]
    assert len(avg_vars) == 1

    # Use region starting at index 2 to test buffering behavior
    region_job = NoaaHrrrAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=avg_vars,
        append_dim=template_config.append_dim,
        region=slice(2, 4),
        reformat_job_name="test",
    )

    processing_region_ds, _output_region_ds = region_job._get_region_datasets()

    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, avg_vars
    )

    # 3 source coords: 1 buffer + 2 output
    assert len(source_coords) == 3

    expected_init_times = pd.date_range(
        "2014-10-01T00:00", "2014-10-01T02:00", freq="1h"
    )

    for coord, expected_init_time in zip(
        source_coords, expected_init_times, strict=True
    ):
        assert isinstance(coord, NoaaHrrrAnalysisSourceFileCoord)
        assert coord.init_time == expected_init_time
        assert coord.lead_time == pd.Timedelta("1h")
        expected_time = coord.init_time + coord.lead_time
        assert coord.out_loc()["time"] == expected_time


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    template_config = NoaaHrrrAnalysisTemplateConfig()
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
        classmethod(lambda *args, **kwargs: pd.Timestamp("2014-10-01T06:34")),
    )
    existing_ds = template_config.get_template(pd.Timestamp("2014-10-01T05:01"))
    template_utils.write_metadata(existing_ds, store_factory)

    jobs, template_ds = NoaaHrrrAnalysisRegionJob.operational_update_jobs(
        primary_store=store_factory.primary_store(),
        tmp_store=tmp_path / "tmp_ds.zarr",
        get_template_fn=template_config.get_template,
        append_dim=template_config.append_dim,
        all_data_vars=template_config.data_vars,
        reformat_job_name="test_job",
    )

    assert template_ds.time.max() == pd.Timestamp("2014-10-01T06:00")

    assert len(jobs) == 1
    for job in jobs:
        assert isinstance(job, NoaaHrrrAnalysisRegionJob)
        assert job.data_vars == template_config.data_vars


def test_update_template_with_results(
    template_config: NoaaHrrrAnalysisTemplateConfig,
) -> None:
    """Test that update_template_with_results removes the last hour of data."""
    template_ds = template_config.get_template(pd.Timestamp("2014-10-01T05:00"))

    region_job = NoaaHrrrAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 5),
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


# Variables not available before HRRRv3 (pre-2018-07-12)
_PRE_V3_UNAVAILABLE = {
    "downward_long_wave_radiation_flux_surface",
    "relative_humidity_2m",
}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("init_time", "unavailable_var_names"),
    [
        # Pre-HRRRv3: uses PRMSL instead of MSLMA for mean sea level pressure, missing DLWRF and RH
        (pd.Timestamp("2016-01-01T00:00"), _PRE_V3_UNAVAILABLE),
        # Post-HRRRv3: all variables available, uses MSLMA
        (pd.Timestamp("2024-01-01T00:00"), set()),
    ],
    ids=["pre-v3", "post-v3"],
)
def test_download_and_read_all_variables(
    init_time: pd.Timestamp,
    unavailable_var_names: set[str],
    tmp_path: Path,
) -> None:
    """Integration test: download and read all available variables before and after the PRMSL→MSLMA switch."""
    config = NoaaHrrrAnalysisTemplateConfig()
    available_vars = [
        v for v in config.data_vars if v.name not in unavailable_var_names
    ]

    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-hrrr-analysis"}
    region_job = NoaaHrrrAnalysisRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=available_vars,
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    for source_group in NoaaHrrrAnalysisRegionJob.source_groups(available_vars):
        is_hour_0 = has_hour_0_values(source_group[0])
        lead_time = pd.Timedelta("0h") if is_hour_0 else pd.Timedelta("1h")
        coord_init_time = init_time if is_hour_0 else init_time - pd.Timedelta("1h")

        coord = NoaaHrrrAnalysisSourceFileCoord(
            init_time=coord_init_time,
            lead_time=lead_time,
            domain="conus",
            file_type=source_group[0].internal_attrs.hrrr_file_type,
            data_vars=source_group,
        )

        downloaded_path = region_job.download_file(coord)
        updated_coord = replace(coord, downloaded_path=downloaded_path)

        for var in source_group:
            data = region_job.read_data(updated_coord, var)
            assert data.shape == (1059, 1799), (
                f"{var.name}: unexpected shape {data.shape}"
            )
            assert not np.all(np.isnan(data)), f"{var.name}: all NaN"
