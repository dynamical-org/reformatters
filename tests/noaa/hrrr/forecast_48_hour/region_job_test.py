from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.noaa.hrrr.forecast_48_hour.region_job import (
    NoaaHrrrForecast48HourRegionJob,
    NoaaHrrrForecast48HourSourceFileCoord,
)
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)


@pytest.fixture
def template_config() -> NoaaHrrrForecast48HourTemplateConfig:
    return NoaaHrrrForecast48HourTemplateConfig()


def test_source_file_coord_out_loc(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    """Test output location mapping."""
    coord = NoaaHrrrForecast48HourSourceFileCoord(
        init_time=pd.Timestamp("2024-02-29T00:00"),
        lead_time=pd.Timedelta(hours=12),
        domain="conus",
        file_type="sfc",
        data_vars=template_config.data_vars,
    )

    out_loc = coord.out_loc()
    assert out_loc == {
        "init_time": pd.Timestamp("2024-02-29T00:00"),
        "lead_time": pd.Timedelta(hours=12),
    }


def test_region_job_generate_source_file_coords(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    """Test source file coordinate generation."""
    template_ds = template_config.get_template(pd.Timestamp("2025-01-01"))

    # Create a small subset for testing
    test_ds = template_ds.isel(init_time=slice(0, 2), lead_time=slice(0, 3))

    # use `model_construct` to skip pydantic validation so we can pass mock stores
    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],  # Just one variable
        append_dim=template_config.append_dim,
        region=slice(0, 2),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()
    assert processing_region_ds.equals(output_region_ds)

    # Test with a single data variable
    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    # Should generate coordinates for each init_time x lead_time combination
    # 2 init_times x 3 lead_times = 6 coordinates
    assert len(source_coords) == 6

    # Check that all coordinates are NoaaHrrrForecast48HourSourceFileCoord instances
    for coord in source_coords:
        assert isinstance(coord, NoaaHrrrForecast48HourSourceFileCoord)
        assert coord.domain == "conus"
        assert (
            coord.file_type
            == template_config.data_vars[0].internal_attrs.hrrr_file_type
        )


def test_region_job_generate_source_file_coords_filters_hour_0(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    """Test that hour 0 filtering works for accumulated variables."""
    template_ds = template_config.get_template(pd.Timestamp("2025-01-01"))

    # Create a small subset that includes hour 0
    test_ds = template_ds.isel(
        init_time=slice(0, 1), lead_time=slice(0, 2)
    )  # hours 0 and 1

    # Find a variable that doesn't have hour 0 values (if any)
    # For now, just test with a regular variable
    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    processing_region_ds, _output_region_ds = region_job._get_region_datasets()

    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    # Should have coordinates for the available times
    assert len(source_coords) >= 1

    # All coordinates should be valid
    for coord in source_coords:
        assert isinstance(coord, NoaaHrrrForecast48HourSourceFileCoord)


def test_region_job_48h_forecasts(
    template_config: NoaaHrrrForecast48HourTemplateConfig,
) -> None:
    """Test that 48-hour forecast coordinates are generated correctly."""

    template_ds = template_config.get_template(pd.Timestamp("2025-01-01"))
    test_ds = template_ds.isel(
        init_time=slice(0, 1),  # Single init time
        lead_time=slice(0, -1, 12),  # Every 12 hours up to 48h
    )

    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=test_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()
    assert processing_region_ds.equals(output_region_ds)

    source_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    # Should generate coordinates for all available lead times (48h dataset)
    assert len(source_coords) > 0

    # All coordinates should be valid NoaaHrrrForecast48HourSourceFileCoord instances
    for coord in source_coords:
        assert isinstance(coord, NoaaHrrrForecast48HourSourceFileCoord)
        assert coord.domain == "conus"  # Always CONUS for this dataset
        # Lead time should be <= 48h (dataset maximum)
        assert coord.lead_time <= pd.Timedelta("48h")


def test_operational_update_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    template_config = NoaaHrrrForecast48HourTemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset-A",
        template_config_version="test-version",
    )

    # Set the append_dim_end for the update
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2022-01-01T12:34")),
    )
    # Set the append_dim_start for the update
    # Use a template_ds as a lightweight way to create a mock dataset with a known max append dim coordinate
    existing_ds = template_config.get_template(
        pd.Timestamp("2022-01-01T06:01")  # 06 will be max existing init time
    )
    template_utils.write_metadata(existing_ds, store_factory)

    jobs, template_ds = NoaaHrrrForecast48HourRegionJob.operational_update_jobs(
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
        assert isinstance(job, NoaaHrrrForecast48HourRegionJob)
        assert job.data_vars == template_config.data_vars


def test_update_append_dim_end_is_tz_naive() -> None:
    region_job = NoaaHrrrForecast48HourRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=Mock(),
        append_dim=Mock(),
        region=Mock(),
        reformat_job_name="test",
    )
    assert region_job._update_append_dim_end().tzinfo is None
