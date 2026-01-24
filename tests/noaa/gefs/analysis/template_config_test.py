from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.gefs.analysis.region_job import (
    GefsAnalysisRegionJob,
    GefsAnalysisSourceFileCoord,
)
from reformatters.noaa.gefs.analysis.template_config import GefsAnalysisTemplateConfig


@pytest.fixture
def template_config() -> GefsAnalysisTemplateConfig:
    """Create a GEFS analysis template config for testing."""
    return GefsAnalysisTemplateConfig()


@pytest.fixture(scope="session")
def gefs_analysis_first_message_path() -> Path:
    cfg = GefsAnalysisTemplateConfig()
    assert cfg.data_vars
    init_time = pd.Timestamp("2024-11-01T00:00")

    coord = GefsAnalysisSourceFileCoord(
        init_time=init_time,
        lead_time=pd.Timedelta("0h"),
        data_vars=(cfg.data_vars[0],),
        ensemble_member=0,
    )

    region_job = GefsAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=cfg.get_template(init_time),
        data_vars=cfg.data_vars,
        append_dim=cfg.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    return region_job.download_file(coord)


def test_dataset_attributes(template_config: GefsAnalysisTemplateConfig) -> None:
    """Test dataset attributes are correctly configured."""
    attrs = template_config.dataset_attributes
    assert attrs.dataset_id == "noaa-gefs-analysis"
    assert attrs.dataset_version == "0.1.2"
    assert attrs.name == "NOAA GEFS analysis"
    assert "Global Ensemble Forecast System" in attrs.description
    assert attrs.spatial_domain == "Global"
    assert attrs.spatial_resolution == "0.25 degrees (~20km)"
    assert attrs.time_resolution == "3.0 hours"
    assert "2000-01-01 00:00:00 UTC" in attrs.time_domain


def test_dimensions_and_append_dim(template_config: GefsAnalysisTemplateConfig) -> None:
    """Test dimension configuration."""
    assert template_config.dims == ("time", "latitude", "longitude")
    assert template_config.append_dim == "time"
    assert template_config.append_dim_start == pd.Timestamp("2000-01-01T00:00")
    assert template_config.append_dim_frequency == pd.Timedelta("3h")


def test_dimension_coordinates(template_config: GefsAnalysisTemplateConfig) -> None:
    """Test dimension coordinates generation."""
    coords = template_config.dimension_coordinates()

    # Check time coordinate
    assert "time" in coords
    time_coord = coords["time"]
    assert isinstance(time_coord, pd.DatetimeIndex)
    assert time_coord.freq == pd.Timedelta("3h")
    # Should start at the append_dim_start time itself
    assert time_coord[0] == template_config.append_dim_start

    # Check shared coordinates are included
    assert "latitude" in coords
    assert "longitude" in coords
    assert isinstance(coords["latitude"], np.ndarray)
    assert isinstance(coords["longitude"], np.ndarray)


def test_derive_coordinates(template_config: GefsAnalysisTemplateConfig) -> None:
    """Test coordinate derivation."""
    # Create a minimal dataset for testing
    ds = xr.Dataset(
        coords={
            "time": pd.date_range("2000-01-01", periods=3, freq="3h"),
            "latitude": np.linspace(-90, 90, 721),
            "longitude": np.linspace(-180, 179.75, 1440),
        }
    )

    derived = template_config.derive_coordinates(ds)

    # Should return spatial_ref at minimum
    assert "spatial_ref" in derived
    assert derived["spatial_ref"][0] == ()  # Empty dimensions tuple


def test_coordinates_configuration(template_config: GefsAnalysisTemplateConfig) -> None:
    """Test coordinate metadata and encoding configuration."""
    coords = template_config.coords

    # Should have time coordinate plus shared coordinates
    coord_names = {coord.name for coord in coords}
    assert "time" in coord_names
    assert "latitude" in coord_names
    assert "longitude" in coord_names
    assert "spatial_ref" in coord_names

    # Test time coordinate specifically
    time_coord = next(coord for coord in coords if coord.name == "time")
    assert time_coord.encoding.dtype == "int64"
    assert time_coord.encoding.calendar == "proleptic_gregorian"
    assert time_coord.encoding.units == "seconds since 1970-01-01 00:00:00"
    assert time_coord.attrs.units == "seconds since 1970-01-01 00:00:00"

    assert time_coord.encoding.chunks == 146000
    assert time_coord.encoding.shards is None


def test_data_variables_configuration(
    template_config: GefsAnalysisTemplateConfig,
) -> None:
    """Test data variable configuration."""
    data_vars = template_config.data_vars

    # Should have the expected GEFS variables
    assert len(data_vars) > 0
    var_names = {var.name for var in data_vars}

    # Check some expected variables are present
    expected_vars = {
        "temperature_2m",
        "precipitation_surface",
        "pressure_surface",
        "wind_u_10m",
        "wind_v_10m",
    }
    assert expected_vars.issubset(var_names)

    # Test chunking configuration
    for var in data_vars:
        assert var.encoding.chunks is not None
        assert var.encoding.shards is not None
        # Chunks should match expected pattern (time, lat, lon)
        chunks = var.encoding.chunks
        shards = var.encoding.shards
        assert isinstance(chunks, tuple)
        assert isinstance(shards, tuple)
        assert len(chunks) == 3
        assert len(shards) == 3
        # Time chunks should be 180 * 8 = 1440 (180 days of 3-hourly data)
        assert chunks[0] == 1440
        # Lat/lon chunks should be 32
        assert chunks[1] == 32
        assert chunks[2] == 32


def test_dataset_id_property(template_config: GefsAnalysisTemplateConfig) -> None:
    """Test dataset ID property."""
    assert template_config.dataset_id == "noaa-gefs-analysis"


def test_append_dim_coordinates_range(
    template_config: GefsAnalysisTemplateConfig,
) -> None:
    """Test append dimension coordinate generation."""
    end_time = pd.Timestamp("2000-01-02T00:00")
    coords = template_config.append_dim_coordinates(end_time)

    # Should be left-inclusive, right-exclusive
    assert coords[0] == template_config.append_dim_start
    assert coords[-1] < end_time
    assert len(coords) == 8  # 24 hours / 3 hours = 8 steps


def test_template_config_immutable(template_config: GefsAnalysisTemplateConfig) -> None:
    """Test that template config maintains expected immutability."""
    # Key properties should be consistent across calls
    dims1 = template_config.dims
    dims2 = template_config.dims
    assert dims1 == dims2

    attrs1 = template_config.dataset_attributes
    attrs2 = template_config.dataset_attributes
    assert attrs1.dataset_id == attrs2.dataset_id

    coords1 = template_config.coords
    coords2 = template_config.coords
    assert len(coords1) == len(coords2)
    assert {c.name for c in coords1} == {c.name for c in coords2}


def test_chunk_and_shard_alignment(template_config: GefsAnalysisTemplateConfig) -> None:
    """Test that chunk and shard sizes are properly aligned."""
    for var in template_config.data_vars:
        chunks = var.encoding.chunks
        shards = var.encoding.shards
        assert chunks is not None
        assert shards is not None
        assert isinstance(chunks, tuple)
        assert isinstance(shards, tuple)
        assert len(chunks) == len(shards)

        # Shards should be larger than or equal to chunks in each dimension
        for chunk, shard in zip(chunks, shards, strict=True):
            assert shard >= chunk
