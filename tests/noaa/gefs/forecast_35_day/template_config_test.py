import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.gefs.forecast_35_day.template_config import (
    GefsForecast35DayTemplateConfig,
)


@pytest.fixture
def template_config() -> GefsForecast35DayTemplateConfig:
    """Create a GEFS 35-day forecast template config for testing."""
    return GefsForecast35DayTemplateConfig()


def test_dataset_attributes(template_config: GefsForecast35DayTemplateConfig) -> None:
    """Test dataset attributes are correctly configured."""
    attrs = template_config.dataset_attributes
    assert attrs.dataset_id == "noaa-gefs-forecast-35-day"
    assert attrs.dataset_version == "0.2.0"
    assert attrs.name == "NOAA GEFS forecast, 35 day"
    assert "Global Ensemble Forecast System" in attrs.description
    assert attrs.spatial_domain == "Global"
    assert "0.25 degrees" in attrs.spatial_resolution
    assert "0.5 degrees" in attrs.spatial_resolution
    assert "Forecasts initialized every 24 hours" in attrs.time_resolution
    assert attrs.forecast_domain == "Forecast lead time 0-840 hours (0-35 days) ahead"
    assert "2020-10-01 00:00:00 UTC" in attrs.time_domain


def test_dimensions_and_append_dim(
    template_config: GefsForecast35DayTemplateConfig,
) -> None:
    """Test dimension configuration."""
    expected_dims = (
        "init_time",
        "ensemble_member",
        "lead_time",
        "latitude",
        "longitude",
    )
    assert template_config.dims == expected_dims
    assert template_config.append_dim == "init_time"
    assert template_config.append_dim_start == pd.Timestamp("2020-10-01T00:00")
    assert template_config.append_dim_frequency == pd.Timedelta("24h")


def test_dimension_coordinates(
    template_config: GefsForecast35DayTemplateConfig,
) -> None:
    """Test dimension coordinates generation."""
    coords = template_config.dimension_coordinates()

    # Check init_time coordinate
    assert "init_time" in coords
    init_time_coord = coords["init_time"]
    assert isinstance(init_time_coord, pd.DatetimeIndex)
    assert init_time_coord.freq == pd.Timedelta("24h")
    assert init_time_coord[0] == template_config.append_dim_start

    # Check ensemble_member coordinate
    assert "ensemble_member" in coords
    ensemble_coord = coords["ensemble_member"]
    assert isinstance(ensemble_coord, np.ndarray)
    assert len(ensemble_coord) == 31  # 31 ensemble members
    assert ensemble_coord[0] == 0
    assert ensemble_coord[-1] == 30

    # Check lead_time coordinate
    assert "lead_time" in coords
    lead_time_coord = coords["lead_time"]
    assert isinstance(lead_time_coord, pd.TimedeltaIndex)
    # Should combine 3-hourly (0-240h) with 6-hourly (246-840h)
    assert pd.Timedelta("0h") in lead_time_coord
    assert pd.Timedelta("240h") in lead_time_coord
    assert pd.Timedelta("246h") in lead_time_coord
    assert pd.Timedelta("840h") in lead_time_coord
    # Check the transition happens at the right point
    three_hourly_part = pd.timedelta_range("0h", "240h", freq="3h")
    six_hourly_part = pd.timedelta_range("246h", "840h", freq="6h")
    expected_combined = three_hourly_part.union(six_hourly_part)
    pd.testing.assert_index_equal(lead_time_coord, expected_combined)

    # Check shared coordinates are included
    assert "latitude" in coords
    assert "longitude" in coords
    assert isinstance(coords["latitude"], np.ndarray)
    assert isinstance(coords["longitude"], np.ndarray)


def test_derive_coordinates(template_config: GefsForecast35DayTemplateConfig) -> None:
    """Test coordinate derivation including forecast-specific coordinates."""
    # Create a minimal dataset for testing with realistic dimensions
    init_times = pd.date_range("2020-10-01", periods=2, freq="24h")
    lead_times = pd.timedelta_range("0h", "12h", freq="3h")
    ensemble_members = np.arange(31)

    ds = xr.Dataset(
        coords={
            "init_time": init_times,
            "ensemble_member": np.arange(31),
            "lead_time": lead_times,
            "latitude": np.linspace(-90, 90, 721),
            "longitude": np.linspace(-180, 179.75, 1440),
        }
    )

    derived = template_config.derive_coordinates(ds)

    # Should return spatial_ref
    assert "spatial_ref" in derived
    assert derived["spatial_ref"][0] == ()  # Empty dimensions tuple

    # Should return valid_time
    assert "valid_time" in derived
    valid_time = derived["valid_time"]
    assert hasattr(valid_time, "dims")  # Should be an xarray DataArray

    # Should return ingested_forecast_length
    assert "ingested_forecast_length" in derived
    ingested_length = derived["ingested_forecast_length"]
    assert isinstance(ingested_length, tuple)
    assert ingested_length[0] == ("init_time", "ensemble_member")
    assert isinstance(ingested_length[1], np.ndarray)
    assert ingested_length[1].shape == (len(init_times), len(ensemble_members))

    # Should return expected_forecast_length
    assert "expected_forecast_length" in derived
    expected_length = derived["expected_forecast_length"]
    assert isinstance(expected_length, tuple)
    assert expected_length[0] == ("init_time",)
    assert isinstance(expected_length[1], np.ndarray)
    assert len(expected_length[1]) == len(init_times)
    # Check that 00 UTC gets 35 days (840 hours), others get 16 days (384 hours)
    for i, init_time in enumerate(pd.to_datetime(init_times)):
        expected_hours = 840 if init_time.hour == 0 else 384
        expected_delta = pd.Timedelta(hours=expected_hours)
        assert expected_length[1][i] == expected_delta


def test_coordinates_configuration(
    template_config: GefsForecast35DayTemplateConfig,
) -> None:
    """Test coordinate metadata and encoding configuration."""
    coords = template_config.coords

    # Should have all expected coordinates
    coord_names = {coord.name for coord in coords}
    expected_coords = {
        "init_time",
        "ensemble_member",
        "lead_time",
        "valid_time",
        "ingested_forecast_length",
        "expected_forecast_length",
        "latitude",
        "longitude",
        "spatial_ref",
    }
    assert expected_coords.issubset(coord_names)

    # Test init_time coordinate
    init_time_coord = next(coord for coord in coords if coord.name == "init_time")
    assert init_time_coord.encoding.dtype == "int64"
    assert init_time_coord.encoding.calendar == "proleptic_gregorian"
    assert init_time_coord.encoding.units == "seconds since 1970-01-01 00:00:00"

    # Test ensemble_member coordinate
    ensemble_coord = next(coord for coord in coords if coord.name == "ensemble_member")
    assert ensemble_coord.encoding.dtype == "uint16"
    assert ensemble_coord.encoding.chunks == 31  # All ensemble members in one chunk
    assert ensemble_coord.encoding.shards is None  # Coordinates don't have shards

    # Test lead_time coordinate
    lead_time_coord = next(coord for coord in coords if coord.name == "lead_time")
    assert lead_time_coord.encoding.dtype == "int64"
    assert lead_time_coord.encoding.units == "seconds"
    # Should chunk all lead times together
    assert lead_time_coord.encoding.chunks == 181


def test_data_variables_configuration(
    template_config: GefsForecast35DayTemplateConfig,
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
        # Chunks should match expected pattern (init_time, ensemble_member, lead_time, lat, lon)
        chunks = var.encoding.chunks
        shards = var.encoding.shards
        assert isinstance(chunks, tuple)
        assert isinstance(shards, tuple)
        assert len(chunks) == 5
        assert len(shards) == 5

        # init_time: 1 forecast per chunk
        assert chunks[0] == 1
        assert shards[0] == 1

        # ensemble_member: all 31 members in one chunk
        assert chunks[1] == 31
        assert shards[1] == 31

        # lead_time: 64 per chunk
        assert chunks[2] == 64
        assert shards[2] == 64 * 3

        # latitude: 17 per chunk
        assert chunks[3] == 17
        assert shards[3] == 17 * 22

        # longitude: 16 per chunk
        assert chunks[4] == 16
        assert shards[4] == 16 * 23


def test_dataset_id_property(template_config: GefsForecast35DayTemplateConfig) -> None:
    """Test dataset ID property."""
    assert template_config.dataset_id == "noaa-gefs-forecast-35-day"


def test_append_dim_coordinates_range(
    template_config: GefsForecast35DayTemplateConfig,
) -> None:
    """Test append dimension coordinate generation."""
    end_time = pd.Timestamp("2020-10-05T00:00")
    coords = template_config.append_dim_coordinates(end_time)

    # Should be left-inclusive, right-exclusive
    assert coords[0] == template_config.append_dim_start
    assert coords[-1] < end_time
    assert len(coords) == 4  # 4 days


def test_forecast_length_logic() -> None:
    """Test expected forecast length calculation logic."""
    template_config = GefsForecast35DayTemplateConfig()

    # Create test dataset with different init hours
    init_times = [
        pd.Timestamp("2020-10-01T00:00"),  # 00 UTC - should get 35 days
        pd.Timestamp("2020-10-01T06:00"),  # 06 UTC - should get 16 days
        pd.Timestamp("2020-10-01T12:00"),  # 12 UTC - should get 16 days
        pd.Timestamp("2020-10-01T18:00"),  # 18 UTC - should get 16 days
    ]

    ds = xr.Dataset(
        coords={
            "init_time": init_times,
            "lead_time": pd.timedelta_range("0h", "12h", freq="3h"),
            "ensemble_member": np.arange(31),
        }
    )

    derived = template_config.derive_coordinates(ds)
    expected_length = derived["expected_forecast_length"]
    assert isinstance(expected_length, tuple)
    expected_values = expected_length[1]

    # Check each init time gets correct expected forecast length
    assert expected_values[0] == pd.Timedelta(hours=840)  # 00 UTC -> 35 days
    assert expected_values[1] == pd.Timedelta(hours=384)  # 06 UTC -> 16 days
    assert expected_values[2] == pd.Timedelta(hours=384)  # 12 UTC -> 16 days
    assert expected_values[3] == pd.Timedelta(hours=384)  # 18 UTC -> 16 days


def test_template_config_immutable(
    template_config: GefsForecast35DayTemplateConfig,
) -> None:
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


def test_chunk_and_shard_alignment(
    template_config: GefsForecast35DayTemplateConfig,
) -> None:
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
