from datetime import timedelta

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr.core.sync
import zarr.storage

from reformatters.common import validation


@pytest.fixture
def forecast_dataset(rng: np.random.Generator) -> xr.Dataset:
    """Create a mock forecast dataset for testing."""
    init_times = pd.date_range("2024-01-01", periods=5, freq="6h")
    lead_times = pd.timedelta_range(start="0h", end="240h", freq="6h")
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)

    ds = xr.Dataset(
        {
            "temperature": (
                ["init_time", "lead_time", "latitude", "longitude"],
                rng.standard_normal(
                    (len(init_times), len(lead_times), len(lats), len(lons))
                ),
            ),
            "precipitation": (
                ["init_time", "lead_time", "latitude", "longitude"],
                rng.standard_normal(
                    (len(init_times), len(lead_times), len(lats), len(lons))
                ),
            ),
        },
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


@pytest.fixture
def analysis_dataset(rng: np.random.Generator) -> xr.Dataset:
    """Create a mock analysis dataset for testing."""
    times = pd.date_range("2024-01-01", periods=48, freq="1h")
    lats = np.linspace(90, -90, 10)  # Decreasing as per convention
    lons = np.linspace(-180, 180, 20)

    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "latitude", "longitude"],
                rng.standard_normal((len(times), len(lats), len(lons))),
            ),
            "humidity": (
                ["time", "latitude", "longitude"],
                rng.standard_normal((len(times), len(lats), len(lons))),
            ),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    return ds


def test_check_forecast_current_data_passes(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """Test that check_forecast_current_data passes when recent data exists."""
    now = pd.Timestamp("2024-01-01 18:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda: now)

    result = validation.check_forecast_current_data(forecast_dataset)

    assert result.passed
    assert "Data found for the latest day" in result.message


def test_check_forecast_current_data_fails(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """Test that check_forecast_current_data fails when no recent data exists."""
    now = pd.Timestamp("2024-01-10")
    monkeypatch.setattr("pandas.Timestamp.now", lambda: now)

    result = validation.check_forecast_current_data(forecast_dataset)

    assert not result.passed
    assert "No data found for the latest day" in result.message


def test_check_forecast_recent_nans_passes(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """Test that check_forecast_recent_nans passes when NaN percentage is acceptable."""
    now = pd.Timestamp("2024-01-01 18:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda: now)

    result = validation.check_forecast_recent_nans(forecast_dataset)

    assert result.passed
    assert "acceptable NaN percentages" in result.message


def test_check_forecast_recent_nans_fails(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """Test that check_forecast_recent_nans fails when NaN percentage is too high."""
    now = pd.Timestamp("2024-01-01 18:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda: now)

    # Add excessive NaNs to the latest init_time
    forecast_dataset["temperature"].loc[
        {"init_time": forecast_dataset.init_time[-1]}
    ] = np.nan

    result = validation.check_forecast_recent_nans(
        forecast_dataset, max_nan_percentage=10
    )

    assert not result.passed
    assert "Excessive NaN values found" in result.message
    assert "temperature" in result.message


def test_check_analysis_current_data_passes(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_current_data passes when recent data exists."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    result = validation.check_analysis_current_data(analysis_dataset)

    assert result.passed
    assert "Data found within" in result.message


def test_check_analysis_current_data_fails(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_current_data fails when no recent data exists."""
    now = pd.Timestamp("2024-01-10")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    result = validation.check_analysis_current_data(analysis_dataset)

    assert not result.passed
    assert "No data found within" in result.message


def test_check_analysis_current_data_custom_delay(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_current_data respects custom max_expected_delay."""
    # Dataset ends at 2024-01-02 23:00:00
    # Check at 2024-01-03 12:00:00 (13 hours after last data)
    now = pd.Timestamp("2024-01-03 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    # Should fail with default 12 hour delay (last data is 13 hours ago)
    result = validation.check_analysis_current_data(analysis_dataset)
    assert not result.passed

    # Should pass with 48 hour delay
    result = validation.check_analysis_current_data(
        analysis_dataset, max_expected_delay=timedelta(hours=48)
    )
    assert result.passed


def test_check_analysis_recent_nans_passes(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans passes when NaN percentage is acceptable."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    result = validation.check_analysis_recent_nans(analysis_dataset)

    assert result.passed
    assert "acceptable NaN percentages" in result.message


def test_check_analysis_recent_nans_fails(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans fails when NaN percentage is too high."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    # Set all recent data to NaN to ensure the random sample will catch it
    analysis_dataset["temperature"].loc[{"time": slice("2024-01-02", None)}] = np.nan

    result = validation.check_analysis_recent_nans(
        analysis_dataset, max_expected_delay=timedelta(hours=12), max_nan_percentage=5
    )

    assert not result.passed
    assert "Excessive NaN values found" in result.message
    assert "temperature" in result.message


def test_check_analysis_recent_nans_custom_parameters(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans respects custom parameters."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    # Set all recent data to NaN to ensure the random sample will catch it
    recent_slice = {"time": slice("2024-01-02", None)}
    analysis_dataset["temperature"].loc[recent_slice] = np.nan

    # Should fail with 5% threshold
    result = validation.check_analysis_recent_nans(
        analysis_dataset, max_expected_delay=timedelta(hours=12), max_nan_percentage=5
    )
    assert not result.passed

    # Should pass with 100% threshold
    result = validation.check_analysis_recent_nans(
        analysis_dataset, max_expected_delay=timedelta(hours=12), max_nan_percentage=100
    )
    assert result.passed


def test_check_analysis_recent_nans_quarter_sampling_passes(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans passes with quarter spatial sampling."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    result = validation.check_analysis_recent_nans(
        analysis_dataset, spatial_sampling="quarter"
    )

    assert result.passed
    assert "acceptable NaN percentages" in result.message


def test_check_analysis_recent_nans_quarter_sampling_fails(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans fails with quarter sampling when NaN percentage is too high."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    # Set all recent data to NaN to ensure the quarter sample will catch it
    analysis_dataset["temperature"].loc[{"time": slice("2024-01-02", None)}] = np.nan

    result = validation.check_analysis_recent_nans(
        analysis_dataset,
        max_expected_delay=timedelta(hours=12),
        max_nan_percentage=5,
        spatial_sampling="quarter",
    )

    assert not result.passed
    assert "Excessive NaN values found" in result.message
    assert "temperature" in result.message


def test_check_analysis_recent_nans_quarter_sampling_different_quarters(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans samples different quarters based on random selection."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    # Add NaNs to all quarters to ensure we can test the sampling
    lat_size = len(analysis_dataset.latitude)
    lon_size = len(analysis_dataset.longitude)

    # Set all recent data to NaN first
    analysis_dataset["temperature"].loc[{"time": slice("2024-01-02", None)}] = np.nan

    # Then set one quarter back to valid data (bottom-right quarter)
    analysis_dataset["temperature"].isel(
        time=slice(-24, None),  # Last 24 hours
        latitude=slice(lat_size // 2, lat_size),
        longitude=slice(lon_size // 2, lon_size),
    ).values[:] = 1.0

    # Mock RNG to always select bottom-right quarter (both integers calls return 1)
    class MockRngBottomRight:
        def integers(self, _low: int, _high: int) -> int:
            return 1

    monkeypatch.setattr(
        "reformatters.common.validation.np.random.default_rng",
        lambda seed=None: MockRngBottomRight(),
    )

    result = validation.check_analysis_recent_nans(
        analysis_dataset,
        max_expected_delay=timedelta(hours=12),
        max_nan_percentage=5,
        spatial_sampling="quarter",
    )

    # Should pass because bottom-right quarter has valid data
    assert result.passed

    # Mock RNG to select top-left quarter (both integers calls return 0)
    class MockRngTopLeft:
        def integers(self, _low: int, _high: int) -> int:
            return 0

    monkeypatch.setattr(
        "reformatters.common.validation.np.random.default_rng",
        lambda seed=None: MockRngTopLeft(),
    )

    result = validation.check_analysis_recent_nans(
        analysis_dataset,
        max_expected_delay=timedelta(hours=12),
        max_nan_percentage=5,
        spatial_sampling="quarter",
    )

    # Should fail because top-left quarter has NaNs
    assert not result.passed


def test_check_analysis_recent_nans_xy_dimensions(
    monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator
) -> None:
    """Test that check_analysis_recent_nans works with x/y dimensions instead of lat/lon."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    times = pd.date_range("2024-01-01", periods=48, freq="1h")
    x = np.arange(20)
    y = np.arange(10)

    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            )
        },
        coords={"time": times, "y": y, "x": x},
    )

    result = validation.check_analysis_recent_nans(ds, spatial_sampling="quarter")

    assert result.passed
    assert "acceptable NaN percentages" in result.message


def test_check_analysis_recent_nans_invalid_spatial_sampling(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans raises error for invalid spatial_sampling mode."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    with pytest.raises(AssertionError, match="Expected code to be unreachable"):
        validation.check_analysis_recent_nans(
            analysis_dataset,
            spatial_sampling="invalid",  # type: ignore[arg-type]
        )


def test_compare_replica_and_primary_coords_divergence(
    rng: np.random.Generator,
) -> None:
    """Test that compare_replica_and_primary detects coordinate divergence."""
    times = pd.date_range("2024-01-01", periods=24, freq="1h")
    x = np.arange(20)
    y = np.arange(10)
    chunk_size = 8

    primary_ds = xr.Dataset(
        {
            "temperature": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
            "humidity": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
        },
        coords={"time": times, "y": y, "x": x},
    )
    for var in primary_ds.data_vars.values():
        var.encoding["chunks"] = (chunk_size, len(y), len(x))

    replica_ds = primary_ds.copy(deep=True).isel(time=slice(None, -1))
    result = validation.compare_replica_and_primary("time", replica_ds, primary_ds)

    assert not result.passed
    assert "different for coords: ['time']" in result.message


def test_compare_replica_and_primary_vars_divergence(
    rng: np.random.Generator,
) -> None:
    """Test that compare_replica_and_primary detects variable data divergence."""
    times = pd.date_range("2024-01-01", periods=24, freq="1h")
    x = np.arange(20)
    y = np.arange(10)
    chunk_size = 8

    primary_ds = xr.Dataset(
        {
            "temperature": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
            "humidity": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
        },
        coords={"time": times, "y": y, "x": x},
    )
    for var in primary_ds.data_vars.values():
        var.encoding["chunks"] = (chunk_size, len(y), len(x))

    replica_ds = primary_ds.copy(deep=True)
    replica_ds["temperature"].values[-chunk_size:, 0, 0] = 999.0

    result = validation.compare_replica_and_primary("time", replica_ds, primary_ds)

    assert not result.passed
    assert (
        "different for at least the following vars: ['temperature']" in result.message
    )


def test_compare_replica_and_primary_passes(
    rng: np.random.Generator,
) -> None:
    """Test that compare_replica_and_primary passes when replica and primary are identical."""
    times = pd.date_range("2024-01-01", periods=24, freq="1h")
    x = np.arange(20)
    y = np.arange(10)
    chunk_size = 8

    primary_ds = xr.Dataset(
        {
            "temperature": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
            "humidity": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
        },
        coords={"time": times, "y": y, "x": x},
    )
    for var in primary_ds.data_vars.values():
        var.encoding["chunks"] = (chunk_size, len(y), len(x))

    replica_ds = primary_ds.copy(deep=True)

    result = validation.compare_replica_and_primary("time", replica_ds, primary_ds)

    assert result.passed
    assert "replica and primary stores is the same" in result.message


def test_check_for_expected_shards_passes(rng: np.random.Generator) -> None:
    """Test that check_for_expected_shards passes when all expected shards are present."""
    times = pd.date_range("2024-01-01", periods=16, freq="1h")
    x = np.arange(10)
    y = np.arange(8)

    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
            "humidity": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
        },
        coords={"time": times, "y": y, "x": x},
        attrs={"dataset_id": "test-dataset"},
    )

    # Set chunking and sharding to create multiple shards
    chunk_sizes = (4, 4, 5)
    shard_sizes = (4, 4, 5)
    for var in ds.data_vars.values():
        var.encoding["chunks"] = chunk_sizes
        var.encoding["shards"] = shard_sizes

    # Write to memory store
    store = zarr.storage.MemoryStore()
    ds.to_zarr(store, mode="w", consolidated=False)

    result = validation.check_for_expected_shards(store, ds)

    assert result.passed
    assert "All variables have expected shards" in result.message


def test_check_for_expected_shards_fails_missing_shards(
    rng: np.random.Generator,
) -> None:
    """Test that check_for_expected_shards fails when expected shards are missing."""
    times = pd.date_range("2024-01-01", periods=16, freq="1h")
    x = np.arange(10)
    y = np.arange(8)

    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
        },
        coords={"time": times, "y": y, "x": x},
        attrs={"dataset_id": "test-dataset"},
    )

    # Set chunking and sharding to create multiple shards (4x2x2 = 16 total shards)
    chunk_sizes = (4, 4, 5)
    shard_sizes = (4, 4, 5)
    for var in ds.data_vars.values():
        var.encoding["chunks"] = chunk_sizes
        var.encoding["shards"] = shard_sizes

    # Write to memory store
    store = zarr.storage.MemoryStore()
    ds.to_zarr(store, mode="w", consolidated=False)

    # Manually delete a shard to simulate missing data
    # Delete shard 0/0/0 for temperature using sync wrapper
    zarr.core.sync.sync(store.delete("temperature/c/0/0/0"))

    result = validation.check_for_expected_shards(store, ds)

    assert not result.passed
    assert "temperature" in result.message
    assert "missing expected shards" in result.message


def test_check_for_expected_shards_passes_with_extra_shards(
    rng: np.random.Generator,
) -> None:
    """Test that check_for_expected_shards passes when extra shards are present."""
    times = pd.date_range("2024-01-01", periods=8, freq="1h")
    x = np.arange(10)
    y = np.arange(8)

    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
        },
        coords={"time": times, "y": y, "x": x},
        attrs={"dataset_id": "test-dataset"},
    )

    chunk_sizes = (4, 4, 5)
    shard_sizes = (4, 4, 5)
    for var in ds.data_vars.values():
        var.encoding["chunks"] = chunk_sizes
        var.encoding["shards"] = shard_sizes

    # Write full dataset to store
    store = zarr.storage.MemoryStore()
    ds.to_zarr(store, mode="w", consolidated=False)

    # Trim the dataset to only include first chunk of time
    ds_trimmed = ds.isel(time=slice(0, 4))
    # Need to preserve encoding on the trimmed dataset
    for var in ds_trimmed.data_vars.values():
        var.encoding["chunks"] = chunk_sizes
        var.encoding["shards"] = shard_sizes

    # The store has shards for all 8 time steps, but metadata only exposes first 4
    # This simulates the operational update scenario where extra shards exist
    result = validation.check_for_expected_shards(store, ds_trimmed)

    assert result.passed
    assert "All variables have expected shards" in result.message


def test_check_for_expected_shards_icechunk_store(rng: np.random.Generator) -> None:
    """Test that check_for_expected_shards works with an IcechunkStore."""
    times = pd.date_range("2024-01-01", periods=8, freq="1h")
    x = np.arange(6)
    y = np.arange(4)

    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
            "pressure": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
        },
        coords={"time": times, "y": y, "x": x},
        attrs={"dataset_id": "test-icechunk-dataset"},
    )

    chunk_sizes = (4, 2, 3)
    shard_sizes = (4, 2, 3)
    for var in ds.data_vars.values():
        var.encoding["chunks"] = chunk_sizes
        var.encoding["shards"] = shard_sizes

    # Create in-memory Icechunk store
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")
    store = session.store

    # Write dataset
    ds.to_zarr(store, mode="w", consolidated=False)

    # Commit the changes
    session.commit("Initial commit")

    result = validation.check_for_expected_shards(store, ds)

    assert result.passed
    assert "All variables have expected shards" in result.message
