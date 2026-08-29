import logging
import warnings
from datetime import timedelta

import icechunk
import numpy as np
import pandas as pd
import pydantic
import pytest
import sentry_sdk
import xarray as xr
import zarr.core.sync
import zarr.storage

from reformatters.common import validation
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)


class NanTestDataVar(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=(1, 1, 1, 1),
        shards=None,
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="K",
        long_name="Test variable",
        short_name="test",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(
        keep_mantissa_bits="no-rounding"
    )


def _context(
    ds: xr.Dataset,
    append_dim: str,
    store: zarr.storage.StoreLike | None = None,
    append_dim_shard_size: int | None = None,
    **kwargs: object,
) -> validation.ValidationContext:
    for var in ds.data_vars.values():
        if append_dim not in var.dims:
            continue
        if append_dim_shard_size is not None or var.encoding.get("shards") is None:
            shard_size = append_dim_shard_size or 1
            var.encoding["shards"] = tuple(
                shard_size if dim == append_dim else size
                for dim, size in var.sizes.items()
            )
    return validation.ValidationContext(
        store=store if store is not None else zarr.storage.MemoryStore(),
        ds=ds,
        append_dim=append_dim,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


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
                {"step_type": "instant"},
            ),
            "precipitation": (
                ["init_time", "lead_time", "latitude", "longitude"],
                rng.standard_normal(
                    (len(init_times), len(lead_times), len(lats), len(lons))
                ),
                {"step_type": "accum"},
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


def test_check_current_data_passes_and_fails(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """A position missing past its deadline fails; present positions pass."""
    # Dataset has 6-hourly init_times through 2024-01-02 00:00.
    context = _context(
        forecast_dataset, "init_time", append_dim_frequency=pd.Timedelta("6h")
    )
    check = validation.CheckCurrentData(max_delay=timedelta(hours=2))

    # 2024-01-02 00:00 was due at 02:00 and is present.
    monkeypatch.setattr(
        "pandas.Timestamp.now", lambda: pd.Timestamp("2024-01-02 03:00")
    )
    assert check.check(context).passed

    # 2024-01-02 06:00 was due at 08:00 and is missing.
    monkeypatch.setattr(
        "pandas.Timestamp.now", lambda: pd.Timestamp("2024-01-02 08:01")
    )
    result = check.check(context)
    assert not result.passed
    assert "Missing init_time=2024-01-02T06:00:00" in result.message


def test_check_current_data_tight_deadline_holds_off_schedule(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """Deadlines attach to grid positions, so a tight max_delay does not false-alarm
    when the check runs mid-cycle, long after the newest position's timestamp."""
    context = _context(
        forecast_dataset, "init_time", append_dim_frequency=pd.Timedelta("6h")
    )
    check = validation.CheckCurrentData(max_delay=timedelta(hours=2, minutes=31))

    # Latest present init is 2024-01-02 00:00; mid-cycle its age is ~5h59m but the
    # next init (06:00) is not yet due, so this passes.
    monkeypatch.setattr(
        "pandas.Timestamp.now", lambda: pd.Timestamp("2024-01-02 05:59")
    )
    assert check.check(context).passed

    # One minute past 06:00's deadline it fails.
    monkeypatch.setattr(
        "pandas.Timestamp.now", lambda: pd.Timestamp("2024-01-02 08:32")
    )
    assert not check.check(context).passed


def test_check_current_data_grid_anchored_to_dataset_positions(
    monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator
) -> None:
    """Deadlines fall on the dataset's own grid, not epoch-aligned multiples."""
    # 6-hourly grid offset from midnight: 03:00, 09:00, 15:00, 21:00.
    init_times = pd.date_range("2024-01-01 03:00", periods=3, freq="6h")
    ds = xr.Dataset(
        {"temperature": (["init_time"], rng.standard_normal(len(init_times)))},
        coords={"init_time": init_times},
    )
    context = _context(ds, "init_time", append_dim_frequency=pd.Timedelta("6h"))
    check = validation.CheckCurrentData(max_delay=timedelta(hours=2))

    # Latest present is 15:00. At 17:00 an epoch-aligned floor would demand a
    # nonexistent 12:00 position; the newest due grid position is 15:00, present.
    monkeypatch.setattr(
        "pandas.Timestamp.now", lambda: pd.Timestamp("2024-01-01 17:00")
    )
    assert check.check(context).passed

    # 21:00 is due at 23:00 and missing.
    monkeypatch.setattr(
        "pandas.Timestamp.now", lambda: pd.Timestamp("2024-01-01 23:01")
    )
    assert not check.check(context).passed


def test_check_current_data_analysis_dim(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """The same check works on a time append dim."""
    # Dataset ends at 2024-01-02 23:00:00; check 13 hours later.
    now = pd.Timestamp("2024-01-03 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    context = _context(
        analysis_dataset, "time", append_dim_frequency=pd.Timedelta("1h")
    )
    assert (
        not validation.CheckCurrentData(max_delay=timedelta(hours=12))
        .check(context)
        .passed
    )
    assert (
        validation.CheckCurrentData(max_delay=timedelta(hours=14)).check(context).passed
    )


def test_check_current_data_requires_append_dim_frequency(
    forecast_dataset: xr.Dataset,
) -> None:
    with pytest.raises(AssertionError, match="append_dim_frequency"):
        validation.CheckCurrentData(max_delay=timedelta(hours=2)).check(
            _context(forecast_dataset, "init_time")
        )


def test_check_recent_nans_passes(forecast_dataset: xr.Dataset) -> None:
    """Default (no NaNs allowed) passes for a clean dataset."""
    result = validation.CheckRecentNans().check(_context(forecast_dataset, "init_time"))

    assert result.passed
    assert "NaN fraction within" in result.message


def test_check_recent_nans_fails(forecast_dataset: xr.Dataset) -> None:
    """Fails when NaN fraction exceeds threshold."""
    # Add excessive NaNs to the latest init_time
    forecast_dataset["temperature"].loc[
        {"init_time": forecast_dataset.init_time[-1]}
    ] = np.nan

    result = validation.CheckRecentNans(max_nan_fraction=0.1).check(
        _context(forecast_dataset, "init_time")
    )

    assert not result.passed
    assert "Excessive NaN fraction" in result.message
    assert "temperature" in result.message


def test_check_recent_nans_skips_lead_time_0_for_non_instant(
    forecast_dataset: xr.Dataset,
) -> None:
    """lead_time=0 is dropped for vars with step_type != instant."""
    # NaN out lead_time=0 of the precipitation (step_type=accum) var at latest init_time
    forecast_dataset["precipitation"].loc[
        {
            "init_time": forecast_dataset.init_time[-1],
            "lead_time": pd.Timedelta(0),
        }
    ] = np.nan

    result = validation.CheckRecentNans().check(_context(forecast_dataset, "init_time"))

    # Should still pass because lead_time=0 is dropped for non-instant vars.
    assert result.passed


def test_check_recent_nans_skips_lead_time_0_from_template_config(
    forecast_dataset: xr.Dataset,
) -> None:
    """A template var declaring hour_0_values_override=False also drops lead_time=0."""
    # temperature is step_type=instant in the store's attrs but its template config
    # declares it has no hour 0 values (e.g. HRRR categorical vars).
    forecast_dataset["temperature"].loc[
        {
            "init_time": forecast_dataset.init_time[-1],
            "lead_time": pd.Timedelta(0),
        }
    ] = np.nan

    data_vars = (
        NanTestDataVar(name="temperature"),
        NanTestDataVar(name="precipitation"),
    )
    failed = validation.CheckRecentNans().check(
        _context(forecast_dataset, "init_time", data_vars=data_vars)
    )
    assert not failed.passed

    data_vars_with_override = (
        NanTestDataVar(
            name="temperature",
            internal_attrs=BaseInternalAttrs(
                keep_mantissa_bits="no-rounding", hour_0_values_override=False
            ),
        ),
        NanTestDataVar(name="precipitation"),
    )
    passed = validation.CheckRecentNans().check(
        _context(forecast_dataset, "init_time", data_vars=data_vars_with_override)
    )
    assert passed.passed


def test_check_recent_nans_skips_lead_time_0_for_deaccumulated_var(
    forecast_dataset: xr.Dataset,
) -> None:
    """A deaccumulated rate has no lead-0 value even where its source provides a lead-0
    accumulation, so that slice is dropped."""
    forecast_dataset["precipitation"].loc[
        {
            "init_time": forecast_dataset.init_time[-1],
            "lead_time": pd.Timedelta(0),
        }
    ] = np.nan
    forecast_dataset["precipitation"].attrs["step_type"] = "instant"

    data_vars = (
        NanTestDataVar(name="temperature"),
        NanTestDataVar(
            name="precipitation",
            internal_attrs=BaseInternalAttrs(
                keep_mantissa_bits="no-rounding",
                deaccumulate_to_rate=True,
                hour_0_values_override=True,
            ),
        ),
    )

    result = validation.CheckRecentNans().check(
        _context(forecast_dataset, "init_time", data_vars=data_vars)
    )

    assert result.passed


def test_check_recent_nans_include_exclude_vars(
    forecast_dataset: xr.Dataset,
) -> None:
    """include_vars / exclude_vars limit which vars are checked."""
    forecast_dataset["temperature"].loc[
        {"init_time": forecast_dataset.init_time[-1]}
    ] = np.nan

    # Excluding temperature should make the check pass
    result = validation.CheckRecentNans(exclude_vars=["temperature"]).check(
        _context(forecast_dataset, "init_time")
    )
    assert result.passed

    # Including only precipitation should also pass
    result = validation.CheckRecentNans(include_vars=["precipitation"]).check(
        _context(forecast_dataset, "init_time")
    )
    assert result.passed


def test_check_recent_nans_unknown_var_raises(forecast_dataset: xr.Dataset) -> None:
    """A typo'd variable name raises instead of silently checking nothing."""
    with pytest.raises(ValueError, match=r"unknown variables.*not_a_var"):
        validation.CheckRecentNans(include_vars=["not_a_var"]).check(
            _context(forecast_dataset, "init_time")
        )
    with pytest.raises(ValueError, match="unknown variables"):
        validation.CheckRecentNans(exclude_vars=["not_a_var"]).check(
            _context(forecast_dataset, "init_time")
        )


def test_check_recent_nans_empty_selection_raises(
    forecast_dataset: xr.Dataset,
) -> None:
    """A selection excluding every variable is a config error."""
    with pytest.raises(ValueError, match="selects no variables"):
        validation.CheckRecentNans(
            include_vars=["temperature"], exclude_vars=["temperature"]
        ).check(_context(forecast_dataset, "init_time"))


def test_check_recent_nans_selected_vars_missing_from_store_fails(
    forecast_dataset: xr.Dataset,
) -> None:
    """A template var the store does not carry yet fails, not vacuously passes.

    Names are validated against the template's catalog so a partially backfilled
    store is not a config error, but measuring nothing must not report a pass.
    """
    data_vars = (
        NanTestDataVar(name="temperature"),
        NanTestDataVar(name="not_written_yet"),
    )
    result = validation.CheckRecentNans(include_vars=["not_written_yet"]).check(
        _context(forecast_dataset, "init_time", data_vars=data_vars)
    )
    assert not result.passed
    assert "None of the selected variables are in the store" in result.message


def test_check_recent_nans_leading_tier_excuses_newest(
    forecast_dataset: xr.Dataset,
) -> None:
    """A leading 1.0 tier skips the newest position, checking only older ones."""
    # NaN only the latest init_time
    forecast_dataset["temperature"].loc[
        {"init_time": forecast_dataset.init_time[-1]}
    ] = np.nan

    context = _context(forecast_dataset, "init_time")
    # Default targets the bad init_time and fails
    assert not validation.CheckRecentNans().check(context).passed

    # A leading 1.0 tier excuses the still-filling newest init_time and passes
    assert (
        validation.CheckRecentNans(max_nan_fraction=(1.0, 0.0), append_dim_window=2)
        .check(context)
        .passed
    )


def test_check_recent_nans_intermediate_tier(forecast_dataset: xr.Dataset) -> None:
    """A fractional leading tier loosens the newest position without excusing it."""
    # Make the newest init_time ~50% NaN and an older one clean.
    forecast_dataset["temperature"].loc[
        {
            "init_time": forecast_dataset.init_time[-1],
            "lead_time": forecast_dataset.lead_time[::2],
        }
    ] = np.nan

    context = _context(forecast_dataset, "init_time")
    assert (
        validation.CheckRecentNans(max_nan_fraction=(0.6, 0.0), append_dim_window=2)
        .check(context)
        .passed
    )
    assert (
        not validation.CheckRecentNans(max_nan_fraction=(0.3, 0.0), append_dim_window=2)
        .check(context)
        .passed
    )


def test_check_recent_nans_window_catches_older_init(
    forecast_dataset: xr.Dataset,
) -> None:
    """A wider window catches NaNs in a recent-but-not-newest init_time."""
    # NaN a whole init_time that is not the newest (index 2 of 5, 3rd newest)
    bad_init = forecast_dataset.init_time[2]
    forecast_dataset["temperature"].loc[{"init_time": bad_init}] = np.nan

    context = _context(forecast_dataset, "init_time")
    # The test store has one position per shard, so the default two shards miss it.
    assert validation.CheckRecentNans().check(context).passed

    # A window reaching back to it catches the gap and names the offending init_time.
    result = validation.CheckRecentNans(append_dim_shards=3).check(context)
    assert not result.passed
    assert "Excessive NaN fraction" in result.message
    assert pd.Timestamp(bad_init.values).isoformat() in result.message


def test_check_recent_nans_window_all_clean_passes(
    forecast_dataset: xr.Dataset,
) -> None:
    """A clean window passes and reports how many positions were checked."""
    result = validation.CheckRecentNans(append_dim_shards=3).check(
        _context(forecast_dataset, "init_time")
    )
    assert result.passed
    assert "All 3 checked init_time positions" in result.message


def test_check_recent_nans_empty_selection_fails(
    rng: np.random.Generator,
) -> None:
    """A variable left with no values to check must fail, not vacuously pass.

    A non-instant variable's lead_time=0 slice is dropped, so a single-lead
    dataset leaves it empty. An empty selection yields a NaN fraction of NaN,
    which no threshold comparison can catch.
    """
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)
    ds = xr.Dataset(
        {
            "precipitation": (
                ["init_time", "lead_time", "latitude", "longitude"],
                rng.standard_normal((2, 1, len(lats), len(lons))),
                {"step_type": "accum"},
            ),
        },
        coords={
            "init_time": pd.date_range("2024-01-01", periods=2, freq="6h"),
            "lead_time": pd.timedelta_range(start="0h", periods=1, freq="6h"),
            "latitude": lats,
            "longitude": lons,
        },
    )

    result = validation.CheckRecentNans().check(_context(ds, "init_time"))

    assert not result.passed
    assert "precipitation" in result.message


def test_check_recent_nans_fewer_positions_than_tiers_fails(
    rng: np.random.Generator,
) -> None:
    """A store with fewer positions than threshold tiers fails, not silently skips.

    The strict older-position tiers would otherwise never run.
    """
    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "latitude", "longitude"],
                rng.standard_normal((1, 3, 4)),
            )
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=1, freq="1h"),
            "latitude": np.linspace(90, -90, 3),
            "longitude": np.linspace(-180, 180, 4),
        },
    )
    result = validation.CheckRecentNans(
        max_nan_fraction=(1.0, 0.0), append_dim_window=2
    ).check(_context(ds, "time"))
    assert not result.passed
    assert "need at least 2" in result.message


def test_check_recent_nans_selects_by_position_not_clock(
    analysis_dataset: xr.Dataset,
) -> None:
    """Positions are chosen positionally, so recency does not affect what is checked.

    Recency is CheckCurrentData's question; an off-schedule run still checks real data.
    """
    context = _context(analysis_dataset, "time")
    assert validation.CheckRecentNans().check(context).passed

    # Proves it selected real values rather than passing vacuously.
    analysis_dataset["temperature"].loc[{"time": slice("2024-01-02 22:00", None)}] = (
        np.nan
    )
    result = validation.CheckRecentNans().check(context)
    assert not result.passed
    assert "temperature" in result.message


def test_check_recent_nans_checks_each_position_independently(
    analysis_dataset: xr.Dataset,
) -> None:
    """One ruined timestep fails even when its neighbours are clean.

    A single fraction over the whole window would average it away, which is what
    forced structural thresholds to encode expected emptiness.
    """
    analysis_dataset["temperature"].loc[{"time": "2024-01-02 23:00"}] = np.nan

    result = validation.CheckRecentNans(
        append_dim_window=2, max_nan_fraction=0.4
    ).check(_context(analysis_dataset, "time"))

    assert not result.passed
    assert "2024-01-02T23:00:00 exceeds 0.4" in result.message


def test_check_recent_nans_deep_window_checks_every_position(
    analysis_dataset: xr.Dataset,
) -> None:
    """A configured shard depth catches a gap anywhere in its range."""
    context = _context(analysis_dataset, "time")

    # Ruin a deep position the default two-shard range would never see.
    analysis_dataset["temperature"].loc[{"time": "2024-01-01 12:00"}] = np.nan
    assert validation.CheckRecentNans().check(context).passed

    result = validation.CheckRecentNans(append_dim_shards=48).check(context)
    assert not result.passed
    assert "2024-01-01T12:00:00" in result.message


def test_check_recent_nans_defaults_to_two_trailing_shards(
    analysis_dataset: xr.Dataset,
) -> None:
    """Two shards cover the previous shard when an update crosses a boundary."""
    ds = analysis_dataset.isel(time=slice(0, 7))
    ds["temperature"].loc[{"time": "2024-01-01 03:00"}] = np.nan
    context = _context(ds, "time", append_dim_shard_size=3)

    assert validation.CheckRecentNans(append_dim_shards=1).check(context).passed
    result = validation.CheckRecentNans().check(context)
    assert not result.passed
    assert "2024-01-01T03:00:00" in result.message


def test_check_recent_nans_explicit_window_bounds_the_read(
    analysis_dataset: xr.Dataset,
) -> None:
    """Whole-grid strategies can bound the read to an explicit position count."""
    context = _context(analysis_dataset, "time", append_dim_shard_size=6)

    result = validation.CheckRecentNans(
        append_dim_window=2, spatial_sampling="all"
    ).check(context)
    assert result.passed
    assert "2 most recent" in result.message


def test_check_recent_nans_partial_trailing_shard(
    analysis_dataset: xr.Dataset,
) -> None:
    """The operational norm: an update has appended part of a new shard.

    The window must reach back over the whole previous shard, not just the positions
    written into the partial one.
    """
    ds = analysis_dataset.isel(time=slice(0, 8))  # shards of 3 -> 3, 3, 2
    ds["temperature"].loc[{"time": "2024-01-01 03:00"}] = np.nan  # oldest of shard 2
    context = _context(ds, "time", append_dim_shard_size=3)

    result = validation.CheckRecentNans().check(context)
    assert not result.passed
    assert "2024-01-01T03:00:00" in result.message
    # Shards 2 and 3 span positions 3..7, so five positions are checked.
    assert validation.CheckRecentNans()._resolve_window(ds, "time") == 5


def test_check_recent_nans_shards_deeper_than_the_store(
    analysis_dataset: xr.Dataset,
) -> None:
    """A young store holds fewer shards than asked for; check all of it, don't fail."""
    ds = analysis_dataset.isel(time=slice(0, 4))
    context = _context(ds, "time", append_dim_shard_size=3)

    assert (
        validation.CheckRecentNans(append_dim_shards=10)._resolve_window(ds, "time")
        == 4
    )
    assert validation.CheckRecentNans(append_dim_shards=10).check(context).passed


def test_check_recent_nans_window_deeper_than_the_store(
    analysis_dataset: xr.Dataset,
) -> None:
    """An explicit window past the store's extent is clamped to what exists."""
    ds = analysis_dataset.isel(time=slice(0, 3))
    result = validation.CheckRecentNans(append_dim_window=100).check(
        _context(ds, "time")
    )
    assert result.passed
    assert "3 most recent" in result.message


def test_check_recent_nans_window_from_selected_vars_shards(
    analysis_dataset: xr.Dataset,
) -> None:
    """Shard size is read from the selected variables, so a store whose variables
    shard differently along the append dim still resolves a window."""
    ds = analysis_dataset.isel(time=slice(0, 6))
    context = _context(ds, "time", append_dim_shard_size=3)
    # humidity shards differently; a check over both variables could not resolve.
    ds["humidity"].encoding["shards"] = tuple(
        2 if dim == "time" else size for dim, size in ds["humidity"].sizes.items()
    )

    assert (
        validation.CheckRecentNans(include_vars=["temperature"]).check(context).passed
    )
    with pytest.raises(AssertionError, match="Inconsistent shards sizes"):
        validation.CheckRecentNans().check(context)


def test_check_recent_nans_rejects_non_float_vars(
    analysis_dataset: xr.Dataset,
) -> None:
    """An integer variable holds no NaN, so a NaN check on one would always pass."""
    ds = analysis_dataset.copy()
    ds["flags"] = ds["temperature"].astype("int16")
    with pytest.raises(ValueError, match="cannot hold NaN"):
        validation.CheckRecentNans(include_vars=["flags"]).check(_context(ds, "time"))
    # Excluding it leaves the float variables checkable.
    assert (
        validation.CheckRecentNans(exclude_vars=["flags"])
        .check(_context(ds, "time"))
        .passed
    )


def test_check_recent_nans_logged_worst_excludes_excused_positions(
    caplog: pytest.LogCaptureFixture, analysis_dataset: xr.Dataset
) -> None:
    """Point sampling reads an excused position anyway (one pass), so the logged
    worst fraction must still ignore it, or a still-filling newest position looks
    like a problem in the logs."""
    analysis_dataset["temperature"].loc[{"time": "2024-01-02 23:00"}] = np.nan
    context = _context(analysis_dataset, "time", append_dim_shard_size=6)

    with caplog.at_level(logging.INFO, logger="reformatters.common.validation"):
        assert (
            validation.CheckRecentNans(max_nan_fraction=(1.0, 0.0))
            .check(context)
            .passed
        )

    (record,) = [r for r in caplog.records if "NaN fractions" in r.message]
    assert "temperature=0.000000" in record.message


def test_check_recent_nans_excludes_structurally_dead_points(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A point that is NaN at every position is a structural hole, not a gap.

    Without excluding it, an ocean/out-of-domain cell would fail every run; with only
    dead points sampled there is nothing to measure, which must fail rather than pass.
    """
    times = pd.date_range("2024-01-01", periods=6, freq="1h")
    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "latitude", "longitude"],
                rng.standard_normal((len(times), 2, 2)),
            )
        },
        coords={
            "time": times,
            "latitude": [10.0, 20.0],
            "longitude": [30.0, 40.0],
        },
    )
    # One column is NaN at every position (structural), the rest is clean.
    ds["temperature"].loc[{"latitude": 10.0, "longitude": 30.0}] = np.nan
    context = _context(ds, "time")

    # Sample every cell: points are drawn independently, so a 2x2 draw can land wholly
    # on the dead cell, which is the case below rather than this one.
    def sample_every_cell(
        ds: xr.Dataset,
        sampling_strategy: validation.SpatialSamplingStrategy,
        num_points: int = 0,
    ) -> xr.Dataset:
        return ds.isel(
            longitude=xr.DataArray([0, 0, 1, 1], dims="point"),
            latitude=xr.DataArray([0, 1, 0, 1], dims="point"),
        )

    monkeypatch.setattr(validation, "_apply_spatial_sampling", sample_every_cell)

    assert validation.CheckRecentNans().check(context).passed

    # With every sampled point dead there is nothing measurable: fail, don't pass.
    all_dead = ds.copy(deep=True)
    all_dead["temperature"].values[:] = np.nan
    result = validation.CheckRecentNans().check(_context(all_dead, "time"))
    assert not result.passed
    assert "No values selected" in result.message


def test_check_recent_nans_sampled_points_config_validation() -> None:
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(sampled_points=0)


def test_check_recent_nans_config_validation() -> None:
    """Misconfiguration is rejected at construction, not at the first cron fire."""
    # Unknown field name
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(max_nan_fracton=0.1)  # ty: ignore[unknown-argument]
    # Invalid sampling strategy
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(spatial_sampling="invalid")  # ty: ignore[invalid-argument-type]
    # More tiers than window
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(
            max_nan_fraction=(1.0, 0.5, 0.0), append_dim_window=2
        )
    # Threshold out of range
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(max_nan_fraction=1.5)
    # All tiers 1.0 would check nothing
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(max_nan_fraction=(1.0, 1.0), append_dim_window=2)
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(append_dim_window=0)
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(append_dim_shards=0)
    # Both bounds set at once is ambiguous
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(append_dim_window=2, append_dim_shards=2)
    # No thresholds to check against
    with pytest.raises(pydantic.ValidationError):
        validation.CheckRecentNans(max_nan_fraction=())


def test_check_nan_fractions_logs_one_record_for_all_variables(
    caplog: pytest.LogCaptureFixture, analysis_dataset: xr.Dataset
) -> None:
    """Per-variable fractions arrive in a single log record.

    Sentry drops log records emitted in bursts within the same second, so a line
    per variable loses exactly the detail a later inspection needs.
    """
    with caplog.at_level(logging.INFO, logger="reformatters.common.validation"):
        assert (
            validation.CheckRecentNans(append_dim_window=1)
            .check(_context(analysis_dataset, "time"))
            .passed
        )

    fraction_records = [
        r.getMessage() for r in caplog.records if "NaN fractions" in r.getMessage()
    ]
    assert len(fraction_records) == 1
    assert "temperature=0.000000" in fraction_records[0]
    assert "humidity=0.000000" in fraction_records[0]


def test_check_recent_nans_quarter_sampling_passes(
    analysis_dataset: xr.Dataset,
) -> None:
    result = validation.CheckRecentNans(spatial_sampling="quarter").check(
        _context(analysis_dataset, "time")
    )

    assert result.passed


def test_check_recent_nans_quarter_sampling_fails(
    analysis_dataset: xr.Dataset,
) -> None:
    """Quarter sampling catches excessive NaNs covering the dataset."""
    analysis_dataset["temperature"].loc[{"time": slice("2024-01-02", None)}] = np.nan

    result = validation.CheckRecentNans(
        max_nan_fraction=0.05, spatial_sampling="quarter"
    ).check(_context(analysis_dataset, "time"))

    assert not result.passed
    assert "Excessive NaN fraction" in result.message
    assert "temperature" in result.message


def test_check_recent_nans_quarter_sampling_different_quarters(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Quarter sampling selects different quarters based on RNG."""
    lat_size = len(analysis_dataset.latitude)
    lon_size = len(analysis_dataset.longitude)

    analysis_dataset["temperature"].loc[{"time": slice("2024-01-02", None)}] = np.nan
    analysis_dataset["temperature"].isel(
        time=slice(-24, None),
        latitude=slice(lat_size // 2, lat_size),
        longitude=slice(lon_size // 2, lon_size),
    ).values[:] = 1.0

    class MockRngBottomRight:
        def integers(
            self, _low: int, _high: int, size: int | None = None
        ) -> int | np.ndarray:
            if size is not None:
                return np.full(size, 1)
            return 1

    monkeypatch.setattr(
        "reformatters.common.validation.np.random.default_rng",
        lambda seed=None: MockRngBottomRight(),
    )

    result = validation.CheckRecentNans(
        max_nan_fraction=0.05, spatial_sampling="quarter"
    ).check(_context(analysis_dataset, "time"))
    assert result.passed

    class MockRngTopLeft:
        def integers(
            self, _low: int, _high: int, size: int | None = None
        ) -> int | np.ndarray:
            if size is not None:
                return np.zeros(size, dtype=int)
            return 0

    monkeypatch.setattr(
        "reformatters.common.validation.np.random.default_rng",
        lambda seed=None: MockRngTopLeft(),
    )

    result = validation.CheckRecentNans(
        max_nan_fraction=0.05, spatial_sampling="quarter"
    ).check(_context(analysis_dataset, "time"))
    assert not result.passed


def test_check_recent_nans_random_points_sampling(
    analysis_dataset: xr.Dataset,
) -> None:
    """random_points strategy selects N spatial points."""
    result = validation.CheckRecentNans(spatial_sampling="random_points").check(
        _context(analysis_dataset, "time")
    )
    assert result.passed

    # Inject NaNs at every point so any random sample sees them
    analysis_dataset["temperature"].loc[{"time": slice("2024-01-02", None)}] = np.nan
    result = validation.CheckRecentNans(
        max_nan_fraction=0.05, spatial_sampling="random_points"
    ).check(_context(analysis_dataset, "time"))
    assert not result.passed


def test_check_recent_nans_xy_dimensions(rng: np.random.Generator) -> None:
    """CheckRecentNans works with x/y dimensions instead of lat/lon."""
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

    result = validation.CheckRecentNans(spatial_sampling="quarter").check(
        _context(ds, "time")
    )
    assert result.passed


def test_check_recent_nans_all_sampling(analysis_dataset: xr.Dataset) -> None:
    """spatial_sampling='all' reads the full spatial grid."""
    analysis_dataset["temperature"].loc[{"time": slice("2024-01-02", None)}] = np.nan

    result = validation.CheckRecentNans(
        spatial_sampling="all", max_nan_fraction=0.05
    ).check(_context(analysis_dataset, "time"))
    assert not result.passed
    assert "temperature" in result.message


def test_spatial_dims_raises_when_unknown(rng: np.random.Generator) -> None:
    """Sampling on a dataset without lat/lon or x/y dims raises ValueError."""
    times = pd.date_range("2024-01-01", periods=48, freq="1h")
    ds = xr.Dataset(
        {"temperature": (["time", "row", "col"], rng.standard_normal((48, 10, 10)))},
        coords={"time": times, "row": np.arange(10), "col": np.arange(10)},
    )

    with pytest.raises(ValueError, match="Can't infer spatial dimensions"):
        validation.CheckRecentNans(spatial_sampling="quarter").check(
            _context(ds, "time")
        )


def test_truncate_shards_truncation() -> None:
    """_truncate_shards collapses long lists with head/tail and an ellipsis."""
    shards = [str(i) for i in range(10)]
    out = validation._truncate_shards(shards, keep=3)
    assert out == "[0, 1, 2, ..., 7, 8, 9]"


def test_summarize_coords_multidimensional() -> None:
    """Multi-dimensional coords (e.g. valid_time) collapse to a first..last range."""
    init_times = pd.date_range("2026-06-05T12:00", periods=1, freq="6h")
    lead_times = pd.timedelta_range(start="0h", periods=209, freq="1h")
    valid_time = xr.DataArray(
        init_times.values[:, None] + lead_times.values[None, :],
        dims=("init_time", "lead_time"),
    )
    ds = xr.Dataset(
        coords={
            "init_time": ("init_time", init_times),
            "lead_time": ("lead_time", lead_times),
            "valid_time": valid_time,
        }
    )

    summary = validation._summarize_coords(ds)

    assert "valid_time=[2026-06-05T12:00:00..2026-06-14T04:00:00] (n=209)" in summary
    # The full nested array must not be dumped.
    assert "\n" not in summary
    assert "init_time=2026-06-05T12:00:00" in summary
    assert "lead_time=[0 days 00:00:00..8 days 16:00:00] (n=209)" in summary


class PassingCheck(validation.Validator):
    def check(
        self,
        context: validation.ValidationContext,  # noqa: ARG002
    ) -> validation.ValidationResult:
        return validation.ValidationResult(passed=True, message="ok")


class FailingCheck(validation.Validator):
    def check(
        self,
        context: validation.ValidationContext,  # noqa: ARG002
    ) -> validation.ValidationResult:
        return validation.ValidationResult(passed=False, message="bad thing")


class AlsoFailingCheck(validation.Validator):
    def check(
        self,
        context: validation.ValidationContext,  # noqa: ARG002
    ) -> validation.ValidationResult:
        return validation.ValidationResult(passed=False, message="another bad thing")


def _write_test_store(rng: np.random.Generator) -> zarr.storage.MemoryStore:
    times = pd.date_range("2024-01-01", periods=4, freq="1h")
    ds = xr.Dataset(
        {"temperature": (["time", "y", "x"], rng.standard_normal((4, 4, 4)))},
        coords={"time": times, "y": np.arange(4), "x": np.arange(4)},
    )
    store = zarr.storage.MemoryStore()
    # Match production stores, which carry consolidated metadata (see template_utils.write_metadata),
    # so validate_dataset opens them without the consolidated-fallback warning. Writing it emits a
    # spec-compatibility UserWarning that production suppresses too.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part in the Zarr format 3 specification",
            category=UserWarning,
        )
        ds.to_zarr(store, mode="w")
    return store


def test_validate_dataset_raises_on_failed_validator(
    rng: np.random.Generator,
) -> None:
    """validate_dataset raises naming the dataset and listing failed check names and messages."""
    store = _write_test_store(rng)

    with pytest.raises(
        validation.OperationalValidationError,
        match="d validation failed:\n- FailingCheck: bad thing",
    ):
        validation.validate_dataset(
            [PassingCheck(), FailingCheck()],
            store=store,
            append_dim="time",
            dataset_id="d",
        )

    # Passing-only validators should not raise.
    validation.validate_dataset(
        [PassingCheck()], store=store, append_dim="time", dataset_id="d"
    )


def test_validate_dataset_requires_region_job_for_virtual_checks(
    rng: np.random.Generator,
) -> None:
    """A manifest-probing check without a region job is caught before any check runs."""
    store = _write_test_store(rng)
    with pytest.raises(AssertionError, match="require a region_job"):
        validation.validate_dataset(
            [validation.CheckVirtualManifestCompleteness()],
            store=store,
            append_dim="time",
            dataset_id="d",
        )


def test_validate_dataset_fingerprints_by_dataset_and_check(
    caplog: pytest.LogCaptureFixture, rng: np.random.Generator
) -> None:
    """A failure fingerprints by (dataset_id, failed check names), not by message text, so
    repeated failures carrying different per-run details (a NaN fraction, an init_time,
    ...) group into one Sentry issue instead of each filing a new one. The fingerprint
    goes on the isolation scope, which outlives validate_dataset's frame, so it is still
    in effect when Sentry captures the raised exception at process exit — that capture
    would otherwise group by the dataset-agnostic raise site. And no failure is logged at
    ERROR, which the Sentry logging integration would report as a second, differently
    grouped issue.
    """
    store = _write_test_store(rng)

    with sentry_sdk.isolation_scope():
        with pytest.raises(validation.OperationalValidationError):
            validation.validate_dataset(
                [FailingCheck(), AlsoFailingCheck()],
                store=store,
                append_dim="time",
                dataset_id="noaa-gfs-forecast",
            )

        assert sentry_sdk.get_isolation_scope()._fingerprint == [
            "noaa-gfs-forecast",
            "FailingCheck",
            "AlsoFailingCheck",
        ]

    failure_logs = [r for r in caplog.records if "Failed " in r.getMessage()]
    assert [r.levelno for r in failure_logs] == [logging.WARNING, logging.WARNING]


def test_validator_name() -> None:
    """Instance names distinguish multiple instances of one check class."""
    assert PassingCheck().name == "PassingCheck"
    assert validation.CheckRecentNans().name == "CheckRecentNans"
    assert (
        validation.CheckRecentNans(include_vars=["a", "b"]).name
        == "CheckRecentNans(include=a,b)"
    )
    assert (
        validation.CheckRecentNans(exclude_vars=["a", "b", "c", "d"]).name
        == "CheckRecentNans(exclude=a,b,c,+1)"
    )
    assert (
        validation.CheckVirtualManifestCompleteness().name
        == "CheckVirtualManifestCompleteness"
    )


def _replica_and_primary(
    rng: np.random.Generator,
) -> tuple[xr.Dataset, xr.Dataset]:
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

    return primary_ds.copy(deep=True), primary_ds


def test_check_replica_matches_primary_coords_divergence(
    rng: np.random.Generator,
) -> None:
    replica_ds, primary_ds = _replica_and_primary(rng)
    replica_ds = replica_ds.isel(time=slice(None, -1))

    result = validation.CheckReplicaMatchesPrimary().check(
        _context(replica_ds, "time", primary_ds=primary_ds)
    )

    assert not result.passed
    assert "different for coords: ['time']" in result.message


def test_check_replica_matches_primary_vars_divergence(
    rng: np.random.Generator,
) -> None:
    replica_ds, primary_ds = _replica_and_primary(rng)
    chunk_size = 8
    replica_ds["temperature"].values[-chunk_size:, 0, 0] = 999.0

    result = validation.CheckReplicaMatchesPrimary().check(
        _context(replica_ds, "time", primary_ds=primary_ds)
    )

    assert not result.passed
    assert (
        "different for at least the following vars: ['temperature']" in result.message
    )


def test_check_replica_matches_primary_passes(
    rng: np.random.Generator,
) -> None:
    replica_ds, primary_ds = _replica_and_primary(rng)

    result = validation.CheckReplicaMatchesPrimary().check(
        _context(replica_ds, "time", primary_ds=primary_ds)
    )

    assert result.passed
    assert "replica and primary stores is the same" in result.message


def _sharded_dataset(rng: np.random.Generator, periods: int = 16) -> xr.Dataset:
    times = pd.date_range("2024-01-01", periods=periods, freq="1h")
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

    chunk_sizes = (4, 4, 5)
    shard_sizes = (4, 4, 5)
    for var in ds.data_vars.values():
        var.encoding["chunks"] = chunk_sizes
        var.encoding["shards"] = shard_sizes
    return ds


def test_check_expected_shards_passes(rng: np.random.Generator) -> None:
    ds = _sharded_dataset(rng)
    store = zarr.storage.MemoryStore()
    ds.to_zarr(store, mode="w", consolidated=False)

    result = validation.CheckExpectedShards().check(_context(ds, "time", store=store))

    assert result.passed
    assert "All variables have expected shards" in result.message


def test_check_expected_shards_fails_missing_shards(
    rng: np.random.Generator,
) -> None:
    ds = _sharded_dataset(rng)
    store = zarr.storage.MemoryStore()
    ds.to_zarr(store, mode="w", consolidated=False)

    # Delete different shards per variable to exercise the per-var details branch
    zarr.core.sync.sync(store.delete("temperature/c/0/0/0"))
    zarr.core.sync.sync(store.delete("humidity/c/1/0/0"))
    zarr.core.sync.sync(store.delete("humidity/c/2/0/0"))

    result = validation.CheckExpectedShards().check(_context(ds, "time", store=store))

    assert not result.passed
    assert result.message == (
        "Missing shards: temperature (1 missing), humidity (2 missing). "
        "temperature: [0/0/0], humidity: [1/0/0, 2/0/0]"
    )


def test_check_expected_shards_fails_same_missing_shards_across_vars(
    rng: np.random.Generator,
) -> None:
    """When all problem vars are missing the same shards the message is collapsed."""
    ds = _sharded_dataset(rng)
    store = zarr.storage.MemoryStore()
    ds.to_zarr(store, mode="w", consolidated=False)

    # Delete the same shards for both variables
    for var in ("temperature", "humidity"):
        zarr.core.sync.sync(store.delete(f"{var}/c/0/0/0"))
        zarr.core.sync.sync(store.delete(f"{var}/c/1/0/0"))

    result = validation.CheckExpectedShards().check(_context(ds, "time", store=store))

    assert not result.passed
    assert result.message == (
        "Missing shards: temperature (2 missing), humidity (2 missing). "
        "all missing the same shards: [0/0/0, 1/0/0]"
    )


def test_check_expected_shards_passes_with_extra_shards(
    rng: np.random.Generator,
) -> None:
    """Extra shards beyond the metadata's extent are fine (operational trim)."""
    ds = _sharded_dataset(rng, periods=8)
    chunk_sizes = (4, 4, 5)
    shard_sizes = (4, 4, 5)

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
    result = validation.CheckExpectedShards().check(
        _context(ds_trimmed, "time", store=store)
    )

    assert result.passed
    assert "All variables have expected shards" in result.message


def test_check_expected_shards_icechunk_store(rng: np.random.Generator) -> None:
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

    result = validation.CheckExpectedShards().check(_context(ds, "time", store=store))

    assert result.passed
    assert "All variables have expected shards" in result.message


def _grouped_store(rng: np.random.Generator) -> zarr.storage.MemoryStore:
    """A store with a root variable plus a `pressure_level` group variable carrying a
    dimension the root variable does not have."""
    times = pd.date_range("2024-01-01", periods=16, freq="1h")
    y = np.arange(8)
    x = np.arange(10)
    levels = np.array([500, 850])

    root_ds = xr.Dataset(
        {
            "temperature_2m": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))),
            ),
        },
        coords={"time": times, "y": y, "x": x},
        attrs={"dataset_id": "test-dataset"},
    )
    root_ds["temperature_2m"].encoding.update(
        {"chunks": (4, 4, 5), "shards": (4, 4, 5)}
    )

    group_ds = xr.Dataset(
        {
            "temperature": (
                ["time", "pressure_level", "y", "x"],
                rng.standard_normal((len(times), len(levels), len(y), len(x))),
            ),
        },
        coords={"time": times, "pressure_level": levels, "y": y, "x": x},
    )
    group_ds["temperature"].encoding.update(
        {"chunks": (4, 1, 4, 5), "shards": (4, 1, 4, 5)}
    )

    tree = xr.DataTree.from_dict({"/": root_ds, "/pressure_level": group_ds})
    store = zarr.storage.MemoryStore()
    tree.to_zarr(store, mode="w", consolidated=False, write_inherited_coords=True)
    return store


def test_check_expected_shards_passes_with_vertical_group(
    rng: np.random.Generator,
) -> None:
    store = _grouped_store(rng)
    ds = validation.open_flattened_dataset(store, consolidated=False)

    result = validation.CheckExpectedShards().check(_context(ds, "time", store=store))

    assert result.passed
    assert "All variables have expected shards" in result.message


def test_check_expected_shards_fails_missing_group_shards(
    rng: np.random.Generator,
) -> None:
    store = _grouped_store(rng)
    ds = validation.open_flattened_dataset(store, consolidated=False)

    zarr.core.sync.sync(store.delete("pressure_level/temperature/c/0/1/0/0"))

    result = validation.CheckExpectedShards().check(_context(ds, "time", store=store))

    assert not result.passed
    assert result.message == (
        "Missing shards: pressure_level/temperature (1 missing). "
        "pressure_level/temperature: [0/1/0/0]"
    )


def test_check_replica_matches_primary_passes_with_vertical_group(
    rng: np.random.Generator,
) -> None:
    store = _grouped_store(rng)
    primary_ds = validation.open_flattened_dataset(store, consolidated=False)
    replica_ds = primary_ds.copy(deep=True)

    result = validation.CheckReplicaMatchesPrimary().check(
        _context(replica_ds, "time", primary_ds=primary_ds)
    )

    assert result.passed
    assert "replica and primary stores is the same" in result.message
