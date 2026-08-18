from __future__ import annotations

import abc
import itertools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    assert_never,
    cast,
)

import numpy as np
import pandas as pd
import pydantic
import sentry_sdk
import xarray as xr
import zarr
import zarr.core.sync
import zarr.storage
from icechunk.store import IcechunkStore
from zarr.abc.store import Store

from reformatters.common import iterating
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.retry import retry

if TYPE_CHECKING:
    from reformatters.common.config_models import DataVar
    from reformatters.common.region_job import SourceFileCoord
    from reformatters.common.virtual_region_job import VirtualRegionJob

log = get_logger(__name__)


SpatialSamplingStrategy = Literal["all", "quarter", "random_points"]

# Default ThreadPool size per sampling strategy. Random points are tiny so we
# can saturate S3 with many concurrent reads; "all" reads big slabs so keep
# parallelism low to bound memory; "quarter" is in between.
_DEFAULT_MAX_WORKERS: dict[SpatialSamplingStrategy, int] = {
    "random_points": 12,
    "quarter": 4,
    "all": 2,
}
_NUM_RANDOM_POINTS = 2


class ValidationResult(pydantic.BaseModel):
    """Result of a validation check."""

    passed: bool
    message: str
    checked_count: int | None = (
        None  # items/references checked, when the validator tracks it
    )


class OperationalValidationError(ValueError):
    """Raised by validate_dataset when one or more checks fail."""


@dataclass(frozen=True)
class ValidationContext:
    """Everything a validator may draw on. Each validator uses the subset it needs."""

    store: zarr.storage.StoreLike
    # The store's contents as one flat Dataset covering every group
    # (see open_flattened_dataset); variables are keyed by store path.
    ds: xr.Dataset
    append_dim: str
    # The template config's append_dim_frequency; None when no template is in play.
    append_dim_frequency: pd.Timedelta | None = None
    # The template config's declared variables; empty when no template is in play
    # (offline scripts, tests). Lets a check consult config that is not written to
    # the store (has_hour_0_values, internal attrs) and validate variable names.
    data_vars: Sequence[DataVar[Any]] = ()
    # The operational-window job, present when validating a virtual dataset.
    region_job: VirtualRegionJob[Any, Any] | None = None
    # The primary store's dataset, present when `store` is a replica.
    primary_ds: xr.Dataset | None = None

    def virtual_region_job(self) -> VirtualRegionJob[Any, Any]:
        assert self.region_job is not None, (
            "this check requires a virtual dataset's operational-window region job"
        )
        return self.region_job

    def known_var_paths(self) -> Sequence[str]:
        """The variable paths a check's include/exclude filters are validated against:
        the template's when available (the complete catalog, even when a partially
        backfilled store carries a subset), else the opened store's."""
        if self.data_vars:
            return [var.path for var in self.data_vars]
        return [str(name) for name in self.ds.data_vars]


class Validator(FrozenBaseModel, abc.ABC):
    """One operational check: configuration (validated pydantic fields) plus logic.

    Datasets list configured instances in DynamicalDataset.validators(); validate_dataset
    runs each against a ValidationContext. Unknown field names are rejected at
    construction, so a config typo fails when the dataset module imports, not at the
    first cron fire.
    """

    model_config = pydantic.ConfigDict(frozen=True, strict=True, extra="forbid")

    # True for checks that probe a virtual dataset's manifest; validate_dataset
    # requires a region_job when any listed check sets this.
    requires_virtual_dataset: ClassVar[bool] = False

    @property
    def name(self) -> str:
        """Identifies this check in log lines, failure messages, and the Sentry
        fingerprint. Includes the variable selection so multiple instances of one
        class are distinguishable."""
        label = self.selection_label() if isinstance(self, VariableSelection) else None
        return f"{type(self).__name__}({label})" if label else type(self).__name__

    @abc.abstractmethod
    def check(self, context: ValidationContext) -> ValidationResult: ...


class VariableSelection(FrozenBaseModel):
    """include_vars/exclude_vars fields for checks that can target a variable subset.

    Variables are named by store path (`<group>/<name>`, or the bare name at the root).
    Names are validated against the known catalog — a typo raises instead of silently
    checking nothing.
    """

    include_vars: Sequence[str] | Literal["all"] = "all"
    exclude_vars: Sequence[str] = ()

    def selects(self, var_path: str) -> bool:
        return (
            self.include_vars == "all" or var_path in self.include_vars
        ) and var_path not in self.exclude_vars

    def validate_var_names(self, known_var_paths: Sequence[str]) -> None:
        named = set(self.exclude_vars)
        if self.include_vars != "all":
            named |= set(self.include_vars)
        unknown = sorted(named - set(known_var_paths))
        if unknown:
            raise ValueError(
                f"{type(self).__name__} names unknown variables {unknown}. "
                f"Known variable paths: {sorted(known_var_paths)}"
            )
        if not any(self.selects(path) for path in known_var_paths):
            raise ValueError(f"{type(self).__name__} selects no variables")

    def selection_label(self) -> str | None:
        if self.include_vars != "all":
            return f"include={self._truncate(self.include_vars)}"
        if self.exclude_vars:
            return f"exclude={self._truncate(self.exclude_vars)}"
        return None

    @staticmethod
    def _truncate(names: Sequence[str], keep: int = 3) -> str:
        if len(names) <= keep:
            return ",".join(names)
        return f"{','.join(names[:keep])},+{len(names) - keep}"


def open_flattened_dataset(
    store: zarr.storage.StoreLike, *, consolidated: bool
) -> xr.Dataset:
    """Open a store as one flat Dataset covering every group.

    xr.open_zarr reads only the root group, so a multi-group dataset's vertical-group
    variables (e.g. ``pressure_level/temperature``) would be invisible to validators —
    silently shrinking coverage to root-only. Opening the whole DataTree and flattening
    it (iterating.flatten_groups) exposes every group var keyed by its store path; root
    vars keep their bare names. Validators key variables — and their include/exclude
    filters — by that path, which is unique across groups. A single-group store flattens
    to exactly its root dataset.
    """
    tree = xr.open_datatree(
        store,  # ty: ignore[invalid-argument-type]
        engine="zarr",
        chunks=None,
        consolidated=consolidated,
        decode_timedelta=True,  # so lead_time selects by pd.Timedelta label
    )
    return iterating.flatten_groups(tree)


def validate_dataset(
    validators: Sequence[Validator],
    *,
    store: zarr.storage.StoreLike,
    append_dim: str,
    dataset_id: str,
    append_dim_frequency: pd.Timedelta | None = None,
    data_vars: Sequence[DataVar[Any]] = (),
    region_job: VirtualRegionJob[Any, Any] | None = None,
    primary_ds: xr.Dataset | None = None,
) -> None:
    """
    Validate a zarr dataset by running a series of quality checks.

    Args:
        validators: the checks to run.
        store: the zarr/icechunk store to validate.
        append_dim: the dataset's append dimension.
        dataset_id: identifies the dataset in the Sentry fingerprint of a failure.
        append_dim_frequency: the template config's append-dim frequency.
        data_vars: the template config's declared variables (see ValidationContext).
        region_job: the operational-window job, required when any validator sets
            requires_virtual_dataset.
        primary_ds: the primary store's dataset, when `store` is a replica.

    Raises:
        OperationalValidationError: If any validation checks fail
    """
    log.info(f"Validating zarr {store}")

    virtual_checks = [v.name for v in validators if v.requires_virtual_dataset]
    assert not virtual_checks or region_job is not None, (
        f"{virtual_checks} require a region_job but validate_dataset was called without one"
    )

    consolidated = not isinstance(store, IcechunkStore)

    failures: list[tuple[str, str]] = []
    for validator in validators:
        # A fresh open per check bounds memory: nothing one check loaded keeps the
        # next check's working set alive.
        ds = open_flattened_dataset(store, consolidated=consolidated)
        context = ValidationContext(
            store=store,
            ds=ds,
            append_dim=append_dim,
            append_dim_frequency=append_dim_frequency,
            data_vars=tuple(data_vars),
            region_job=region_job,
            primary_ds=primary_ds,
        )
        result = validator.check(context)

        if not result.passed:
            # Warn don't error; the raise below creates a single Sentry issue.
            log.warning(f"Failed {validator.name}: {result.message}")
            failures.append((validator.name, result.message))
        else:
            log.info(f"Passed {validator.name}: {result.message}")

        ds.close()
        del ds

    if failures:
        message = "Zarr validation failed:\n" + "\n".join(
            f"- {name}: {message}" for name, message in failures
        )
        # Fingerprint by (dataset_id, failed check names), not message text, so repeated
        # failures carrying different per-run details group into one Sentry issue.
        sentry_sdk.get_isolation_scope().fingerprint = [
            dataset_id,
            *[name for name, _ in failures],
        ]
        raise OperationalValidationError(message)

    log.info("Zarr validation passed all checks")


class CheckCurrentData(Validator):
    """Fail when an append-dim position that is due has not been ingested.

    A position is due `max_delay` after its own timestamp: the source's publication
    delay plus our update duration plus slack — typically the validation cron's
    offset after the cycle it validates. Deadlines attach to grid positions (the
    template's append_dim_frequency, anchored to the dataset's own positions), not
    to the moment the check runs, so a tight deadline holds at any wall-clock time
    and an off-schedule run never alerts on a gap that is merely the normal wait
    for the next cycle.
    """

    max_delay: timedelta

    def check(self, context: ValidationContext) -> ValidationResult:
        append_dim = context.append_dim
        frequency = context.append_dim_frequency
        assert frequency is not None, (
            "CheckCurrentData requires the template's append_dim_frequency on the context"
        )
        index = context.ds.get_index(append_dim)
        if len(index) == 0:
            return ValidationResult(
                passed=False, message=f"Dataset has no {append_dim} positions"
            )
        first = pd.Timestamp(index.min())
        latest = pd.Timestamp(index.max())
        # The newest grid position whose deadline has passed.
        due = (
            first
            + ((pd.Timestamp.now() - self.max_delay - first) // frequency) * frequency
        )
        if due < first:
            return ValidationResult(
                passed=True,
                message=f"No {append_dim} position is due yet (positions are due "
                f"{self.max_delay} after their timestamp)",
            )
        if latest < due:
            return ValidationResult(
                passed=False,
                message=f"Missing {append_dim}={due.isoformat()}, which was due "
                f"{self.max_delay} after its timestamp "
                f"(latest present is {latest.isoformat()})",
            )
        return ValidationResult(
            passed=True,
            message=f"{append_dim}={due.isoformat()}, the newest position due "
            f"{self.max_delay} after its timestamp, is present "
            f"(latest is {latest.isoformat()})",
        )


class CheckRecentNans(VariableSelection, Validator):
    """Check the NaN fraction of recent append-dim positions.

    Checks the newest `window` positions, each independently so a per-position
    threshold is not diluted by a wider window. A window > 1 catches a gap that lands
    in a recent-but-no-longer-newest position (a late source file, a catch-up or
    re-backfill run): with a window of 1 each position is validated only while it is
    the newest, then never looked at again.

    `max_nan_fraction` is one threshold for every checked position, or a tuple indexed
    newest-first with the last value extending through the rest of the window — the
    same shape as CheckVirtualManifestCompleteness.min_present_fraction. A leading tier
    loosens positions the source is still filling in: `(0.45, 0.0)` allows the newest
    position 45% NaN while every older one must be complete, and a leading `1.0`
    excuses the newest position entirely (it is skipped, not read). For variables that
    fill in over several positions on different schedules, separate instances with
    different include_vars/tiers check each group against what is complete by then,
    instead of one threshold loose enough for all of them.

    Default `spatial_sampling="random_points"` reads all non-spatial dims (lead times,
    ensemble members) at 2 random spatial points per variable — cheap when data is
    chunked along the append dim. Use `"quarter"` for structural-NaN datasets (random
    points are bimodal/unstable) and `"all"` only for small datasets.

    When the operational update rewrites a deep window (e.g. a year of positions each
    run), set `window` to that depth and `sampled_positions` to a small count: each
    validation then checks that many randomly chosen positions within the window
    instead of every one, so older rewritten positions get eventual coverage at
    bounded cost. Sampled positions are chosen randomly, so per-recency tiers do not
    apply — a sampled check takes a single max_nan_fraction.

    A variable without hour-0 values (`step_type != "instant"`, or the template's
    `has_hour_0_values()` is false) has its lead_time=0 slice dropped before computing
    the NaN fraction.
    """

    max_nan_fraction: float | tuple[float, ...] = 0.0
    window: int = 2
    sampled_positions: int | None = None
    spatial_sampling: SpatialSamplingStrategy = "random_points"
    max_workers: int | None = None

    @pydantic.model_validator(mode="after")
    def _validate_thresholds(self) -> CheckRecentNans:
        tiers = self._tiers
        assert self.window >= 1, "window must be >= 1"
        assert 1 <= len(tiers) <= self.window, (
            f"max_nan_fraction has {len(tiers)} tiers which must fit in window={self.window}"
        )
        assert all(0.0 <= t <= 1.0 for t in tiers), (
            f"max_nan_fraction values must be within [0, 1], got {tiers}"
        )
        assert min(tiers) < 1.0, (
            "every max_nan_fraction tier is 1.0, which no NaN fraction can exceed — "
            "this check would test nothing"
        )
        if self.sampled_positions is not None:
            assert 1 <= self.sampled_positions <= self.window, (
                f"sampled_positions ({self.sampled_positions}) must be within "
                f"[1, window={self.window}]"
            )
            assert len(tiers) == 1, (
                "sampled positions are randomly chosen, so per-recency max_nan_fraction "
                "tiers do not apply; use a single threshold"
            )
        return self

    @property
    def name(self) -> str:
        if self.sampled_positions is None:
            return super().name
        parts = [f"sampled_positions={self.sampled_positions}", f"window={self.window}"]
        if label := self.selection_label():
            parts.insert(0, label)
        return f"{type(self).__name__}({', '.join(parts)})"

    @property
    def _tiers(self) -> tuple[float, ...]:
        if isinstance(self.max_nan_fraction, tuple):
            return self.max_nan_fraction
        return (self.max_nan_fraction,)

    def check(self, context: ValidationContext) -> ValidationResult:
        ds = context.ds
        append_dim = context.append_dim
        self.validate_var_names(context.known_var_paths())
        # A partially backfilled store (e2e tests) may carry a subset of the template.
        var_paths = [str(name) for name in ds.data_vars if self.selects(str(name))]
        if not var_paths:
            return ValidationResult(
                passed=False,
                message="None of the selected variables are in the store",
            )
        skip_lead_time_0_vars = {
            var.path for var in context.data_vars if not var.has_hour_0_values()
        }

        tiers = self._tiers
        size = ds.sizes[append_dim]
        if size < len(tiers):
            return ValidationResult(
                passed=False,
                message=(
                    f"Only {size} {append_dim} position(s), need at least "
                    f"{len(tiers)} to check the {tiers} NaN thresholds"
                ),
            )

        recencies: Sequence[int] = range(min(self.window, size))
        if self.sampled_positions is not None and self.sampled_positions < len(
            recencies
        ):
            rng = np.random.default_rng()
            recencies = sorted(
                rng.choice(len(recencies), size=self.sampled_positions, replace=False)
            )

        failures = []
        checked = 0
        for recency in recencies:
            threshold = tiers[min(recency, len(tiers) - 1)]
            if threshold >= 1.0:
                continue  # no NaN fraction can exceed 1.0; skip the read
            index = size - 1 - recency
            checked += 1
            result = _check_nan_fractions(
                _apply_spatial_sampling(
                    ds.isel({append_dim: [index]}), self.spatial_sampling
                ),
                max_nan_fraction=threshold,
                var_paths=var_paths,
                skip_lead_time_0_vars=skip_lead_time_0_vars,
                max_workers=self.max_workers
                or _DEFAULT_MAX_WORKERS[self.spatial_sampling],
            )
            if not result.passed:
                position = _format_coord_value(ds[append_dim].values[index])
                failures.append(f"{append_dim}={position}: {result.message}")

        if failures:
            return ValidationResult(
                passed=False,
                message=f"Excessive NaN fraction in {len(failures)} of {checked} "
                "recent positions:\n" + "\n".join(failures),
            )
        return ValidationResult(
            passed=True,
            message=f"All {checked} checked recent {append_dim} positions have "
            f"NaN fraction within {tiers}",
        )


def _spatial_dims(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.dims and "longitude" in ds.dims:
        return "longitude", "latitude"
    if "x" in ds.dims and "y" in ds.dims:
        return "x", "y"
    raise ValueError("Can't infer spatial dimensions from dataset")


def _apply_spatial_sampling(
    ds: xr.Dataset,
    sampling_strategy: SpatialSamplingStrategy,
) -> xr.Dataset:
    rng = np.random.default_rng()

    if sampling_strategy == "all":
        return ds

    x_dim, y_dim = _spatial_dims(ds)
    x_size = ds.sizes[x_dim]
    y_size = ds.sizes[y_dim]

    if sampling_strategy == "quarter":
        x_slice = (
            slice(0, x_size // 2)
            if rng.integers(0, 2) == 0
            else slice(x_size // 2, x_size)
        )
        y_slice = (
            slice(0, y_size // 2)
            if rng.integers(0, 2) == 0
            else slice(y_size // 2, y_size)
        )
        return ds.isel({x_dim: x_slice, y_dim: y_slice})

    if sampling_strategy == "random_points":
        x_idxs = rng.integers(0, x_size, size=_NUM_RANDOM_POINTS)
        y_idxs = rng.integers(0, y_size, size=_NUM_RANDOM_POINTS)
        # Pair each x with each y to form N points (use a synthetic "point" dim).
        return ds.isel(
            {
                x_dim: xr.DataArray(x_idxs, dims="point"),
                y_dim: xr.DataArray(y_idxs, dims="point"),
            }
        )

    assert_never(sampling_strategy)


def _format_coord_value(value: object) -> str:
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.timedelta64):
        return str(pd.Timedelta(value))
    if isinstance(value, float | np.floating):
        return f"{float(value):.4f}"
    return str(value)


def _summarize_coords(ds: xr.Dataset) -> str:
    parts = []
    for name in ds.coords:
        values = ds.coords[name].values.ravel()
        if values.size == 0:
            parts.append(f"{name}=<empty>")
        elif values.size == 1:
            parts.append(f"{name}={_format_coord_value(values[0])}")
        elif values.size <= 4:
            joined = ", ".join(_format_coord_value(v) for v in values)
            parts.append(f"{name}=[{joined}]")
        else:
            parts.append(
                f"{name}=[{_format_coord_value(values[0])}..{_format_coord_value(values[-1])}] (n={values.size})"
            )
    return ", ".join(parts)


def _check_nan_fractions(
    sample_ds: xr.Dataset,
    *,
    max_nan_fraction: float,
    var_paths: Sequence[str],
    skip_lead_time_0_vars: set[str],
    max_workers: int,
) -> ValidationResult:
    log.info(
        f"Computing NaN fraction for {len(var_paths)} variables: {sorted(var_paths)} "
        f"over coordinates: {_summarize_coords(sample_ds)}"
    )

    fractions: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_var = {
            executor.submit(
                _compute_var_nan_fraction,
                sample_ds,
                var_path,
                skip_lead_time_0_vars=skip_lead_time_0_vars,
            ): var_path
            for var_path in var_paths
        }
        for future in as_completed(future_to_var):
            fractions[future_to_var[future]] = future.result()

    # Combine: many info records in the same second are dropped before reaching Sentry.
    summary = ", ".join(f"{var}={fractions[var]:.6f}" for var in sorted(fractions))
    log.info(f"NaN fractions: {summary}")

    # An empty selection means the check measured nothing. Its NaN fraction is NaN,
    # which is not > any threshold, so without this it reports a pass.
    unmeasured_vars = sorted(
        var_path
        for var_path, fraction in fractions.items()
        if not np.isfinite(fraction)
    )
    problem_vars = {
        var_path: fraction
        for var_path, fraction in fractions.items()
        if np.isfinite(fraction) and fraction > max_nan_fraction
    }

    messages = []
    if unmeasured_vars:
        messages.append(
            "No values selected to compute a NaN fraction for: "
            + ", ".join(unmeasured_vars)
        )
    if problem_vars:
        messages.append(
            f"Excessive NaN fraction (> {max_nan_fraction}):\n"
            + "\n".join(
                f"- {var}: {fraction:.6f} NaN fraction"
                for var, fraction in sorted(problem_vars.items())
            )
        )
    if messages:
        return ValidationResult(passed=False, message="\n".join(messages))

    return ValidationResult(
        passed=True,
        message=f"All {len(var_paths)} variables have NaN fraction <= {max_nan_fraction}",
    )


def _compute_var_nan_fraction(
    ds: xr.Dataset,
    var_path: str,
    *,
    skip_lead_time_0_vars: set[str],
) -> float:
    da = ds[var_path]
    if "lead_time" in da.dims and (
        var_path in skip_lead_time_0_vars
        or da.attrs.get("step_type", "instant") != "instant"
    ):
        da = da.isel(lead_time=slice(1, None))
    # Deep copy after slicing to force eager load of just the needed region
    # (helps avoid memory leaks observed iterating null checks across vars).
    da = da.copy(deep=True)
    return float(da.isnull().mean().compute().item())


class CheckReplicaMatchesPrimary(Validator):
    """Compare a replica store's data against the primary's.

    Coordinates are compared exactly; a random subset of variables is compared over
    the last append-dim chunk at a random spatial window.
    """

    def check(self, context: ValidationContext) -> ValidationResult:
        replica_ds = context.ds
        primary_ds = context.primary_ds
        assert primary_ds is not None, (
            "CheckReplicaMatchesPrimary runs on replica stores; validate_dataset "
            "was called without a primary_ds"
        )
        append_dim = context.append_dim

        rng = np.random.default_rng()
        problem_coords = []
        for coord in primary_ds.coords:
            try:
                xr.testing.assert_equal(primary_ds[coord], replica_ds[coord])
            except AssertionError as e:
                log.exception(e)
                problem_coords.append(coord)
        if problem_coords:
            return ValidationResult(
                passed=False,
                message=f"Data in replica and primary stores are different for coords: {problem_coords}",
            )

        num_variables_to_check = min(5, len(primary_ds.data_vars))
        data_var_names = [str(k) for k in primary_ds.data_vars]
        variables_to_check = rng.choice(
            data_var_names, num_variables_to_check, replace=False
        )

        last_chunk = iterating.dimension_slices(primary_ds, append_dim, "chunks")[-1]
        problem_vars = []

        window_size = 100

        for var in variables_to_check:
            # Create random slices of `window_size` along non-append dimensions
            non_append_dim_slices = {
                dim_name: slice(
                    *(
                        start := int(
                            rng.integers(0, max(1, dim_size - window_size - 1))
                        ),
                        start + window_size,
                    )
                )
                for dim_name, dim_size in replica_ds[var].sizes.items()
                if dim_name != append_dim
            }

            # We create deep copies here to avoid sharing memory with the original dataset
            replica_ds_last_chunk = (
                replica_ds[var]
                .isel({append_dim: last_chunk, **non_append_dim_slices})
                .copy(deep=True)
            )
            primary_ds_last_chunk = (
                primary_ds[var]
                .isel({append_dim: last_chunk, **non_append_dim_slices})
                .copy(deep=True)
            )

            try:
                log.info(f"Comparing {var} in replica and primary stores")
                xr.testing.assert_equal(replica_ds_last_chunk, primary_ds_last_chunk)
            except AssertionError as e:
                log.exception(e)
                problem_vars.append(str(var))

            replica_ds_last_chunk.close()
            primary_ds_last_chunk.close()
            del replica_ds_last_chunk
            del primary_ds_last_chunk

        if problem_vars:
            return ValidationResult(
                passed=False,
                message=f"Data in replica and primary stores are different for at least the following vars: {problem_vars}",
            )
        return ValidationResult(
            passed=True,
            message="Data in tested subset of replica and primary stores is the same",
        )


class CheckExpectedShards(Validator):
    """Check that every shard the store's metadata declares is present.

    The write path passes write_empty_chunks=True, so even an all-fill shard is
    written and a missing shard always means a failed write.
    """

    def check(self, context: ValidationContext) -> ValidationResult:
        store = context.store
        ds = context.ds
        assert isinstance(store, Store), (
            f"CheckExpectedShards lists store keys, which requires a zarr Store, got {type(store)}"
        )
        log.info(f"Checking for expected shards in {store}")

        problem_vars = []
        var_missing_shard_indexes = {}

        for var in map(str, ds.data_vars):  # our keys are strs, xr types as Hashable
            shard_counts_per_dim = [
                len(iterating.chunk_slices(size, shard_size))
                for size, shard_size in zip(
                    ds[var].shape, ds[var].encoding["shards"], strict=True
                )
            ]
            ranges = [range(shard_count) for shard_count in shard_counts_per_dim]
            expected_shard_indexes = {
                "/".join(map(str, index)) for index in itertools.product(*ranges)
            }

            actual_var_shard_indexes = retry(
                partial(_sync_list_shards, store, var),
                max_attempts=3,
            )

            # During operational updates we trim down the dataset to only include
            # data that was fully processed. This means there may be some extra shards present
            # in the store, but the metadata has been trimmed such that they are not exposed.
            # As such, we don't expect these two sets to necessarily be equal, but we do expect
            # that expected_shard_indexes should be a proper subset of actual_var_shard_indexes.
            missing_shard_indexes = expected_shard_indexes - actual_var_shard_indexes
            if len(missing_shard_indexes) > 0:
                problem_vars.append(var)
                var_missing_shard_indexes[var] = sorted(missing_shard_indexes)

        if len(problem_vars) > 0:
            summary = ", ".join(
                f"{var} ({len(var_missing_shard_indexes[var])} missing)"
                for var in problem_vars
            )
            shard_lists = [var_missing_shard_indexes[var] for var in problem_vars]
            if len(problem_vars) > 1 and all(
                s == shard_lists[0] for s in shard_lists[1:]
            ):
                details = (
                    f"all missing the same shards: {_truncate_shards(shard_lists[0])}"
                )
            else:
                details = ", ".join(
                    f"{var}: {_truncate_shards(var_missing_shard_indexes[var])}"
                    for var in problem_vars
                )
            return ValidationResult(
                passed=False,
                message=f"Missing shards: {summary}. {details}",
            )

        return ValidationResult(
            passed=True,
            message="All variables have expected shards",
        )


def _truncate_shards(shards: Sequence[str], keep: int = 3) -> str:
    if len(shards) <= keep * 2:
        return f"[{', '.join(shards)}]"
    head = ", ".join(shards[:keep])
    tail = ", ".join(shards[-keep:])
    return f"[{head}, ..., {tail}]"


def _sync_list_shards(store: Store, var: str) -> set[str]:
    return zarr.core.sync.sync(_list_shards(store, var))


async def _list_shards(store: Store, var: str) -> set[str]:
    return {key.split(f"{var}/c/")[-1] async for key in store.list_prefix(f"{var}")}


class CheckVirtualManifestCompleteness(VariableSelection, Validator):
    """Assert each recent append-dim position is sufficiently present in the manifest.

    The virtual analog of CheckExpectedShards. Re-runs the region job's own
    source_file_coords + filter_already_present over its window (a handful of recent
    steps; whole-archive coverage is manifest_scan) and checks, per position, the fraction
    of expected source files present against `min_present_fraction` — indexed newest-first,
    older positions held to its last value. Regenerating the dataset's own coords excludes
    structural absences (hour-0 accumulated vars, etc.). One ref-existence probe per source
    file, no decode.

    Positions past the store's append-dim extent are skipped: the window runs ahead of what
    the source has published, and an update grows the append dim only as far as the refs it
    writes. So a wholly un-ingested recent stretch is CheckCurrentData's job, while an
    interior gap still fails here.

      (1.0,)      every append-dim position whole (default).
      (0.5, 1.0)  the newest may be half-published (e.g. GEFS 35-day's slow long lead
                  times); older append-dim positions whole.
      (0.0, 1.0)  the newest is excused entirely; every older position whole.

    `include_vars` / `exclude_vars` narrow the check to the source files carrying those
    variables — a file is expected when it carries any of them — so variables publishing
    on different schedules can be checked separately, each held to a whole 1.0 from the
    position where its own files are expected. The split must fall on source-file
    boundaries (asserted): presence is probed through each file's representative
    variable, so a file carrying both checked and unchecked variables could pass on a
    variable this instance does not cover.
    """

    min_present_fraction: tuple[float, ...] = (1.0,)

    requires_virtual_dataset: ClassVar[bool] = True

    @pydantic.model_validator(mode="after")
    def _validate_thresholds(self) -> CheckVirtualManifestCompleteness:
        assert self.min_present_fraction, "min_present_fraction must be non-empty"
        # If a real source ever leaves append-dim positions permanently incomplete, add
        # an explicit opt-out field rather than relaxing this into a soft convention.
        assert self.min_present_fraction[-1] == 1.0, (
            "the last min_present_fraction tier holds for every older append-dim "
            "position, so it must be 1.0; loosen only the leading tiers, which cover "
            f"positions the source may still be filling in (got {self.min_present_fraction})"
        )
        return self

    def check(self, context: ValidationContext) -> ValidationResult:
        region_job = context.virtual_region_job()
        store = context.store
        assert isinstance(store, IcechunkStore)
        append_dim = region_job.append_dim
        self.validate_var_names([var.path for var in region_job.data_vars])

        ingested_through = context.ds.get_index(append_dim).max()
        candidates = [
            coord
            for coord in region_job.source_file_coords()
            if coord.out_loc()[append_dim] <= ingested_through
            and self._carries_selected_var(coord, region_job)
        ]
        expected_per_position = Counter(c.out_loc()[append_dim] for c in candidates)
        positions = sorted(expected_per_position, reverse=True)  # newest first
        if len(positions) < len(self.min_present_fraction):
            return ValidationResult(
                passed=False,
                message=(
                    f"Only {len(positions)} ingested {append_dim} position(s) in the "
                    f"validation window, need at least {len(self.min_present_fraction)} "
                    f"to check the {self.min_present_fraction} completeness thresholds"
                ),
            )

        missing_per_position = Counter(
            c.out_loc()[append_dim]
            for c in region_job.filter_already_present(candidates, store)
        )
        problems = []
        for recency, position in enumerate(positions):
            required = self.min_present_fraction[
                min(recency, len(self.min_present_fraction) - 1)
            ]
            expected = expected_per_position[position]
            present = expected - missing_per_position[position]
            if present / expected < required:
                problems.append(
                    f"{append_dim}={position}: {present}/{expected} present "
                    f"({present / expected:.1%} < required {required:.0%})"
                )
        if problems:
            return ValidationResult(
                passed=False,
                message="Incomplete manifest:\n"
                + "\n".join(f"- {p}" for p in problems),
            )
        return ValidationResult(
            passed=True,
            message=(
                f"All {len(positions)} ingested {append_dim} positions (through "
                f"{ingested_through}) meet completeness thresholds "
                f"{self.min_present_fraction}"
            ),
        )

    def _carries_selected_var(
        self, coord: SourceFileCoord, region_job: VirtualRegionJob[Any, Any]
    ) -> bool:
        if self.include_vars == "all" and not self.exclude_vars:
            return True
        file_vars = getattr(coord, "data_vars", None) or region_job.data_vars
        selected = [self.selects(var.path) for var in file_vars]
        assert all(selected) or not any(selected), (
            f"{coord.get_url()} carries both selected and unselected variables "
            f"({self.name}). Presence is probed through the file's representative "
            "variable, which may be one this instance does not cover, so a "
            "partially-checked file could pass while a checked variable has no "
            "references. Split checks along source-file boundaries."
        )
        return any(selected)


class CheckVirtualDecodeHealth(Validator):
    """Decode the references that are present and assert they are readable.

    Completeness checks that references *exist*; this checks the ones that exist actually
    decode — the per-variable serializer (e.g. GribberishCodec) and virtual-container
    authorization, end to end. Over the recent window it keeps only the source files
    present in the manifest (filter_already_present), so a not-yet-published ref is never
    mistaken for a decode failure, then decodes a bounded sample of them. `positions`
    selects which append-dim positions to check: an int targets that many of the newest
    positions with data (default 1 — so a broken newest reference is caught at the next
    validation, not a cycle later; use more when the newest position does not carry every
    variable), while "all" covers the whole window, optionally capped to `max_positions`
    evenly spaced positions for a whole-archive offline sweep. Within a position it
    samples `sampled_leads` lead times (first + last + evenly spaced interior) across
    every member, and `sampled_levels` levels of any vertical dim (e.g. pressure_level)
    so a group var is decode-checked at a bounded set of levels rather than every one.
    A variable fails if any sampled chunk errors or all of its sampled chunks decode
    entirely NaN. Fails — never silently passes — when no references are present.
    """

    positions: int | Literal["all"] = 1
    sampled_leads: int = 5
    sampled_levels: int = 3
    max_positions: int | None = None
    max_workers: int = 32
    # Offline opt-in. Given (var_path, out_loc), returns whether a chunk reference actually
    # exists. When provided, a variable with no reference at a sampled position is skipped
    # (not decoded, not a failure) -- reference existence is the availability check's
    # concern. When None (operational default) every declared variable is decoded and a
    # missing reference reads as fill NaN and fails, which is how the operational check
    # catches removed/renamed/unpulled vars.
    reference_exists: Callable[[str, Mapping[str, Any]], bool] | None = None

    requires_virtual_dataset: ClassVar[bool] = True

    @pydantic.model_validator(mode="after")
    def _validate_positions(self) -> CheckVirtualDecodeHealth:
        if isinstance(self.positions, int):
            assert self.positions >= 1, "positions must be >= 1"
            assert self.max_positions is None, (
                "max_positions caps positions='all'; an int positions already bounds "
                "the check"
            )
        return self

    def check(self, context: ValidationContext) -> ValidationResult:
        region_job = context.virtual_region_job()
        store = context.store
        assert isinstance(store, IcechunkStore)
        ds = context.ds
        append_dim = region_job.append_dim
        candidates = region_job.source_file_coords()
        if not candidates:
            return ValidationResult(
                passed=False,
                checked_count=0,
                message=f"No source files in the {append_dim} window to decode-check",
            )
        absent = {id(c) for c in region_job.filter_already_present(candidates, store)}
        present = [c for c in candidates if id(c) not in absent]
        if not present:
            return ValidationResult(
                passed=False,
                checked_count=0,
                message=f"No present references in the {append_dim} window to decode",
            )

        present_positions = sorted({c.out_loc()[append_dim] for c in present})
        targets = self._select_targets(present_positions)
        to_decode = self._sample_leads(
            [c for c in present if c.out_loc()[append_dim] in targets]
        )

        min_nan_fraction: dict[str, float] = {}
        first_error: dict[str, str] = {}
        no_reference_vars: set[str] = set()
        decoded_refs = 0
        decode = partial(self._decode_coord, region_job=region_job, ds=ds)
        with ThreadPoolExecutor(self.max_workers) as pool:
            for results, skipped in pool.map(decode, to_decode):
                decoded_refs += len(results)
                for var_path, nan_fraction, error in results:
                    min_nan_fraction[var_path] = min(
                        min_nan_fraction.get(var_path, float("inf")), nan_fraction
                    )
                    if error is not None and var_path not in first_error:
                        first_error[var_path] = error
                if self.reference_exists is not None:
                    no_reference_vars |= skipped

        problems = []
        for var_path in sorted(min_nan_fraction):
            if var_path in first_error:
                problems.append(f"{var_path}: decode error ({first_error[var_path]})")
            elif min_nan_fraction[var_path] >= 1.0:
                problems.append(f"{var_path}: every sampled chunk decoded entirely NaN")

        target_label = ", ".join(str(p) for p in sorted(targets))
        if problems:
            return ValidationResult(
                passed=False,
                checked_count=decoded_refs,
                message=f"Decode health failures at {append_dim}={target_label}:\n"
                + "\n".join(f"- {p}" for p in problems),
            )
        message = (
            f"Decoded {len(to_decode)} present source files across "
            f"{len(min_nan_fraction)} variables at {append_dim}={target_label} "
            "— all readable"
        )
        if self.reference_exists is not None and no_reference_vars:
            message += (
                f" ({len(no_reference_vars)} variable(s) had no reference at sampled "
                "positions — reference existence is reported by the "
                "availability/manifest check)"
            )
        return ValidationResult(
            passed=True, message=message, checked_count=decoded_refs
        )

    def _select_targets(self, present_positions: Sequence[Any]) -> set[Any]:
        if isinstance(self.positions, int):
            return set(present_positions[-self.positions :])
        if self.max_positions and len(present_positions) > self.max_positions:
            return {
                present_positions[i]
                for i in np.unique(
                    np.linspace(0, len(present_positions) - 1, self.max_positions)
                    .round()
                    .astype(int)
                )
            }
        return set(present_positions)

    def _decode_coord(
        self,
        coord: SourceFileCoord,
        region_job: VirtualRegionJob[Any, Any],
        ds: xr.Dataset,
    ) -> tuple[list[tuple[str, float, str | None]], set[str]]:
        loc = coord.out_loc()
        # A coord's data_vars are exactly the variables its file carries (e.g. no
        # accumulated vars at hour 0), so every one should decode to data.
        file_vars = getattr(coord, "data_vars", None) or region_job.data_vars
        results = []
        skipped: set[str] = set()
        for var in file_vars:
            if self.reference_exists is not None and not self.reference_exists(
                var.path, cast("Mapping[str, Any]", loc)
            ):
                skipped.add(var.path)
                continue
            da = ds[var.path]
            selection = {dim: value for dim, value in loc.items() if dim in da.dims}
            da = self._sample_levels(da.sel(selection))
            try:
                # Retried so a transient object store failure is not reported as
                # a decode failure; a genuine decode error still fails fast.
                values = retry(
                    lambda da=da: da.copy(deep=True).load().values,
                    max_attempts=3,
                )
                results.append((var.path, float(np.isnan(values).mean()), None))
            except Exception as e:  # noqa: BLE001 - any decode failure is a validation failure
                results.append((var.path, 1.0, f"{type(e).__name__}: {e}"))
        return results, skipped

    def _sample_leads(
        self, coords: Sequence[SourceFileCoord]
    ) -> Sequence[SourceFileCoord]:
        """Down-sample to `sampled_leads` lead times (first + last + evenly spaced
        interior), keeping every other coordinate (e.g. all members). Coords without a
        lead_time (analysis) are returned unchanged."""
        leads = sorted(
            {c.out_loc()["lead_time"] for c in coords if "lead_time" in c.out_loc()}
        )
        if len(leads) <= self.sampled_leads:
            return coords
        keep = {
            leads[i]
            for i in np.linspace(0, len(leads) - 1, self.sampled_leads)
            .round()
            .astype(int)
        }
        return [c for c in coords if c.out_loc().get("lead_time") in keep]

    def _sample_levels(self, da: xr.DataArray) -> xr.DataArray:
        """Down-sample any vertical (non-spatial) dim to `sampled_levels` evenly spaced
        levels, so a group var is decode-checked at a bounded set of levels rather than
        all of them. Single-level vars (only spatial dims left) are returned unchanged."""
        spatial = ("y", "x", "latitude", "longitude")
        isel: dict[Any, Any] = {}
        for dim in da.dims:
            if dim in spatial:
                continue
            size = da.sizes[dim]
            if size > self.sampled_levels:
                isel[dim] = np.unique(
                    np.linspace(0, size - 1, self.sampled_levels).round().astype(int)
                )
        return da.isel(isel) if isel else da
