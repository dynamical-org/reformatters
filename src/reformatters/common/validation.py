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
    Literal,
    Protocol,
    assert_never,
    cast,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import pydantic
import xarray as xr
import zarr
import zarr.core.sync
import zarr.storage
from icechunk.store import IcechunkStore
from zarr.abc.store import Store

from reformatters.common import iterating
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

if TYPE_CHECKING:
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


@runtime_checkable
class XarrayDataValidator(Protocol):
    """A validator that runs on an opened xarray Dataset.

    The common, generic kind — works on any dataset (materialized or virtual) and
    needs nothing beyond the data it reads (lag, NaN-fraction, ...).
    """

    def __call__(self, ds: xr.Dataset) -> ValidationResult: ...


class VirtualDataValidator(abc.ABC):
    """A validator that needs manifest/store access, not just an opened Dataset.

    Virtual datasets list these in DynamicalDataset.validators() alongside the plain
    XarrayDataValidator functions; validate_dataset dispatches by type, handing these the
    operational-window region job (to regenerate source-file coords and probe the
    manifest), the icechunk store, and the opened dataset — each validator uses the subset
    it needs. Tuning (e.g. the per-position completeness thresholds) is a field on the
    concrete validator, set where it is listed in validators() — one place.
    """

    @abc.abstractmethod
    def __call__(
        self,
        region_job: VirtualRegionJob[Any, Any],
        store: IcechunkStore,
        ds: xr.Dataset,
    ) -> ValidationResult: ...


# A dataset's validators() may mix the two kinds; validate_dataset dispatches on type.
DataValidator = XarrayDataValidator | VirtualDataValidator


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
    store: zarr.storage.StoreLike,
    validators: Sequence[DataValidator],
    *,
    region_job: VirtualRegionJob[Any, Any] | None = None,
) -> None:
    """
    Validate a zarr dataset by running a series of quality checks.

    Args:
        store: the zarr/icechunk store to validate.
        validators: the checks to run; XarrayDataValidators receive the opened dataset,
            VirtualDataValidators receive (region_job, store, ds).
        region_job: the operational-window job, required when any validator is a
            VirtualDataValidator (it supplies the source-file coords + manifest probe).

    Raises:
        ValueError: If any validation checks fail
    """
    log.info(f"Validating zarr {store}")

    consolidated = not isinstance(store, IcechunkStore)

    # Run all validators
    failed_validations = []
    for validator in validators:
        ds = open_flattened_dataset(store, consolidated=consolidated)

        if isinstance(validator, VirtualDataValidator):
            assert region_job is not None, (
                f"{type(validator).__name__} needs a region_job but validate_dataset "
                "was called without one"
            )
            assert isinstance(store, IcechunkStore)
            result = validator(region_job, store, ds)
        else:
            result = validator(ds)

        if not result.passed:
            log.error(f"Failed validation: {result.message}")
            failed_validations.append(result.message)
        else:
            log.info(f"Passed validation: {result.message}")

        ds.close()
        del ds

    if failed_validations:
        raise ValueError(
            "Zarr validation failed:\n"
            + "\n".join(f"- {msg}" for msg in failed_validations)
        )

    log.info("Zarr validation passed all checks")


def check_forecast_current_data(
    ds: xr.Dataset,
    max_latest_init_time_age: timedelta = timedelta(days=1),
) -> ValidationResult:
    """Check that the latest init_time is within max_latest_init_time_age. Fails if no recent init_time."""
    now = pd.Timestamp.now()
    latest_init_time_ds = ds.sel(init_time=slice(now - max_latest_init_time_age, None))
    if latest_init_time_ds.sizes["init_time"] == 0:
        return ValidationResult(
            passed=False,
            message=f"No data found within {max_latest_init_time_age} of now",
        )

    return ValidationResult(
        passed=True,
        message=f"Data found within {max_latest_init_time_age} of now",
    )


def check_analysis_current_data(
    ds: xr.Dataset, max_expected_delay: timedelta = timedelta(hours=12)
) -> ValidationResult:
    """Check for data in the most recent day. Fails if no data is found."""
    now = pd.Timestamp.now()
    latest_init_time_ds = ds.sel(time=slice(now - max_expected_delay, None))
    if latest_init_time_ds.sizes["time"] == 0:
        return ValidationResult(
            passed=False,
            message=f"No data found within {max_expected_delay} of now",
        )

    return ValidationResult(
        passed=True,
        message=f"Data found within {max_expected_delay} of now",
    )


def check_forecast_recent_nans(
    ds: xr.Dataset,
    *,
    init_time_offset: int = -1,
    max_nan_fraction: float = 0.0,
    include_vars: Sequence[str] | Literal["all"] = "all",
    exclude_vars: Sequence[str] = (),
    spatial_sampling: SpatialSamplingStrategy = "random_points",
    additional_skip_lead_time_0_vars: Sequence[str] = (),
    max_workers: int | None = None,
) -> ValidationResult:
    """
    Check the NaN fraction of a recent init_time in a forecast dataset.

    `init_time_offset` selects which init_time to check from the end
    (`-1` = latest, `-2` = previous, etc.). Use `-2` for datasets whose
    latest init is still being filled in (e.g. long-horizon ensembles).

    Default `spatial_sampling="random_points"` reads all lead_times (and any
    ensemble members) at 2 random spatial points per variable — cheap when
    data is chunked by init_time. Use `"all"` only for small datasets.

    Variables with `step_type != "instant"` always have their lead_time=0 slice
    dropped before computing the NaN fraction (these vars do not have valid
    hour 0 data). `additional_skip_lead_time_0_vars` adds extra names on top
    (e.g. HRRR categorical vars which are step_type=instant but have no hour 0 data).
    """
    sample_ds = ds.isel(init_time=[init_time_offset])
    sample_ds = _apply_spatial_sampling(sample_ds, spatial_sampling)

    return _check_nan_fractions(
        sample_ds,
        max_nan_fraction=max_nan_fraction,
        include_vars=include_vars,
        exclude_vars=exclude_vars,
        additional_skip_lead_time_0_vars=additional_skip_lead_time_0_vars,
        max_workers=max_workers or _DEFAULT_MAX_WORKERS[spatial_sampling],
    )


def check_analysis_recent_nans(
    ds: xr.Dataset,
    *,
    max_expected_delay: timedelta = timedelta(hours=12),
    max_nan_fraction: float = 0.0,
    include_vars: Sequence[str] | Literal["all"] = "all",
    exclude_vars: Sequence[str] = (),
    spatial_sampling: SpatialSamplingStrategy = "random_points",
    max_workers: int | None = None,
) -> ValidationResult:
    """
    Check the NaN fraction of recent timesteps in an analysis dataset.

    Default `spatial_sampling="random_points"` reads 2 random spatial points
    (across all timesteps in the window) — cheap and covers independent
    locations. Use `"quarter"` for structural-NaN datasets and `"all"` only
    when small.
    """
    now = pd.Timestamp.now()
    sample_ds = ds.sel(time=slice(now - max_expected_delay, None))
    sample_ds = _apply_spatial_sampling(sample_ds, spatial_sampling)

    return _check_nan_fractions(
        sample_ds,
        max_nan_fraction=max_nan_fraction,
        include_vars=include_vars,
        exclude_vars=exclude_vars,
        additional_skip_lead_time_0_vars=(),
        max_workers=max_workers or _DEFAULT_MAX_WORKERS[spatial_sampling],
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
    include_vars: Sequence[str] | Literal["all"],
    exclude_vars: Sequence[str],
    additional_skip_lead_time_0_vars: Sequence[str],
    max_workers: int,
) -> ValidationResult:
    var_names = [
        var_name
        for var_name in map(str, sample_ds.data_vars)
        if (include_vars == "all" or var_name in include_vars)
        and var_name not in exclude_vars
    ]

    log.info(
        f"Computing NaN fraction for {len(var_names)} variables: {sorted(var_names)} "
        f"over coordinates: {_summarize_coords(sample_ds)}"
    )

    skip_lead_time_0_vars = set(additional_skip_lead_time_0_vars)
    fractions: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_var = {
            executor.submit(
                _compute_var_nan_fraction,
                sample_ds,
                var_name,
                additional_skip_lead_time_0_vars=skip_lead_time_0_vars,
            ): var_name
            for var_name in var_names
        }
        for future in as_completed(future_to_var):
            var_name = future_to_var[future]
            fraction = future.result()
            fractions[var_name] = fraction
            log.info(f"NaN fraction for {var_name}: {fraction:.6f}")

    problem_vars = {
        var_name: fraction
        for var_name, fraction in fractions.items()
        if fraction > max_nan_fraction
    }

    if problem_vars:
        message = f"Excessive NaN fraction (> {max_nan_fraction}):\n" + "\n".join(
            f"- {var}: {fraction:.6f} NaN fraction"
            for var, fraction in sorted(problem_vars.items())
        )
        return ValidationResult(passed=False, message=message)

    return ValidationResult(
        passed=True,
        message=f"All {len(var_names)} variables have NaN fraction <= {max_nan_fraction}",
    )


def _compute_var_nan_fraction(
    ds: xr.Dataset,
    var_name: str,
    *,
    additional_skip_lead_time_0_vars: set[str],
) -> float:
    da = ds[var_name]
    if "lead_time" in da.dims and (
        var_name in additional_skip_lead_time_0_vars
        or da.attrs.get("step_type", "instant") != "instant"
    ):
        da = da.isel(lead_time=slice(1, None))
    # Deep copy after slicing to force eager load of just the needed region
    # (helps avoid memory leaks observed iterating null checks across vars).
    da = da.copy(deep=True)
    return float(da.isnull().mean().compute().item())


def compare_replica_and_primary(
    append_dim: str, replica_ds: xr.Dataset, primary_ds: xr.Dataset
) -> ValidationResult:
    """Compare the data in the replica and primary stores."""
    rng = np.random.default_rng()
    problem_coords = []
    for coord in primary_ds.coords:
        try:
            xr.testing.assert_equal(primary_ds[coord], replica_ds[coord])
        except AssertionError as e:
            log.exception(e)
            problem_coords.append(coord)
    if problem_coords:
        message = f"Data in replica and primary stores are different for coords: {problem_coords}"
        return ValidationResult(passed=False, message=message)

    num_variables_to_check = min(5, len(primary_ds.data_vars))
    data_var_names = [str(k) for k in primary_ds.data_vars]
    variables_to_check = rng.choice(
        data_var_names, num_variables_to_check, replace=False
    )

    last_chunk = iterating.dimension_slices(primary_ds, append_dim, "chunks")[-1]
    problem_vars = []

    window_size = 100

    rng = np.random.default_rng()

    for var in variables_to_check:
        # Create random slices of `window_size` along non-append dimensions
        non_append_dim_slices = {
            dim_name: slice(
                *(
                    start := int(rng.integers(0, max(1, dim_size - window_size - 1))),
                    start + window_size,
                )
            )
            for dim_name, dim_size in replica_ds.sizes.items()
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
    else:
        return ValidationResult(
            passed=True,
            message="Data in tested subset of replica and primary stores is the same",
        )


def check_for_expected_shards(store: Store, ds: xr.Dataset) -> ValidationResult:
    """Check that the expected shards are present in the store."""
    log.info(f"Checking for expected shards in {store}")

    problem_vars = []
    var_missing_shard_indexes = {}

    for var in map(str, ds.data_vars):  # our keys are strs, xr types as Hashable
        ordered_dims = ds[var].dims

        shard_counts_per_dim = [
            len(iterating.dimension_slices(ds, str(dim), "shards"))
            for dim in ordered_dims
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
            # HRRR categorical variables have enough 0s that some shards are not written
            # We will remove this skip when fill_value is updated to nan / write_empty_chunks is true
            if (
                ds.attrs["dataset_id"] == "noaa-hrrr-forecast-48-hour"
                and "categorical" in var
            ):
                log.info(
                    f"Expecting to find fewer than the maximum shards for categorical hrrr variable ({var}) due to fill value 0 and write empty chunks false"
                )
                continue

            problem_vars.append(var)
            var_missing_shard_indexes[var] = sorted(missing_shard_indexes)

    if len(problem_vars) > 0:
        summary = ", ".join(
            f"{var} ({len(var_missing_shard_indexes[var])} missing)"
            for var in problem_vars
        )
        shard_lists = [var_missing_shard_indexes[var] for var in problem_vars]
        if len(problem_vars) > 1 and all(s == shard_lists[0] for s in shard_lists[1:]):
            details = f"all missing the same shards: {_truncate_shards(shard_lists[0])}"
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


@dataclass(frozen=True)
class CheckVirtualManifestCompleteness(VirtualDataValidator):
    """Assert recent append-dim positions are sufficiently ingested in the manifest.

    Re-runs the operational filter (the region job's own source_file_coords +
    filter_already_present) over the recent window and checks, per position, the fraction
    of expected source files present against `min_present_fraction` — indexed newest-first,
    with positions older than the tuple held to its last value. Cheap: one ref-existence
    probe per source file, no decode. Reusing the dataset's own coord generation means
    structural absences (hour-0 accumulated vars, etc.) are not in the expected set. The
    virtual analog of check_for_expected_shards.

    Examples (validation typically runs once each recent position should be ingested):
      (1.0,)      every position in the window must be fully ingested (default).
      (0.5, 1.0)  the newest position may be half-published (e.g. GEFS 35-day's slow long
                  lead times); every older position must be complete.
      (0.0, 1.0)  ignore the newest (still publishing); require the rest complete.
      (0.8,)      every position at least 80% present (a source that trickles in).
    """

    min_present_fraction: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        assert self.min_present_fraction, "min_present_fraction must be non-empty"

    def __call__(
        self,
        region_job: VirtualRegionJob[Any, Any],
        store: IcechunkStore,
        ds: xr.Dataset,  # noqa: ARG002 - completeness reads the manifest, not the dataset
    ) -> ValidationResult:
        append_dim = region_job.append_dim
        candidates = region_job.source_file_coords()
        expected_per_position = Counter(c.out_loc()[append_dim] for c in candidates)
        positions = sorted(expected_per_position, reverse=True)  # newest first
        if len(positions) < len(self.min_present_fraction):
            return ValidationResult(
                passed=False,
                message=(
                    f"Only {len(positions)} {append_dim} position(s) in the validation "
                    f"window, need at least {len(self.min_present_fraction)} to check the "
                    f"{self.min_present_fraction} completeness thresholds"
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
                f"All {len(positions)} {append_dim} positions meet completeness "
                f"thresholds {self.min_present_fraction}"
            ),
        )


@dataclass(frozen=True)
class CheckVirtualDecodeHealth(VirtualDataValidator):
    """Decode the references that are present and assert they are readable.

    Completeness checks that references *exist*; this checks the ones that exist actually
    decode — the per-variable serializer (e.g. GribberishCodec) and virtual-container
    authorization, end to end. Over the recent window it keeps only the source files
    present in the manifest (filter_already_present), so a not-yet-published ref is never
    mistaken for a decode failure, then decodes a bounded sample of them. `positions`
    selects which append-dim positions to check: "latest" (default) targets the newest
    position with data — so a broken newest reference is caught at the next validation, not
    a cycle later — while "all" covers the whole window. Within a position it samples
    `sampled_leads` lead times (first + last + evenly spaced interior) across every member,
    and `sampled_levels` levels of any vertical dim (e.g. pressure_level) so a group var is
    decode-checked at a bounded set of levels rather than every one. `max_positions`
    optionally caps "all" to an evenly spaced subset of positions for a whole-archive
    offline sweep. A variable fails if any sampled chunk errors or all of its sampled chunks
    decode entirely NaN. Fails — never silently passes — when no references are present.
    """

    positions: Literal["latest", "all"] = "latest"
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

    def __call__(
        self,
        region_job: VirtualRegionJob[Any, Any],
        store: IcechunkStore,
        ds: xr.Dataset,
    ) -> ValidationResult:
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
        if self.positions == "latest":
            return {present_positions[-1]}
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
