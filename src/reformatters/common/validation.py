from __future__ import annotations

import abc
import itertools
from collections.abc import Sequence
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
    it needs. Tuning (e.g. how many newest positions to skip) is a field on the concrete
    validator, set where it is listed in validators() — one place.
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


def open_validation_dataset(
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
        ds = open_validation_dataset(store, consolidated=consolidated)

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
    """Assert every source file that should already be ingested is in the manifest.

    Re-runs the operational filter (the region job's own source_file_coords +
    filter_already_present) over the recent window, excluding the newest
    `skip_newest_positions` append-dim positions, which may still be mid-publication.
    Anything still missing is a real hole — a source file that failed to ingest.
    Cheap: one ref-existence probe per source file, no decode. Reusing the dataset's own
    coord generation means structural absences (hour-0 accumulated vars, etc.) are not in
    the expected set. The virtual analog of check_for_expected_shards.
    """

    skip_newest_positions: int = 1

    def __call__(
        self,
        region_job: VirtualRegionJob[Any, Any],
        store: IcechunkStore,
        ds: xr.Dataset,  # noqa: ARG002 - completeness reads the manifest, not the dataset
    ) -> ValidationResult:
        append_dim = region_job.append_dim
        candidates = region_job.source_file_coords()
        positions = sorted({c.out_loc()[append_dim] for c in candidates})
        if len(positions) <= self.skip_newest_positions:
            return ValidationResult(
                passed=True,
                message=f"Too few {append_dim} positions ({len(positions)}) to check completeness",
            )

        cutoff = positions[-1 - self.skip_newest_positions]
        expected = [c for c in candidates if c.out_loc()[append_dim] <= cutoff]
        missing = region_job.filter_already_present(expected, store)
        if missing:
            missing_positions = sorted({str(c.out_loc()[append_dim]) for c in missing})
            return ValidationResult(
                passed=False,
                message=(
                    f"{len(missing)} of {len(expected)} expected source files missing from "
                    f"the manifest (through {append_dim}={cutoff}); "
                    f"affected {append_dim}: {missing_positions}"
                ),
            )
        return ValidationResult(
            passed=True,
            message=f"All {len(expected)} expected source files present through {append_dim}={cutoff}",
        )


@dataclass(frozen=True)
class CheckVirtualDecodeHealth(VirtualDataValidator):
    """Decode a bounded sample of the latest complete position and assert every variable
    decodes without error and isn't entirely NaN.

    Exercises the live read path — the per-variable serializer (e.g. GribberishCodec) and
    virtual-container authorization — end to end, catching a regression that leaves refs
    present but unreadable. Bounded by sampling: `sampled_leads` lead times (first + last
    + evenly spaced interior) across every ensemble member of one position, decoded
    concurrently. The whole-archive NaN/coverage scan stays offline (it is one full-message
    decode per chunk across all leads x members — hours).
    """

    skip_newest_positions: int = 1
    sampled_leads: int = 5
    max_workers: int = 32

    def __call__(
        self,
        region_job: VirtualRegionJob[Any, Any],
        store: IcechunkStore,  # noqa: ARG002 - decode health reads through the opened dataset
        ds: xr.Dataset,
    ) -> ValidationResult:
        # Decode a position the store actually holds (the append dim grows lazily as refs
        # arrive), skipping the newest still-publishing ones — not a template position
        # that may not be ingested yet (which would index-miss and silently decode another).
        append_dim = region_job.append_dim
        n_positions = ds.sizes.get(append_dim, 0)
        if n_positions <= self.skip_newest_positions:
            return ValidationResult(
                passed=True,
                message=f"Too few {append_dim} positions ({n_positions}) to sample-decode",
            )
        target_index = n_positions - 1 - self.skip_newest_positions
        target = ds.get_index(append_dim)[target_index]

        lead_indexes: list[int] | None = None
        if "lead_time" in ds.dims:
            n_leads = ds.sizes["lead_time"]
            lead_indexes = sorted(
                set(
                    np.linspace(0, n_leads - 1, min(self.sampled_leads, n_leads))
                    .round()
                    .astype(int)
                    .tolist()
                )
            )
        members: list[int | None] = (
            list(range(ds.sizes["ensemble_member"]))
            if "ensemble_member" in ds.dims
            else [None]
        )
        var_names = [str(v) for v in ds.data_vars]

        def decode(task: tuple[str, int | None]) -> tuple[str, float, str | None]:
            var_name, member = task
            da = ds[var_name]
            # Sample only along dims this var has — group vars carry an extra vertical
            # dim, others may lack lead_time/ensemble_member — so isel never hits a
            # missing dim.
            selection: dict[str, Any] = {append_dim: target_index}
            if member is not None and "ensemble_member" in da.dims:
                selection["ensemble_member"] = member
            if lead_indexes is not None and "lead_time" in da.dims:
                selection["lead_time"] = lead_indexes
            da = da.isel(selection)
            # Accumulated/non-instant vars have no valid hour-0 data; drop it before
            # judging NaN-ness (matches _compute_var_nan_fraction).
            if (
                "lead_time" in da.dims
                and lead_indexes is not None
                and lead_indexes[0] == 0
                and da.attrs.get("step_type", "instant") != "instant"
            ):
                da = da.isel(lead_time=slice(1, None))
            if da.size == 0:
                return var_name, float("nan"), None  # nothing left to judge
            try:
                values = da.copy(deep=True).load().values
                return var_name, float(np.isnan(values).mean()), None
            except Exception as e:  # noqa: BLE001 - any decode failure is a validation failure
                return var_name, 1.0, f"{type(e).__name__}: {e}"

        tasks = [(var_name, member) for var_name in var_names for member in members]
        # inf until a non-empty chunk is judged; NaN fractions (empty slices) are skipped
        # so a var with nothing to judge is not mistaken for entirely-NaN.
        min_nan_fraction: dict[str, float] = dict.fromkeys(var_names, float("inf"))
        first_error: dict[str, str] = {}
        with ThreadPoolExecutor(self.max_workers) as pool:
            for var_name, nan_fraction, error in pool.map(decode, tasks):
                if not np.isnan(nan_fraction):
                    min_nan_fraction[var_name] = min(
                        min_nan_fraction[var_name], nan_fraction
                    )
                if error is not None and var_name not in first_error:
                    first_error[var_name] = error

        problems = []
        for var_name in var_names:
            if var_name in first_error:
                problems.append(f"{var_name}: decode error ({first_error[var_name]})")
            elif np.isfinite(min_nan_fraction[var_name]) and (
                min_nan_fraction[var_name] >= 1.0
            ):
                problems.append(f"{var_name}: every sampled chunk decoded entirely NaN")

        if problems:
            return ValidationResult(
                passed=False,
                message=f"Decode health failures at {append_dim}={target}:\n"
                + "\n".join(f"- {p}" for p in problems),
            )
        n_decodes = len(tasks) * (len(lead_indexes) if lead_indexes else 1)
        return ValidationResult(
            passed=True,
            message=(
                f"All {len(var_names)} variables decoded with data at "
                f"{append_dim}={target} ({n_decodes} chunk decodes)"
            ),
        )
