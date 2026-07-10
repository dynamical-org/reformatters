import functools
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import icechunk
import numpy as np
import obstore
import pandas as pd
import typer
import xarray as xr
import zarr
from zarr.storage import ObjectStore, StoreLike

from reformatters.common.retry import retry
from reformatters.common.validation import open_flattened_dataset

# Whole-archive scans issue many concurrent object-store reads. Raise zarr's async
# concurrency once here; every validation entry point imports this module.
zarr.config.set({"async.concurrency": 32})

OUTPUT_DIR = "data/output"

STAC_CATALOG_URL = "https://stac.dynamical.org/catalog.json"
GEFS_ANALYSIS_COLLECTION_ID = "noaa-gefs-analysis"


@functools.cache
def get_gefs_analysis_reference_url() -> str:
    """Look up the GEFS analysis icechunk asset URL from the STAC catalog."""
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        catalog = client.get(STAC_CATALOG_URL).raise_for_status().json()
        for link in catalog["links"]:
            if link["rel"] != "child":
                continue
            collection = client.get(link["href"]).raise_for_status().json()
            if collection["id"] == GEFS_ANALYSIS_COLLECTION_ID:
                return collection["assets"]["icechunk"]["href"]
    raise ValueError(
        f"Collection {GEFS_ANALYSIS_COLLECTION_ID!r} not found in STAC catalog at {STAC_CATALOG_URL}"
    )


def resolve_reference_url(reference_url: str | None) -> str:
    if reference_url is not None:
        return reference_url
    return get_gefs_analysis_reference_url()


reference_url_option = typer.Option(
    None,
    "--reference-url",
    help="Reference dataset URL "
    "(default: GEFS analysis icechunk asset looked up from the STAC catalog)",
)


variables_option = typer.Option(
    None,
    "--variable",
    "-v",
    help="Variable to plot (can be used multiple times). "
    "If not provided, will plot all common variables.",
)

start_date_option = typer.Option(
    None,
    "--start-date",
    help="Scope analysis to times after this date",
)

end_date_option = typer.Option(
    None,
    "--end-date",
    help="Scope analysis to times before this date",
)

output_dir_option = typer.Option(
    None,
    "--output-dir",
    help="Write outputs into this directory instead of creating a new run directory.",
)

level_option = typer.Option(
    None,
    "--level",
    help="Vertical level value to sample for variables on a level dim (e.g. 500 for "
    "500 mb). Selects the nearest level on each variable's own vertical dim. "
    "Default: the middle level. Single-level variables ignore this.",
)


@dataclass
class AvailabilitySeries:
    """Fraction of one variable's data available per append-dim position.

    Built from manifest ref probes on a virtual store and from value (null) scans on a
    materialized store — see scripts/validation/availability.py. NaN fraction means the
    position was not probed (no present source file to probe against).
    """

    positions: np.ndarray  # datetime64[ns], sorted
    fraction: np.ndarray  # float in [0, 1], NaN = not probed


@dataclass
class VariableStats:
    """Stats + metadata accumulated for one variable across plot types."""

    name: str
    units: str | None = None
    long_name: str | None = None
    short_name: str | None = None
    standard_name: str | None = None
    step_type: str | None = None

    # Vertical level sampled for this variable (None for single-level variables)
    level_dim: str | None = None
    level_value: float | None = None

    # Availability over the append dim (manifest-probed on virtual stores,
    # value-scanned on materialized stores).
    availability_plot: str | None = None
    positions_total: int | None = None
    positions_complete: int | None = None
    first_incomplete: str | None = None
    last_incomplete: str | None = None

    # Null value counts at the two run points (materialized stores only)
    null_count_p1: int | None = None
    null_count_p2: int | None = None
    total_count_p1: int | None = None
    total_count_p2: int | None = None
    unavailable_timestamps_p1: list[str] = field(default_factory=list)
    unavailable_timestamps_p2: list[str] = field(default_factory=list)

    # Spatial comparison
    spatial_plot: str | None = None
    spatial_time_label: str | None = None
    val_spatial_min: float | None = None
    val_spatial_max: float | None = None
    val_spatial_mean: float | None = None
    ref_spatial_min: float | None = None
    ref_spatial_max: float | None = None
    ref_spatial_mean: float | None = None
    ref_available_spatial: bool = False

    # Value time series over the full period (per-timestep mean ± std at each point)
    value_ts_plot: str | None = None
    value_mean_p1: float | None = None
    value_std_p1: float | None = None
    value_min_p1: float | None = None
    value_max_p1: float | None = None
    value_mean_p2: float | None = None
    value_std_p2: float | None = None
    value_min_p2: float | None = None
    value_max_p2: float | None = None

    # Timeseries comparison
    temporal_plot: str | None = None
    val_temporal_min_p1: float | None = None
    val_temporal_max_p1: float | None = None
    val_temporal_mean_p1: float | None = None
    val_temporal_min_p2: float | None = None
    val_temporal_max_p2: float | None = None
    val_temporal_mean_p2: float | None = None
    ref_temporal_min_p1: float | None = None
    ref_temporal_max_p1: float | None = None
    ref_temporal_mean_p1: float | None = None
    ref_temporal_min_p2: float | None = None
    ref_temporal_max_p2: float | None = None
    ref_temporal_mean_p2: float | None = None
    ref_available_temporal: bool = False


@dataclass
class RunContext:
    """Shared state for a validation run. Built by run-all or lazily by individual commands."""

    output_dir: Path
    validation_url: str
    reference_url: str | None
    validation_ds: xr.Dataset
    reference_ds: xr.Dataset | None
    started_at: pd.Timestamp
    # Spatial points used by null + timeseries plots.
    point1_sel: dict[str, int]
    point2_sel: dict[str, int]
    point1_lat: float
    point1_lon: float
    point2_lat: float
    point2_lon: float
    ensemble_member: int | None
    variables: list[str]
    start_date: str | None = None
    # Virtual stores decode source files on read, inverting the cost model: a spatial
    # snapshot is cheap, an append-dim point column is expensive. Routes availability /
    # value_timeseries to manifest- and sample-based paths instead of full reads.
    is_virtual: bool = False
    level_override: float | None = None
    spatial_time_label: str | None = None
    ref_spatial_time_label: str | None = None
    temporal_period_label: str | None = None
    # Pinned lead/member of the virtual value-timeseries sample, for the summary header.
    value_ts_lead_label: str | None = None
    value_ts_member: int | None = None
    unavailable_timestamps_file: str | None = None
    # One-sentence description of how availability was measured, for the report.
    availability_method_note: str | None = None
    # Sampled decode health (virtual stores); None until run_decode_scan runs.
    decode_note: str | None = None
    decode_failures: list[str] | None = None
    availability: dict[str, AvailabilitySeries] = field(default_factory=dict)
    combined_availability_plot: str | None = None
    # Point arrays loaded once by run_value_availability (var -> (point1, point2)) and reused
    # by run_value_timeseries to avoid reading the point data a second time.
    loaded_point_data: dict[str, tuple[xr.DataArray, xr.DataArray]] = field(
        default_factory=dict
    )
    stats: dict[str, VariableStats] = field(default_factory=dict)

    def stats_for(self, var: str) -> VariableStats:
        if var not in self.stats:
            self.stats[var] = VariableStats(
                name=var, **extract_variable_metadata(self.validation_ds, var)
            )
        return self.stats[var]


def is_forecast_dataset(ds: xr.Dataset) -> bool:
    """Check if dataset is a forecast (has init_time and lead_time) or analysis (has time)."""
    return "init_time" in ds.dims and "lead_time" in ds.dims


def scope_time_period(
    ds: xr.Dataset, start_date: str | None, end_date: str | None
) -> xr.Dataset:
    append_dim = "init_time" if is_forecast_dataset(ds) else "time"
    if start_date or end_date:
        ds = ds.sel({append_dim: slice(start_date, end_date)})
    return ds


def var_slug(var: str) -> str:
    """Filesystem/anchor-safe form of a (possibly group-pathed) variable name.

    Flattened group vars are keyed by store path (e.g. `pressure_level/temperature`); the
    `/` would otherwise be read as a directory separator in plot filenames. Display text
    keeps the original name; only filenames and HTML ids go through here.
    """
    return var.replace("/", "__")


def _anonymous_virtual_credentials(
    storage: icechunk.Storage,
) -> dict[str, Any] | None:
    """Anonymous read credentials for a virtual store's persisted chunk containers.

    A virtual icechunk store decodes chunks from source files, which requires authorizing
    access to the source buckets. The container set is persisted in the repo config (see
    docs/virtual_datasets.md); every dynamical source is public S3, so anonymous S3
    credentials per container prefix suffice. Returns None for a materialized store.
    """
    config = icechunk.Repository.fetch_config(storage)
    containers = config.virtual_chunk_containers if config is not None else None
    if not containers:
        return None
    items = containers.values() if isinstance(containers, dict) else containers
    prefixes = [c.url_prefix if hasattr(c, "url_prefix") else c for c in items]
    return icechunk.containers_credentials(
        {prefix: icechunk.s3_anonymous_credentials() for prefix in prefixes}
    )


def open_icechunk_readonly(url: str) -> icechunk.IcechunkStore:
    """Open an s3 icechunk store read-only and anonymously (virtual chunk access included).

    Unlike StoreFactory.primary_store this needs no credentials or Kubernetes secret
    access, so offline validation runs anywhere the bucket is publicly readable.
    """
    assert url.startswith("s3://"), url
    assert url.endswith(".icechunk"), url
    path = url.removeprefix("s3://")
    assert "/" in path
    bucket, prefix = path.split("/", 1)
    storage = icechunk.s3_storage(
        bucket=bucket, prefix=prefix, anonymous=True, region="us-west-2"
    )
    repo = icechunk.Repository.open(
        storage,
        authorize_virtual_chunk_access=_anonymous_virtual_credentials(storage),
    )
    return repo.readonly_session("main").store


def load_retried(da: xr.DataArray) -> xr.DataArray:
    """Hours-long runs make enough object store reads that a transient failure is
    near-certain; reads are idempotent so retry them."""
    return retry(da.load)


def load_zarr_dataset(url: str) -> xr.Dataset:
    url = url.removesuffix("/")
    if url.startswith("s3://") and url.endswith(".icechunk"):
        store: StoreLike = open_icechunk_readonly(url)
        consolidated = False
    elif url.startswith("s3://"):
        store = ObjectStore(
            obstore.store.from_url(
                url,
                region="us-west-2",
                skip_signature=True,
                retry_config={
                    "max_retries": 16,
                    "backoff": {
                        "base": 2,
                        "init_backoff": timedelta(seconds=1),
                        "max_backoff": timedelta(seconds=16),
                    },
                    # A backstop, shouldn't hit this with the above backoff settings
                    "retry_timeout": timedelta(minutes=5),
                },
            )
        )
        consolidated = True
    else:
        store = url
        consolidated = True

    # open_flattened_dataset exposes every vertical group's vars (e.g.
    # pressure_level/temperature) keyed by store path, not just the root group.
    ds = open_flattened_dataset(store, consolidated=consolidated)
    if "longitude" in ds.coords and "latitude" in ds.coords:
        ds.longitude.load()
        ds.latitude.load()
    return ds


def get_spatial_dimensions(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.dims and "longitude" in ds.dims:
        return "latitude", "longitude"
    return "y", "x"


def get_random_spatial_indices(
    ds: xr.Dataset, lat_dim: str, lon_dim: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Get two random spatial indices for plotting."""
    rng = np.random.default_rng()
    lat_size = ds.sizes[lat_dim]
    lon_size = ds.sizes[lon_dim]

    lat_lo, lat_hi = 0, lat_size
    if lat_dim == "latitude":
        lats = ds.latitude.values
        # Avoid polar grid points where many variables have edge-case behavior
        # (e.g. wind components degenerate, projections distort).
        if lats.min() < -80 and lats.max() > 80:
            valid = np.where((lats >= -70) & (lats <= 70))[0]
            lat_lo, lat_hi = int(valid.min()), int(valid.max()) + 1

    lat_range = lat_hi - lat_lo
    lat1_idx = int(rng.integers(lat_lo, lat_lo + lat_range // 4))
    lat2_idx = int(rng.integers(lat_lo + 3 * lat_range // 4, lat_hi))
    lon1_idx = int(rng.integers(0, lon_size // 4))
    lon2_idx = int(rng.integers(3 * lon_size // 4, lon_size))
    point1_sel = {lat_dim: lat1_idx, lon_dim: lon1_idx}
    point2_sel = {lat_dim: lat2_idx, lon_dim: lon2_idx}
    return point1_sel, point2_sel


def get_two_random_points(
    ds: xr.Dataset,
) -> tuple[dict[str, int], dict[str, int], tuple[float, float], tuple[float, float]]:
    """Get two random spatial points (indices and coordinates)."""
    lat_dim, lon_dim = get_spatial_dimensions(ds)
    point1_sel, point2_sel = get_random_spatial_indices(ds, lat_dim, lon_dim)
    if lat_dim == "latitude" and lon_dim == "longitude":
        lat1 = float(ds.latitude[point1_sel["latitude"]])
        lon1 = float(ds.longitude[point1_sel["longitude"]])
        lat2 = float(ds.latitude[point2_sel["latitude"]])
        lon2 = float(ds.longitude[point2_sel["longitude"]])
    else:
        lat1 = float(ds.latitude[point1_sel["y"], point1_sel["x"]])
        lon1 = float(ds.longitude[point1_sel["y"], point1_sel["x"]])
        lat2 = float(ds.latitude[point2_sel["y"], point2_sel["x"]])
        lon2 = float(ds.longitude[point2_sel["y"], point2_sel["x"]])
    return point1_sel, point2_sel, (lat1, lon1), (lat2, lon2)


def select_variables_for_plotting(
    ds: xr.Dataset, requested_vars: list[str] | None
) -> list[str]:
    """Select and validate variables for plotting.

    Raises on any unknown requested variable: silently dropping one makes a partial
    validation run look complete. Group vars are addressed by store path
    (`pressure_level/temperature`).
    """
    available_vars = [str(k) for k in ds.data_vars]
    if requested_vars:
        missing = [var for var in requested_vars if var not in available_vars]
        if missing:
            raise ValueError(
                f"Variables not in dataset: {missing}. Group vars are addressed by "
                f"store path (e.g. pressure_level/temperature); available: {available_vars}"
            )
        selected_vars = list(requested_vars)
    else:
        selected_vars = available_vars
    selected_vars.sort()
    return selected_vars


def select_random_ensemble_member(ds: xr.Dataset) -> tuple[xr.Dataset, int | None]:
    """Select a random ensemble member and return the member index."""
    if "ensemble_member" not in ds.dims:
        return ds, None
    rng = np.random.default_rng()
    ensemble_member = int(rng.choice(ds.ensemble_member, 1)[0])
    return (
        ds.sel(ensemble_member=ensemble_member),
        ensemble_member,
    )


SPATIAL_DIMS = ("y", "x", "latitude", "longitude")
_NON_VERTICAL_DIMS = (
    "init_time",
    "time",
    "valid_time",
    "lead_time",
    "ensemble_member",
    *SPATIAL_DIMS,
)


def vertical_dims(ds: xr.Dataset, var: str) -> list[str]:
    """Vertical (level) dims of a variable — anything that isn't time/lead/member/spatial.

    A var in a vertical group (e.g. ``pressure_level/temperature``) keeps its level dim
    after flattening; single-level vars (``temperature_2m``) have none. Generic across
    level types (pressure_level, model_level, …) — no level name is hard-coded.
    """
    return [str(d) for d in ds[var].dims if d not in _NON_VERTICAL_DIMS]


def choose_level(ds: xr.Dataset, var: str, override: float | None) -> dict[str, Any]:
    """Pick one representative level for a vertical var, or ``{}`` for a single-level var.

    The report samples a single level per vertical variable (like it samples one ensemble
    member and one spatial snapshot); use ``override`` to inspect a specific level. Default
    is the middle level of the var's own vertical dim; override selects the nearest. Returns
    a ``{level_dim: label}`` mapping ready for ``.sel()``.
    """
    vdims = vertical_dims(ds, var)
    if not vdims:
        return {}
    assert len(vdims) == 1, f"{var} has multiple vertical dims {vdims}, not supported"
    dim = vdims[0]
    coord = ds[dim].values
    if override is not None:
        idx = int(np.abs(coord - override).argmin())
    else:
        idx = len(coord) // 2
    return {dim: coord[idx].item()}


def select_var_level(ctx: RunContext, var: str, stats: VariableStats) -> dict[str, Any]:
    """Resolve and record the level to plot for `var`; ``{}`` for single-level vars."""
    sel = choose_level(ctx.validation_ds, var, ctx.level_override)
    if sel:
        [(dim, value)] = sel.items()
        stats.level_dim = dim
        stats.level_value = float(value)
    return sel


def level_label(stats: VariableStats) -> str:
    """Display suffix for the sampled level, or empty string for single-level vars."""
    if stats.level_dim is None:
        return ""
    return f" [{stats.level_dim}={stats.level_value:g}]"


def is_virtual_store(url: str) -> bool:
    """True if `url` is an icechunk store with persisted virtual chunk containers."""
    url = url.removesuffix("/")
    if not (url.startswith("s3://") and url.endswith(".icechunk")):
        return False
    path = url.removeprefix("s3://")
    assert "/" in path
    bucket, prefix = path.split("/", 1)
    storage = icechunk.s3_storage(
        bucket=bucket, prefix=prefix, anonymous=True, region="us-west-2"
    )
    return _anonymous_virtual_credentials(storage) is not None


def extract_variable_metadata(ds: xr.Dataset, var: str) -> dict[str, Any]:
    """Pull commonly-referenced attrs (units, long_name, etc.) from a variable."""
    attrs = ds[var].attrs
    return {
        "units": attrs.get("units"),
        "long_name": attrs.get("long_name"),
        "short_name": attrs.get("short_name"),
        "standard_name": attrs.get("standard_name"),
        "step_type": attrs.get("step_type"),
    }


def dataset_id_and_version(url: str) -> tuple[str, str]:
    """Parse `.../<dataset-id>/<version>{.zarr|.icechunk}` from a URL.

    Strips the `.zarr` / `.icechunk` suffix from the version so it's usable in file paths.
    """
    url_clean = url.removesuffix("/")
    parts = url_clean.split("/")
    version = parts[-1].removesuffix(".zarr").removesuffix(".icechunk")
    dataset_id = parts[-2]
    return dataset_id, version


def create_run_output_dir(
    validation_url: str, base_timestamp: pd.Timestamp | None = None
) -> Path:
    """Create the per-run output directory: data/output/<dataset-id>/<version>_<YYYY-MM-DDTHH-MM>/"""
    ts = base_timestamp if base_timestamp is not None else pd.Timestamp.now(tz="UTC")
    timestamp_str = ts.strftime("%Y-%m-%dT%H-%M")
    dataset_id, version = dataset_id_and_version(validation_url)
    run_dir = Path(OUTPUT_DIR) / dataset_id / f"{version}_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_output_dir(validation_url: str, output_dir: Path | str | None) -> Path:
    """Resolve the output dir: use the provided one, or create a new per-run dir."""
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return create_run_output_dir(validation_url)
