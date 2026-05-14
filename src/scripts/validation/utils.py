from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import obstore
import pandas as pd
import typer
import xarray as xr
from zarr.storage import ObjectStore, StoreLike

OUTPUT_DIR = "data/output"

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


@dataclass
class VariableStats:
    """Stats + metadata accumulated for one variable across plot types."""

    name: str
    units: str | None = None
    long_name: str | None = None
    short_name: str | None = None
    standard_name: str | None = None
    step_type: str | None = None

    # Null analysis
    null_plot: str | None = None
    null_count_p1: int | None = None
    null_count_p2: int | None = None
    total_count_p1: int | None = None
    total_count_p2: int | None = None
    missing_timestamps_p1: list[str] = field(default_factory=list)
    missing_timestamps_p2: list[str] = field(default_factory=list)

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
    end_date: str | None = None
    spatial_time_label: str | None = None
    ref_spatial_time_label: str | None = None
    temporal_period_label: str | None = None
    missing_timestamps_file: str | None = None
    combined_nulls_plot: str | None = None
    combined_spatial_plot: str | None = None
    combined_temporal_plot: str | None = None
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


def load_zarr_dataset(url: str) -> xr.Dataset:
    url = url.removesuffix("/")
    if url.startswith("s3://"):
        if url.endswith(".icechunk"):
            path = url.removeprefix("s3://")
            assert "/" in path
            i = path.index("/")
            bucket = path[:i]
            prefix = path[i + 1 :]
            storage = icechunk.s3_storage(
                bucket=bucket, prefix=prefix, anonymous=True, region="us-west-2"
            )
            repo = icechunk.Repository.open(storage)
            session = repo.readonly_session("main")
            store: StoreLike = session.store
        else:
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
    else:
        store = url

    ds: xr.Dataset = xr.open_zarr(store, chunks=None, decode_timedelta=True)
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
    lat1_idx = int(rng.integers(0, lat_size // 4))
    lon1_idx = int(rng.integers(0, lon_size // 4))
    lat2_idx = int(rng.integers(3 * lat_size // 4, lat_size))
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
    """Select and validate variables for plotting."""
    available_vars = [str(k) for k in ds.data_vars]
    if requested_vars:
        selected_vars = [var for var in requested_vars if var in available_vars]
        if not selected_vars:
            raise ValueError("No valid variables specified")
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
