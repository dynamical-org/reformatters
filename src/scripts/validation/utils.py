from datetime import timedelta
from pathlib import Path

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


def is_forecast_dataset(ds: xr.Dataset) -> bool:
    """Check if dataset is a forecast (has init_time and lead_time) or analysis (has time)."""
    return "init_time" in ds.dims and "lead_time" in ds.dims


def scope_time_period(
    ds: xr.Dataset, start_date: str | None, end_date: str | None
) -> xr.Dataset:
    append_dim = "init_time" if is_forecast_dataset(ds) else "time"
    if start_date:
        ds = ds.sel({append_dim: slice(start_date, end_date)})
    if end_date:
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
    ensemble_member = rng.choice(ds.ensemble_member, 1)[0]
    return (
        ds.sel(ensemble_member=ensemble_member),
        ensemble_member,
    )


def get_output_filepath(base_filename: str, url: str) -> Path:
    """
    Get output filepath by generating a filename with timestamp and
    version suffix and creating the parent directory organized by dataset id.
    """
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M-%S")
    url_clean = url.removesuffix("/")
    path_components = url_clean.split("/")
    url_suffix = path_components[-1].removesuffix("/")
    dataset_id = path_components[-2]
    filename = f"{base_filename}_{url_suffix}_{timestamp_str}.png"
    filepath = Path(OUTPUT_DIR) / dataset_id / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath
