import numpy as np
import typer
import xarray as xr

# Common constants
OUTPUT_DIR = "data/output"

# Common typer options
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
    if is_forecast_dataset(ds):
        append_dim = "init_time"
    else:
        append_dim = "time"
    if start_date:
        ds = ds.sel({append_dim: slice(start_date, end_date)})
    if end_date:
        ds = ds.sel({append_dim: slice(start_date, end_date)})
    return ds


# Utility: Load a zarr dataset as xarray.Dataset
def load_zarr_dataset(url: str, decode_timedelta: bool = False) -> xr.Dataset:
    ds: xr.Dataset = xr.open_zarr(url, chunks=None, decode_timedelta=decode_timedelta)
    return ds


# Utility: Get spatial dimension names
def get_spatial_dimensions(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.dims and "longitude" in ds.dims:
        return "latitude", "longitude"
    return "y", "x"


# Utility: Get two random spatial indices for plotting
def get_random_spatial_indices(
    ds: xr.Dataset, lat_dim: str, lon_dim: str
) -> tuple[dict[str, int], dict[str, int]]:
    lat_size = ds.sizes[lat_dim]
    lon_size = ds.sizes[lon_dim]
    lat1_idx = np.random.randint(0, lat_size // 4)
    lon1_idx = np.random.randint(0, lon_size // 4)
    lat2_idx = np.random.randint(3 * lat_size // 4, lat_size)
    lon2_idx = np.random.randint(3 * lon_size // 4, lon_size)
    point1_sel = {lat_dim: lat1_idx, lon_dim: lon1_idx}
    point2_sel = {lat_dim: lat2_idx, lon_dim: lon2_idx}
    return point1_sel, point2_sel


# Utility: Get two random spatial points (indices and coordinates)
def get_two_random_points(
    ds: xr.Dataset,
) -> tuple[dict[str, int], dict[str, int], tuple[float, float], tuple[float, float]]:
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


# Utility: Select and validate variables for plotting
def select_variables_for_plotting(
    ds: xr.Dataset, requested_vars: list[str] | None
) -> list[str]:
    available_vars = list(ds.data_vars.keys())
    if requested_vars:
        selected_vars = [var for var in requested_vars if var in available_vars]
        if not selected_vars:
            raise ValueError("No valid variables specified")
    else:
        selected_vars = available_vars
    return selected_vars


# Utility: Select a random ensemble member and return the member index
def select_random_ensemble_member(ds: xr.Dataset) -> tuple[xr.Dataset, int | None]:
    if "ensemble_member" not in ds.dims:
        return ds, None
    ensemble_member = np.random.choice(ds.ensemble_member, 1)[0]
    return (
        ds.sel(ensemble_member=ensemble_member),
        ensemble_member,
    )
