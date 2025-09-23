import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xarray as xr
import zarr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from reformatters.common.logging import get_logger
from scripts.validation.utils import OUTPUT_DIR, variables_option

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 128})

GEFS_ANALYSIS_URL = "https://data.dynamical.org/noaa/gefs/analysis/latest.zarr"


def is_forecast_dataset(ds: xr.Dataset) -> bool:
    """Check if dataset is a forecast (has init_time and lead_time) or analysis (has time)."""
    return "init_time" in ds.dims and "lead_time" in ds.dims


def load_zarr_dataset(url: str, decode_timedelta: bool = False) -> xr.Dataset:
    """Loads an xarray dataset from a Zarr URL."""
    log.info(f"Loading dataset from: {url}")
    ds: xr.Dataset = xr.open_zarr(url, chunks=None, decode_timedelta=decode_timedelta)
    return ds


def get_spatial_dimensions(ds: xr.Dataset) -> tuple[str, str]:
    """Determine spatial dimension names based on dataset."""
    if "latitude" in ds.dims and "longitude" in ds.dims:
        return "latitude", "longitude"
    return "y", "x"


def get_random_spatial_indices(
    ds: xr.Dataset, lat_dim: str, lon_dim: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Get two random spatial indices for plotting, ensuring different quartiles."""
    lat_size = ds.sizes[lat_dim]
    lon_size = ds.sizes[lon_dim]

    lat1_idx = np.random.randint(0, lat_size // 4)
    lon1_idx = np.random.randint(0, lon_size // 4)
    lat2_idx = np.random.randint(3 * lat_size // 4, lat_size)
    lon2_idx = np.random.randint(3 * lon_size // 4, lon_size)

    point1_sel = {lat_dim: lat1_idx, lon_dim: lon1_idx}
    point2_sel = {lat_dim: lat2_idx, lon_dim: lon2_idx}

    log.info(f"Selected indices: Point 1: {point1_sel}, Point 2: {point2_sel}")
    log.info(f"Spatial dimensions: {lat_dim}={lat_size}, {lon_dim}={lon_size}")

    return point1_sel, point2_sel


def get_two_random_points(
    ds: xr.Dataset,
) -> tuple[dict[str, int], dict[str, int], tuple[float, float], tuple[float, float]]:
    """Get two random spatial points (indices and coordinates) for plotting."""
    lat_dim, lon_dim = get_spatial_dimensions(ds)
    point1_sel, point2_sel = get_random_spatial_indices(ds, lat_dim, lon_dim)

    if lat_dim == "latitude" and lon_dim == "longitude":
        lat1 = float(ds.latitude[point1_sel["latitude"]])
        lon1 = float(ds.longitude[point1_sel["longitude"]])
        lat2 = float(ds.latitude[point2_sel["latitude"]])
        lon2 = float(ds.longitude[point2_sel["longitude"]])
    else:  # Projected grids (y, x)
        lat1 = float(ds.latitude[point1_sel["y"], point1_sel["x"]])
        lon1 = float(ds.longitude[point1_sel["y"], point1_sel["x"]])
        lat2 = float(ds.latitude[point2_sel["y"], point2_sel["x"]])
        lon2 = float(ds.longitude[point2_sel["y"], point2_sel["x"]])

    log.info(f"Point 1: lat={lat1:.2f}, lon={lon1:.2f}")
    log.info(f"Point 2: lat={lat2:.2f}, lon={lon2:.2f}")
    return point1_sel, point2_sel, (lat1, lon1), (lat2, lon2)


def select_variables_for_plotting(
    ds: xr.Dataset, requested_vars: list[str] | None
) -> list[str]:
    """Selects and validates variables for plotting."""
    available_vars = list(ds.data_vars.keys())
    log.info(f"Found {len(available_vars)} variables in dataset")

    if requested_vars:
        selected_vars = [var for var in requested_vars if var in available_vars]
        if not selected_vars:
            typer.echo("Error: No valid variables specified", err=True)
            raise typer.Exit(1)
    else:
        selected_vars = available_vars

    log.info(f"Plotting variables: {selected_vars}")
    return selected_vars


def select_ensemble_member(ds: xr.Dataset) -> xr.Dataset:
    """Selects a random ensemble member if the dimension exists."""
    if "ensemble_member" in ds.dims:
        ensemble_member = np.random.choice(ds.ensemble_member, 1)[0]
        ds = ds.sel(ensemble_member=ensemble_member)
        log.info(f"Selected ensemble member: {ensemble_member}")
    return ds


def select_time_period_for_comparison(
    validation_ds: xr.Dataset, reference_ds: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset, str, str, str]:
    """Selects appropriate time periods for validation and reference datasets."""
    if is_forecast_dataset(validation_ds):
        log.info("Detected forecast dataset - selecting random init_time")
        selected_init_time = pd.Timestamp(
            np.random.choice(validation_ds.init_time, 1)[0]
        )
        validation_subset = validation_ds.sel(init_time=selected_init_time)
        log.info(f"Selected init_time: {selected_init_time}")

        valid_time_start = validation_subset.valid_time.min().item()
        valid_time_end = validation_subset.valid_time.max().item()
        reference_subset = reference_ds.sel(
            time=slice(pd.Timestamp(valid_time_start), pd.Timestamp(valid_time_end))
        )

        title_suffix = (
            f"Forecast init_time: {selected_init_time.strftime('%Y-%m-%dT%H:%M')}"
        )
        time_coord = "valid_time"
        ref_time_coord = "time"

    else:
        log.info("Detected analysis dataset - selecting random 10-day period")

        time_start = pd.Timestamp(validation_ds.time.min().item())
        time_end = pd.Timestamp(validation_ds.time.max().item())
        ten_days = pd.Timedelta(days=10)

        if time_end - time_start < ten_days:
            validation_subset = validation_ds
            selected_start = time_start
            selected_end = time_end
            log.info("Dataset shorter than 10 days, using entire time range")
        else:
            latest_start = time_end - ten_days
            time_range_seconds = (latest_start - time_start).total_seconds()
            random_offset = np.random.randint(0, int(time_range_seconds) + 1)
            selected_start = time_start + pd.Timedelta(seconds=random_offset)
            selected_end = selected_start + ten_days
            log.info(f"Selected time period: {selected_start} to {selected_end}")

        validation_subset = validation_ds.sel(time=slice(selected_start, selected_end))
        reference_subset = reference_ds.sel(time=slice(selected_start, selected_end))

        title_suffix = "Analysis period: 10-day sample"
        time_coord = "time"
        ref_time_coord = "time"

    return validation_subset, reference_subset, title_suffix, time_coord, ref_time_coord


def plot_single_variable_at_point(
    ax: Axes,
    var: str,
    point_sel: dict[str, int],
    lat: float,
    lon: float,
    validation_subset: xr.Dataset,
    reference_subset: xr.Dataset,
    time_coord: str,
    ref_time_coord: str,
) -> None:
    """Plots a single variable at a specific spatial point for both datasets."""
    # Select validation data for this point
    val_point_data = validation_subset[var].isel(point_sel)

    # Debug the data
    log.info(f"Validation data shape for {var}: {val_point_data.shape}")
    log.info(
        f"Validation data min/max for {var}: {float(val_point_data.min()):.3f} / {float(val_point_data.max()):.3f}"
    )
    log.info(f"Point selection: {point_sel}")
    log.info(f"First few values for {var}: {val_point_data.values[:5]}")

    # Plot validation data
    ax.plot(
        val_point_data[time_coord],
        val_point_data,
        marker="o",
        linestyle="-",
        markersize=3,
        color="red",
        label=validation_subset.attrs.get("name", "Validation"),
        alpha=0.8,
    )

    # Plot reference data if variable exists
    if var in reference_subset.data_vars:
        ref_point_data = reference_subset[var].sel(
            latitude=lat, longitude=lon, method="nearest"
        )

        ax.plot(
            ref_point_data[ref_time_coord],
            ref_point_data,
            marker="s",
            linestyle="-",
            markersize=3,
            color="blue",
            label=reference_subset.attrs.get("name", "Reference"),
            alpha=0.8,
        )

    # Formatting
    ax.set_xlabel("Valid Time" if time_coord == "valid_time" else "Time")
    ax.set_ylabel(f"{var}")
    ax.set_title(f"{var} - (lat={lat:.2f}, lon={lon:.2f})", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.relim()
    ax.autoscale_view()


def save_and_show_plot(fig: Figure, dataset_id: str, show_plot: bool) -> None:
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    filename = f"{dataset_id}_timeseries_comparison.png"
    filepath = f"{OUTPUT_DIR}/{filename}"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    log.info(f"Saved to {filepath}")
    if show_plot:
        fig.show()


def compare_timeseries(
    validation_url: str,
    reference_url: str = GEFS_ANALYSIS_URL,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
) -> None:
    """Compare timeseries between validation and reference datasets."""

    validation_ds = load_zarr_dataset(validation_url, decode_timedelta=True)
    # TODO: Make the time slice configurable or remove if not always needed.
    # validation_ds = validation_ds.sel(init_time=slice("2025-09-01T00", "2025-09-22T00"))

    reference_ds = load_zarr_dataset(reference_url)

    selected_vars = select_variables_for_plotting(validation_ds, variables)

    validation_ds = select_ensemble_member(validation_ds)

    point1_sel, point2_sel, (lat1, lon1), (lat2, lon2) = get_two_random_points(
        validation_ds
    )

    (
        validation_subset,
        reference_subset,
        title_suffix,
        time_coord,
        ref_time_coord,
    ) = select_time_period_for_comparison(validation_ds, reference_ds)

    fig, axes = plt.subplots(
        len(selected_vars), 2, figsize=(12, 4 * len(selected_vars)), squeeze=False
    )

    for i, var in enumerate(selected_vars):
        log.info(f"Plotting {var}")

        plot_single_variable_at_point(
            axes[i, 0],
            var,
            point1_sel,
            lat1,
            lon1,
            validation_subset,
            reference_subset,
            time_coord,
            ref_time_coord,
        )

        plot_single_variable_at_point(
            axes[i, 1],
            var,
            point2_sel,
            lat2,
            lon2,
            validation_subset,
            reference_subset,
            time_coord,
            ref_time_coord,
        )

    dataset_id = validation_ds.attrs.get("dataset_id", "")
    fig.suptitle(f"Timeseries Comparison\n{title_suffix}", fontsize=14, y=0.95)

    save_and_show_plot(fig, dataset_id, show_plot)
