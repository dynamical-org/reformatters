import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from reformatters.common.logging import get_logger
from scripts.validation.utils import (
    OUTPUT_DIR,
    end_date_option,
    get_two_random_points,
    is_forecast_dataset,
    load_zarr_dataset,
    scope_time_period,
    select_random_ensemble_member,
    select_variables_for_plotting,
    start_date_option,
    variables_option,
)

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 128})

GEFS_ANALYSIS_URL = "https://data.dynamical.org/noaa/gefs/analysis/latest.zarr"


def select_time_period_for_comparison(
    validation_ds: xr.Dataset, reference_ds: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset, str, str, str]:
    """Selects appropriate time periods for validation and reference datasets."""
    rng = np.random.default_rng()
    if is_forecast_dataset(validation_ds):
        log.info("Detected forecast dataset - selecting random init_time")
        selected_init_time = pd.Timestamp(rng.choice(validation_ds.init_time, 1)[0])
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
            random_offset = rng.integers(0, int(time_range_seconds) + 1)
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
        assert "latitude" in reference_subset.dims, (
            "Reference datasets must have latitude dimension"
        )
        assert "longitude" in reference_subset.dims, (
            "Reference datasets must have longitude dimension"
        )

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
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
) -> None:
    """Compare timeseries between validation and reference datasets."""

    validation_ds = load_zarr_dataset(validation_url)
    if start_date or end_date:
        validation_ds = scope_time_period(validation_ds, start_date, end_date)
    reference_ds = load_zarr_dataset(reference_url)

    selected_vars = select_variables_for_plotting(validation_ds, variables)

    validation_ds, ensemble_member = select_random_ensemble_member(validation_ds)

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
        len(selected_vars),
        2,
        figsize=(12, 4 * len(selected_vars)),
        squeeze=False,
        constrained_layout=True,
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
    ds_title = validation_ds.attrs.get("name", "")
    if ensemble_member is not None:
        ds_title += f" (Ensemble Member {ensemble_member})"
    ref_title = reference_ds.attrs.get("name", "")
    fig.suptitle(
        f"Timeseries Comparison\n{ds_title} vs {ref_title}\n{title_suffix}", fontsize=14
    )

    save_and_show_plot(fig, dataset_id, show_plot)
