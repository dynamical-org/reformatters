import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xarray as xr
import zarr

from reformatters.common.logging import get_logger
from reformatters.common.time_utils import whole_hours
from scripts.validation.utils import (
    end_date_option,
    get_output_filepath,
    is_forecast_dataset,
    load_zarr_dataset,
    scope_time_period,
    select_random_ensemble_member,
    start_date_option,
    variables_option,
)

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 128})

GEFS_ANALYSIS_URL = "https://data.dynamical.org/noaa/gefs/analysis/latest.zarr"


def align_reference_spatially(ds: xr.Dataset, reference_ds: xr.Dataset) -> xr.Dataset:
    return reference_ds.sel(
        latitude=slice(ds.latitude.max(), ds.latitude.min()),
        longitude=slice(ds.longitude.min(), ds.longitude.max()),
    )


def align_to_valid_time_forecast(
    ds: xr.Dataset,
    reference_ds: xr.Dataset,
    init_time: str | None,
    lead_time: str | None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Align forecast dataset to reference dataset by selecting init_time/lead_time."""
    rng = np.random.default_rng()

    if init_time is None:
        selected_init_time = pd.Timestamp(rng.choice(ds.init_time, 1)[0])
    else:
        selected_init_time = pd.Timestamp(init_time)

    if lead_time is None:
        selected_lead_time = rng.choice(ds.lead_time, 1)[0]
    else:
        selected_lead_time = lead_time

    ds = ds.sel(
        init_time=selected_init_time,
        lead_time=selected_lead_time,
    )

    valid_time = pd.Timestamp(ds.valid_time.item())

    reference_ds = reference_ds.sel(time=valid_time, method="nearest")
    return ds, reference_ds


def align_to_valid_time_analysis(
    ds: xr.Dataset,
    reference_ds: xr.Dataset,
    time: str | None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Align analysis dataset to reference dataset by selecting a time."""
    rng = np.random.default_rng()

    if time is None:
        selected_time = pd.Timestamp(rng.choice(ds.time, 1)[0])
    else:
        selected_time = pd.Timestamp(time)

    ds = ds.sel(time=selected_time)
    reference_ds = reference_ds.sel(time=selected_time, method="nearest")
    return ds, reference_ds


def create_comparison_plot(  # noqa: PLR0915 PLR0912
    validation_ds: xr.Dataset,
    reference_ds: xr.Dataset,
    variables: list[str],
    validation_url: str,
    ensemble_member: int | None = None,
    init_time: str | None = None,
    lead_time: str | None = None,
    time: str | None = None,
) -> None:
    """Create comparison plot matching the example image format"""
    is_forecast = is_forecast_dataset(validation_ds)

    # Align datasets to a common time point (done once for all variables)
    if is_forecast:
        ds, ref_ds = align_to_valid_time_forecast(
            validation_ds,
            reference_ds,
            init_time,
            lead_time,
        )
    else:
        ds, ref_ds = align_to_valid_time_analysis(
            validation_ds,
            reference_ds,
            time,
        )

    # Format timestamps for titles (done once for all variables)
    if is_forecast:
        ds_init_time = pd.Timestamp(ds.init_time.item()).strftime("%Y-%m-%dT%H:%M")
        ds_lead_time_hours = whole_hours(pd.Timedelta(ds.lead_time.item()))
        ds_lead_time_str = f"{ds_lead_time_hours:g}h"
        ds_time = f"{ds_init_time}+{ds_lead_time_str}"
    else:
        ds_time = pd.Timestamp(ds.time.item()).strftime("%Y-%m-%dT%H:%M")
    ref_time = pd.Timestamp(ref_ds.time.item()).strftime("%Y-%m-%dT%H:%M")

    n_vars = len(variables)
    plt.figure(figsize=(15, 3 * n_vars))

    for i, var in enumerate(variables):
        # Get validation data array
        data = ds[var].load()

        # Check if variable exists in reference dataset
        var_in_reference = var in reference_ds.data_vars

        if var_in_reference:
            ref_data = ref_ds[var].load()
            vmin = min(float(data.min()), float(ref_data.min()))
            vmax = max(float(data.max()), float(ref_data.max()))
        else:
            vmin = float(data.min())
            vmax = float(data.max())

        if var_in_reference:
            log.info(
                f"Plotting {var} for {ds.attrs['name']} ({ds_time}) and {ref_ds.attrs['name']} ({ref_time})"
            )
        else:
            log.info(
                f"Plotting {var} for {ds.attrs['name']} ({ds_time}) - variable not found in reference dataset"
            )

        ds_title = f"{ds.attrs['name']}"
        if ensemble_member is not None:
            ds_title += f" (Ensemble Member {ensemble_member})"
        ds_title += f" - {var}\n{ds_time}"

        if var_in_reference:
            ref_title = f"{ref_ds.attrs['name']} - {var}\n{ref_time}"
        else:
            ref_title = f"{ref_ds.attrs['name']} - {var}\n(Variable not available)"

        # Left plot - reference dataset (or empty if variable doesn't exist)
        ax1 = plt.subplot(n_vars, 3, i * 3 + 1)

        if var_in_reference:
            im1 = ax1.pcolormesh(
                ref_data.longitude,  # ty: ignore[possibly-unresolved-reference]
                ref_data.latitude,  # ty: ignore[possibly-unresolved-reference]
                ref_data.values,  # ty: ignore[possibly-unresolved-reference]
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(im1, ax=ax1)

            # Use combined bounds from both datasets for consistent axis ranges
            lon_min = min(float(ref_data.longitude.min()), float(data.longitude.min()))  # ty: ignore[possibly-unresolved-reference]
            lon_max = max(float(ref_data.longitude.max()), float(data.longitude.max()))  # ty: ignore[possibly-unresolved-reference]
            lat_min = min(float(ref_data.latitude.min()), float(data.latitude.min()))  # ty: ignore[possibly-unresolved-reference]
            lat_max = max(float(ref_data.latitude.max()), float(data.latitude.max()))  # ty: ignore[possibly-unresolved-reference]
        else:
            # Show empty plot for missing variable
            ax1.text(
                0.5,
                0.5,
                "Variable not\navailable in\nreference dataset",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=12,
                color="gray",
            )
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)

            # Use validation dataset bounds when reference doesn't have the variable
            lon_min = float(data.longitude.min())
            lon_max = float(data.longitude.max())
            lat_min = float(data.latitude.min())
            lat_max = float(data.latitude.max())

        ax1.set_title(ref_title)
        ax1.set_aspect("auto")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")

        # Middle plot - validation dataset
        ax2 = plt.subplot(n_vars, 3, i * 3 + 2)
        im2 = ax2.pcolormesh(
            data.longitude,
            data.latitude,
            data.values,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im2, ax=ax2)
        ax2.set_title(ds_title)
        ax2.set_aspect("auto")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")

        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)

        # Right plot - histogram comparison
        ax3 = plt.subplot(n_vars, 3, i * 3 + 3)

        # Flatten validation data and remove NaN values in one step
        data_values_flat = data.values.flat
        data_clean = data_values_flat[~np.isnan(data_values_flat)]

        if len(data_clean) == 0:
            ax3.text(
                0.5,
                0.5,
                "All data in validated\ndataset is nan at this step",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
                color="gray",
            )
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            continue

        if var_in_reference and not np.isnan(ref_data.values).all():  # ty: ignore[possibly-unresolved-reference]
            # Include reference data in histogram if available
            ref_values_flat = ref_data.values.flat  # ty: ignore[possibly-unresolved-reference]
            ref_clean = ref_values_flat[~np.isnan(ref_values_flat)]

            # Calculate combined range for consistent bins
            data_min, data_max = (
                min(np.min(data_clean), np.min(ref_clean)),
                max(np.max(data_clean), np.max(ref_clean)),
            )

            # Create histograms with consistent bins
            ax3.hist(
                ref_clean,
                bins=40,
                alpha=0.7,
                label=ref_ds.attrs["name"],
                color="blue",
                range=(data_min, data_max),
                density=True,
                stacked=True,
            )
        else:
            # Only validation data available
            data_min, data_max = np.min(data_clean), np.max(data_clean)

        ax3.hist(
            data_clean,
            bins=40,
            alpha=0.7,
            label=ds.attrs["name"],
            color="red",
            range=(data_min, data_max),
            density=True,
            stacked=True,
        )

        # Set x-axis limits to match the data range (only if there's actual range)
        if data_min != data_max:
            ax3.set_xlim(data_min, data_max)

        ax3.set_xlabel(f"{var}")
        ax3.set_ylabel("Frequency")
        ax3.set_title(f"Distribution - {var}")

        # Position legend below the title
        ax3.legend(
            bbox_to_anchor=(0.5, 0.95),
            loc="center",
            ncol=2,
            fontsize="small",
            frameon=False,
            columnspacing=1.0,
        )
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filename_parts = [
        validation_ds.attrs["dataset_id"],
        validation_ds.attrs["dataset_version"],
        ds_time,
        reference_ds.attrs["dataset_id"],
        reference_ds.attrs["dataset_version"],
        ref_time,
        f"{len(variables)}_variables",
    ]
    base_filename = "_".join(filename_parts)
    filepath = get_output_filepath(base_filename, validation_url)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    log.info(f"Comparison plot saved to {filepath}")


def compare_spatial(
    validation_url: str,
    reference_url: str = GEFS_ANALYSIS_URL,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
    init_time: str | None = None,
    lead_time: str | None = None,
    time: str | None = None,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
) -> None:
    """Create comparison plots between two zarr datasets.

    For forecast datasets (with init_time and lead_time dimensions), use
    --init-time and --lead-time to specify the point to plot.

    For analysis datasets (with time dimension), use --time to specify
    the point to plot.
    """

    log.info(f"Loading validation dataset from: {validation_url}")
    validation_ds = load_zarr_dataset(validation_url)
    if start_date or end_date:
        validation_ds = scope_time_period(validation_ds, start_date, end_date)

    is_forecast = is_forecast_dataset(validation_ds)
    log.info(
        f"Detected {'forecast' if is_forecast else 'analysis'} dataset "
        f"(dimensions: {list(validation_ds.sizes.keys())})"
    )

    log.info(f"Loading reference dataset from: {reference_url}")
    reference_ds = load_zarr_dataset(reference_url)

    validation_vars = [str(k) for k in validation_ds.data_vars]
    log.info(f"Found {len(validation_vars)} variables in validation dataset")

    if variables:
        # Use specified variables that exist in validation dataset
        selected_vars = [var for var in variables if var in validation_ds.data_vars]
        missing_vars = set(variables) - set(validation_vars)

        if missing_vars:
            log.warning(f"Variables not found in validation dataset: {missing_vars}")

        if not selected_vars:
            typer.echo("Error: No valid variables specified", err=True)
            raise typer.Exit(1)
    else:
        selected_vars = validation_vars

    log.info(f"Plotting variables: {selected_vars}")

    validation_ds, ensemble_member = select_random_ensemble_member(validation_ds)
    spatially_aligned_reference_ds = align_reference_spatially(
        validation_ds, reference_ds
    )

    create_comparison_plot(
        validation_ds,
        spatially_aligned_reference_ds,
        selected_vars,
        validation_url=validation_url,
        ensemble_member=ensemble_member,
        init_time=init_time,
        lead_time=lead_time,
        time=time,
    )

    if show_plot:
        plt.show()
