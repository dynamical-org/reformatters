import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xarray as xr
import zarr

from reformatters.common.logging import get_logger
from scripts.validation.utils import OUTPUT_DIR, variables_option

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 128})

GEFS_ANALYSIS_URL = "https://data.dynamical.org/noaa/gefs/analysis/latest.zarr"


def align_reference_spatially(ds: xr.Dataset, reference_ds: xr.Dataset) -> xr.Dataset:
    return reference_ds.sel(
        latitude=slice(ds.latitude.max(), ds.latitude.min()),
        longitude=slice(ds.longitude.min(), ds.longitude.max()),
    )


def align_to_valid_time(
    ds: xr.Dataset,
    reference_ds: xr.Dataset,
    init_time: str | None,
    lead_time: str | None,
) -> tuple[xr.Dataset, xr.Dataset]:
    selected_init_time: pd.Timestamp
    selected_lead_time: pd.Timedelta

    if init_time is None:
        selected_init_time = pd.Timestamp(np.random.choice(ds.init_time, 1)[0])
    else:
        selected_init_time = pd.Timestamp(init_time)

    if lead_time is None:
        selected_lead_time = pd.Timedelta(np.random.choice(ds.lead_time, 1)[0])
    else:
        selected_lead_time = pd.Timedelta(lead_time)

    ds = ds.sel(
        init_time=selected_init_time,
        lead_time=selected_lead_time,
    )

    valid_time = pd.Timestamp(ds.valid_time.item())

    reference_ds = reference_ds.sel(time=valid_time, method="nearest")
    return ds, reference_ds


def select_random_enseble_member(ds: xr.Dataset) -> tuple[xr.Dataset, int | None]:
    if "ensemble_member" not in ds.dims:
        return ds, None

    ensemble_member = np.random.choice(ds.ensemble_member, 1)[0]
    return (
        ds.sel(ensemble_member=ensemble_member).squeeze("ensemble_member"),
        ensemble_member,
    )


def create_comparison_plot(
    validation_ds: xr.Dataset,
    reference_ds: xr.Dataset,
    variables: list[str],
    ensemble_member: int | None = None,
    init_time: str | None = None,
    lead_time: str | None = None,
) -> None:
    """Create comparison plot matching the example image format"""

    n_vars = len(variables)
    plt.figure(figsize=(15, 3 * n_vars))

    for i, var in enumerate(variables):
        ds, ref_ds = align_to_valid_time(
            validation_ds,
            reference_ds,
            init_time,
            lead_time,
        )

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

        # Dataset titles with timestamps
        ds_init_time = pd.Timestamp(ds.init_time.item()).strftime("%Y-%m-%dT%H:%M")
        ds_lead_time = (
            f"{(pd.Timedelta(ds.lead_time.item()).total_seconds()) / 3600:g}h"
        )
        ds_time = f"{ds_init_time}+{ds_lead_time}"
        ref_time = pd.Timestamp(ref_ds.time.item()).strftime("%Y-%m-%dT%H:%M")

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
                ref_data.longitude,
                ref_data.latitude,
                ref_data.values,
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(im1, ax=ax1)

            # Use combined bounds from both datasets for consistent axis ranges
            lon_min = min(float(ref_data.longitude.min()), float(data.longitude.min()))
            lon_max = max(float(ref_data.longitude.max()), float(data.longitude.max()))
            lat_min = min(float(ref_data.latitude.min()), float(data.latitude.min()))
            lat_max = max(float(ref_data.latitude.max()), float(data.latitude.max()))
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

        if var_in_reference and not np.isnan(ref_data.values).all():
            # Include reference data in histogram if available
            ref_values_flat = ref_data.values.flat
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
    filename = "_".join(filename_parts) + ".png"
    filepath = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    log.info(f"Comparison plot saved to {filepath}")


def compare_vars(
    validation_url: str,
    reference_url: str | None = GEFS_ANALYSIS_URL,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
    init_time: str | None = None,
    lead_time: str | None = None,
) -> None:
    """Create comparison plots between two zarr datasets."""

    log.info(f"Loading validation dataset from: {validation_url}")
    validation_ds = xr.open_zarr(
        validation_url,
        chunks=None,
        decode_timedelta=True,
    )

    log.info(f"Loading reference dataset from: {reference_url}")
    reference_ds = xr.open_zarr(reference_url, chunks=None)

    validation_vars = list(validation_ds.data_vars.keys())
    log.info(f"Found {len(validation_vars)} variables in validation dataset")

    if variables:
        # Use specified variables that exist in validation dataset
        selected_vars = [var for var in variables if var in validation_ds.data_vars]
        missing_vars = set(variables) - set(validation_ds.data_vars.keys())

        if missing_vars:
            log.warning(f"Variables not found in validation dataset: {missing_vars}")

        if not selected_vars:
            typer.echo("Error: No valid variables specified", err=True)
            raise typer.Exit(1)
    else:
        selected_vars = validation_vars

    log.info(f"Plotting variables: {selected_vars}")

    validation_ds, ensemble_member = select_random_enseble_member(validation_ds)
    spatially_aligned_reference_ds = align_reference_spatially(
        validation_ds, reference_ds
    )

    create_comparison_plot(
        validation_ds,
        spatially_aligned_reference_ds,
        selected_vars,
        ensemble_member,
        init_time,
        lead_time,
    )

    if show_plot:
        plt.show()
