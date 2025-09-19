import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xarray as xr
import zarr

from reformatters.common.logging import get_logger

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 128})

GEFS_ANALYSIS_URL = "https://data.dynamical.org/noaa/gefs/analysis/latest.zarr"
OUTPUT_DIR = "data/output"

app = typer.Typer()

variables_option = typer.Option(
    None,
    "--variable",
    "-v",
    help="Variable to plot (can be used multiple times). "
    "If not provided, will plot all common variables.",
)


def align_reference_spatially(ds: xr.Dataset, reference_ds: xr.Dataset) -> xr.Dataset:
    return reference_ds.sel(
        latitude=slice(ds.latitude.max(), ds.latitude.min()),
        longitude=slice(ds.longitude.min(), ds.longitude.max()),
    )


def align_to_random_valid_time(
    ds: xr.Dataset, reference_ds: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    ds = ds.sel(
        init_time=np.random.choice(ds.init_time, 1),
        lead_time=np.random.choice(ds.lead_time, 1),
    )
    ds = ds.squeeze(["init_time", "lead_time"])

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


def get_common_variables(ds1: xr.Dataset, ds2: xr.Dataset) -> list[str]:
    """Get variables that exist in both datasets"""
    return [str(var) for var in ds1.data_vars if var in ds2.data_vars]


def create_comparison_plot(
    validation_ds: xr.Dataset,
    reference_ds: xr.Dataset,
    variables: list[str],
    ensemble_member: int | None = None,
) -> None:
    """Create comparison plot matching the example image format"""

    n_vars = len(variables)
    plt.figure(figsize=(15, 3 * n_vars))

    for i, var in enumerate(variables):
        ds, ref_ds = align_to_random_valid_time(validation_ds, reference_ds)

        # Get data arrays
        data = ds[var].load()
        ref_data = ref_ds[var].load()

        # Dataset titles with timestamps
        ds_time = pd.Timestamp(ds.valid_time.item()).strftime("%Y-%m-%dT%H:%M")
        ref_time = pd.Timestamp(ref_ds.time.item()).strftime("%Y-%m-%dT%H:%M")
        log.info(
            f"Plotting {var} for {ds.attrs['name']} ({ds_time}) and {ref_ds.attrs['name']} ({ref_time})"
        )

        ds_title = f"{ds.attrs['name']}"
        if ensemble_member is not None:
            ds_title += f" (Ensemble Member {ensemble_member})"
        ds_title += f" - {var}\n{ds_time}"
        ref_title = f"{ref_ds.attrs['name']} - {var}\n{ref_time}"

        # Determine shared color limits for consistent comparison
        vmin = min(float(data.min()), float(ref_data.min()))
        vmax = max(float(data.max()), float(ref_data.max()))

        # Left plot - ref dataset
        ax1 = plt.subplot(n_vars, 3, i * 3 + 1)
        im1 = ax1.pcolormesh(
            ref_data.longitude,
            ref_data.latitude,
            ref_data.values,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(ref_title)
        ax1.set_aspect("auto")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        lon_min, lon_max = (
            float(ref_data.longitude.min()),
            float(ref_data.longitude.max()),
        )
        lat_min, lat_max = (
            float(ref_data.latitude.min()),
            float(ref_data.latitude.max()),
        )

        # Create reasonable tick spacing within data bounds
        lon_start = np.ceil(lon_min / 50) * 50
        lon_end = np.floor(lon_max / 50) * 50
        lat_start = np.ceil(lat_min / 25) * 25
        lat_end = np.floor(lat_max / 25) * 25

        lon_ticks = np.arange(lon_start, lon_end + 1, 50)
        lat_ticks = np.arange(lat_start, lat_end + 1, 25)

        ax1.set_xticks(lon_ticks)
        ax1.set_yticks(lat_ticks)

        # Middle plot - validation dataset (was first dataset)
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
        ax2.set_xticks(lon_ticks)
        ax2.set_yticks(lat_ticks)

        # Right plot - histogram comparison
        ax3 = plt.subplot(n_vars, 3, i * 3 + 3)

        # Flatten arrays and remove NaN values
        data_flat = data.values.flatten()
        ref_flat = ref_data.values.flatten()
        data_clean = data_flat[~np.isnan(data_flat)]
        ref_clean = ref_flat[~np.isnan(ref_flat)]

        # Calculate combined range for consistent bins
        data_min, data_max = (
            min(np.min(data_clean), np.min(ref_clean)),
            max(np.max(data_clean), np.max(ref_clean)),
        )
        # Create histograms with consistent bins
        ax3.hist(
            ref_clean,
            bins="auto",
            alpha=0.7,
            label=ref_ds.attrs["name"],
            color="blue",
            range=(data_min, data_max),
        )
        ax3.hist(
            data_clean,
            bins="auto",
            alpha=0.7,
            label=ds.attrs["name"],
            color="red",
            range=(data_min, data_max),
        )

        # Set x-axis limits to match the data range
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


@app.command()
def compare(
    validation_url: str,
    reference_url: str | None = GEFS_ANALYSIS_URL,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
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

    common_variables = get_common_variables(validation_ds, reference_ds)
    log.info(f"Found {len(common_variables)} common variables")

    if variables:
        # Use specified variables that exist in both datasets
        selected_vars = list(set(variables) & set(common_variables))
        missing_vars = set(variables) - set(common_variables)

        if missing_vars:
            log.warning(f"Variables not found in both datasets: {missing_vars}")

        if not selected_vars:
            typer.echo("Error: No valid variables specified", err=True)
            raise typer.Exit(1)
    else:
        selected_vars = common_variables

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
    )

    if show_plot:
        plt.show()


@app.command()
def list_variables(
    dataset_url: str = typer.Argument(help="URL of the dataset to examine"),
) -> None:
    """List all variables in a zarr dataset."""

    log.info(f"Loading dataset from: {dataset_url}")
    ds = xr.open_zarr(dataset_url, chunks=None, decode_timedelta=True)

    variables = list(ds.data_vars.keys())
    typer.echo(f"Dataset contains {len(variables)} variables:")
    for var in sorted(variables):
        typer.echo(f"  {var}")


if __name__ == "__main__":
    app()
