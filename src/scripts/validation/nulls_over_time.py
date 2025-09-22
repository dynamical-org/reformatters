import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr
import zarr

from reformatters.common.logging import get_logger
from scripts.validation.utils import OUTPUT_DIR, variables_option

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 128})


def get_spatial_points_and_select(ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Get two random spatial points: one in <=25% quartile, one in >=75% quartile."""

    # Determine spatial dimension names
    if "latitude" in ds.dims and "longitude" in ds.dims:
        dim1, dim2 = "latitude", "longitude"
    else:
        dim1, dim2 = "y", "x"

    size1, size2 = ds.sizes[dim1], ds.sizes[dim2]

    # Random indices in each quartile
    idx1_p1 = np.random.randint(0, size1 // 4)
    idx2_p1 = np.random.randint(0, size2 // 4)
    idx1_p2 = np.random.randint(3 * size1 // 4, size1)
    idx2_p2 = np.random.randint(3 * size2 // 4, size2)

    ds_p1 = ds.isel({dim1: idx1_p1, dim2: idx2_p1})
    ds_p2 = ds.isel({dim1: idx1_p2, dim2: idx2_p2})

    return ds_p1, ds_p2


def plot_nulls(
    dataset_url: str,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
) -> None:
    """Analyze null values at two spatial points across time dimensions."""

    ds = xr.open_zarr(dataset_url, chunks=None, decode_timedelta=True)
    ds = ds.sel(init_time=slice("2025-09-01T00", "2025-09-22T00"))

    if variables:
        selected_vars = [var for var in variables if var in ds.data_vars]
        if not selected_vars:
            typer.echo("Error: No valid variables specified", err=True)
            raise typer.Exit(1)
    else:
        selected_vars = list(ds.data_vars.keys())

    # Get spatial points and select data
    ds_p1, ds_p2 = get_spatial_points_and_select(ds)

    log.info(
        f"Point 1: lat={float(ds_p1.latitude):.2f}, lon={float(ds_p1.longitude):.2f}"
    )
    log.info(
        f"Point 2: lat={float(ds_p2.latitude):.2f}, lon={float(ds_p2.longitude):.2f}"
    )

    # Create plots
    fig, axes = plt.subplots(
        len(selected_vars), 2, figsize=(12, 4 * len(selected_vars)), squeeze=False
    )

    for i, var in enumerate(selected_vars):
        log.info(f"Plotting {var}")

        # Calculate null data for both points (for plotting)
        non_time_dims = [
            dim for dim in ds_p1[var].dims if dim not in ["time", "init_time"]
        ]

        # Load null masks once for both points
        null_mask_p1 = ds_p1[var].isnull()
        null_mask_p2 = ds_p2[var].isnull()

        # Calculate null fractions for plotting
        null_p1 = null_mask_p1.mean(dim=non_time_dims)
        null_p2 = null_mask_p2.mean(dim=non_time_dims)

        # Find time dimension for x-axis
        time_dim = next(dim for dim in ["time", "init_time"] if dim in null_p1.dims)

        # Plot both points
        for j, (null_data, null_mask, ds_point, point_name, color) in enumerate(
            [
                (
                    null_p1,
                    null_mask_p1,
                    ds_p1,
                    f"Point 1 (lat={float(ds_p1.latitude):.2f}, lon={float(ds_p1.longitude):.2f})",
                    "blue",
                ),
                (
                    null_p2,
                    null_mask_p2,
                    ds_p2,
                    f"Point 2 (lat={float(ds_p2.latitude):.2f}, lon={float(ds_p2.longitude):.2f})",
                    "orange",
                ),
            ]
        ):
            ax = axes[i, j]

            ax.plot(
                null_data[time_dim],
                null_data,
                marker="o",
                linestyle="-",
                markersize=3,
                color=color,
            )
            ax.set_xlabel(time_dim.replace("_", " ").title())
            ax.set_title(f"{var} - {point_name}")
            ax.set_ylabel("Null Fraction")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # For logging, filter lead_time if needed
            check_mask = null_mask
            if (
                ds_point[var].attrs.get("step_type") != "instant"
                and "lead_time" in ds_point[var].dims
            ):
                # For accumulated variables, exclude the first lead_time (analysis)
                check_mask = null_mask.isel(lead_time=slice(1, None))

            # Check if there are any nulls in the filtered data
            if check_mask.any():
                # Calculate null fraction for the filtered data to get specific coordinates
                log_null_data = check_mask.mean(dim=non_time_dims)

                # Report all times with any missing values
                missing_times = log_null_data[time_dim].where(
                    log_null_data > 0, drop=True
                )
                if len(missing_times) > 0:
                    log.info(
                        f"{point_name} - {var} missing at: {list(missing_times.values)}"
                    )
            else:
                log.info(f"{point_name} - {var}: No missing values found")

    plt.tight_layout()

    # Save plot
    filename = f"{ds.attrs.get('dataset_id', 'unknown')}_null_analysis.png"
    filepath = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    log.info(f"Saved to {filepath}")

    if show_plot:
        plt.show()
