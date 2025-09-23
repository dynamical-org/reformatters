import matplotlib.pyplot as plt
import zarr

from reformatters.common.logging import get_logger
from scripts.validation.utils import (
    OUTPUT_DIR,
    end_date_option,
    get_two_random_points,
    load_zarr_dataset,
    scope_time_period,
    select_variables_for_plotting,
    start_date_option,
    variables_option,
)

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 128})


def report_nulls(
    dataset_url: str,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
) -> None:
    """Analyze null values at two spatial points across time dimensions."""

    ds = load_zarr_dataset(dataset_url, decode_timedelta=True)
    if start_date or end_date:
        ds = scope_time_period(ds, start_date, end_date)

    selected_vars = select_variables_for_plotting(ds, variables)

    # Get two random spatial points (indices and coordinates)
    point1_sel, point2_sel, (lat1, lon1), (lat2, lon2) = get_two_random_points(ds)
    ds_p1 = ds.isel(point1_sel)
    ds_p2 = ds.isel(point2_sel)

    log.info(f"Point 1: lat={lat1:.2f}, lon={lon1:.2f}")
    log.info(f"Point 2: lat={lat2:.2f}, lon={lon2:.2f}")

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
                    f"Point 1 (lat={lat1:.2f}, lon={lon1:.2f})",
                    "blue",
                ),
                (
                    null_p2,
                    null_mask_p2,
                    ds_p2,
                    f"Point 2 (lat={lat2:.2f}, lon={lon2:.2f})",
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
