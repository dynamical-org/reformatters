from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import zarr
from matplotlib.axes import Axes

from reformatters.common.logging import get_logger
from scripts.validation.utils import (
    RunContext,
    end_date_option,
    get_two_random_points,
    load_zarr_dataset,
    output_dir_option,
    resolve_output_dir,
    scope_time_period,
    select_variables_for_plotting,
    start_date_option,
    variables_option,
)

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 32})


def _compute_nulls_for_point(
    ds_point: xr.Dataset, var: str
) -> tuple[xr.DataArray, list[str], int, int]:
    """Compute null fraction over time, plus (filtered) missing timestamps + counts.

    For accumulated/avg variables, excludes the first lead_time (analysis step) from the
    "unexpected nulls" tally — it's structurally NaN by design.
    """
    non_time_dims = [
        dim for dim in ds_point[var].dims if dim not in ("time", "init_time")
    ]
    null_mask = ds_point[var].isnull()
    null_frac = null_mask.mean(dim=non_time_dims)

    check_mask = null_mask
    if (
        ds_point[var].attrs.get("step_type") != "instant"
        and "lead_time" in ds_point[var].dims
    ):
        check_mask = null_mask.isel(lead_time=slice(1, None))

    if check_mask.any():
        time_dim = next(d for d in ("time", "init_time") if d in null_frac.dims)
        log_null = check_mask.mean(dim=non_time_dims)
        missing = log_null[time_dim].where(log_null > 0, drop=True)
        missing_strs = missing.dt.strftime("%Y-%m-%dT%H:%M:%S").values.tolist()
    else:
        missing_strs = []

    return (
        null_frac,
        missing_strs,
        int(check_mask.sum().item()),
        int(check_mask.size),
    )


def _draw_null_trace(ax: Axes, null_data: xr.DataArray, color: str, title: str) -> None:
    time_dim = next(d for d in ("time", "init_time") if d in null_data.dims)
    ax.plot(
        null_data[time_dim],
        null_data,
        marker="o",
        linestyle="-",
        markersize=3,
        color=color,
    )
    ax.set_xlabel(time_dim.replace("_", " ").title())
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Null Fraction")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def _format_missing_summary(missing: list[str]) -> str:
    if not missing:
        return "none"
    if len(missing) <= 6:
        return f"{len(missing)} ({', '.join(missing)})"
    head = ", ".join(missing[:3])
    tail = ", ".join(missing[-3:])
    return f"{len(missing)} (first: {head} … last: {tail})"


def write_missing_timestamps_file(output_dir: Path, ctx: RunContext) -> str | None:
    """Write missing_timestamps.txt aggregating every (var, point) with nulls. Returns filename or None."""
    entries = []
    for var, stats in ctx.stats.items():
        for point_label, missing in (
            (
                f"Point 1 (lat={ctx.point1_lat:.2f}, lon={ctx.point1_lon:.2f})",
                stats.missing_timestamps_p1,
            ),
            (
                f"Point 2 (lat={ctx.point2_lat:.2f}, lon={ctx.point2_lon:.2f})",
                stats.missing_timestamps_p2,
            ),
        ):
            if missing:
                entries.append((var, point_label, missing))

    if not entries:
        return None

    filename = "missing_timestamps.txt"
    path = output_dir / filename
    total = sum(len(m) for _, _, m in entries)
    lines = [
        "# Missing timestamps",
        f"# Total: {total} across {len(entries)} (variable, point) combinations.",
        "# Use --filter-contains <timestamp> to retry those source files with backfill.",
        "",
    ]
    for var, point_label, missing in entries:
        lines.append(f"## {var} @ {point_label}")
        lines.append(f"count: {len(missing)}")
        lines.extend(missing)
        lines.append("")
        lines.append(
            "retry-filter: " + " ".join(f"--filter-contains {m}" for m in missing)
        )
        lines.append("")
    path.write_text("\n".join(lines))
    return filename


def run_report_nulls(ctx: RunContext) -> None:
    """Produce per-variable null plots + a combined plot in ctx.output_dir and update ctx.stats."""
    ds_p1 = ctx.validation_ds.isel(ctx.point1_sel)
    ds_p2 = ctx.validation_ds.isel(ctx.point2_sel)

    n_vars = len(ctx.variables)
    p1_label = f"Point 1 (lat={ctx.point1_lat:.2f}, lon={ctx.point1_lon:.2f})"
    p2_label = f"Point 2 (lat={ctx.point2_lat:.2f}, lon={ctx.point2_lon:.2f})"

    log.info(f"report-nulls: {n_vars} variables at {p1_label} / {p2_label}")

    fig_c, axes_c = plt.subplots(n_vars, 2, figsize=(12, 2.625 * n_vars), squeeze=False)

    for i, var in enumerate(ctx.variables):
        stats = ctx.stats_for(var)

        null_p1, missing_p1, n_p1, total_p1 = _compute_nulls_for_point(ds_p1, var)
        null_p2, missing_p2, n_p2, total_p2 = _compute_nulls_for_point(ds_p2, var)

        stats.missing_timestamps_p1 = missing_p1
        stats.missing_timestamps_p2 = missing_p2
        stats.null_count_p1 = n_p1
        stats.null_count_p2 = n_p2
        stats.total_count_p1 = total_p1
        stats.total_count_p2 = total_p2

        # Per-variable figure.
        fig_v, axes_v = plt.subplots(1, 2, figsize=(12, 3), squeeze=False)
        _draw_null_trace(axes_v[0, 0], null_p1, "blue", p1_label)
        _draw_null_trace(axes_v[0, 1], null_p2, "orange", p2_label)
        fig_v.suptitle(var, fontsize=11)
        fig_v.tight_layout()
        out_path = ctx.output_dir / f"nulls_{var}.png"
        fig_v.savefig(out_path, dpi=80, bbox_inches="tight")
        plt.close(fig_v)
        stats.null_plot = out_path.name

        # Combined figure row.
        _draw_null_trace(axes_c[i, 0], null_p1, "blue", f"{var} — {p1_label}")
        _draw_null_trace(axes_c[i, 1], null_p2, "orange", f"{var} — {p2_label}")

        p1_fmt = _format_missing_summary(missing_p1)
        p2_fmt = _format_missing_summary(missing_p2)
        log.info(f"  nulls {var}: P1 missing={p1_fmt} | P2 missing={p2_fmt}")

    fig_c.suptitle("Null analysis — all variables", fontsize=13)
    fig_c.tight_layout()
    combined_path = ctx.output_dir / "combined_nulls.png"
    fig_c.savefig(combined_path, dpi=120, bbox_inches="tight")
    plt.close(fig_c)
    ctx.combined_nulls_plot = combined_path.name

    missing_file = write_missing_timestamps_file(ctx.output_dir, ctx)
    ctx.missing_timestamps_file = missing_file
    if missing_file:
        log.info(f"  wrote missing timestamp list -> {missing_file}")


def report_nulls(
    dataset_url: str,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
    output_dir: Path | None = output_dir_option,
) -> None:
    """Analyze null values for each data variable at two spatial points across time."""
    ds = load_zarr_dataset(dataset_url)
    if start_date or end_date:
        ds = scope_time_period(ds, start_date, end_date)

    selected_vars = select_variables_for_plotting(ds, variables)
    point1_sel, point2_sel, (lat1, lon1), (lat2, lon2) = get_two_random_points(ds)

    out = resolve_output_dir(dataset_url, output_dir)
    log.info(f"output dir: {out}")

    ctx = RunContext(
        output_dir=out,
        validation_url=dataset_url,
        reference_url=None,
        validation_ds=ds,
        reference_ds=None,
        started_at=pd.Timestamp.now(tz="UTC"),
        point1_sel=point1_sel,
        point2_sel=point2_sel,
        point1_lat=lat1,
        point1_lon=lon1,
        point2_lat=lat2,
        point2_lon=lon2,
        ensemble_member=None,
        variables=selected_vars,
    )
    run_report_nulls(ctx)

    if show_plot:
        plt.show()
