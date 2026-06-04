from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from matplotlib.axes import Axes

from reformatters.common.logging import get_logger
from scripts.validation.utils import (
    RunContext,
    VariableStats,
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


def _compute_value_series(da_point: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Per-timestep mean and std of values at one point, reduced over non-time dims.

    For analysis datasets (a single value per timestep) the std is zero and the mean is
    just the value. For forecast/ensemble datasets the band spans lead_time/ensemble spread.
    """
    non_time_dims = [dim for dim in da_point.dims if dim not in ("time", "init_time")]
    mean_series = da_point.mean(dim=non_time_dims)
    std_series = da_point.std(dim=non_time_dims)
    return mean_series, std_series


def _draw_value_trace(
    ax: Axes,
    mean_series: xr.DataArray,
    std_series: xr.DataArray,
    color: str,
    std_color: str,
    title: str,
    units: str,
    has_std: bool,
) -> None:
    time_dim = next(d for d in ("time", "init_time") if d in mean_series.dims)
    times = mean_series[time_dim]
    mean = mean_series.values
    (mean_line,) = ax.plot(
        times, mean, marker="o", linestyle="-", markersize=2, color=color, label="mean"
    )
    ax.set_xlabel(time_dim.replace("_", " ").title())
    ax.set_ylabel(units or "")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    if has_std:
        # std on a secondary right axis, in a contrasting color drawn on top of the mean
        # at partial opacity so the mean still shows through where they overlap.
        ax_std = ax.twinx()
        (std_line,) = ax_std.plot(
            times,
            std_series.values,
            linestyle="-",
            color=std_color,
            alpha=0.4,
            label="std dev",
        )
        ax_std.set_ylabel(f"std dev {units}".strip())
        ax.legend(handles=[mean_line, std_line], fontsize=8, loc="upper right")


def _store_value_stats(
    stats: VariableStats,
    mean_series: xr.DataArray,
    std_series: xr.DataArray,
    point: int,
) -> None:
    mean = mean_series.values
    mean = mean[~np.isnan(mean)]
    std = std_series.values
    std = std[~np.isnan(std)]
    overall_mean = float(np.mean(mean)) if mean.size else None
    overall_std = float(np.mean(std)) if std.size else None
    overall_min = float(np.min(mean)) if mean.size else None
    overall_max = float(np.max(mean)) if mean.size else None
    if point == 1:
        stats.value_mean_p1 = overall_mean
        stats.value_std_p1 = overall_std
        stats.value_min_p1 = overall_min
        stats.value_max_p1 = overall_max
    else:
        stats.value_mean_p2 = overall_mean
        stats.value_std_p2 = overall_std
        stats.value_min_p2 = overall_min
        stats.value_max_p2 = overall_max


def _point_arrays(ctx: RunContext, var: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Reuse arrays loaded by run_report_nulls, else load them (standalone command)."""
    if var in ctx.loaded_point_data:
        return ctx.loaded_point_data[var]
    da_p1 = ctx.validation_ds.isel(ctx.point1_sel)[var].load()
    da_p2 = ctx.validation_ds.isel(ctx.point2_sel)[var].load()
    return da_p1, da_p2


def run_value_timeseries(ctx: RunContext) -> None:
    """Produce per-variable + combined full-period value time series plots in ctx.output_dir."""
    n_vars = len(ctx.variables)
    p1_label = f"Point 1 (lat={ctx.point1_lat:.2f}, lon={ctx.point1_lon:.2f})"
    p2_label = f"Point 2 (lat={ctx.point2_lat:.2f}, lon={ctx.point2_lon:.2f})"

    log.info(f"value-timeseries: {n_vars} variables at {p1_label} / {p2_label}")

    fig_c, axes_c = plt.subplots(n_vars, 2, figsize=(14, 2.625 * n_vars), squeeze=False)

    for i, var in enumerate(ctx.variables):
        stats = ctx.stats_for(var)
        units = stats.units or ""

        da_p1, da_p2 = _point_arrays(ctx, var)
        mean_p1, std_p1 = _compute_value_series(da_p1)
        mean_p2, std_p2 = _compute_value_series(da_p2)
        _store_value_stats(stats, mean_p1, std_p1, 1)
        _store_value_stats(stats, mean_p2, std_p2, 2)

        # std is only meaningful when there are non-time dims (lead_time / ensemble) to
        # reduce over — for analysis datasets there's a single value per timestep.
        has_std = any(d not in ("time", "init_time") for d in da_p1.dims)

        # Per-variable figure.
        fig_v, axes_v = plt.subplots(1, 2, figsize=(14, 3.375), squeeze=False)
        _draw_value_trace(
            axes_v[0, 0], mean_p1, std_p1, "blue", "fuchsia", p1_label, units, has_std
        )
        _draw_value_trace(
            axes_v[0, 1], mean_p2, std_p2, "orange", "red", p2_label, units, has_std
        )
        fig_v.suptitle(
            f"{var} — full-period mean ± std" if has_std else var, fontsize=11
        )
        fig_v.tight_layout()
        out_path = ctx.output_dir / f"value_timeseries_{var}.png"
        fig_v.savefig(out_path, dpi=80, bbox_inches="tight")
        plt.close(fig_v)
        stats.value_ts_plot = out_path.name

        # Combined figure row.
        _draw_value_trace(
            axes_c[i, 0],
            mean_p1,
            std_p1,
            "blue",
            "fuchsia",
            f"{var} — {p1_label}",
            units,
            has_std,
        )
        _draw_value_trace(
            axes_c[i, 1],
            mean_p2,
            std_p2,
            "orange",
            "red",
            f"{var} — {p2_label}",
            units,
            has_std,
        )

    fig_c.suptitle("Full-period value time series — all variables", fontsize=13)
    fig_c.tight_layout()
    combined_path = ctx.output_dir / "combined_value_timeseries.png"
    fig_c.savefig(combined_path, dpi=120, bbox_inches="tight")
    plt.close(fig_c)
    ctx.combined_value_timeseries_plot = combined_path.name


def value_timeseries(
    dataset_url: str,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
    output_dir: Path | None = output_dir_option,
) -> None:
    """Plot per-timestep mean ± std of each variable at two spatial points over all time."""
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
    run_value_timeseries(ctx)

    if show_plot:
        plt.show()
