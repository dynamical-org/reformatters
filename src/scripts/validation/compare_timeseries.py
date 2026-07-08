from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from matplotlib.axes import Axes

from reformatters.common.logging import get_logger
from scripts.validation.utils import (
    COMBINED_PLOT_MAX_VARS,
    RunContext,
    VariableStats,
    end_date_option,
    get_two_random_points,
    is_forecast_dataset,
    is_virtual_store,
    level_label,
    level_option,
    load_retried,
    load_zarr_dataset,
    output_dir_option,
    reference_url_option,
    resolve_output_dir,
    resolve_reference_url,
    scope_time_period,
    select_random_ensemble_member,
    select_var_level,
    select_variables_for_plotting,
    start_date_option,
    var_slug,
    variables_option,
)

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 32})


def select_time_period_for_comparison(
    validation_ds: xr.Dataset, reference_ds: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset, str, str, str]:
    """Select appropriate time periods for validation and reference datasets."""
    rng = np.random.default_rng()
    if is_forecast_dataset(validation_ds):
        selected_init_time = pd.Timestamp(rng.choice(validation_ds.init_time, 1)[0])
        validation_subset = validation_ds.sel(init_time=selected_init_time)

        valid_time_start = validation_subset.valid_time.min().item()
        valid_time_end = validation_subset.valid_time.max().item()
        reference_subset = reference_ds.sel(
            time=slice(pd.Timestamp(valid_time_start), pd.Timestamp(valid_time_end))
        )
        title_suffix = (
            f"Forecast init_time: {selected_init_time.strftime('%Y-%m-%dT%H:%M')}"
        )
        return validation_subset, reference_subset, title_suffix, "valid_time", "time"

    time_start = pd.Timestamp(validation_ds.time.min().item())
    time_end = pd.Timestamp(validation_ds.time.max().item())
    ten_days = pd.Timedelta(days=10)

    if time_end - time_start < ten_days:
        selected_start = time_start
        selected_end = time_end
    else:
        latest_start = time_end - ten_days
        time_range_seconds = (latest_start - time_start).total_seconds()
        random_offset = rng.integers(0, int(time_range_seconds) + 1)
        selected_start = time_start + pd.Timedelta(seconds=random_offset)
        selected_end = selected_start + ten_days

    validation_subset = validation_ds.sel(time=slice(selected_start, selected_end))
    reference_subset = reference_ds.sel(time=slice(selected_start, selected_end))
    title_suffix = (
        f"Analysis period: {selected_start.strftime('%Y-%m-%dT%H:%M')} - "
        f"{selected_end.strftime('%Y-%m-%dT%H:%M')}"
    )
    return validation_subset, reference_subset, title_suffix, "time", "time"


def _compute_point_stats(
    da: xr.DataArray,
) -> tuple[float, float, float] | tuple[None, None, None]:
    vals = da.values
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None, None, None
    return float(np.min(vals)), float(np.max(vals)), float(np.mean(vals))


def _draw_timeseries_at_point(
    ax: Axes,
    val_series: xr.DataArray,
    ref_series: xr.DataArray | None,
    time_coord: str,
    ref_time_coord: str,
    val_label: str,
    ref_label: str,
    units: str,
    title: str,
) -> None:
    ax.plot(
        val_series[time_coord],
        val_series,
        marker="o",
        linestyle="-",
        markersize=3,
        color="red",
        label=val_label,
        alpha=0.8,
    )
    if ref_series is not None:
        ax.plot(
            ref_series[ref_time_coord],
            ref_series,
            marker="s",
            linestyle="-",
            markersize=3,
            color="blue",
            label=ref_label,
            alpha=0.8,
        )
    ax.set_xlabel("Valid Time" if time_coord == "valid_time" else "Time")
    ax.set_ylabel(units or "")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.relim()
    ax.autoscale_view()


def _load_timeseries_for_var(
    var: str,
    ctx: RunContext,
    validation_subset: xr.Dataset,
    reference_subset: xr.Dataset,
    level_sel: dict[str, object],
) -> tuple[xr.DataArray, xr.DataArray | None, xr.DataArray, xr.DataArray | None]:
    val = validation_subset[var]
    if level_sel:
        val = val.sel(level_sel)
    val_p1 = load_retried(val.isel(ctx.point1_sel))
    val_p2 = load_retried(val.isel(ctx.point2_sel))
    ref_p1: xr.DataArray | None = None
    ref_p2: xr.DataArray | None = None
    if var in reference_subset.data_vars and (
        not level_sel or next(iter(level_sel)) in reference_subset[var].dims
    ):
        assert "latitude" in reference_subset.dims
        assert "longitude" in reference_subset.dims
        ref = reference_subset[var]
        if level_sel:
            ref = ref.sel(level_sel, method="nearest")
        ref_p1 = load_retried(
            ref.sel(latitude=ctx.point1_lat, longitude=ctx.point1_lon, method="nearest")
        )
        ref_p2 = load_retried(
            ref.sel(latitude=ctx.point2_lat, longitude=ctx.point2_lon, method="nearest")
        )
    return val_p1, ref_p1, val_p2, ref_p2


def _store_temporal_stats(
    stats: VariableStats,
    val_p1: xr.DataArray,
    val_p2: xr.DataArray,
    ref_p1: xr.DataArray | None,
    ref_p2: xr.DataArray | None,
) -> None:
    (
        stats.val_temporal_min_p1,
        stats.val_temporal_max_p1,
        stats.val_temporal_mean_p1,
    ) = _compute_point_stats(val_p1)
    (
        stats.val_temporal_min_p2,
        stats.val_temporal_max_p2,
        stats.val_temporal_mean_p2,
    ) = _compute_point_stats(val_p2)
    if ref_p1 is not None:
        stats.ref_available_temporal = True
        (
            stats.ref_temporal_min_p1,
            stats.ref_temporal_max_p1,
            stats.ref_temporal_mean_p1,
        ) = _compute_point_stats(ref_p1)
    if ref_p2 is not None:
        stats.ref_available_temporal = True
        (
            stats.ref_temporal_min_p2,
            stats.ref_temporal_max_p2,
            stats.ref_temporal_mean_p2,
        ) = _compute_point_stats(ref_p2)


def _fmt(v: float | None) -> str:
    return f"{v:.3g}" if v is not None else "n/a"


def run_compare_timeseries(ctx: RunContext) -> None:
    """Produce per-variable + combined timeseries comparison plots in ctx.output_dir."""
    assert ctx.reference_ds is not None, (
        "compare-timeseries requires a reference dataset"
    )

    # Temporal plots are over a single member; ctx.validation_ds keeps the full
    # ensemble dim so the availability scan can see every member.
    validation_ds = ctx.validation_ds
    if "ensemble_member" in validation_ds.dims:
        if ctx.ensemble_member is None:
            validation_ds, ctx.ensemble_member = select_random_ensemble_member(
                validation_ds
            )
            log.info(f"Ensemble member (random): {ctx.ensemble_member}")
        else:
            validation_ds = validation_ds.sel(ensemble_member=ctx.ensemble_member)

    (
        validation_subset,
        reference_subset,
        title_suffix,
        time_coord,
        ref_time_coord,
    ) = select_time_period_for_comparison(validation_ds, ctx.reference_ds)

    val_label = validation_ds.attrs.get("name", "validation")
    ref_label = ctx.reference_ds.attrs.get("name", "reference")
    ctx.temporal_period_label = title_suffix

    n_vars = len(ctx.variables)
    log.info(f"compare-timeseries: {n_vars} variables — {title_suffix}")

    if n_vars <= COMBINED_PLOT_MAX_VARS:
        fig_c, axes_c = plt.subplots(
            n_vars,
            2,
            figsize=(14, 3.0 * n_vars),
            squeeze=False,
            constrained_layout=True,
        )
    else:
        fig_c = axes_c = None
        log.info(
            f"compare-timeseries: {n_vars} variables exceeds combined-plot cap "
            f"{COMBINED_PLOT_MAX_VARS}; skipping combined_temporal.png (per-variable "
            "PNGs unaffected)"
        )

    p1_title_suffix = f"(lat={ctx.point1_lat:.2f}, lon={ctx.point1_lon:.2f})"
    p2_title_suffix = f"(lat={ctx.point2_lat:.2f}, lon={ctx.point2_lon:.2f})"

    for i, var in enumerate(ctx.variables):
        stats = ctx.stats_for(var)
        level_sel = select_var_level(ctx, var, stats)
        level_note = level_label(stats)
        val_p1, ref_p1, val_p2, ref_p2 = _load_timeseries_for_var(
            var, ctx, validation_subset, reference_subset, level_sel
        )
        _store_temporal_stats(stats, val_p1, val_p2, ref_p1, ref_p2)
        units = stats.units or ""

        # Per-variable figure
        fig_v, axes_v = plt.subplots(
            1, 2, figsize=(14, 3.375), squeeze=False, constrained_layout=True
        )
        _draw_timeseries_at_point(
            axes_v[0, 0],
            val_p1,
            ref_p1,
            time_coord,
            ref_time_coord,
            val_label,
            ref_label,
            units,
            f"{var}{level_note} — {p1_title_suffix}",
        )
        _draw_timeseries_at_point(
            axes_v[0, 1],
            val_p2,
            ref_p2,
            time_coord,
            ref_time_coord,
            val_label,
            ref_label,
            units,
            f"{var}{level_note} — {p2_title_suffix}",
        )
        fig_v.suptitle(
            f"{var}{level_note}\n{val_label} vs {ref_label}\n{title_suffix}",
            fontsize=11,
        )
        out_path = ctx.output_dir / f"temporal_{var_slug(var)}.png"
        fig_v.savefig(out_path, dpi=80, bbox_inches="tight")
        plt.close(fig_v)
        stats.temporal_plot = out_path.name

        # Combined row
        if axes_c is not None:
            _draw_timeseries_at_point(
                axes_c[i, 0],
                val_p1,
                ref_p1,
                time_coord,
                ref_time_coord,
                val_label,
                ref_label,
                units,
                f"{var}{level_note} — {p1_title_suffix}",
            )
            _draw_timeseries_at_point(
                axes_c[i, 1],
                val_p2,
                ref_p2,
                time_coord,
                ref_time_coord,
                val_label,
                ref_label,
                units,
                f"{var}{level_note} — {p2_title_suffix}",
            )

        log.info(
            f"  temporal {var}: "
            f"P1 val=[{_fmt(stats.val_temporal_min_p1)}, {_fmt(stats.val_temporal_max_p1)}] "
            f"P2 val=[{_fmt(stats.val_temporal_min_p2)}, {_fmt(stats.val_temporal_max_p2)}]"
        )

    if fig_c is not None:
        fig_c.suptitle(
            f"Timeseries comparison — all variables\n{val_label} vs {ref_label}\n{title_suffix}",
            fontsize=13,
        )
        combined_path = ctx.output_dir / "combined_temporal.png"
        fig_c.savefig(combined_path, dpi=120, bbox_inches="tight")
        plt.close(fig_c)
        ctx.combined_temporal_plot = combined_path.name


def compare_timeseries(
    validation_url: str,
    reference_url: str | None = reference_url_option,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
    level: float | None = level_option,
    output_dir: Path | None = output_dir_option,
) -> None:
    """Create per-variable + combined timeseries comparison plots."""
    validation_ds = load_zarr_dataset(validation_url)
    if start_date or end_date:
        validation_ds = scope_time_period(validation_ds, start_date, end_date)
    reference_url = resolve_reference_url(reference_url)
    reference_ds = load_zarr_dataset(reference_url)

    selected_vars = select_variables_for_plotting(validation_ds, variables)
    point1_sel, point2_sel, (lat1, lon1), (lat2, lon2) = get_two_random_points(
        validation_ds
    )

    out = resolve_output_dir(validation_url, output_dir)
    log.info(f"output dir: {out}")

    ctx = RunContext(
        output_dir=out,
        validation_url=validation_url,
        reference_url=reference_url,
        validation_ds=validation_ds,
        reference_ds=reference_ds,
        started_at=pd.Timestamp.now(tz="UTC"),
        point1_sel=point1_sel,
        point2_sel=point2_sel,
        point1_lat=lat1,
        point1_lon=lon1,
        point2_lat=lat2,
        point2_lon=lon2,
        ensemble_member=None,
        variables=selected_vars,
        is_virtual=is_virtual_store(validation_url),
        level_override=level,
    )
    run_compare_timeseries(ctx)

    if show_plot:
        plt.show()
