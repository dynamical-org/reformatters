from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from matplotlib.axes import Axes

from reformatters.common.logging import get_logger
from reformatters.common.time_utils import whole_hours
from scripts.validation.utils import (
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
    rng = np.random.default_rng()

    if init_time is None:
        # Default to a recent-but-settled position (10 steps back) so the snapshot has
        # complete forecasts and all mid-archive-introduced variables present, instead
        # of a random init that may land in an early era with many not-yet-introduced
        # (all-NaN) variables. Tiny archives (< 10 inits) fall back to the first init.
        idx = max(-10, -ds.sizes["init_time"])
        selected_init_time = pd.Timestamp(ds.init_time.isel(init_time=idx).item())
    else:
        selected_init_time = pd.Timestamp(init_time)
    if lead_time is None:
        # Prefer leads whose valid time the reference covers; a lead past the
        # reference's last analysis compares against a snapshot hours older.
        candidate_leads = ds.lead_time.values
        ref_max = pd.Timestamp(reference_ds.time.max().item())
        covered = [
            lead
            for lead in candidate_leads
            if selected_init_time + pd.Timedelta(lead) <= ref_max
        ]
        selected_lead_time = rng.choice(covered or candidate_leads, 1)[0]
    else:
        selected_lead_time = lead_time

    ds = ds.sel(init_time=selected_init_time, lead_time=selected_lead_time)
    valid_time = pd.Timestamp(ds.valid_time.item())
    reference_ds = reference_ds.sel(time=valid_time, method="nearest")
    return ds, reference_ds


def align_to_valid_time_analysis(
    ds: xr.Dataset, reference_ds: xr.Dataset, time: str | None
) -> tuple[xr.Dataset, xr.Dataset]:
    if time is None:
        # Default to a recent-but-settled position (10 steps back); see
        # align_to_valid_time_forecast. Tiny archives fall back to the first time.
        idx = max(-10, -ds.sizes["time"])
        selected_time = pd.Timestamp(ds.time.isel(time=idx).item())
    else:
        selected_time = pd.Timestamp(time)
    ds = ds.sel(time=selected_time)
    reference_ds = reference_ds.sel(time=selected_time, method="nearest")
    return ds, reference_ds


def _downsample_for_plot(ds: xr.Dataset, max_plot_dim: int = 1000) -> xr.Dataset:
    strides: dict[str, int] = {}
    for dim in ("latitude", "longitude", "y", "x"):
        if dim in ds.dims and ds.sizes[dim] > max_plot_dim:
            strides[dim] = ds.sizes[dim] // max_plot_dim
    if strides:
        ds = ds.isel(
            {dim: slice(None, None, stride) for dim, stride in strides.items()}
        )
    return ds


def _format_spatial_time_label(ds: xr.Dataset, is_forecast: bool) -> str:
    if is_forecast:
        init_str = pd.Timestamp(ds.init_time.item()).strftime("%Y-%m-%dT%H:%M")
        lead_hours = whole_hours(pd.Timedelta(ds.lead_time.item()))
        return f"init={init_str}, lead={lead_hours:g}h"
    return pd.Timestamp(ds.time.item()).strftime("%Y-%m-%dT%H:%M")


def _compute_spatial_stats(
    data: xr.DataArray, ref_data: xr.DataArray | None, stats: VariableStats
) -> tuple[np.ndarray, np.ndarray]:
    """Update stats and return (data_clean, ref_clean) flattened arrays."""
    data_clean = data.values.flat[~np.isnan(data.values.flat)]
    if data_clean.size == 0:
        stats.val_spatial_min = stats.val_spatial_max = stats.val_spatial_mean = float(
            "nan"
        )
    else:
        stats.val_spatial_min = float(np.min(data_clean))
        stats.val_spatial_max = float(np.max(data_clean))
        stats.val_spatial_mean = float(np.mean(data_clean))

    if ref_data is not None:
        ref_clean = ref_data.values.flat[~np.isnan(ref_data.values.flat)]
        stats.ref_available_spatial = True
        if ref_clean.size:
            stats.ref_spatial_min = float(np.min(ref_clean))
            stats.ref_spatial_max = float(np.max(ref_clean))
            stats.ref_spatial_mean = float(np.mean(ref_clean))
    else:
        ref_clean = np.array([])
        stats.ref_available_spatial = False

    return data_clean, ref_clean


def _draw_spatial_triplet(
    ax_ref: Axes,
    ax_val: Axes,
    ax_hist: Axes,
    var: str,
    data: xr.DataArray,
    ref_data: xr.DataArray | None,
    data_clean: np.ndarray,
    ref_clean: np.ndarray,
    units: str | None,
    ds_title: str,
    ref_title: str,
) -> None:
    """Draw reference map, validation map, and histogram onto provided axes."""
    if ref_data is not None:
        vmin = min(float(data.min()), float(ref_data.min()))
        vmax = max(float(data.max()), float(ref_data.max()))
    else:
        vmin = float(data.min()) if data_clean.size else 0.0
        vmax = float(data.max()) if data_clean.size else 1.0

    # Reference map
    if ref_data is not None:
        im1 = ax_ref.pcolormesh(
            ref_data.longitude, ref_data.latitude, ref_data.values, vmin=vmin, vmax=vmax
        )
        plt.colorbar(im1, ax=ax_ref, label=units or "")
        lon_min = min(float(ref_data.longitude.min()), float(data.longitude.min()))
        lon_max = max(float(ref_data.longitude.max()), float(data.longitude.max()))
        lat_min = min(float(ref_data.latitude.min()), float(data.latitude.min()))
        lat_max = max(float(ref_data.latitude.max()), float(data.latitude.max()))
    else:
        ax_ref.text(
            0.5,
            0.5,
            "Variable not\navailable in\nreference dataset",
            ha="center",
            va="center",
            transform=ax_ref.transAxes,
            fontsize=12,
            color="gray",
        )
        lon_min = float(data.longitude.min())
        lon_max = float(data.longitude.max())
        lat_min = float(data.latitude.min())
        lat_max = float(data.latitude.max())
    ax_ref.set_title(ref_title, fontsize=10)
    ax_ref.set_aspect("auto")
    ax_ref.set_xlabel("Longitude")
    ax_ref.set_ylabel("Latitude")

    # Validation map
    im2 = ax_val.pcolormesh(
        data.longitude, data.latitude, data.values, vmin=vmin, vmax=vmax
    )
    plt.colorbar(im2, ax=ax_val, label=units or "")
    ax_val.set_title(ds_title, fontsize=10)
    ax_val.set_aspect("auto")
    ax_val.set_xlabel("Longitude")
    ax_val.set_ylabel("Latitude")
    ax_val.set_xlim(lon_min, lon_max)
    ax_val.set_ylim(lat_min, lat_max)

    # Histogram
    if data_clean.size == 0:
        ax_hist.text(
            0.5,
            0.5,
            "All validation data is\nNaN at this step",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
            fontsize=12,
            color="gray",
        )
        return

    if ref_data is not None and ref_clean.size:
        data_min = min(np.min(data_clean), np.min(ref_clean))
        data_max = max(np.max(data_clean), np.max(ref_clean))
        ax_hist.hist(
            ref_clean,
            bins=40,
            alpha=0.7,
            label=ref_title.splitlines()[0] if ref_title else "reference",
            color="blue",
            range=(data_min, data_max),
            density=True,
            stacked=True,
        )
    else:
        data_min = float(np.min(data_clean))
        data_max = float(np.max(data_clean))

    ax_hist.hist(
        data_clean,
        bins=40,
        alpha=0.7,
        label=ds_title.splitlines()[0] if ds_title else "validation",
        color="red",
        range=(data_min, data_max),
        density=True,
        stacked=True,
    )
    if data_min != data_max:
        ax_hist.set_xlim(data_min, data_max)
    ax_hist.set_xlabel(units or "")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Distribution\n{var}", fontsize=10)
    ax_hist.legend(fontsize="small", frameon=False)
    ax_hist.grid(True, alpha=0.3)


def _reference_data(
    ref_ds: xr.Dataset, var: str, level_sel: dict[str, object]
) -> xr.DataArray | None:
    """Reference field for `var` at the sampled level, or None if not comparable."""
    if var not in ref_ds.data_vars:
        return None
    ref_data = ref_ds[var]
    if level_sel:
        dim = next(iter(level_sel))
        if dim not in ref_data.dims:
            return None  # reference lacks this level dim; nothing to compare against
        ref_data = ref_data.sel(level_sel, method="nearest")
    return load_retried(ref_data)


def run_compare_spatial(
    ctx: RunContext,
    init_time: str | None = None,
    lead_time: str | None = None,
    time: str | None = None,
) -> None:
    """Produce per-variable spatial comparison plots in ctx.output_dir."""
    assert ctx.reference_ds is not None, "compare-spatial requires a reference dataset"

    # Spatial plots are over a single member; ctx.validation_ds keeps the full
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

    is_forecast = is_forecast_dataset(validation_ds)
    spatially_aligned_ref = align_reference_spatially(validation_ds, ctx.reference_ds)

    if is_forecast:
        ds, ref_ds = align_to_valid_time_forecast(
            validation_ds, spatially_aligned_ref, init_time, lead_time
        )
    else:
        ds, ref_ds = align_to_valid_time_analysis(
            validation_ds, spatially_aligned_ref, time
        )
    ds = _downsample_for_plot(ds)

    val_time_label = _format_spatial_time_label(ds, is_forecast)
    ref_time_label = pd.Timestamp(ref_ds.time.item()).strftime("%Y-%m-%dT%H:%M")
    val_label = validation_ds.attrs.get("name", "validation")
    ref_label = ctx.reference_ds.attrs.get("name", "reference")
    ctx.spatial_time_label = val_time_label
    ctx.ref_spatial_time_label = ref_time_label

    n_vars = len(ctx.variables)
    log.info(
        f"compare-spatial: {n_vars} variables at {val_time_label} "
        f"(ref {ref_time_label})"
    )

    for var in ctx.variables:
        stats = ctx.stats_for(var)
        level_sel = select_var_level(ctx, var, stats)
        level_note = level_label(stats)

        data = ds[var]
        if level_sel:
            data = data.sel(level_sel)
        data = load_retried(data)
        ref_data = _reference_data(ref_ds, var, level_sel)
        data_clean, ref_clean = _compute_spatial_stats(data, ref_data, stats)

        stats.spatial_time_label = val_time_label
        stats.spatial_plot = f"spatial_{var_slug(var)}.png"

        ds_title = val_label
        if ctx.ensemble_member is not None:
            ds_title += f" (ensemble {ctx.ensemble_member})"
        ds_title += f"\n{var}{level_note} @ {val_time_label}"
        ref_title = (
            f"{ref_label}\n{var}{level_note} @ {ref_time_label}"
            if ref_data is not None
            else f"{ref_label}\n{var}{level_note} (not available)"
        )

        # Per-variable figure
        fig_v, axes_v = plt.subplots(1, 3, figsize=(15, 3.375), squeeze=False)
        _draw_spatial_triplet(
            axes_v[0, 0],
            axes_v[0, 1],
            axes_v[0, 2],
            var,
            data,
            ref_data,
            data_clean,
            ref_clean,
            stats.units,
            ds_title,
            ref_title,
        )
        fig_v.tight_layout()
        fig_v.savefig(ctx.output_dir / stats.spatial_plot, dpi=80, bbox_inches="tight")
        plt.close(fig_v)

        ref_note = (
            f"ref[{stats.ref_spatial_min:.3g}, {stats.ref_spatial_max:.3g}]"
            if stats.ref_available_spatial and stats.ref_spatial_min is not None
            else "ref=n/a"
        )
        log.info(
            f"  spatial {var}: val[{stats.val_spatial_min:.3g}, "
            f"{stats.val_spatial_max:.3g}] mean={stats.val_spatial_mean:.3g} {ref_note}"
        )


def compare_spatial(
    validation_url: str,
    reference_url: str | None = reference_url_option,
    variables: list[str] | None = variables_option,
    show_plot: bool = False,
    init_time: str | None = None,
    lead_time: str | None = None,
    time: str | None = None,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
    level: float | None = level_option,
    output_dir: Path | None = output_dir_option,
) -> None:
    """Create per-variable spatial comparison plots between two zarr datasets."""
    log.info(f"Loading validation dataset: {validation_url}")
    validation_ds = load_zarr_dataset(validation_url)
    if start_date or end_date:
        validation_ds = scope_time_period(validation_ds, start_date, end_date)

    is_forecast = is_forecast_dataset(validation_ds)
    log.info(f"Detected {'forecast' if is_forecast else 'analysis'} dataset")

    reference_url = resolve_reference_url(reference_url)
    log.info(f"Loading reference dataset: {reference_url}")
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
    run_compare_spatial(ctx, init_time=init_time, lead_time=lead_time, time=time)

    if show_plot:
        plt.show()
