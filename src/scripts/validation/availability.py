"""Per-variable availability over the append dim, rendered uniformly for both store types.

Availability is measured differently per store type — a materialized store is
value-scanned (`.isnull()`) at the two run points, while on a virtual store that would
decode every chunk it touches, so availability comes from manifest ref probes
(`manifest_scan.scan_manifest`), which are exhaustive per source file and per-variable
per position. Both paths produce an `AvailabilitySeries` per variable and render the
same artifacts: an all-variable heatmap and a trace plot for each incomplete variable.
See docs/validation.md.
"""

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xarray as xr
import zarr

from reformatters.common.logging import get_logger
from scripts.validation.utils import (
    AvailabilitySeries,
    RunContext,
    end_date_option,
    get_two_random_points,
    is_virtual_store,
    level_option,
    load_retried,
    load_zarr_dataset,
    open_icechunk_readonly,
    output_dir_option,
    resolve_output_dir,
    scope_time_period,
    select_var_level,
    select_variables_for_plotting,
    start_date_option,
    var_slug,
    variables_option,
)

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 32})

HEATMAP_FILENAME = "availability_heatmap.png"
MAX_HEATMAP_COLUMNS = 1200


@dataclass
class VarAvailabilitySummary:
    plot: str
    positions_total: int
    positions_complete: int
    first_incomplete: str | None
    last_incomplete: str | None

    @property
    def is_complete(self) -> bool:
        return self.positions_complete == self.positions_total


def _position_labels(positions: np.ndarray) -> list[str]:
    return [p.isoformat()[:16] for p in pd.DatetimeIndex(positions)]


def _plot_var_availability(
    series: AvailabilitySeries, out_path: Path, var: str
) -> None:
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(
        series.positions,
        series.fraction,
        marker="o",
        markersize=2,
        linestyle="-",
        color="green",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Fraction available")
    ax.set_title(f"{var} — availability over append dim", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def _heatmap_grid(
    series_by_var: dict[str, AvailabilitySeries],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(positions, fraction grid [var, position], var names) on the union position axis."""
    all_positions = np.unique(
        np.concatenate([s.positions for s in series_by_var.values()])
    )
    grid = np.full((len(series_by_var), len(all_positions)), np.nan)
    for row, series in enumerate(series_by_var.values()):
        grid[row, np.searchsorted(all_positions, series.positions)] = series.fraction
    return all_positions, grid, list(series_by_var)


def _downsample_columns(grid: np.ndarray, max_columns: int) -> np.ndarray:
    n_cols = grid.shape[1]
    if n_cols <= max_columns:
        return grid
    edges = np.linspace(0, n_cols, max_columns + 1).astype(int)
    blocks = [grid[:, lo:hi] for lo, hi in pairwise(edges)]
    with np.errstate(invalid="ignore"):
        return np.column_stack([np.nanmean(b, axis=1) for b in blocks])


def _plot_heatmap(series_by_var: dict[str, AvailabilitySeries], out_path: Path) -> None:
    positions, grid, var_names = _heatmap_grid(series_by_var)
    display = _downsample_columns(grid, MAX_HEATMAP_COLUMNS)

    fig_height = max(3.0, 0.16 * len(var_names) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.set_facecolor("lightgrey")  # masked (not probed) cells show the axes background
    ax.imshow(
        np.ma.masked_invalid(display),
        aspect="auto",
        interpolation="nearest",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_yticks(range(len(var_names)))
    ax.set_yticklabels(var_names, fontsize=max(4, min(8, 900 // len(var_names))))
    tick_idx = np.unique(np.linspace(0, display.shape[1] - 1, 8).astype(int))
    column_to_position = np.linspace(0, len(positions) - 1, display.shape[1]).astype(
        int
    )
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [
            str(p)[:10]
            for p in pd.DatetimeIndex(positions[column_to_position[tick_idx]])
        ],
        fontsize=7,
    )
    ax.set_title(
        "Availability by variable over the append dim (green=available, red=missing)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_availability_artifacts(
    output_dir: Path, series_by_var: dict[str, AvailabilitySeries]
) -> tuple[str, dict[str, VarAvailabilitySummary]]:
    """Write the heatmap + a trace plot per variable. Returns (heatmap filename, per-var summaries)."""
    assert series_by_var, "no availability series to render"
    _plot_heatmap(series_by_var, output_dir / HEATMAP_FILENAME)

    summaries: dict[str, VarAvailabilitySummary] = {}
    for var, series in series_by_var.items():
        probed = ~np.isnan(series.fraction)
        complete = probed & (series.fraction >= 1.0)
        incomplete_positions = series.positions[probed & ~complete]
        plot_name = f"availability_{var_slug(var)}.png"
        _plot_var_availability(series, output_dir / plot_name, var)
        labels = _position_labels(incomplete_positions)
        summaries[var] = VarAvailabilitySummary(
            plot=plot_name,
            positions_total=int(probed.sum()),
            positions_complete=int(complete.sum()),
            first_incomplete=labels[0] if labels else None,
            last_incomplete=labels[-1] if labels else None,
        )
    return HEATMAP_FILENAME, summaries


def apply_availability(ctx: RunContext) -> None:
    """Render availability artifacts from ctx.availability and record them on ctx.stats."""
    heatmap, summaries = write_availability_artifacts(ctx.output_dir, ctx.availability)
    ctx.combined_availability_plot = heatmap
    for var, summary in summaries.items():
        stats = ctx.stats_for(var)
        stats.availability_plot = summary.plot
        stats.positions_total = summary.positions_total
        stats.positions_complete = summary.positions_complete
        stats.first_incomplete = summary.first_incomplete
        stats.last_incomplete = summary.last_incomplete
        if not summary.is_complete:
            log.info(
                f"  availability {var}: {summary.positions_complete}/"
                f"{summary.positions_total} positions complete "
                f"({summary.first_incomplete} → {summary.last_incomplete})"
            )


# ---------------------------------------------------------------------------
# Materialized stores: value (null) scan at the two run points.
# ---------------------------------------------------------------------------


def _compute_nulls_for_point(
    da_point: xr.DataArray,
) -> tuple[xr.DataArray, list[str], int, int]:
    """Null fraction per position, plus unavailable timestamps + counts.

    For accumulated/avg variables, excludes the first lead_time (analysis step) from the
    tally — it's structurally NaN by design.
    """
    non_time_dims = [dim for dim in da_point.dims if dim not in ("time", "init_time")]
    null_mask = da_point.isnull()

    check_mask = null_mask
    if da_point.attrs.get("step_type") != "instant" and "lead_time" in da_point.dims:
        check_mask = null_mask.isel(lead_time=slice(1, None))

    time_dim = next(d for d in ("time", "init_time") if d in da_point.dims)
    null_frac = check_mask.mean(dim=[d for d in non_time_dims if d in check_mask.dims])

    if check_mask.any():
        unavailable = null_frac[time_dim].where(null_frac > 0, drop=True)
        unavailable_strs = unavailable.dt.strftime("%Y-%m-%dT%H:%M:%S").values.tolist()
    else:
        unavailable_strs = []

    return (
        null_frac,
        unavailable_strs,
        int(check_mask.sum().item()),
        int(check_mask.size),
    )


def _format_unavailable_summary(unavailable: list[str]) -> str:
    if not unavailable:
        return "none"
    if len(unavailable) <= 6:
        return f"{len(unavailable)} ({', '.join(unavailable)})"
    head = ", ".join(unavailable[:3])
    tail = ", ".join(unavailable[-3:])
    return f"{len(unavailable)} (first: {head} … last: {tail})"


def write_unavailable_timestamps_file(output_dir: Path, ctx: RunContext) -> str | None:
    """Write unavailable_timestamps.txt aggregating every (var, point) with nulls. Returns filename or None."""
    entries = []
    for var, stats in ctx.stats.items():
        for point_label, unavailable in (
            (
                f"Point 1 (lat={ctx.point1_lat:.2f}, lon={ctx.point1_lon:.2f})",
                stats.unavailable_timestamps_p1,
            ),
            (
                f"Point 2 (lat={ctx.point2_lat:.2f}, lon={ctx.point2_lon:.2f})",
                stats.unavailable_timestamps_p2,
            ),
        ):
            if unavailable:
                entries.append((var, point_label, unavailable))

    if not entries:
        return None

    filename = "unavailable_timestamps.txt"
    path = output_dir / filename
    total = sum(len(m) for _, _, m in entries)
    all_unavailable = sorted(
        {ts for _, _, unavailable in entries for ts in unavailable}
    )
    combined_filter = " ".join(f"--filter-contains {ts}" for ts in all_unavailable)
    lines = [
        "# Unavailable timestamps",
        f"# Total: {total} across {len(entries)} (variable, point) combinations.",
        "# Use --filter-contains <timestamp> to retry those source files with backfill.",
        "",
        f"# Combined retry filter ({len(all_unavailable)} unique timestamps across all variables):",
        f"combined-retry-filter: {combined_filter}",
        "",
    ]
    for var, point_label, unavailable in entries:
        lines.append(f"## {var} @ {point_label}")
        lines.append(f"count: {len(unavailable)}")
        lines.extend(unavailable)
        lines.append("")
        lines.append(
            "retry-filter: " + " ".join(f"--filter-contains {m}" for m in unavailable)
        )
        lines.append("")
    path.write_text("\n".join(lines))
    return filename


def run_value_availability(ctx: RunContext) -> None:
    """Value-scan availability at the two run points (materialized stores)."""
    assert not ctx.is_virtual, (
        "value-scan availability decodes every chunk on a virtual store; "
        "use run_manifest_availability"
    )

    ds_p1 = ctx.validation_ds.isel(ctx.point1_sel)
    ds_p2 = ctx.validation_ds.isel(ctx.point2_sel)
    p1_label = f"Point 1 (lat={ctx.point1_lat:.2f}, lon={ctx.point1_lon:.2f})"
    p2_label = f"Point 2 (lat={ctx.point2_lat:.2f}, lon={ctx.point2_lon:.2f})"
    log.info(f"availability: {len(ctx.variables)} variables at {p1_label} / {p2_label}")

    for var in ctx.variables:
        stats = ctx.stats_for(var)
        level_sel = select_var_level(ctx, var, stats)

        # Load each point's values once and cache them so run_value_timeseries can
        # reuse the same read instead of loading the point data a second time.
        da_p1 = ds_p1[var]
        da_p2 = ds_p2[var]
        if level_sel:
            da_p1 = da_p1.sel(level_sel)
            da_p2 = da_p2.sel(level_sel)
        da_p1 = load_retried(da_p1)
        da_p2 = load_retried(da_p2)
        ctx.loaded_point_data[var] = (da_p1, da_p2)

        null_p1, unavailable_p1, n_p1, total_p1 = _compute_nulls_for_point(da_p1)
        null_p2, unavailable_p2, n_p2, total_p2 = _compute_nulls_for_point(da_p2)

        stats.unavailable_timestamps_p1 = unavailable_p1
        stats.unavailable_timestamps_p2 = unavailable_p2
        stats.null_count_p1 = n_p1
        stats.null_count_p2 = n_p2
        stats.total_count_p1 = total_p1
        stats.total_count_p2 = total_p2

        time_dim = next(d for d in ("time", "init_time") if d in null_p1.dims)
        fraction = 1.0 - (null_p1.values + null_p2.values) / 2.0
        ctx.availability[var] = AvailabilitySeries(
            positions=null_p1[time_dim].values, fraction=fraction
        )

        p1_fmt = _format_unavailable_summary(unavailable_p1)
        p2_fmt = _format_unavailable_summary(unavailable_p2)
        log.info(f"  nulls {var}: P1 unavailable={p1_fmt} | P2 unavailable={p2_fmt}")

    ctx.availability_method_note = (
        "Availability is value-scanned at the two run points: the fraction of non-null "
        "values per position, averaged over both points (structural hour-0 NaNs of "
        "accumulated variables excluded)."
    )
    ctx.unavailable_timestamps_file = write_unavailable_timestamps_file(
        ctx.output_dir, ctx
    )
    if ctx.unavailable_timestamps_file:
        log.info(
            f"  wrote unavailable timestamp list -> {ctx.unavailable_timestamps_file}"
        )
    apply_availability(ctx)


# ---------------------------------------------------------------------------
# Virtual stores: manifest ref probes, whole archive.
# ---------------------------------------------------------------------------


def run_manifest_scan(ctx: RunContext) -> dict[pd.Timestamp, tuple[int, int]]:
    """Manifest-probe availability across the whole archive (virtual stores),
    filling ctx.availability but rendering nothing — matplotlib is main-thread-only,
    so run-all runs this concurrently with the decode + plot phases and calls
    apply_availability after joining. Returns the per-position source file
    availability so callers can gate on it.
    """
    # Late import: manifest_scan imports this module for the shared rendering.
    from scripts.validation.manifest_scan import (  # noqa: PLC0415
        result_availability_series,
        scan_manifest,
        write_unavailable_timestamps_file,
    )
    from scripts.validation.scan_common import resolve_virtual_dataset  # noqa: PLC0415

    dataset_id = ctx.validation_ds.attrs["dataset_id"]
    dataset = resolve_virtual_dataset(dataset_id)
    append_dim = dataset.template_config.append_dim
    # End is exclusive; extend one step past the committed extent so the newest
    # position is scanned without flagging not-yet-published ones.
    end = (
        pd.Timestamp(ctx.validation_ds[append_dim].max().item())
        + dataset.template_config.append_dim_frequency
    )
    start = pd.Timestamp(ctx.start_date) if ctx.start_date else None

    store = open_icechunk_readonly(ctx.validation_url)
    result = scan_manifest(
        dataset, store, start=start, end=end, variables=ctx.variables
    )
    series = result_availability_series(result)
    ctx.availability = {var: series[var] for var in ctx.variables if var in series}

    incomplete_files = {
        position: (present, expected)
        for position, (present, expected) in result.file_availability.items()
        if present < expected
    }
    if incomplete_files:
        ctx.unavailable_timestamps_file = write_unavailable_timestamps_file(
            incomplete_files, ctx.output_dir
        )
    return result.file_availability


def run_manifest_availability(ctx: RunContext) -> dict[pd.Timestamp, tuple[int, int]]:
    """Manifest-probe availability plus rendered artifacts (the standalone command)."""
    file_availability = run_manifest_scan(ctx)
    apply_availability(ctx)
    return file_availability


def build_run_context(
    dataset_url: str,
    variables: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    level: float | None = None,
    output_dir: Path | None = None,
) -> RunContext:
    """The RunContext a standalone command runs against (run-all builds its own)."""
    ds = load_zarr_dataset(dataset_url)
    if start_date or end_date:
        ds = scope_time_period(ds, start_date, end_date)

    selected_vars = select_variables_for_plotting(ds, variables)
    point1_sel, point2_sel, (lat1, lon1), (lat2, lon2) = get_two_random_points(ds)

    out = resolve_output_dir(dataset_url, output_dir)
    log.info(f"output dir: {out}")

    return RunContext(
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
        start_date=start_date,
        end_date=end_date,
        is_virtual=is_virtual_store(dataset_url),
        level_override=level,
    )


def availability(
    dataset_url: str,
    variables: list[str] | None = variables_option,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
    level: float | None = level_option,
    output_dir: Path | None = output_dir_option,
    min_fraction: float = typer.Option(
        1.0,
        "--min-fraction",
        help="Virtual stores: exit non-zero if any position has less than this "
        "fraction of its expected source files (the post-backfill completeness gate)",
    ),
) -> None:
    """Per-variable availability over the append dim (manifest-probed for virtual stores)."""
    ctx = build_run_context(
        dataset_url, variables, start_date, end_date, level, output_dir
    )
    if ctx.is_virtual:
        file_availability = run_manifest_availability(ctx)
        below = [
            position
            for position, (present, expected) in file_availability.items()
            if present / expected < min_fraction
        ]
        if below:
            log.error(
                f"Manifest incomplete: {len(below)} of {len(file_availability)} "
                f"positions below {min_fraction:.0%} of expected source files"
            )
            raise typer.Exit(1)
        log.info(
            f"Manifest complete: all {len(file_availability)} positions "
            f"≥ {min_fraction:.0%}"
        )
    else:
        run_value_availability(ctx)
