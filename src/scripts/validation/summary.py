import math
from pathlib import Path

from scripts.validation.utils import (
    RunContext,
    VariableStats,
    dataset_id_and_version,
    is_forecast_dataset,
)


def _fmt_num(v: float | None) -> str:
    if v is None:
        return "n/a"
    if math.isnan(v):
        return "NaN"
    return f"{v:.4g}"


def _fmt_count(n: int | None, total: int | None) -> str:
    if n is None or total is None:
        return "n/a"
    return f"{n}/{total}"


def _unavailable_summary(unavailable: list[str]) -> str:
    if not unavailable:
        return "none"
    if len(unavailable) <= 6:
        return f"{len(unavailable)} unavailable: {', '.join(unavailable)}"
    head = ", ".join(unavailable[:3])
    tail = ", ".join(unavailable[-3:])
    return f"{len(unavailable)} unavailable (first: {head} … last: {tail})"


def _dataset_time_range(ctx: RunContext) -> str:
    ds = ctx.validation_ds
    if is_forecast_dataset(ds):
        init_min = str(ds.init_time.min().values)[:16]
        init_max = str(ds.init_time.max().values)[:16]
        return f"init_time {init_min} → {init_max}"
    time_min = str(ds.time.min().values)[:16]
    time_max = str(ds.time.max().values)[:16]
    return f"time {time_min} → {time_max}"


def _reference_time_range(ctx: RunContext) -> str:
    if ctx.reference_ds is None or "time" not in ctx.reference_ds.dims:
        return "n/a"
    t_min = str(ctx.reference_ds.time.min().values)[:16]
    t_max = str(ctx.reference_ds.time.max().values)[:16]
    return f"time {t_min} → {t_max}"


def _stats_line(
    label: str, mn: float | None, mean: float | None, mx: float | None
) -> str:
    return f"- {label}: min={_fmt_num(mn)}, mean={_fmt_num(mean)}, max={_fmt_num(mx)}"


def _variable_section(stats: VariableStats, ctx: RunContext) -> str:
    lines = [
        f"### `{stats.name}`",
        "",
        "**Metadata**",
        "",
        f"- units: `{stats.units or 'n/a'}`",
        f"- long_name: {stats.long_name or 'n/a'}",
        f"- short_name: {stats.short_name or 'n/a'}",
        f"- standard_name: {stats.standard_name or 'n/a'}",
        f"- step_type: {stats.step_type or 'n/a'}",
        "",
        "**Spatial comparison**",
        "",
        f"- plot: `{stats.spatial_plot or 'n/a'}`",
        f"- time: {stats.spatial_time_label or 'n/a'} "
        f"(reference at {ctx.ref_spatial_time_label or 'n/a'})",
        _stats_line(
            "validation",
            stats.val_spatial_min,
            stats.val_spatial_mean,
            stats.val_spatial_max,
        ),
    ]
    if stats.ref_available_spatial:
        lines.append(
            _stats_line(
                "reference",
                stats.ref_spatial_min,
                stats.ref_spatial_mean,
                stats.ref_spatial_max,
            )
        )
    else:
        lines.append("- reference:  variable not available in reference dataset")
    lines += [
        "",
        "**Temporal comparison**",
        "",
        f"- plot: `{stats.temporal_plot or 'n/a'}`",
        f"- period: {ctx.temporal_period_label or 'n/a'}",
        "",
        f"P1 (lat={ctx.point1_lat:.2f}, lon={ctx.point1_lon:.2f}):",
        "",
        _stats_line(
            "validation",
            stats.val_temporal_min_p1,
            stats.val_temporal_mean_p1,
            stats.val_temporal_max_p1,
        ),
    ]
    if stats.ref_available_temporal:
        lines.append(
            _stats_line(
                "reference",
                stats.ref_temporal_min_p1,
                stats.ref_temporal_mean_p1,
                stats.ref_temporal_max_p1,
            )
        )
    else:
        lines.append("- reference:  variable not available in reference dataset")
    lines += [
        "",
        f"P2 (lat={ctx.point2_lat:.2f}, lon={ctx.point2_lon:.2f}):",
        "",
        _stats_line(
            "validation",
            stats.val_temporal_min_p2,
            stats.val_temporal_mean_p2,
            stats.val_temporal_max_p2,
        ),
    ]
    if stats.ref_available_temporal:
        lines.append(
            _stats_line(
                "reference",
                stats.ref_temporal_min_p2,
                stats.ref_temporal_mean_p2,
                stats.ref_temporal_max_p2,
            )
        )
    else:
        lines.append("- reference:  variable not available in reference dataset")

    lines += [
        "",
        "**Nulls**",
        "",
        f"- P1 nulls: {_fmt_count(stats.null_count_p1, stats.total_count_p1)} — "
        f"{_unavailable_summary(stats.unavailable_timestamps_p1)}",
        f"- P2 nulls: {_fmt_count(stats.null_count_p2, stats.total_count_p2)} — "
        f"{_unavailable_summary(stats.unavailable_timestamps_p2)}",
        "",
    ]
    return "\n".join(lines)


def write_summary_md(ctx: RunContext) -> Path:  # noqa: PLR0915
    """Write validation_summary.md aggregating a full validation run. Returns the path."""
    ds = ctx.validation_ds
    ref = ctx.reference_ds

    val_id, val_ver = dataset_id_and_version(ctx.validation_url)
    val_name = ds.attrs.get("name", val_id)
    ref_id = ref.attrs.get("dataset_id", "n/a") if ref is not None else "n/a"
    ref_ver = ref.attrs.get("dataset_version", "n/a") if ref is not None else "n/a"
    ref_name = ref.attrs.get("name", ref_id) if ref is not None else "n/a"

    is_forecast = is_forecast_dataset(ds)

    scope_line = "full dataset"
    if ctx.start_date or ctx.end_date:
        scope_line = f"start={ctx.start_date or 'dataset start'}, end={ctx.end_date or 'dataset end'}"

    unavailable_rows: list[tuple[str, str, int, int, float, str, str]] = []
    for var in ctx.variables:
        stats = ctx.stats[var]
        for point, unavailable, total in (
            ("P1", stats.unavailable_timestamps_p1, stats.total_count_p1),
            ("P2", stats.unavailable_timestamps_p2, stats.total_count_p2),
        ):
            if unavailable and total:
                pct = len(unavailable) / total * 100
                unavailable_rows.append(
                    (
                        var,
                        point,
                        len(unavailable),
                        total,
                        pct,
                        unavailable[0],
                        unavailable[-1],
                    )
                )

    lines: list[str] = []
    # No H1 here — the dynamical.org page template generates the title from the catalog entry.
    lines.append(
        f"This dataset validation report plots a sample of values from the "
        f"{val_name} dataset over time and across space, comparing where possible "
        f"to a previously validated reference dataset. It also reports the quantity "
        f"of unavailable values and their associated timestamps. These analyses are one "
        f"layer of a multi-layered dataset validation process we perform at "
        f"dynamical.org and also provide users a preview of the dataset contents."
    )
    lines.append("")
    lines.append(
        f"Report generation start time: "
        f"{ctx.started_at.strftime('%Y-%m-%dT%H:%M:%S')} UTC"
    )
    lines.append("")
    lines.append("## Datasets")
    lines.append("")
    lines.append("| Role | Name | ID | Version | URL |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Validation | {val_name} | `{val_id}` | `{val_ver}` | "
        f"`{ctx.validation_url}` |"
    )
    lines.append(
        f"| Reference  | {ref_name} | `{ref_id}` | `{ref_ver}` | "
        f"`{ctx.reference_url or 'n/a'}` |"
    )
    lines.append("")
    lines.append("## Run parameters")
    lines.append("")
    lines.append(
        f"- Validation dataset type: {'forecast' if is_forecast else 'analysis'}"
    )
    lines.append(f"- Validation time range: {_dataset_time_range(ctx)}")
    lines.append(f"- Reference time range:  {_reference_time_range(ctx)}")
    lines.append(f"- Time scope: {scope_line}")
    lines.append("")
    lines.append("### Unavailable values")
    lines.append("")
    lines.append(f"- Point 1: lat={ctx.point1_lat:.4f}, lon={ctx.point1_lon:.4f}")
    lines.append(f"- Point 2: lat={ctx.point2_lat:.4f}, lon={ctx.point2_lon:.4f}")
    lines.append("")
    lines.append("### Spatial and distribution")
    lines.append("")
    if ctx.ensemble_member is not None:
        lines.append(f"- Ensemble member: {ctx.ensemble_member}")
    lines.append(
        f"- Spatial comparison time: "
        f"{ctx.spatial_time_label or 'n/a'} (reference at {ctx.ref_spatial_time_label or 'n/a'})"
    )
    lines.append("")
    lines.append("### Time series")
    lines.append("")
    if ctx.ensemble_member is not None:
        lines.append(f"- Ensemble member: {ctx.ensemble_member}")
    lines.append(f"- Point 1: lat={ctx.point1_lat:.4f}, lon={ctx.point1_lon:.4f}")
    lines.append(f"- Point 2: lat={ctx.point2_lat:.4f}, lon={ctx.point2_lon:.4f}")
    lines.append(f"- Timeseries period: {ctx.temporal_period_label or 'n/a'}")
    lines.append("")

    lines.append("## Combined plots")
    lines.append("")
    lines.append("All variables combined into a single plot for each type of analysis.")
    lines.append("")
    combined_items = [
        ("Unavailable values", ctx.combined_nulls_plot),
        ("Spatial and distributions", ctx.combined_spatial_plot),
        ("Time series", ctx.combined_temporal_plot),
    ]
    for label, filename in combined_items:
        if filename:
            lines.append(f"- {label}: [`{filename}`]({filename})")
    lines.append("")

    lines.append("## Unavailable timestamps")
    lines.append("")
    if not unavailable_rows:
        lines.append("None detected at the two sampled points.")
    else:
        lines.append(
            f"Full list: [`{ctx.unavailable_timestamps_file or 'unavailable_timestamps.txt'}`]"
            f"({ctx.unavailable_timestamps_file or 'unavailable_timestamps.txt'})"
        )
        lines.append("")
        lines.append(
            "| Variable | Point | Unavailable count | Total count | Unavailable % "
            "| Earliest unavailable | Latest unavailable |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for var, point, count, total, pct, earliest, latest in unavailable_rows:
            lines.append(
                f"| `{var}` | {point} | {count} | {total} | {pct:.2f}% "
                f"| {earliest} | {latest} |"
            )
    lines.append("")

    lines.append("## Per-variable details")
    lines.append("")
    lines.extend(_variable_section(ctx.stats[var], ctx) for var in ctx.variables)

    path = ctx.output_dir / "validation_summary.md"
    path.write_text("\n".join(lines))
    return path
