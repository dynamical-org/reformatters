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


def _missing_summary(missing: list[str]) -> str:
    if not missing:
        return "none"
    if len(missing) <= 6:
        return f"{len(missing)} missing: {', '.join(missing)}"
    head = ", ".join(missing[:3])
    tail = ", ".join(missing[-3:])
    return f"{len(missing)} missing (first: {head} … last: {tail})"


def _link(label: str, filename: str | None) -> str:
    return f"[{label}]({filename})" if filename else f"~~{label}~~"


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


def _variable_row(stats: VariableStats) -> str:
    plots = (
        f"{_link('nulls', stats.null_plot)} · "
        f"{_link('spatial', stats.spatial_plot)} · "
        f"{_link('temporal', stats.temporal_plot)}"
    )
    null_p1 = _fmt_count(stats.null_count_p1, stats.total_count_p1)
    null_p2 = _fmt_count(stats.null_count_p2, stats.total_count_p2)
    return (
        f"| `{stats.name}` | {stats.units or ''} | {stats.long_name or ''} "
        f"| {null_p1} | {null_p2} | {plots} |"
    )


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
        f"- validation: min={_fmt_num(stats.val_spatial_min)}, "
        f"max={_fmt_num(stats.val_spatial_max)}, mean={_fmt_num(stats.val_spatial_mean)}",
    ]
    if stats.ref_available_spatial:
        lines.append(
            f"- reference:  min={_fmt_num(stats.ref_spatial_min)}, "
            f"max={_fmt_num(stats.ref_spatial_max)}, mean={_fmt_num(stats.ref_spatial_mean)}"
        )
    else:
        lines.append("- reference:  variable not available in reference dataset")
    lines += [
        "",
        "**Temporal comparison**",
        "",
        f"- plot: `{stats.temporal_plot or 'n/a'}`",
        f"- period: {ctx.temporal_period_label or 'n/a'}",
        f"- validation @ P1 (lat={ctx.point1_lat:.2f}, lon={ctx.point1_lon:.2f}): "
        f"min={_fmt_num(stats.val_temporal_min_p1)}, "
        f"max={_fmt_num(stats.val_temporal_max_p1)}, "
        f"mean={_fmt_num(stats.val_temporal_mean_p1)}",
        f"- validation @ P2 (lat={ctx.point2_lat:.2f}, lon={ctx.point2_lon:.2f}): "
        f"min={_fmt_num(stats.val_temporal_min_p2)}, "
        f"max={_fmt_num(stats.val_temporal_max_p2)}, "
        f"mean={_fmt_num(stats.val_temporal_mean_p2)}",
    ]
    if stats.ref_available_temporal:
        lines += [
            f"- reference  @ P1: min={_fmt_num(stats.ref_temporal_min_p1)}, "
            f"max={_fmt_num(stats.ref_temporal_max_p1)}, "
            f"mean={_fmt_num(stats.ref_temporal_mean_p1)}",
            f"- reference  @ P2: min={_fmt_num(stats.ref_temporal_min_p2)}, "
            f"max={_fmt_num(stats.ref_temporal_max_p2)}, "
            f"mean={_fmt_num(stats.ref_temporal_mean_p2)}",
        ]
    else:
        lines.append("- reference:  variable not available in reference dataset")

    lines += [
        "",
        "**Nulls**",
        "",
        f"- P1 nulls: {_fmt_count(stats.null_count_p1, stats.total_count_p1)} — "
        f"{_missing_summary(stats.missing_timestamps_p1)}",
        f"- P2 nulls: {_fmt_count(stats.null_count_p2, stats.total_count_p2)} — "
        f"{_missing_summary(stats.missing_timestamps_p2)}",
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

    missing_entries = []
    for var, stats in ctx.stats.items():
        if stats.missing_timestamps_p1:
            missing_entries.append((var, "P1", len(stats.missing_timestamps_p1)))
        if stats.missing_timestamps_p2:
            missing_entries.append((var, "P2", len(stats.missing_timestamps_p2)))
    missing_total = sum(n for _, _, n in missing_entries)

    lines: list[str] = []
    lines.append(f"# Validation run — `{val_id}` `{val_ver}`")
    lines.append("")
    lines.append(f"Started: {ctx.started_at.strftime('%Y-%m-%dT%H:%M:%S')} local")
    lines.append("")
    lines.append("## Datasets")
    lines.append("")
    lines.append("| Role | Name | ID / Version | URL |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Validation | {val_name} | `{val_id}` `{val_ver}` | `{ctx.validation_url}` |"
    )
    lines.append(
        f"| Reference  | {ref_name} | `{ref_id}` `{ref_ver}` | "
        f"`{ctx.reference_url or 'n/a'}` |"
    )
    lines.append("")
    lines.append("## Run parameters")
    lines.append("")
    lines.append(
        f"- Validation dataset type: **{'forecast' if is_forecast else 'analysis'}**"
    )
    lines.append(f"- Validation time range: {_dataset_time_range(ctx)}")
    lines.append(f"- Reference time range:  {_reference_time_range(ctx)}")
    lines.append(f"- Time scope: {scope_line}")
    lines.append(
        f"- Ensemble member: "
        f"{ctx.ensemble_member if ctx.ensemble_member is not None else 'n/a'}"
    )
    lines.append(f"- Point 1: lat={ctx.point1_lat:.4f}, lon={ctx.point1_lon:.4f}")
    lines.append(f"- Point 2: lat={ctx.point2_lat:.4f}, lon={ctx.point2_lon:.4f}")
    lines.append(
        f"- Spatial comparison time: "
        f"{ctx.spatial_time_label or 'n/a'} (reference at {ctx.ref_spatial_time_label or 'n/a'})"
    )
    lines.append(f"- Timeseries period: {ctx.temporal_period_label or 'n/a'}")
    lines.append("")

    lines.append("## Combined plots")
    lines.append("")
    combined_items = [
        ("nulls", ctx.combined_nulls_plot),
        ("spatial", ctx.combined_spatial_plot),
        ("temporal", ctx.combined_temporal_plot),
    ]
    for label, filename in combined_items:
        if filename:
            lines.append(f"- {label}: [`{filename}`]({filename})")
    lines.append("")

    lines.append("## Missing timestamps")
    lines.append("")
    if not missing_entries:
        lines.append("None detected at the two sampled points.")
    else:
        lines.append(
            f"**{missing_total}** missing timestamps across **{len(missing_entries)}** "
            f"(variable, point) combinations."
        )
        lines.append("")
        lines.append(
            f"Full list: [`{ctx.missing_timestamps_file or 'missing_timestamps.txt'}`]"
            f"({ctx.missing_timestamps_file or 'missing_timestamps.txt'})"
        )
        lines.append("")
        lines.append("| Variable | Point | Count |")
        lines.append("|---|---|---|")
        for var, point, count in missing_entries:
            lines.append(f"| `{var}` | {point} | {count} |")
    lines.append("")

    lines.append("## Variables overview")
    lines.append("")
    lines.append("| Variable | Units | Long name | Nulls @ P1 | Nulls @ P2 | Plots |")
    lines.append("|---|---|---|---|---|---|")
    lines.extend(_variable_row(ctx.stats[var]) for var in ctx.variables)
    lines.append("")

    lines.append("## Per-variable details")
    lines.append("")
    lines.extend(_variable_section(ctx.stats[var], ctx) for var in ctx.variables)

    path = ctx.output_dir / "validation_summary.md"
    path.write_text("\n".join(lines))
    return path
