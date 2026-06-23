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


def _append_dim_range(ctx: RunContext) -> tuple[str, str]:
    """Min and max of the append dimension (already scoped if start/end were applied)."""
    ds = ctx.validation_ds
    dim = "init_time" if is_forecast_dataset(ds) else "time"
    return str(ds[dim].min().values)[:16], str(ds[dim].max().values)[:16]


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


def _stats_row(
    label: str, mn: float | None, mean: float | None, mx: float | None
) -> str:
    return f"| {label} | {_fmt_num(mn)} | {_fmt_num(mean)} | {_fmt_num(mx)} |"


def _metadata_table(stats: VariableStats) -> list[str]:
    return [
        "**Metadata**",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| units | `{stats.units or 'n/a'}` |",
        f"| long_name | {stats.long_name or 'n/a'} |",
        f"| short_name | {stats.short_name or 'n/a'} |",
        f"| standard_name | {stats.standard_name or 'n/a'} |",
        f"| step_type | {stats.step_type or 'n/a'} |",
        "",
    ]


def _spatial_table(stats: VariableStats, ctx: RunContext) -> list[str]:
    if stats.ref_available_spatial:
        ref_clause = f"reference at {ctx.ref_spatial_time_label or 'n/a'}"
    else:
        ref_clause = "reference not available"
    lines = [
        f"**Spatial** — snapshot at {stats.spatial_time_label or 'n/a'} ({ref_clause})",
        "",
        "| Source | min | mean | max |",
        "|---|---|---|---|",
        _stats_row(
            "Validation",
            stats.val_spatial_min,
            stats.val_spatial_mean,
            stats.val_spatial_max,
        ),
    ]
    if stats.ref_available_spatial:
        lines.append(
            _stats_row(
                "Reference",
                stats.ref_spatial_min,
                stats.ref_spatial_mean,
                stats.ref_spatial_max,
            )
        )
    lines.append("")
    return lines


def _value_ts_table(stats: VariableStats, ctx: RunContext) -> list[str]:
    lo, hi = _append_dim_range(ctx)
    return [
        f"**Point time series statistics for the full period ({lo} - {hi})**",
        "",
        "| Point | min | mean | std | max |",
        "|---|---|---|---|---|",
        f"| P1 | {_fmt_num(stats.value_min_p1)} | {_fmt_num(stats.value_mean_p1)} "
        f"| {_fmt_num(stats.value_std_p1)} | {_fmt_num(stats.value_max_p1)} |",
        f"| P2 | {_fmt_num(stats.value_min_p2)} | {_fmt_num(stats.value_mean_p2)} "
        f"| {_fmt_num(stats.value_std_p2)} | {_fmt_num(stats.value_max_p2)} |",
        "",
    ]


def _temporal_table(stats: VariableStats, ctx: RunContext) -> list[str]:
    lines = [
        f"**Temporal** — period {ctx.temporal_period_label or 'n/a'}",
        "",
        "| Source | min | mean | max |",
        "|---|---|---|---|",
        _stats_row(
            "P1 Validation",
            stats.val_temporal_min_p1,
            stats.val_temporal_mean_p1,
            stats.val_temporal_max_p1,
        ),
    ]
    if stats.ref_available_temporal:
        lines.append(
            _stats_row(
                "P1 Reference",
                stats.ref_temporal_min_p1,
                stats.ref_temporal_mean_p1,
                stats.ref_temporal_max_p1,
            )
        )
    lines.append(
        _stats_row(
            "P2 Validation",
            stats.val_temporal_min_p2,
            stats.val_temporal_mean_p2,
            stats.val_temporal_max_p2,
        )
    )
    if stats.ref_available_temporal:
        lines.append(
            _stats_row(
                "P2 Reference",
                stats.ref_temporal_min_p2,
                stats.ref_temporal_mean_p2,
                stats.ref_temporal_max_p2,
            )
        )
    lines.append("")
    return lines


def _nulls_line(stats: VariableStats) -> str:
    p1 = (
        f"P1: {_fmt_count(stats.null_count_p1, stats.total_count_p1)} "
        f"({_unavailable_summary(stats.unavailable_timestamps_p1)})"
    )
    p2 = (
        f"P2: {_fmt_count(stats.null_count_p2, stats.total_count_p2)} "
        f"({_unavailable_summary(stats.unavailable_timestamps_p2)})"
    )
    return f"**Nulls** — {p1}; {p2}"


def _variable_section(stats: VariableStats, ctx: RunContext) -> str:
    lines = [f"### `{stats.name}`", ""]
    lines += _metadata_table(stats)
    lines += _value_ts_table(stats, ctx)
    lines += _spatial_table(stats, ctx)
    lines += _temporal_table(stats, ctx)
    lines += [_nulls_line(stats), ""]
    return "\n".join(lines)


def _run_parameters_table(ctx: RunContext) -> list[str]:
    is_forecast = is_forecast_dataset(ctx.validation_ds)

    spatial_time = (
        f"{ctx.spatial_time_label or 'n/a'} "
        f"(reference at {ctx.ref_spatial_time_label or 'n/a'})"
    )

    rows: list[tuple[str, str]] = [
        ("Validation dataset type", "forecast" if is_forecast else "analysis"),
        ("Validation time range", _dataset_time_range(ctx)),
        ("Reference time range", _reference_time_range(ctx)),
        ("Point 1", f"lat={ctx.point1_lat:.4f}, lon={ctx.point1_lon:.4f}"),
        ("Point 2", f"lat={ctx.point2_lat:.4f}, lon={ctx.point2_lon:.4f}"),
    ]
    if ctx.ensemble_member is not None:
        rows.append(("Ensemble member", str(ctx.ensemble_member)))
    rows += [
        ("Spatial comparison time", spatial_time),
        ("Timeseries period", ctx.temporal_period_label or "n/a"),
    ]

    lines = ["| Parameter | Value |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return lines


def write_summary_md(ctx: RunContext) -> Path:  # noqa: PLR0915
    """Write validation_summary.md aggregating a full validation run. Returns the path."""
    ds = ctx.validation_ds
    ref = ctx.reference_ds

    val_id, val_ver = dataset_id_and_version(ctx.validation_url)
    val_name = ds.attrs.get("name", val_id)
    ref_id = ref.attrs.get("dataset_id", "n/a") if ref is not None else "n/a"
    ref_ver = ref.attrs.get("dataset_version", "n/a") if ref is not None else "n/a"
    ref_name = ref.attrs.get("name", ref_id) if ref is not None else "n/a"

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
    lines.extend(_run_parameters_table(ctx))
    lines.append("")

    lines.append("## Combined plots")
    lines.append("")
    combined_items = [
        ("Unavailable values", ctx.combined_nulls_plot),
        ("Value time series (full period)", ctx.combined_value_timeseries_plot),
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
