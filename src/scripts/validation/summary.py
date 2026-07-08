import math
from pathlib import Path

from scripts.validation.decode_scan import decode_summary_lines
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
    rows = [
        "**Metadata**",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| units | `{stats.units or 'n/a'}` |",
        f"| long_name | {stats.long_name or 'n/a'} |",
        f"| short_name | {stats.short_name or 'n/a'} |",
        f"| standard_name | {stats.standard_name or 'n/a'} |",
        f"| step_type | {stats.step_type or 'n/a'} |",
    ]
    if stats.level_dim is not None:
        rows.append(f"| sampled level | {stats.level_dim}={stats.level_value:g} |")
    rows.append("")
    return rows


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
    sampled = " — sampled (pinned lead/level/member)" if ctx.is_virtual else ""
    return [
        f"**Point time series statistics for the full period ({lo} - {hi}){sampled}**",
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


def _availability_line(stats: VariableStats) -> str:
    if stats.positions_total is None:
        return "**Availability** — n/a"
    if stats.positions_complete == stats.positions_total:
        detail = (
            f"{stats.positions_total} of {stats.positions_total} positions complete"
        )
    else:
        detail = (
            f"{stats.positions_complete} of {stats.positions_total} positions complete "
            f"(incomplete {stats.first_incomplete} → {stats.last_incomplete}, "
            f"see [plot]({stats.availability_plot}))"
        )
    if stats.null_count_p1 is not None:
        detail += (
            f"; nulls P1 {_fmt_count(stats.null_count_p1, stats.total_count_p1)}, "
            f"P2 {_fmt_count(stats.null_count_p2, stats.total_count_p2)}"
        )
    return f"**Availability** — {detail}"


def _variable_section(stats: VariableStats, ctx: RunContext) -> str:
    lines = [f"### `{stats.name}`", ""]
    lines += _metadata_table(stats)
    lines += _value_ts_table(stats, ctx)
    lines += _spatial_table(stats, ctx)
    lines += _temporal_table(stats, ctx)
    lines += [_availability_line(stats), ""]
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
        (
            "Vertical level",
            f"override {ctx.level_override:g}"
            if ctx.level_override is not None
            else "middle level per vertical dim",
        ),
    ]
    if ctx.is_virtual:
        rows.append(
            (
                "Store type",
                "virtual — availability manifest-probed; value series sampled",
            )
        )

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

    lines.append("## Availability")
    lines.append("")
    if ctx.availability_method_note:
        lines.append(ctx.availability_method_note)
        lines.append("")
    if ctx.combined_availability_plot:
        lines.append(f"![availability heatmap]({ctx.combined_availability_plot})")
        lines.append("")
    incomplete = [
        ctx.stats[var]
        for var in ctx.variables
        if var in ctx.stats
        and ctx.stats[var].positions_total is not None
        and ctx.stats[var].positions_complete != ctx.stats[var].positions_total
    ]
    if not incomplete:
        lines.append("Every variable is complete at every scanned position.")
    else:
        lines.append(
            "| Variable | Complete positions | Incomplete % "
            "| First incomplete | Last incomplete |"
        )
        lines.append("|---|---|---|---|---|")
        for stats in incomplete:
            assert stats.positions_total is not None
            assert stats.positions_complete is not None
            pct = 100 * (1 - stats.positions_complete / stats.positions_total)
            lines.append(
                f"| `{stats.name}` | {stats.positions_complete}/{stats.positions_total} "
                f"| {pct:.2f}% | {stats.first_incomplete} | {stats.last_incomplete} |"
            )
    lines.append("")
    if ctx.unavailable_timestamps_file:
        lines.append(
            f"- Unavailable timestamps + retry filters: "
            f"[`{ctx.unavailable_timestamps_file}`]({ctx.unavailable_timestamps_file})"
        )
        lines.append("")

    if ctx.decode_note is not None:
        lines.append("## Decode health (sampled)")
        lines.append("")
        lines.extend(decode_summary_lines(ctx))
        lines.append("")

    lines.append("## Per-variable details")
    lines.append("")
    lines.extend(_variable_section(ctx.stats[var], ctx) for var in ctx.variables)

    path = ctx.output_dir / "validation_summary.md"
    path.write_text("\n".join(lines))
    return path
