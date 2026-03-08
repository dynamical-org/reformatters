# ruff: noqa: T201, B008, PLC0415
"""Compression audit tool for Zarr v3 sharded datasets.

Collects shard sizes from S3 and computes compression ratios, then generates
reports with histograms and summary statistics.

Usage:
    uv run python src/scripts/compression_audit.py collect [--dataset-ids ...]
    uv run python src/scripts/compression_audit.py report <parquet_path> --output-dir <dir>
    uv run python src/scripts/compression_audit.py all --output-dir <dir> [--dataset-ids ...]
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import typer

from reformatters.common.dynamical_dataset import DynamicalDataset

mpl.use("Agg")

NUM_BINS = 1000
DTYPE_SIZES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "uint16": 2,
    "int16": 2,
    "int64": 8,
    "bool": 1,
}

app = typer.Typer(pretty_exceptions_show_locals=False)


def _get_non_contrib_datasets() -> list[DynamicalDataset[Any, Any]]:
    os.environ.setdefault("DYNAMICAL_ENV", "prod")
    from reformatters.__main__ import DYNAMICAL_DATASETS

    return [d for d in DYNAMICAL_DATASETS if "contrib" not in d.__class__.__module__]


def _shard_shape_as_tuple(shards: tuple[int, ...] | int) -> tuple[int, ...]:
    if isinstance(shards, int):
        return (shards,)
    return shards


def _collect_dataset(
    dataset: DynamicalDataset[Any, Any], fs: s3fs.S3FileSystem
) -> list[dict[str, str | float | int]]:
    dataset_id: str = dataset.dataset_id
    sf = dataset.store_factory
    base_path = dataset.primary_storage_config.base_path
    zarr_path = f"{base_path}/{dataset_id}/v{sf.template_config_version}.zarr"
    zarr_prefix = zarr_path.removeprefix("s3://")

    rows: list[dict[str, str | float | int]] = []

    for data_var in dataset.template_config.data_vars:
        assert data_var.encoding.shards is not None, (
            f"{dataset_id}/{data_var.name} has no shard encoding"
        )
        shard_shape = _shard_shape_as_tuple(data_var.encoding.shards)
        dtype_size = DTYPE_SIZES[data_var.encoding.dtype]
        uncompressed_shard_bytes = math.prod(shard_shape) * dtype_size

        var_prefix = f"{zarr_prefix}/{data_var.name}/c"
        print(f"  Listing shards for {data_var.name}...", end="", flush=True)

        shard_files = fs.find(var_prefix, detail=True)
        sizes = [info["size"] for info in shard_files.values() if info["size"] > 0]
        print(f" {len(sizes)} shards")

        if not sizes:
            continue

        fractions = np.array(sizes, dtype=np.float64) / uncompressed_shard_bytes
        fractions = np.clip(fractions, 0.0, 1.0)

        bin_edges = np.linspace(0.0, 1.0, NUM_BINS + 1)
        counts, _ = np.histogram(fractions, bins=bin_edges)

        rows.extend(
            {
                "dataset_id": dataset_id,
                "data_var": data_var.name,
                "compression_fraction_bin_start": float(bin_edges[i]),
                "count": int(counts[i]),
            }
            for i in range(NUM_BINS)
            if counts[i] > 0
        )

    return rows


@app.command()
def collect(
    dataset_ids: list[str] | None = typer.Option(None, "--dataset-id", "-d"),
    output_dir: Path = typer.Option(Path("reports"), "--output-dir", "-o"),
) -> Path:
    """Collect shard sizes from S3 and write a histogram parquet file."""
    datasets = _get_non_contrib_datasets()
    if dataset_ids:
        datasets = [d for d in datasets if d.dataset_id in dataset_ids]
        found = {d.dataset_id for d in datasets}
        missing = set(dataset_ids) - found
        assert not missing, f"Unknown dataset IDs: {missing}"

    fs = s3fs.S3FileSystem(anon=True)
    all_rows: list[dict[str, str | float | int]] = []

    for dataset in datasets:
        print(f"Collecting {dataset.dataset_id}...")
        all_rows.extend(_collect_dataset(dataset, fs))

    df = pd.DataFrame(all_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    parquet_path = output_dir / f"compression_audit_{timestamp}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path}")
    return parquet_path


def _expand_histogram(df: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Expand histogram bin counts back to individual fraction values (bin midpoints)."""
    bin_width = 1.0 / NUM_BINS
    midpoints = df["compression_fraction_bin_start"].to_numpy() + bin_width / 2
    counts = df["count"].to_numpy().astype(int)
    return np.repeat(midpoints, counts)


def _compute_stats(values: np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, float]:
    if len(values) == 0:
        return {
            "min": float("nan"),
            "q0.1": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "q0.9": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    return {
        "min": float(np.min(values)),
        "q0.1": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q0.9": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
        "count": len(values),
    }


def _plot_histogram(
    values: np.ndarray[Any, np.dtype[np.float64]],
    title: str,
    output_path: Path,
    stats: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=100, range=(0, 1), edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Compression Fraction (compressed / uncompressed)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    stat_lines = {
        "min": ("red", "--"),
        "q0.1": ("orange", ":"),
        "median": ("green", "-"),
        "mean": ("blue", "-."),
        "q0.9": ("orange", ":"),
        "max": ("red", "--"),
    }
    for stat_name, (color, linestyle) in stat_lines.items():
        val = stats[stat_name]
        if not np.isnan(val):
            ax.axvline(
                val,
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
                label=f"{stat_name}={val:.4f}",
            )

    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _print_stats(label: str, stats: dict[str, float]) -> None:
    count = stats.get("count", 0)
    print(
        f"{label}: min={stats['min']:.4f} q0.1={stats['q0.1']:.4f} "
        f"median={stats['median']:.4f} mean={stats['mean']:.4f} "
        f"q0.9={stats['q0.9']:.4f} max={stats['max']:.4f} (n={count})"
    )


def _stats_table(stats_list: list[dict[str, Any]]) -> str:
    lines = [
        "| Facet | Min | Q0.1 | Median | Mean | Q0.9 | Max | Count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {s['facet']} | {s['min']:.4f} | {s['q0.1']:.4f} | "
        f"{s['median']:.4f} | {s['mean']:.4f} | {s['q0.9']:.4f} | "
        f"{s['max']:.4f} | {int(s['count'])} |"
        for s in stats_list
    )
    return "\n".join(lines)


def _report_overall(
    df: pd.DataFrame,
    png_dir: Path,
    report_lines: list[str],
) -> None:
    values = _expand_histogram(df)
    stats = _compute_stats(values)
    labeled = {"facet": "overall", **stats}
    _print_stats("Overall", stats)
    _plot_histogram(
        values, "Overall Compression Fractions", png_dir / "overall.png", stats
    )
    report_lines.append("## Overall\n")
    report_lines.append(_stats_table([labeled]))
    report_lines.append("\n![Overall](overall.png)\n")


def _report_by_dataset(
    df: pd.DataFrame,
    png_dir: Path,
    report_lines: list[str],
) -> None:
    report_lines.append("## By Dataset\n")
    dataset_stats: list[dict[str, Any]] = []
    for dataset_id in sorted(df["dataset_id"].unique()):
        sub = df[df["dataset_id"] == dataset_id]
        values = _expand_histogram(sub)
        stats = _compute_stats(values)
        labeled = {"facet": dataset_id, **stats}
        dataset_stats.append(labeled)
        _print_stats(f"Dataset: {dataset_id}", stats)
        png_name = f"dataset_{dataset_id}.png"
        _plot_histogram(values, f"Compression: {dataset_id}", png_dir / png_name, stats)
        report_lines.append(f"### {dataset_id}\n")
        report_lines.append(_stats_table([labeled]))
        report_lines.append(f"\n![{dataset_id}]({png_name})\n")


def _report_by_var(
    df: pd.DataFrame,
    png_dir: Path,
    report_lines: list[str],
) -> None:
    report_lines.append("## By Data Variable\n")
    for var_name in sorted(df["data_var"].unique()):
        sub = df[df["data_var"] == var_name]
        values = _expand_histogram(sub)
        stats = _compute_stats(values)
        labeled = {"facet": var_name, **stats}
        _print_stats(f"Variable: {var_name}", stats)
        png_name = f"var_{var_name}.png"
        _plot_histogram(values, f"Compression: {var_name}", png_dir / png_name, stats)
        report_lines.append(f"### {var_name}\n")
        report_lines.append(_stats_table([labeled]))
        report_lines.append(f"\n![{var_name}]({png_name})\n")


def _report_by_pair(
    df: pd.DataFrame,
    png_dir: Path,
    report_lines: list[str],
) -> None:
    report_lines.append("## By (Dataset, Data Variable)\n")
    pair_stats: list[dict[str, Any]] = []
    for (dataset_id, var_name), sub in df.groupby(["dataset_id", "data_var"]):
        values = _expand_histogram(sub)
        stats = _compute_stats(values)
        labeled = {"facet": f"{dataset_id} / {var_name}", **stats}
        pair_stats.append(labeled)
        _print_stats(f"{dataset_id} / {var_name}", stats)
        png_name = f"pair_{dataset_id}_{var_name}.png"
        _plot_histogram(values, f"{dataset_id} / {var_name}", png_dir / png_name, stats)

    report_lines.append(_stats_table(pair_stats))
    report_lines.append("\n### Individual Histograms\n")
    for (dataset_id, var_name), _ in df.groupby(["dataset_id", "data_var"]):
        png_name = f"pair_{dataset_id}_{var_name}.png"
        report_lines.append(f"#### {dataset_id} / {var_name}\n")
        report_lines.append(f"![{dataset_id}/{var_name}]({png_name})\n")


@app.command()
def report(
    parquet_path: Path = typer.Argument(..., help="Path to the histogram parquet file"),
    output_dir: Path = typer.Option(
        Path("reports/compression_audit"), "--output-dir", "-o"
    ),
) -> None:
    """Generate statistics, plots, and a markdown report from collected data."""
    df = pd.read_parquet(parquet_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = ["# Compression Audit Report\n"]
    _report_overall(df, output_dir, report_lines)
    _report_by_dataset(df, output_dir, report_lines)
    _report_by_var(df, output_dir, report_lines)
    _report_by_pair(df, output_dir, report_lines)

    report_path = output_dir / "compression_audit.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport written to {report_path}")


@app.command("all")
def collect_and_report(
    output_dir: Path = typer.Option(
        Path("reports/compression_audit"), "--output-dir", "-o"
    ),
    dataset_ids: list[str] | None = typer.Option(None, "--dataset-id", "-d"),
) -> None:
    """Collect data and generate report in one step."""
    parquet_path = collect(dataset_ids=dataset_ids, output_dir=output_dir)
    report(parquet_path=parquet_path, output_dir=output_dir)


if __name__ == "__main__":
    app()
