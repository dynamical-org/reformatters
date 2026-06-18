#!/usr/bin/env python3
"""Benchmark candidate compressors on real, already binary-rounded data chunks.

We sample random inner zarr *chunks* (the per-chunk compression unit) directly
from the public Source Coop Zarr v3 copies of two production datasets, then
re-compress the values - which are already lossily binary-rounded on write, see
reformatters.common.binary_rounding - with several lossless codecs. For every
sampled chunk we record the compression ratio plus compress and decompress
throughput, so the full per-chunk distributions can be inspected, not just means.

The baseline ``blosc-zstd3-shuffle`` codec is exactly what these datasets ship
with today (reformatters.common.zarr.BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE); the other
codecs are candidates for squeezing more out of the already-rounded data.

The candidate compressors are not project dependencies, so run with:

    uv run --with openzl --with pcodec --with zfpy \
        python src/scripts/compression_benchmark.py --n-chunks 100

Outputs (under data/compression_benchmark/):
    results.csv   one row per (dataset, variable, chunk, compressor)
    summary.csv   aggregates per compressor and per compressor x variable
    *.png         distribution box plots
"""

from __future__ import annotations

import argparse
import functools
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numcodecs
import numcodecs.abc
import numpy as np
import openzl.ext as zl  # ty: ignore[unresolved-import]
import pandas as pd
import zarr
import zarr.storage
import zfpy  # ty: ignore[unresolved-import]
from pcodec import ChunkConfig, standalone  # ty: ignore[unresolved-import]

from reformatters.common.logging import get_logger
from reformatters.common.zarr import BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE

log = get_logger(__name__)

SOURCE_COOP_BASE = "s3://us-west-2.opendata.source.coop/dynamical"
VERSION = "0.1.0"

# Friendly name -> dataset_id (the production Source Coop Zarr v3 stores).
DATASETS: dict[str, str] = {
    "gfs-analysis": "noaa-gfs-analysis",
    "ifs-ens": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
}

VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "precipitation_surface",
    "wind_u_10m",
    "pressure_reduced_to_mean_sea_level",  # mslp
)

OUTPUT_DIR = Path("data/compression_benchmark")

NDArrayF32 = np.ndarray
T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Compressors
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Compressor:
    name: str
    # encode: chunk array -> compressed bytes
    encode: Callable[[NDArrayF32], bytes]
    # decode: (compressed bytes, original chunk array) -> reconstructed array
    decode: Callable[[bytes, NDArrayF32], NDArrayF32]


def _build_baseline_blosc() -> numcodecs.abc.Codec:
    """The exact codec these datasets ship with today; assert it hasn't drifted."""
    config = BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE["configuration"]
    assert config == {
        "typesize": 4,
        "cname": "zstd",
        "clevel": 3,
        "shuffle": "shuffle",
        "blocksize": 0,
    }, f"baseline codec drifted: {config}"
    # numcodecs.Blosc infers typesize from the float32 array itemsize (4).
    return numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)


_BLOSC = _build_baseline_blosc()


def _blosc_encode(a: NDArrayF32) -> bytes:
    return bytes(_BLOSC.encode(np.ascontiguousarray(a)))


def _blosc_decode(buf: bytes, template: NDArrayF32) -> NDArrayF32:
    return np.frombuffer(_BLOSC.decode(buf), dtype=template.dtype).reshape(
        template.shape
    )


def _pcodec_encode(a: NDArrayF32) -> bytes:
    return standalone.simple_compress(np.ascontiguousarray(a).ravel(), ChunkConfig())


def _pcodec_decode(buf: bytes, template: NDArrayF32) -> NDArrayF32:
    return standalone.simple_decompress(buf).reshape(template.shape)


def _zfp_encode(a: NDArrayF32) -> bytes:
    # zfp supports <=4 dims; chunks carry a leading length-1 init/time dim we squeeze.
    return zfpy.compress_numpy(np.ascontiguousarray(np.squeeze(a)))


def _zfp_decode(buf: bytes, template: NDArrayF32) -> NDArrayF32:
    return zfpy.decompress_numpy(buf).reshape(template.shape)


def _build_openzl_compressor() -> zl.Compressor:
    compressor = zl.Compressor()
    compressor.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    # graphs.Compress is OpenZL's generic, format-aware backend; fed a typed
    # Numeric input it applies its numeric pipeline (transpose/delta/entropy).
    compressor.select_starting_graph(zl.graphs.Compress()(compressor))
    return compressor


_OPENZL = _build_openzl_compressor()


def _openzl_encode(a: NDArrayF32) -> bytes:
    cctx = zl.CCtx()
    cctx.ref_compressor(_OPENZL)
    return cctx.compress([zl.Input(zl.Type.Numeric, np.ascontiguousarray(a).ravel())])


def _openzl_decode(buf: bytes, template: NDArrayF32) -> NDArrayF32:
    output = zl.DCtx().decompress(buf)[0]
    return np.frombuffer(output.content.as_bytes(), dtype=template.dtype).reshape(
        template.shape
    )


COMPRESSORS: tuple[Compressor, ...] = (
    Compressor("blosc-zstd3-shuffle", _blosc_encode, _blosc_decode),
    Compressor("pcodec", _pcodec_encode, _pcodec_decode),
    Compressor("zfp-reversible", _zfp_encode, _zfp_decode),
    Compressor("openzl-numeric", _openzl_encode, _openzl_decode),
)


# --------------------------------------------------------------------------- #
# Chunk sampling and reading
# --------------------------------------------------------------------------- #
def _store_url(dataset_id: str) -> str:
    return f"{SOURCE_COOP_BASE}/{dataset_id}/v{VERSION}.zarr"


def open_array(dataset_id: str, variable: str) -> zarr.Array:
    store = zarr.storage.FsspecStore.from_url(
        _store_url(dataset_id), storage_options={"anon": True}, read_only=True
    )
    array = zarr.open_group(store, mode="r")[variable]
    assert isinstance(array, zarr.Array)
    return array


def sample_full_chunk_starts(
    shape: Sequence[int],
    chunks: Sequence[int],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Random start indices of distinct full-size interior chunks (no partial edges)."""
    per_dim_starts = [
        [i * chunk for i in range(max(length // chunk, 1))]
        for length, chunk in zip(shape, chunks, strict=True)
    ]
    sizes = [len(starts) for starts in per_dim_starts]
    total = math.prod(sizes)
    n = min(n, total)
    flat_indices = rng.choice(total, size=n, replace=False)

    starts: list[tuple[int, ...]] = []
    for flat in flat_indices:
        remainder = int(flat)
        dim_indices = []
        for size in reversed(sizes):
            remainder, position = divmod(remainder, size)
            dim_indices.append(position)
        dim_indices.reverse()
        starts.append(
            tuple(per_dim_starts[d][dim_indices[d]] for d in range(len(sizes)))
        )
    return starts


def read_chunk(
    array: zarr.Array, start: Sequence[int], chunks: Sequence[int]
) -> NDArrayF32:
    selection = tuple(slice(s, s + c) for s, c in zip(start, chunks, strict=True))
    return np.ascontiguousarray(array[selection])


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def _best_time(fn: Callable[[], T], repeats: int) -> tuple[T, float]:
    """Run fn `repeats` times; return its result and the fastest elapsed seconds."""
    start = time.perf_counter()
    result = fn()
    best = time.perf_counter() - start
    for _ in range(repeats - 1):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return result, best


def benchmark_chunk(
    dataset: str,
    variable: str,
    chunk_id: str,
    data: NDArrayF32,
    repeats: int,
) -> list[dict[str, object]]:
    uncompressed_bytes = data.nbytes
    megabytes = uncompressed_bytes / 1e6
    nan_fraction = float(np.isnan(data).mean())
    flat = data.ravel()

    rows: list[dict[str, object]] = []
    for compressor in COMPRESSORS:
        compressed, compress_s = _best_time(
            lambda comp=compressor, d=data: comp.encode(d), repeats
        )
        compressed_bytes = len(compressed)
        decoded, decompress_s = _best_time(
            lambda comp=compressor, b=compressed, d=data: comp.decode(b, d),
            repeats,
        )
        assert np.array_equal(flat, np.asarray(decoded).ravel(), equal_nan=True), (
            f"{compressor.name} not lossless on {dataset}/{variable}/{chunk_id}"
        )
        rows.append(
            {
                "dataset": dataset,
                "variable": variable,
                "chunk_id": chunk_id,
                "compressor": compressor.name,
                "uncompressed_bytes": uncompressed_bytes,
                "compressed_bytes": compressed_bytes,
                "ratio": uncompressed_bytes / compressed_bytes,
                "compress_s": compress_s,
                "decompress_s": decompress_s,
                "compress_mbps": megabytes / compress_s,
                "decompress_mbps": megabytes / decompress_s,
                "nan_fraction": nan_fraction,
            }
        )
    return rows


def _batches(items: Sequence[T], size: int) -> list[Sequence[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    results_path = OUTPUT_DIR / "results.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    header_written = False
    for dataset in args.datasets:
        dataset_id = DATASETS[dataset]
        for variable in args.variables:
            array = open_array(dataset_id, variable)
            chunks = array.chunks
            starts = sample_full_chunk_starts(array.shape, chunks, args.n_chunks, rng)
            log.info(
                f"{dataset}/{variable}: chunk shape {chunks}, "
                f"sampling {len(starts)} of {array.shape} grid"
            )

            read = functools.partial(read_chunk, array, chunks=chunks)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for batch in _batches(starts, args.batch_size):
                    arrays = list(pool.map(read, batch))
                    batch_rows: list[dict[str, object]] = []
                    for start, data in zip(batch, arrays, strict=True):
                        chunk_id = "_".join(str(s) for s in start)
                        batch_rows.extend(
                            benchmark_chunk(
                                dataset, variable, chunk_id, data, args.repeats
                            )
                        )
                    pd.DataFrame(batch_rows).to_csv(
                        results_path,
                        mode="a",
                        header=not header_written,
                        index=False,
                    )
                    header_written = True
                    all_rows.extend(batch_rows)
                    done = len({r["chunk_id"] for r in batch_rows})
                    log.info(f"  {dataset}/{variable}: +{done} chunks")

    df = pd.DataFrame(all_rows)
    log.info(f"Wrote {len(df)} rows to {results_path}")
    return df


# --------------------------------------------------------------------------- #
# Summary and plots
# --------------------------------------------------------------------------- #
STATS: tuple[tuple[str, str], ...] = (
    ("ratio", "compression ratio (higher = smaller)"),
    ("compress_mbps", "compress throughput (MB/s)"),
    ("decompress_mbps", "decompress throughput (MB/s)"),
)


# Distinct, fixed color per compressor, reused across every plot.
_PALETTE: tuple[str, ...] = ("#4c72b0", "#dd8452", "#55a868", "#c44e52")


def compressor_colors() -> dict[str, str]:
    return {c.name: _PALETTE[i] for i, c in enumerate(COMPRESSORS)}


def write_summary(df: pd.DataFrame) -> None:
    metrics = ["ratio", "compress_mbps", "decompress_mbps"]
    aggs = ["mean", "median", "std", "min", "max"]
    overall = df.groupby("compressor")[metrics].agg(aggs)
    by_variable = df.groupby(["variable", "compressor"])[metrics].agg(aggs)
    summary_path = OUTPUT_DIR / "summary.csv"
    with summary_path.open("w") as f:
        f.write("# Overall (pooled across datasets and variables)\n")
        overall.to_csv(f)
        f.write("\n# By variable (pooled across datasets)\n")
        by_variable.to_csv(f)
    log.info(f"Wrote summary to {summary_path}")

    medians = df.groupby("compressor")[metrics].median()
    log.info(f"Median per compressor (pooled):\n{medians.to_string()}")


def _style_boxes(box: dict[str, Any], colors: Sequence[str]) -> None:
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for median in box["medians"]:
        median.set_color("black")


def plot_overall(df: pd.DataFrame) -> Path:
    colors = compressor_colors()
    names = [c.name for c in COMPRESSORS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, label) in zip(axes, STATS, strict=True):
        data = [df.loc[df["compressor"] == name, metric].to_numpy() for name in names]
        box = ax.boxplot(data, patch_artist=True, showfliers=False, tick_labels=names)
        _style_boxes(box, [colors[name] for name in names])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Compressor benchmark on already-rounded chunks "
        "(pooled across datasets + variables)"
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "box_overall.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info(f"Wrote {path}")
    return path


def plot_by_variable(df: pd.DataFrame) -> Path:
    colors = compressor_colors()
    names = [c.name for c in COMPRESSORS]
    variables = [v for v in VARIABLES if v in set(df["variable"])]
    n_comp = len(names)
    group_width = n_comp + 1

    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    for ax, (metric, label) in zip(axes, STATS, strict=True):
        for j, name in enumerate(names):
            positions = [v_i * group_width + j for v_i in range(len(variables))]
            data = [
                df.loc[
                    (df["variable"] == var) & (df["compressor"] == name), metric
                ].to_numpy()
                for var in variables
            ]
            box = ax.boxplot(
                data,
                positions=positions,
                widths=0.8,
                patch_artist=True,
                showfliers=False,
            )
            _style_boxes(box, [colors[name]] * len(variables))
        ax.set_xticks(
            [v_i * group_width + (n_comp - 1) / 2 for v_i in range(len(variables))]
        )
        ax.set_xticklabels(variables, rotation=15)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    handles = [mpatches.Patch(color=colors[name], label=name) for name in names]
    axes[0].legend(handles=handles, ncol=len(names), loc="upper right")
    fig.suptitle("Compressor benchmark by variable (pooled across datasets)")
    fig.tight_layout()
    path = OUTPUT_DIR / "box_by_variable.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info(f"Wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-chunks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--variables", nargs="+", choices=list(VARIABLES), default=list(VARIABLES)
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmarking; re-plot from data/compression_benchmark/results.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        df = pd.read_csv(OUTPUT_DIR / "results.csv")
    else:
        df = run_benchmark(args)
    write_summary(df)
    plot_overall(df)
    plot_by_variable(df)


if __name__ == "__main__":
    main()
