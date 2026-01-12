#!/usr/bin/env python3
# ruff: noqa: T201, PLR0912, PLR0915
"""Zarr V3 chunk and shard layout diagnostic tool for geospatial datasets.

This tool calculates, optimizes, and audits storage layouts for large-scale
geospatial datasets stored in Zarr V3 format.

Usage:
    # Analysis mode with search
    uv run python src/scripts/chunk_shard_size.py \
        --time 8760:1:hour --latitude 721:0.25:degrees --longitude 1440:0.25:degrees --search

    # Forecast mode with manual shapes
    uv run python src/scripts/chunk_shard_size.py \
        --init_time 365:24:hour --ensemble_member 31 --lead_time 181:1:hour \
        --latitude 721:0.25:degrees --longitude 1440:0.25:degrees \
        --chunk_shape 1,31,64,17,16 --shard_shape 1,31,192,374,368
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Literal, get_args

VALID_DIMS = get_args(
    Literal[
        "time",
        "init_time",
        "ensemble_member",
        "lead_time",
        "latitude",
        "longitude",
        "x",
        "y",
        "statistic",
    ]
)

# Storage constants (defaults, can be overridden via CLI)
DEFAULT_BYTES_PER_ELEMENT = 4  # float32
DEFAULT_COMPRESSION_RATIO = 0.2  # 20% of raw size after compression

# Mutable storage params container (set from CLI args in main)
STORAGE_PARAMS: dict[str, int | float] = {
    "bytes_per_element": DEFAULT_BYTES_PER_ELEMENT,
    "compression_ratio": DEFAULT_COMPRESSION_RATIO,
}

# Target sizes in bytes (compressed)
CHUNK_MIN_MB = 2.5
CHUNK_MAX_MB = 6.0
CHUNK_TARGET_MB = 3.5
SHARD_MIN_MB = 100.0
SHARD_MAX_MB = 600.0

# Convert to bytes
CHUNK_MIN_BYTES = CHUNK_MIN_MB * 1024 * 1024
CHUNK_MAX_BYTES = CHUNK_MAX_MB * 1024 * 1024
CHUNK_TARGET_BYTES = CHUNK_TARGET_MB * 1024 * 1024
SHARD_MIN_BYTES = SHARD_MIN_MB * 1024 * 1024
SHARD_MAX_BYTES = SHARD_MAX_MB * 1024 * 1024


@dataclass
class DimensionSpec:
    """Specification for a single dimension."""

    name: str
    length: int
    step: float
    units: str

    @classmethod
    def from_string(cls, name: str, spec: str) -> DimensionSpec:
        """Parse dimension spec from CLI string format 'length:step:units' or 'length'."""
        parts = spec.split(":")
        if len(parts) == 1:
            return cls(name=name, length=int(parts[0]), step=1.0, units="values")
        if len(parts) == 3:
            return cls(
                name=name, length=int(parts[0]), step=float(parts[1]), units=parts[2]
            )
        raise ValueError(
            f"Invalid dimension spec '{spec}'. Expected 'length' or 'length:step:units'"
        )


@dataclass
class LayoutConfig:
    """Configuration for chunk and shard layout."""

    chunk_shape: tuple[int, ...]
    shard_shape: tuple[int, ...]
    dim_names: tuple[str, ...]
    dim_specs: tuple[DimensionSpec, ...]

    def __post_init__(self) -> None:
        # Validate shards are multiples of chunks
        for chunk, shard, dim in zip(
            self.chunk_shape, self.shard_shape, self.dim_names, strict=True
        ):
            if shard % chunk != 0:
                raise ValueError(
                    f"Shard size ({shard}) must be a multiple of chunk size ({chunk}) "
                    f"for dimension '{dim}'"
                )


@dataclass
class StorageMetrics:
    """Storage metrics for chunks and shards."""

    raw_bytes: int
    compressed_bytes: float

    @property
    def raw_mb(self) -> float:
        return self.raw_bytes / (1024 * 1024)

    @property
    def compressed_mb(self) -> float:
        return self.compressed_bytes / (1024 * 1024)


def calculate_storage(num_elements: int) -> StorageMetrics:
    """Calculate raw and compressed storage for given number of elements."""
    bytes_per_element = int(STORAGE_PARAMS["bytes_per_element"])
    compression_ratio = float(STORAGE_PARAMS["compression_ratio"])
    raw_bytes = num_elements * bytes_per_element
    compressed_bytes = raw_bytes * compression_ratio
    return StorageMetrics(raw_bytes, compressed_bytes)


def product(values: tuple[int, ...]) -> int:
    """Calculate product of tuple values."""
    result = 1
    for v in values:
        result *= v
    return result


def get_divisors(n: int) -> list[int]:
    """Get all divisors of n."""
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def count_chunks_for_dim(dim_length: int, chunk_size: int) -> int:
    """Count number of chunks needed to cover a dimension."""
    return math.ceil(dim_length / chunk_size)


def count_shards_for_dim(dim_length: int, shard_size: int) -> int:
    """Count number of shards needed to cover a dimension."""
    return math.ceil(dim_length / shard_size)


def count_total_chunks(
    dim_lengths: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> int:
    """Count total number of chunks in the dataset."""
    total = 1
    for length, chunk in zip(dim_lengths, chunk_shape, strict=True):
        total *= count_chunks_for_dim(length, chunk)
    return total


def count_total_shards(
    dim_lengths: tuple[int, ...], shard_shape: tuple[int, ...]
) -> int:
    """Count total number of shards in the dataset."""
    total = 1
    for length, shard in zip(dim_lengths, shard_shape, strict=True):
        total *= count_shards_for_dim(length, shard)
    return total


def format_physical_span(
    shape: tuple[int, ...], dim_specs: tuple[DimensionSpec, ...]
) -> str:
    """Format the physical span covered by a chunk or shard."""
    parts = []
    for size, spec in zip(shape, dim_specs, strict=True):
        span = size * spec.step
        if spec.units == "degrees":
            parts.append(f"{span:.2f}°")
        elif spec.units in ("hour", "hours"):
            if span >= 24:
                days = span / 24
                parts.append(f"{days:.1f} days")
            else:
                parts.append(f"{span:.0f}h")
        elif spec.units == "values":
            parts.append(f"{size}")
        else:
            parts.append(f"{span:.2f} {spec.units}")
    return " x ".join(parts)


def detect_mode(
    dim_names: tuple[str, ...],
) -> Literal["analysis", "forecast"]:
    """Detect optimization mode from dimension names."""
    has_init_time = "init_time" in dim_names
    has_lead_time = "lead_time" in dim_names
    has_time = "time" in dim_names

    if has_init_time and has_lead_time:
        return "forecast"
    if has_time and not (has_init_time and has_lead_time):
        return "analysis"

    raise ValueError(
        "Cannot determine mode. Provide either 'time' for Analysis Mode, "
        "or both 'init_time' AND 'lead_time' for Forecast Mode."
    )


def get_spatial_dims(dim_names: tuple[str, ...]) -> list[str]:
    """Get the spatial dimension names."""
    return [name for name in dim_names if name in ("latitude", "longitude", "x", "y")]


def get_temporal_dims(dim_names: tuple[str, ...]) -> list[str]:
    """Get the temporal dimension names."""
    return [name for name in dim_names if name in ("time", "init_time", "lead_time")]


def calculate_spatial_evenness(
    dim_lengths: tuple[int, ...],
    shape: tuple[int, ...],
    dim_names: tuple[str, ...],
) -> float:
    """Calculate how evenly shards/chunks divide the spatial domain.

    Returns a score from 0 to 1, where 1 is perfectly even (all same size)
    and lower values indicate more variance.
    """
    spatial_dims = get_spatial_dims(dim_names)
    if not spatial_dims:
        return 1.0

    variances = []
    for name in spatial_dims:
        idx = dim_names.index(name)
        dim_len = dim_lengths[idx]
        size = shape[idx]

        num_full = dim_len // size
        remainder = dim_len % size

        if remainder == 0:
            variances.append(0.0)
        else:
            # Calculate coefficient of variation
            sizes = [size] * num_full + ([remainder] if remainder else [])
            if len(sizes) > 1:
                mean = sum(sizes) / len(sizes)
                variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)
                cv = (variance**0.5) / mean if mean > 0 else 0
                variances.append(cv)
            else:
                variances.append(0.0)

    if not variances:
        return 1.0

    avg_cv = sum(variances) / len(variances)
    return max(0.0, 1.0 - avg_cv)


def calculate_spatial_squareness(
    shape: tuple[int, ...],
    dim_names: tuple[str, ...],
) -> float:
    """Calculate how 'square' the spatial dimensions are.

    Returns a score from 0 to 1, where 1 means all spatial dims have equal size
    (e.g., 32x32) and lower values indicate more rectangular shapes (e.g., 32x64).
    """
    spatial_dims = get_spatial_dims(dim_names)
    if len(spatial_dims) < 2:
        return 1.0

    spatial_sizes = [shape[dim_names.index(name)] for name in spatial_dims]

    # Calculate ratio of min to max - 1.0 means perfectly square
    min_size = min(spatial_sizes)
    max_size = max(spatial_sizes)

    if max_size == 0:
        return 1.0

    return min_size / max_size


def search_chunk_shapes(
    dim_specs: tuple[DimensionSpec, ...],
    mode: Literal["analysis", "forecast"],
) -> list[tuple[int, ...]]:
    """Search for valid chunk shapes meeting size constraints."""
    dim_names = tuple(spec.name for spec in dim_specs)
    dim_lengths = tuple(spec.length for spec in dim_specs)

    # Generate candidate divisors for each dimension
    divisor_lists = []
    for spec in dim_specs:
        divisors = get_divisors(spec.length)
        # Add some non-divisor options for flexibility
        additional = [1, 8, 16, 17, 32, 64, 128, 256]
        for a in additional:
            if a not in divisors and a <= spec.length:
                divisors.append(a)
        divisors = sorted(set(divisors))
        divisor_lists.append(divisors)

    # Apply mode-specific constraints to filter divisors
    constrained_divisors = []
    for spec, divisors in zip(dim_specs, divisor_lists, strict=True):
        filtered = divisors.copy()

        if mode == "analysis":
            # Temporal chunk should not exceed 1 year
            if spec.name == "time":
                if spec.units in ("hour", "hours"):
                    max_time_chunk = (
                        365 * 24 // max(1, int(spec.step))
                    )  # 1 year in steps
                else:
                    max_time_chunk = 365  # Assume daily if units unclear
                filtered = [d for d in divisors if d <= max_time_chunk]

            # Spatial dims should be small for analysis mode
            if spec.name in ("latitude", "longitude", "x", "y"):
                filtered = [d for d in divisors if d <= 64]

        elif mode == "forecast":
            # For forecast, 1 init_time per chunk
            if spec.name == "init_time":
                filtered = [1]

            # Try to keep all ensemble members together
            if spec.name == "ensemble_member":
                filtered = [d for d in divisors if d == spec.length]
                if not filtered:
                    filtered = [d for d in divisors if d >= spec.length // 2]

            # For lead_time, prefer larger chunks
            if spec.name == "lead_time":
                filtered = [d for d in divisors if d >= 32 or d == spec.length]

            # Spatial dims can be larger for forecast mode
            if spec.name in ("latitude", "longitude", "x", "y"):
                filtered = [d for d in divisors if 8 <= d <= 64]
                if not filtered:
                    filtered = [d for d in divisors if d <= 128]

        constrained_divisors.append(filtered if filtered else divisors[:10])

    # Search for valid combinations
    valid_shapes: list[tuple[int, ...]] = []

    # Use iterative approach to avoid combinatorial explosion
    def search_recursive(current_shape: list[int], dim_idx: int) -> None:
        if dim_idx == len(dim_specs):
            shape = tuple(current_shape)
            storage = calculate_storage(product(shape))
            if CHUNK_MIN_BYTES <= storage.compressed_bytes <= CHUNK_MAX_BYTES:
                valid_shapes.append(shape)
            return

        for divisor in constrained_divisors[dim_idx]:
            current_shape.append(divisor)
            # Prune: check if current product already exceeds max
            current_elements = product(tuple(current_shape))
            min_remaining_elements = 1
            for j in range(dim_idx + 1, len(dim_specs)):
                min_remaining_elements *= min(constrained_divisors[j])
            total_min = current_elements * min_remaining_elements
            storage_min = calculate_storage(total_min)

            if storage_min.compressed_bytes <= CHUNK_MAX_BYTES:
                search_recursive(current_shape, dim_idx + 1)

            current_shape.pop()

    search_recursive([], 0)

    # Score and sort by distance from target size, spatial evenness, and squareness
    def score_shape(shape: tuple[int, ...]) -> float:
        storage = calculate_storage(product(shape))
        size_diff = abs(storage.compressed_bytes - CHUNK_TARGET_BYTES)
        size_score = 1.0 / (1.0 + size_diff / CHUNK_TARGET_BYTES)

        evenness = calculate_spatial_evenness(dim_lengths, shape, dim_names)
        squareness = calculate_spatial_squareness(shape, dim_names)

        # Weight: 30% size score, 40% evenness, 30% squareness
        return 0.3 * size_score + 0.4 * evenness + 0.3 * squareness

    valid_shapes.sort(key=score_shape, reverse=True)

    return valid_shapes[:20]  # Return top 20 candidates


def search_shard_shapes(
    chunk_shape: tuple[int, ...],
    dim_specs: tuple[DimensionSpec, ...],
    mode: Literal["analysis", "forecast"],
) -> list[tuple[int, ...]]:
    """Search for valid shard shapes that are multiples of chunk shape."""
    dim_names = tuple(spec.name for spec in dim_specs)
    dim_lengths = tuple(spec.length for spec in dim_specs)

    # Generate multipliers for each dimension
    multiplier_lists = []
    for chunk_size, spec in zip(chunk_shape, dim_specs, strict=True):
        max_multiplier = math.ceil(spec.length / chunk_size)
        multipliers = list(range(1, max_multiplier + 1))

        # Apply mode-specific constraints
        if mode == "forecast":
            if spec.name == "init_time":
                # Keep exactly 1 init_time per shard
                multipliers = [1]
            elif spec.name == "lead_time":
                # Try to fit all lead times in one shard
                all_lead_times_mult = math.ceil(spec.length / chunk_size)
                # Prioritize fitting all, then large portions
                multipliers = [m for m in multipliers if m >= all_lead_times_mult]
                if not multipliers:
                    multipliers = [max_multiplier]
            elif spec.name == "ensemble_member":
                # Keep all ensemble members in one shard
                all_ens_mult = math.ceil(spec.length / chunk_size)
                multipliers = [all_ens_mult]

        multiplier_lists.append(multipliers[:20])  # Limit search space

    # Search for valid combinations
    valid_shapes: list[tuple[int, ...]] = []

    def search_recursive(current_mults: list[int], dim_idx: int) -> None:
        if dim_idx == len(dim_specs):
            shape = tuple(
                m * c for m, c in zip(current_mults, chunk_shape, strict=True)
            )
            storage = calculate_storage(product(shape))
            if SHARD_MIN_BYTES <= storage.compressed_bytes <= SHARD_MAX_BYTES * 2:
                # Allow slightly larger shards in search
                valid_shapes.append(shape)
            return

        for mult in multiplier_lists[dim_idx]:
            current_mults.append(mult)
            # Prune check
            current_shape = tuple(
                m * c
                for m, c in zip(current_mults, chunk_shape[: dim_idx + 1], strict=True)
            )
            current_elements = product(current_shape)
            min_remaining = 1
            for j in range(dim_idx + 1, len(dim_specs)):
                min_remaining *= chunk_shape[j] * min(multiplier_lists[j])
            total_min = current_elements * min_remaining
            storage_min = calculate_storage(total_min)

            if storage_min.compressed_bytes <= SHARD_MAX_BYTES * 2:
                search_recursive(current_mults, dim_idx + 1)

            current_mults.pop()

    search_recursive([], 0)

    # Score by size, spatial evenness, and squareness
    def score_shape(shape: tuple[int, ...]) -> float:
        storage = calculate_storage(product(shape))
        # Prefer shards in the 200-400MB range
        target_shard = (SHARD_MIN_MB + SHARD_MAX_MB) / 2 * 1024 * 1024
        size_diff = abs(storage.compressed_bytes - target_shard)
        size_score = 1.0 / (1.0 + size_diff / target_shard)

        evenness = calculate_spatial_evenness(dim_lengths, shape, dim_names)
        squareness = calculate_spatial_squareness(shape, dim_names)

        # Weight: 25% size score, 45% evenness, 30% squareness
        return 0.25 * size_score + 0.45 * evenness + 0.3 * squareness

    valid_shapes.sort(key=score_shape, reverse=True)

    return valid_shapes[:10]


def calculate_access_costs(
    config: LayoutConfig,
    _mode: Literal["analysis", "forecast"],
) -> dict[str, str]:
    """Calculate access pattern costs."""
    dim_names = config.dim_names
    dim_lengths = tuple(spec.length for spec in config.dim_specs)

    spatial_dims = get_spatial_dims(dim_names)
    temporal_dims = get_temporal_dims(dim_names)

    # Spatial slice: read full spatial domain for single time point
    spatial_slice_chunks = 1
    spatial_slice_shard_elements = 1

    for i, name in enumerate(dim_names):
        if name in spatial_dims:
            spatial_slice_chunks *= count_chunks_for_dim(
                dim_lengths[i], config.chunk_shape[i]
            )
            spatial_slice_shard_elements *= config.shard_shape[i]
        else:
            # Single value in non-spatial dims for slice
            pass

    chunk_storage = calculate_storage(product(config.chunk_shape))
    spatial_slice_mb = spatial_slice_chunks * chunk_storage.compressed_mb

    # Shard touch: how many shards touched for spatial slice
    shards_for_spatial = 1
    for i, name in enumerate(dim_names):
        if name in spatial_dims:
            shards_for_spatial *= count_shards_for_dim(
                dim_lengths[i], config.shard_shape[i]
            )
        # Non-spatial dims: just 1 shard slice
    shard_storage = calculate_storage(product(config.shard_shape))
    shard_touch_gb = shards_for_spatial * shard_storage.compressed_mb / 1024

    # Full time series: read all temporal data for single spatial pixel
    time_series_shards = 1
    time_series_chunks = 1
    for i, name in enumerate(dim_names):
        if name in temporal_dims or name == "ensemble_member":
            time_series_shards *= count_shards_for_dim(
                dim_lengths[i], config.shard_shape[i]
            )
            time_series_chunks *= count_chunks_for_dim(
                dim_lengths[i], config.chunk_shape[i]
            )

    time_series_mb = time_series_chunks * chunk_storage.compressed_mb

    return {
        "spatial_slice_chunks": str(spatial_slice_chunks),
        "spatial_slice_mb": f"{spatial_slice_mb:.2f}",
        "shard_touch_count": str(shards_for_spatial),
        "shard_touch_gb": f"{shard_touch_gb:.3f}",
        "time_series_shards": str(time_series_shards),
        "time_series_mb": f"{time_series_mb:.2f}",
    }


def generate_template_config_code(config: LayoutConfig) -> str:
    """Generate code block for pasting into template config."""
    chunk_storage = calculate_storage(product(config.chunk_shape))
    shard_storage = calculate_storage(product(config.shard_shape))

    lines = []

    # Chunk comment
    lines.append(
        f"        # ~{chunk_storage.raw_mb:.0f}MB uncompressed, "
        f"~{chunk_storage.compressed_mb:.1f}MB compressed"
    )
    lines.append("        var_chunks: dict[Dim, int] = {")

    # Chunk entries
    for i, spec in enumerate(config.dim_specs):
        chunk_size = config.chunk_shape[i]
        chunks_count = count_chunks_for_dim(spec.length, chunk_size)

        # Generate comment based on dimension type
        if spec.name in ("time", "init_time"):
            if spec.units in ("hour", "hours"):
                days = chunk_size * spec.step / 24
                comment = f"# {days:.0f} days of {spec.step:.0f}-hourly data"
            else:
                comment = f"# {chunk_size} {spec.units}"
        elif spec.name in ("latitude", "longitude", "x", "y"):
            comment = f"# {chunks_count} chunks over {spec.length} pixels"
        elif spec.name == "lead_time":
            if spec.units in ("hour", "hours"):
                days = chunk_size * spec.step / 24
                comment = f"# {days:.1f} days of lead time"
            else:
                comment = f"# {chunk_size} lead time steps"
        elif spec.name == "ensemble_member":
            comment = f"# all {chunk_size} ensemble members"
        else:
            comment = f"# {chunks_count} chunks over {spec.length} values"

        lines.append(f'            "{spec.name}": {chunk_size},  {comment}')

    lines.append("        }")
    lines.append("")

    # Shard comment
    lines.append(
        f"        # ~{shard_storage.raw_mb:.0f}MB uncompressed, "
        f"~{shard_storage.compressed_mb:.0f}MB compressed"
    )
    lines.append("        var_shards: dict[Dim, int] = {")

    # Shard entries
    for i, spec in enumerate(config.dim_specs):
        chunk_size = config.chunk_shape[i]
        shard_size = config.shard_shape[i]
        multiplier = shard_size // chunk_size
        shards_count = count_shards_for_dim(spec.length, shard_size)

        if multiplier == 1:
            value_str = f'var_chunks["{spec.name}"]'
        else:
            value_str = f'var_chunks["{spec.name}"] * {multiplier}'

        comment = f"# {shards_count} shards over {spec.length} pixels"

        lines.append(f'            "{spec.name}": {value_str},  {comment}')

    lines.append("        }")

    return "\n".join(lines)


def print_diagnostic_table(
    config: LayoutConfig,
    mode: Literal["analysis", "forecast"],
) -> None:
    """Print comprehensive diagnostic ASCII table."""
    dim_lengths = tuple(spec.length for spec in config.dim_specs)

    chunk_elements = product(config.chunk_shape)
    shard_elements = product(config.shard_shape)

    chunk_storage = calculate_storage(chunk_elements)
    shard_storage = calculate_storage(shard_elements)

    total_chunks = count_total_chunks(dim_lengths, config.chunk_shape)
    total_shards = count_total_shards(dim_lengths, config.shard_shape)

    chunk_span = format_physical_span(config.chunk_shape, config.dim_specs)
    shard_span = format_physical_span(config.shard_shape, config.dim_specs)

    access_costs = calculate_access_costs(config, mode)

    # Calculate chunks per shard
    chunks_per_shard = 1
    for c, s in zip(config.chunk_shape, config.shard_shape, strict=True):
        chunks_per_shard *= s // c

    # Print header
    print("\n" + "=" * 80)
    print(f" ZARR V3 LAYOUT DIAGNOSTICS - {mode.upper()} MODE")
    print("=" * 80)

    # Grid Layout Section
    print("\n┌" + "─" * 78 + "┐")
    print("│ GRID LAYOUT" + " " * 66 + "│")
    print("├" + "─" * 78 + "┤")

    dim_header = (
        "│ Dimension        │ Length │ Chunk │ Shard │ Chunks/Dim │ Shards/Dim │"
    )
    print(dim_header)
    print(
        "├"
        + "─" * 18
        + "┼"
        + "─" * 8
        + "┼"
        + "─" * 7
        + "┼"
        + "─" * 7
        + "┼"
        + "─" * 12
        + "┼"
        + "─" * 12
        + "┤"
    )

    for i, spec in enumerate(config.dim_specs):
        chunks_dim = count_chunks_for_dim(spec.length, config.chunk_shape[i])
        shards_dim = count_shards_for_dim(spec.length, config.shard_shape[i])
        row = f"│ {spec.name:<16} │ {spec.length:>6} │ {config.chunk_shape[i]:>5} │ {config.shard_shape[i]:>5} │ {chunks_dim:>10} │ {shards_dim:>10} │"
        print(row)

    print(
        "└"
        + "─" * 18
        + "┴"
        + "─" * 8
        + "┴"
        + "─" * 7
        + "┴"
        + "─" * 7
        + "┴"
        + "─" * 12
        + "┴"
        + "─" * 12
        + "┘"
    )

    # Summary Statistics
    print("\n┌" + "─" * 78 + "┐")
    print("│ SUMMARY STATISTICS" + " " * 59 + "│")
    print("├" + "─" * 78 + "┤")
    print(f"│ Total Chunks:        {total_chunks:>12,}" + " " * 43 + "│")
    print(f"│ Total Shards:        {total_shards:>12,}" + " " * 43 + "│")
    print(f"│ Chunks per Shard:    {chunks_per_shard:>12,}" + " " * 43 + "│")
    print("└" + "─" * 78 + "┘")

    # Physical Span
    print("\n┌" + "─" * 78 + "┐")
    print("│ PHYSICAL SPAN (per unit)" + " " * 53 + "│")
    print("├" + "─" * 78 + "┤")
    chunk_span_line = f"│ Chunk: {chunk_span}"
    print(chunk_span_line + " " * (79 - len(chunk_span_line)) + "│")
    shard_span_line = f"│ Shard: {shard_span}"
    print(shard_span_line + " " * (79 - len(shard_span_line)) + "│")
    print("└" + "─" * 78 + "┘")

    # Storage Specs
    print("\n┌" + "─" * 78 + "┐")
    print("│ STORAGE SPECIFICATIONS" + " " * 55 + "│")
    print("├" + "─" * 78 + "┤")
    print(
        f"│ {'':20} │ {'Raw MB':>12} │ {'Compressed MB':>14} │ {'Status':>12} │"
        + " " * 4
        + "│"
    )
    print(
        "├"
        + "─" * 21
        + "┼"
        + "─" * 14
        + "┼"
        + "─" * 16
        + "┼"
        + "─" * 14
        + "┼"
        + "─" * 5
        + "┤"
    )

    chunk_status = (
        "✓ OK"
        if CHUNK_MIN_BYTES <= chunk_storage.compressed_bytes <= CHUNK_MAX_BYTES
        else "⚠ CHECK"
    )
    shard_status = (
        "✓ OK"
        if SHARD_MIN_BYTES <= shard_storage.compressed_bytes <= SHARD_MAX_BYTES * 2
        else "⚠ CHECK"
    )

    print(
        f"│ {'Chunk':20} │ {chunk_storage.raw_mb:>12.2f} │ {chunk_storage.compressed_mb:>14.2f} │ {chunk_status:>12} │"
        + " " * 4
        + "│"
    )
    print(
        f"│ {'Shard':20} │ {shard_storage.raw_mb:>12.2f} │ {shard_storage.compressed_mb:>14.2f} │ {shard_status:>12} │"
        + " " * 4
        + "│"
    )
    print(
        "└"
        + "─" * 21
        + "┴"
        + "─" * 14
        + "┴"
        + "─" * 16
        + "┴"
        + "─" * 14
        + "┴"
        + "─" * 5
        + "┘"
    )

    print(
        f"\n  Target chunk: {CHUNK_MIN_MB}-{CHUNK_MAX_MB} MB compressed (sweet spot: {CHUNK_TARGET_MB} MB)"
    )
    print(f"  Target shard: {SHARD_MIN_MB}-{SHARD_MAX_MB}+ MB compressed")

    # Access Pattern Costs
    print("\n┌" + "─" * 78 + "┐")
    print("│ ACCESS PATTERN COSTS" + " " * 57 + "│")
    print("├" + "─" * 78 + "┤")

    print(
        "│ 1. SPATIAL SLICE (full spatial domain, single time point)" + " " * 20 + "│"
    )
    print(
        f"│    Chunks to read:      {access_costs['spatial_slice_chunks']:>10}"
        + " " * 42
        + "│"
    )
    print(
        f"│    Compressed data:     {access_costs['spatial_slice_mb']:>10} MB"
        + " " * 39
        + "│"
    )
    print("│" + " " * 78 + "│")

    print("│ 2. SHARD TOUCH (shards opened for spatial slice)" + " " * 28 + "│")
    print(
        f"│    Shards touched:      {access_costs['shard_touch_count']:>10}"
        + " " * 42
        + "│"
    )
    print(
        f"│    Total shard data:    {access_costs['shard_touch_gb']:>10} GB"
        + " " * 39
        + "│"
    )
    print("│" + " " * 78 + "│")

    print("│ 3. FULL TIME SERIES (all time, single spatial pixel)" + " " * 24 + "│")
    print(
        f"│    Shards to read:      {access_costs['time_series_shards']:>10}"
        + " " * 42
        + "│"
    )
    print(
        f"│    Compressed data:     {access_costs['time_series_mb']:>10} MB"
        + " " * 39
        + "│"
    )

    print("└" + "─" * 78 + "┘")

    # Spatial Evenness
    shard_evenness = calculate_spatial_evenness(
        dim_lengths, config.shard_shape, config.dim_names
    )
    chunk_evenness = calculate_spatial_evenness(
        dim_lengths, config.chunk_shape, config.dim_names
    )

    chunk_squareness = calculate_spatial_squareness(
        config.chunk_shape, config.dim_names
    )
    shard_squareness = calculate_spatial_squareness(
        config.shard_shape, config.dim_names
    )

    print("\n┌" + "─" * 78 + "┐")
    print("│ SPATIAL SCORES (1.0 = optimal)" + " " * 46 + "│")
    print("├" + "─" * 78 + "┤")
    print(
        f"│ Chunk evenness:   {chunk_evenness:.3f}    Chunk squareness: {chunk_squareness:.3f}"
        + " " * 32
        + "│"
    )
    print(
        f"│ Shard evenness:   {shard_evenness:.3f}    Shard squareness: {shard_squareness:.3f}"
        + " " * 32
        + "│"
    )
    print("└" + "─" * 78 + "┘")

    # Template config code block
    print("\n" + "─" * 80)
    print(" TEMPLATE CONFIG CODE (copy-paste into your template_config.py)")
    print("─" * 80)
    print("\n```python")
    print(generate_template_config_code(config))
    print("```")

    print("\n" + "=" * 80 + "\n")


def build_cli_command(
    dim_specs: tuple[DimensionSpec, ...],
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
) -> str:
    """Build the CLI command to run with specified shapes."""
    parts = ["uv run python src/scripts/chunk_shard_size.py"]

    for spec in dim_specs:
        if spec.units == "values" and spec.step == 1.0:
            parts.append(f"--{spec.name} {spec.length}")
        else:
            parts.append(f"--{spec.name} {spec.length}:{spec.step}:{spec.units}")

    parts.append(f"--chunk_shape {','.join(str(c) for c in chunk_shape)}")
    parts.append(f"--shard_shape {','.join(str(s) for s in shard_shape)}")

    return " \\\n    ".join(parts)


def print_search_results(
    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]],
    dim_specs: tuple[DimensionSpec, ...],
    mode: Literal["analysis", "forecast"],
) -> None:
    """Print search results as a comparison table."""
    dim_names = tuple(spec.name for spec in dim_specs)
    dim_lengths = tuple(spec.length for spec in dim_specs)

    print("\n" + "=" * 80)
    print(
        f" SEARCH RESULTS - TOP {len(candidates)} CONFIGURATIONS ({mode.upper()} MODE)"
    )
    print("=" * 80)

    for rank, (chunk_shape, shard_shape) in enumerate(candidates, 1):
        chunk_storage = calculate_storage(product(chunk_shape))
        shard_storage = calculate_storage(product(shard_shape))
        shard_evenness = calculate_spatial_evenness(dim_lengths, shard_shape, dim_names)
        chunk_squareness = calculate_spatial_squareness(chunk_shape, dim_names)
        total_shards = count_total_shards(dim_lengths, shard_shape)

        print(f"\n#{rank}")
        print(
            f"  Chunk: {chunk_shape} → {chunk_storage.compressed_mb:.2f} MB compressed"
        )
        print(
            f"  Shard: {shard_shape} → {shard_storage.compressed_mb:.2f} MB compressed"
        )
        print(
            f"  Total shards: {total_shards:,} | Evenness: {shard_evenness:.3f} | Squareness: {chunk_squareness:.3f}"
        )

        if rank == 1:
            print(
                "\n  [RECOMMENDED - Best balance of size, evenness, and spatial squareness]"
            )

    # Print command to run top recommendation
    if candidates:
        best_chunk, best_shard = candidates[0]
        print("\n" + "-" * 80)
        print("Run this command to see detailed analysis of top recommendation:\n")
        print(build_cli_command(dim_specs, best_chunk, best_shard))

    print("\n" + "=" * 80 + "\n")


def parse_shape(shape_str: str, num_dims: int) -> tuple[int, ...]:
    """Parse comma-separated shape string into tuple."""
    parts = shape_str.split(",")
    if len(parts) != num_dims:
        raise ValueError(f"Shape must have {num_dims} values, got {len(parts)}")
    return tuple(int(p.strip()) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zarr V3 chunk and shard layout diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analysis mode with search
  python chunk_shard_size.py --time 8760:1:hour --latitude 721:0.25:degrees --longitude 1440:0.25:degrees --search

  # Forecast mode with search
  python chunk_shard_size.py --init_time 365 --ensemble_member 31 --lead_time 181:3:hour --latitude 721:0.25:degrees --longitude 1440:0.25:degrees --search

  # Manual configuration
  python chunk_shard_size.py --time 8760:1:hour --latitude 721:0.25:degrees --longitude 1440:0.25:degrees --chunk_shape 1440,32,32 --shard_shape 2880,384,384
        """,
    )

    # Dimension arguments
    for dim in VALID_DIMS:
        parser.add_argument(
            f"--{dim}",
            type=str,
            help=f"Spec for {dim} dimension: 'length:step:units' or 'length'",
        )

    # Shape arguments
    parser.add_argument(
        "--chunk_shape",
        type=str,
        help="Manual chunk shape as comma-separated values",
    )
    parser.add_argument(
        "--shard_shape",
        type=str,
        help="Manual shard shape as comma-separated values",
    )

    # Search flag
    parser.add_argument(
        "--search",
        action="store_true",
        help="Search for optimal chunk/shard configurations",
    )

    # Optional: number of search results
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top configurations to show in search mode (default: 5)",
    )

    # Storage parameters
    parser.add_argument(
        "--bytes_per_element",
        type=int,
        default=DEFAULT_BYTES_PER_ELEMENT,
        help=f"Bytes per array element (default: {DEFAULT_BYTES_PER_ELEMENT} for float32)",
    )
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=DEFAULT_COMPRESSION_RATIO,
        help=f"Expected compression ratio (default: {DEFAULT_COMPRESSION_RATIO} = {DEFAULT_COMPRESSION_RATIO * 100:.0f}%%)",
    )

    args = parser.parse_args()

    # Set storage parameters from CLI args
    STORAGE_PARAMS["bytes_per_element"] = args.bytes_per_element
    STORAGE_PARAMS["compression_ratio"] = args.compression_ratio

    # Collect dimension specs
    dim_specs: list[DimensionSpec] = []
    for dim in VALID_DIMS:
        value = getattr(args, dim)
        if value:
            dim_specs.append(DimensionSpec.from_string(dim, value))

    if not dim_specs:
        parser.error("At least one dimension must be specified")

    dim_names = tuple(spec.name for spec in dim_specs)

    # Detect mode
    try:
        mode = detect_mode(dim_names)
    except ValueError as e:
        parser.error(str(e))

    dim_specs_tuple = tuple(dim_specs)

    # Validate arguments
    if args.search:
        if args.chunk_shape or args.shard_shape:
            parser.error("Cannot use --search with --chunk_shape or --shard_shape")

        print(f"\nSearching for optimal layouts ({mode} mode)...")
        print(f"Dimensions: {dim_names}")
        print(f"Lengths: {tuple(s.length for s in dim_specs)}")

        # Search for chunk shapes
        chunk_candidates = search_chunk_shapes(dim_specs_tuple, mode)

        if not chunk_candidates:
            print("\nNo valid chunk configurations found within constraints.")
            print("Try adjusting dimension sizes or relaxing constraints.")
            sys.exit(1)

        # For each chunk shape, find matching shard shapes
        results: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for chunk_shape in chunk_candidates[:10]:  # Limit chunk candidates
            shard_candidates = search_shard_shapes(chunk_shape, dim_specs_tuple, mode)
            results.extend(
                (chunk_shape, shard_shape) for shard_shape in shard_candidates[:3]
            )

        if not results:
            print("\nNo valid shard configurations found for chunk candidates.")
            sys.exit(1)

        # Sort by combined score
        dim_lengths = tuple(spec.length for spec in dim_specs)

        def combined_score(pair: tuple[tuple[int, ...], tuple[int, ...]]) -> float:
            chunk_shape, shard_shape = pair
            chunk_storage = calculate_storage(product(chunk_shape))
            shard_storage = calculate_storage(product(shard_shape))

            chunk_size_score = 1.0 / (
                1.0
                + abs(chunk_storage.compressed_bytes - CHUNK_TARGET_BYTES)
                / CHUNK_TARGET_BYTES
            )
            shard_target = (SHARD_MIN_MB + SHARD_MAX_MB) / 2 * 1024 * 1024
            shard_size_score = 1.0 / (
                1.0 + abs(shard_storage.compressed_bytes - shard_target) / shard_target
            )

            shard_evenness = calculate_spatial_evenness(
                dim_lengths, shard_shape, dim_names
            )
            chunk_squareness = calculate_spatial_squareness(chunk_shape, dim_names)
            shard_squareness = calculate_spatial_squareness(shard_shape, dim_names)

            return (
                0.15 * chunk_size_score
                + 0.15 * shard_size_score
                + 0.35 * shard_evenness
                + 0.175 * chunk_squareness
                + 0.175 * shard_squareness
            )

        results.sort(key=combined_score, reverse=True)

        print_search_results(results[: args.top], dim_specs_tuple, mode)

        # Show detailed diagnostics for top result
        if results:
            print("\n" + "=" * 80)
            print(" DETAILED DIAGNOSTICS FOR TOP RECOMMENDATION")
            print("=" * 80)
            best_chunk, best_shard = results[0]
            config = LayoutConfig(
                chunk_shape=best_chunk,
                shard_shape=best_shard,
                dim_names=dim_names,
                dim_specs=dim_specs_tuple,
            )
            print_diagnostic_table(config, mode)

    else:
        # Manual mode
        if not args.chunk_shape or not args.shard_shape:
            parser.error(
                "Either --search or both --chunk_shape and --shard_shape required"
            )

        num_dims = len(dim_specs)
        try:
            chunk_shape = parse_shape(args.chunk_shape, num_dims)
            shard_shape = parse_shape(args.shard_shape, num_dims)
        except ValueError as e:
            parser.error(str(e))

        try:
            config = LayoutConfig(
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                dim_names=dim_names,
                dim_specs=dim_specs_tuple,
            )
        except ValueError as e:
            parser.error(str(e))

        print_diagnostic_table(config, mode)


if __name__ == "__main__":
    main()
