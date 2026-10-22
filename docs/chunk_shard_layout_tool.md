# Zarr V3 Chunk & Shard Layout Diagnostic Tool

A Python utility to calculate, optimize, and audit storage layouts for large-scale geospatial datasets stored in Zarr V3 format.

## Overview

This tool helps you:
- **Analyze** existing chunk/shard configurations to understand storage and access costs
- **Search** for optimal chunk/shard layouts based on your dataset dimensions
- **Compare** configurations with detailed diagnostics including access pattern costs

## Quick Start

```bash
# Analysis mode (time-series dataset) - search for optimal layout
uv run src/scripts/chunk_shard_size.py \
  --time 73000:3:hour \
  --latitude 721:0.25:degrees \
  --longitude 1440:0.25:degrees \
  --search

# Forecast mode (init_time + lead_time) - search for optimal layout
uv run src/scripts/chunk_shard_size.py \
  --init_time 365 \
  --ensemble_member 31 \
  --lead_time 181:3:hour \
  --latitude 721:0.25:degrees \
  --longitude 1440:0.25:degrees \
  --search

# Manual configuration - analyze specific chunk/shard shapes
uv run src/scripts/chunk_shard_size.py \
  --time 73000:3:hour \
  --latitude 721:0.25:degrees \
  --longitude 1440:0.25:degrees \
  --chunk_shape 1440,32,32 \
  --shard_shape 2880,384,384
```

## Dimension Input Format

Dimensions are specified using the format: `--dimension_name length:step:units`

| Component | Description | Example |
|-----------|-------------|---------|
| `length` | Number of values in dimension | `721` |
| `step` | Resolution/spacing between values | `0.25` |
| `units` | Units of measurement | `degrees`, `hour`, `values` |

**Shorthand:** If only length is needed, use `--dimension_name length` (defaults to `1:values`).

### Valid Dimensions

The tool validates against these dimension types:
- `time` - Time dimension for analysis datasets
- `init_time` - Forecast initialization time
- `lead_time` - Forecast lead time
- `ensemble_member` - Ensemble member index
- `latitude`, `longitude` - Geographic coordinates
- `x`, `y` - Grid coordinates
- `statistic` - Statistical aggregation dimension

## Operation Modes

### Analysis Mode
Activated when input includes `time` (but not both `init_time` AND `lead_time`).

**Optimization priorities:**
- Optimizes for time-series access
- Temporal chunks limited to ≤1 year
- Smaller spatial chunks (e.g., 32×32)

### Forecast Mode
Activated when input includes both `init_time` AND `lead_time`.

**Optimization priorities:**
- Keeps exactly **1 init_time per shard**
- Attempts to fit all lead_times in a single shard
- Keeps all ensemble members together
- Larger spatial chunks than analysis mode

## Layout Heuristics

### Data Specifications (configurable via flags)
- Element size: 4 bytes (default, `--bytes_per_element`)
- Compression ratio: 0.2 (default, `--compression_ratio`)

### Target Sizes (Compressed)
| Type | Min | Max | Sweet Spot |
|------|-----|-----|------------|
| Chunk | 2.5 MB | 6.0 MB | 3.5 MB |
| Shard | 100 MB | 600+ MB | ~350 MB |

### Constraints
- Shards must be exact integer multiples of chunks
- Spatial evenness prioritizes shard boundaries over chunk boundaries
- **Spatial squareness**: Prefers equal-sized spatial dimensions (e.g., 32×32 over 32×64)

## Command Reference

Run `uv run src/scripts/chunk_shard_size.py --help` for all options and defaults.
Shape entries in `--chunk_shape` and `--shard_shape` follow the script's `VALID_DIMS`
order, restricted to the supplied dimensions.

## Output Diagnostics

The tool outputs a comprehensive ASCII table with:

### Grid Layout
- Per-dimension: length, chunk size, shard size, chunks/dim, shards/dim

### Summary Statistics
- Total chunks and shards in the dataset
- Chunks per shard

### Physical Span
- Real-world coverage of a single chunk and shard (e.g., "180 days × 8° × 8°")

### Storage Specifications
- Raw and compressed sizes for chunks and shards
- Status indicators: ✓ OK / ⚠ CHECK

### Access Pattern Costs
1. **Spatial Slice**: Chunks/data needed to read full spatial domain for single time
2. **Shard Touch**: Total shard data opened for spatial slice
3. **Full Time Series**: Shards/data needed for complete time series at single location
4. **Shared Memory Estimate**: Uncompressed memory needed for a single slice along the append dimension (init_time for forecast mode, time for analysis mode)

### Spatial Scores
- **Evenness**: Score from 0 to 1 (1.0 = perfectly even division across the domain)
- **Squareness**: Score from 0 to 1 (1.0 = spatial dimensions are equal, e.g., 32×32)
- Higher is better for both scores

### Template Config Code Block
- In detailed analysis mode, outputs Python code ready to paste into your `template_config.py`
- Includes properly formatted `var_chunks` and `var_shards` dictionaries with comments

## Interpreting a layout

For the manual configuration in Quick Start, a chunk spans 180 days by 8° by 8°;
a shard spans 360 days by 96° by 96°. At the assumed compression ratio of 0.2,
the chunk is about 1.1 MB and the shard about 324 MB. The chunk is below the tool's
target range, while the shard is within it.

Compare spatial-slice and full-time-series costs for the intended access pattern,
and check the shared-memory estimate against the worker's memory budget. These
are estimates: validate compression and memory use with real data before adopting
a layout.

## Tips for Choosing Layouts

1. **Start with search mode** to get baseline recommendations; it prints a command for detailed analysis of the top result
2. **Prefer square spatial dimensions** (e.g., 32×32 over 32×64) for balanced access patterns
3. **Prioritize spatial evenness** for cloud storage efficiency
4. **Consider access patterns:**
   - Frequent spatial slices → smaller spatial chunks
   - Frequent time series → smaller temporal chunks
5. **Balance chunk/shard sizes:**
   - Too small → HTTP overhead
   - Too large → wasted bandwidth for partial reads
6. **For forecasts:** Always keep 1 init_time per shard for efficient forecast-by-forecast access
7. **Use the template config output** to quickly paste configurations into your `template_config.py`
