# Zarr V3 Chunk & Shard Layout Diagnostic Tool

A Python utility to calculate, optimize, and audit storage layouts for large-scale geospatial datasets stored in Zarr V3 format.

## Overview

This tool helps you:
- **Analyze** existing chunk/shard configurations to understand storage and access costs
- **Search** for optimal chunk/shard layouts based on your dataset dimensions
- **Compare** configurations with detailed diagnostics including access pattern costs

## Installation

The tool is part of the reformatters package. No additional dependencies required.

## Quick Start

```bash
# Run from workspace root
cd /workspace

# Analysis mode (time-series dataset) - search for optimal layout
uv run python src/scripts/chunk_shard_size.py \
  --time 73000:3:hour \
  --latitude 721:0.25:degrees \
  --longitude 1440:0.25:degrees \
  --search

# Forecast mode (init_time + lead_time) - search for optimal layout
uv run python src/scripts/chunk_shard_size.py \
  --init_time 365 \
  --ensemble_member 31 \
  --lead_time 181:3:hour \
  --latitude 721:0.25:degrees \
  --longitude 1440:0.25:degrees \
  --search

# Manual configuration - analyze specific chunk/shard shapes
uv run python src/scripts/chunk_shard_size.py \
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

### Data Specifications
- Element size: `float32` (4 bytes)
- Compression ratio: 20% (0.2)

### Target Sizes (Compressed)
| Type | Min | Max | Sweet Spot |
|------|-----|-----|------------|
| Chunk | 2.5 MB | 6.0 MB | 3.5 MB |
| Shard | 100 MB | 600+ MB | ~350 MB |

### Constraints
- Shards must be exact integer multiples of chunks
- Spatial evenness prioritizes shard boundaries over chunk boundaries

## Command Reference

```
usage: chunk_shard_size.py [-h] [--time TIME] [--init_time INIT_TIME]
                           [--ensemble_member ENSEMBLE_MEMBER]
                           [--lead_time LEAD_TIME] [--latitude LATITUDE]
                           [--longitude LONGITUDE] [--x X] [--y Y]
                           [--statistic STATISTIC] [--chunk_shape CHUNK_SHAPE]
                           [--shard_shape SHARD_SHAPE] [--search] [--top TOP]

Options:
  --time TIME           Spec: 'length:step:units' or 'length'
  --init_time           Spec: 'length:step:units' or 'length'
  --ensemble_member     Spec: 'length:step:units' or 'length'
  --lead_time           Spec: 'length:step:units' or 'length'
  --latitude            Spec: 'length:step:units' or 'length'
  --longitude           Spec: 'length:step:units' or 'length'
  --x, --y              Spec: 'length:step:units' or 'length'
  --statistic           Spec: 'length:step:units' or 'length'
  
  --chunk_shape         Manual chunk shape (comma-separated)
  --shard_shape         Manual shard shape (comma-separated)
  --search              Search for optimal configurations
  --top N               Show top N results in search mode (default: 5)
```

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

### Spatial Evenness Score
- Score from 0 to 1 (1.0 = perfectly even division)
- Higher is better (less wasted space in edge chunks/shards)

## Example Commands and Output

### Example 1: Analysis Mode - Manual Configuration

**Command:**
```bash
uv run python src/scripts/chunk_shard_size.py \
  --time 73000:3:hour \
  --latitude 721:0.25:degrees \
  --longitude 1440:0.25:degrees \
  --chunk_shape 1440,32,32 \
  --shard_shape 2880,384,384
```

**Output:**
```
================================================================================
 ZARR V3 LAYOUT DIAGNOSTICS - ANALYSIS MODE
================================================================================

┌──────────────────────────────────────────────────────────────────────────────┐
│ GRID LAYOUT                                                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ Dimension        │ Length │ Chunk │ Shard │ Chunks/Dim │ Shards/Dim │
├──────────────────┼────────┼───────┼───────┼────────────┼────────────┤
│ time             │  73000 │  1440 │  2880 │         51 │         26 │
│ latitude         │    721 │    32 │   384 │         23 │          2 │
│ longitude        │   1440 │    32 │   384 │         45 │          4 │
└──────────────────┴────────┴───────┴───────┴────────────┴────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ SUMMARY STATISTICS                                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Total Chunks:              52,785                                           │
│ Total Shards:                 208                                           │
│ Chunks per Shard:             288                                           │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHYSICAL SPAN (per unit)                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Chunk: 180.0 days x 8.00° x 8.00°                                            │
│ Shard: 360.0 days x 96.00° x 96.00°                                          │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ STORAGE SPECIFICATIONS                                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                      │       Raw MB │  Compressed MB │       Status │    │
├─────────────────────┼──────────────┼────────────────┼──────────────┼─────┤
│ Chunk                │         5.62 │           1.12 │      ⚠ CHECK │    │
│ Shard                │      1620.00 │         324.00 │         ✓ OK │    │
└─────────────────────┴──────────────┴────────────────┴──────────────┴─────┘

  Target chunk: 2.5-6.0 MB compressed (sweet spot: 3.5 MB)
  Target shard: 100.0-600.0+ MB compressed

┌──────────────────────────────────────────────────────────────────────────────┐
│ ACCESS PATTERN COSTS                                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│ 1. SPATIAL SLICE (full spatial domain, single time point)                    │
│    Chunks to read:            1035                                          │
│    Compressed data:        1164.38 MB                                       │
│                                                                              │
│ 2. SHARD TOUCH (shards opened for spatial slice)                            │
│    Shards touched:               8                                          │
│    Total shard data:         2.531 GB                                       │
│                                                                              │
│ 3. FULL TIME SERIES (all time, single spatial pixel)                        │
│    Shards to read:              26                                          │
│    Compressed data:          57.38 MB                                       │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ SPATIAL EVENNESS SCORE (1.0 = perfectly even)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ Chunk evenness:  0.951                                                        │
│ Shard evenness:  0.910                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Interpretation:**
- ⚠ CHECK on chunk: 1.12 MB compressed is below the 2.5 MB target
- ✓ OK on shard: 324 MB is within the 100-600 MB range
- Spatial slice costs ~1.16 GB and touches 8 shards
- Full time series for one pixel requires 26 shards

---

### Example 2: Analysis Mode - Search

**Command:**
```bash
uv run python src/scripts/chunk_shard_size.py \
  --time 73000:3:hour \
  --latitude 721:0.25:degrees \
  --longitude 1440:0.25:degrees \
  --search --top 3
```

**Output:**
```
Searching for optimal layouts (analysis mode)...
Dimensions: ('time', 'latitude', 'longitude')
Lengths: (73000, 721, 1440)

================================================================================
 SEARCH RESULTS - TOP 3 CONFIGURATIONS (ANALYSIS MODE)
================================================================================

#1
  Chunk: (1825, 64, 40) → 3.56 MB compressed
  Shard: (3650, 768, 160) → 342.19 MB compressed
  Total shards: 180 | Evenness: 1.000

  [RECOMMENDED - Best balance of size and spatial evenness]

#2
  Chunk: (1825, 64, 40) → 3.56 MB compressed
  Shard: (7300, 768, 80) → 342.19 MB compressed
  Total shards: 180 | Evenness: 1.000

#3
  Chunk: (1825, 64, 40) → 3.56 MB compressed
  Shard: (14600, 768, 40) → 342.19 MB compressed
  Total shards: 180 | Evenness: 1.000

--------------------------------------------------------------------------------
To see detailed diagnostics for a configuration, run with --chunk_shape and --shard_shape
================================================================================
```

**Interpretation:**
- All top configurations achieve perfect spatial evenness (1.000)
- Chunk size (3.56 MB) is near the 3.5 MB sweet spot
- Total shards: 180 (manageable for cloud storage)

---

### Example 3: Forecast Mode - Search

**Command:**
```bash
uv run python src/scripts/chunk_shard_size.py \
  --init_time 365 \
  --ensemble_member 31 \
  --lead_time 181:3:hour \
  --latitude 721:0.25:degrees \
  --longitude 1440:0.25:degrees \
  --search --top 3
```

**Output:**
```
Searching for optimal layouts (forecast mode)...
Dimensions: ('init_time', 'ensemble_member', 'lead_time', 'latitude', 'longitude')
Lengths: (365, 31, 181, 721, 1440)

================================================================================
 SEARCH RESULTS - TOP 3 CONFIGURATIONS (FORECAST MODE)
================================================================================

#1
  Chunk: (1, 31, 128, 64, 18) → 3.49 MB compressed
  Shard: (1, 31, 256, 768, 72) → 334.80 MB compressed
  Total shards: 7,300 | Evenness: 1.000

  [RECOMMENDED - Best balance of size and spatial evenness]

#2
  Chunk: (1, 31, 128, 32, 36) → 3.49 MB compressed
  Shard: (1, 31, 256, 384, 144) → 334.80 MB compressed
  Total shards: 7,300 | Evenness: 0.967

#3
  Chunk: (1, 31, 128, 64, 18) → 3.49 MB compressed
  Shard: (1, 31, 256, 384, 144) → 334.80 MB compressed
  Total shards: 7,300 | Evenness: 0.967

--------------------------------------------------------------------------------
To see detailed diagnostics for a configuration, run with --chunk_shape and --shard_shape
================================================================================
```

**Interpretation:**
- All configurations follow forecast mode constraint: 1 init_time per shard
- All 31 ensemble members kept together in chunks and shards
- Lead times grouped: 128 per chunk, ~256 per shard (covering all 181 lead times)
- 7,300 total shards = 365 init_times × 20 spatial shards

---

### Example 4: Existing GEFS Configuration Analysis

**Command:**
```bash
uv run python src/scripts/chunk_shard_size.py \
  --init_time 1825 \
  --ensemble_member 31 \
  --lead_time 181:3:hour \
  --latitude 721:0.25:degrees \
  --longitude 1440:0.25:degrees \
  --chunk_shape 1,31,64,17,16 \
  --shard_shape 1,31,192,374,368
```

This analyzes the actual GEFS 35-day forecast configuration to validate the current setup.

## Tips for Choosing Layouts

1. **Start with search mode** to get baseline recommendations
2. **Prioritize spatial evenness** for cloud storage efficiency
3. **Consider access patterns:**
   - Frequent spatial slices → smaller spatial chunks
   - Frequent time series → smaller temporal chunks
4. **Balance chunk/shard sizes:**
   - Too small → HTTP overhead
   - Too large → wasted bandwidth for partial reads
5. **For forecasts:** Always keep 1 init_time per shard for efficient forecast-by-forecast access
