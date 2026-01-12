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

```
usage: chunk_shard_size.py [-h] [--time TIME] [--init_time INIT_TIME]
                           [--ensemble_member ENSEMBLE_MEMBER]
                           [--lead_time LEAD_TIME] [--latitude LATITUDE]
                           [--longitude LONGITUDE] [--x X] [--y Y]
                           [--statistic STATISTIC] [--chunk_shape CHUNK_SHAPE]
                           [--shard_shape SHARD_SHAPE] [--search] [--top TOP]
                           [--bytes_per_element N] [--compression_ratio R]

Options:
  --time TIME           Spec: 'length:step:units' or 'length'
  --init_time           Spec: 'length:step:units' or 'length'
  --ensemble_member     Spec: 'length:step:units' or 'length'
  --lead_time           Spec: 'length:step:units' or 'length'
  --latitude            Spec: 'length:step:units' or 'length'
  --longitude           Spec: 'length:step:units' or 'length'
  --x, --y              Spec: 'length:step:units' or 'length'
  --statistic           Spec: 'length:step:units' or 'length'
  
  --chunk_shape         Manual chunk shape (comma-separated, order matches dimension flags)
  --shard_shape         Manual shard shape (comma-separated, order matches dimension flags)
  --search              Search for optimal configurations
  --top N               Show top N results in search mode (default: 5)
  
  --bytes_per_element N Bytes per array element (default: 4 for float32)
  --compression_ratio R Expected compression ratio (default: 0.2 = 20%)
```

**Note:** The order of values in `--chunk_shape` and `--shard_shape` must match the order of
dimension flags provided (e.g., if you specify `--time ... --latitude ... --longitude ...`,
then shapes should be `time_val,lat_val,lon_val`).

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

### Spatial Scores
- **Evenness**: Score from 0 to 1 (1.0 = perfectly even division across the domain)
- **Squareness**: Score from 0 to 1 (1.0 = spatial dimensions are equal, e.g., 32×32)
- Higher is better for both scores

### Template Config Code Block
- In detailed analysis mode, outputs Python code ready to paste into your `template_config.py`
- Includes properly formatted `var_chunks` and `var_shards` dictionaries with comments

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
│ SPATIAL SCORES (1.0 = optimal)                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Chunk evenness:   0.951    Chunk squareness: 1.000                                │
│ Shard evenness:   0.910    Shard squareness: 1.000                                │
└──────────────────────────────────────────────────────────────────────────────┘

────────────────────────────────────────────────────────────────────────────────
 TEMPLATE CONFIG CODE (copy-paste into your template_config.py)
────────────────────────────────────────────────────────────────────────────────

```python
        # ~6MB uncompressed, ~1.1MB compressed
        var_chunks: dict[Dim, int] = {
            "time": 1440,  # 180 days of 3-hourly data
            "latitude": 32,  # 23 chunks over 721 pixels
            "longitude": 32,  # 45 chunks over 1440 pixels
        }

        # ~1620MB uncompressed, ~324MB compressed
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"] * 2,  # 26 shards over 73000 pixels
            "latitude": var_chunks["latitude"] * 12,  # 2 shards over 721 pixels
            "longitude": var_chunks["longitude"] * 12,  # 4 shards over 1440 pixels
        }
```
```

**Interpretation:**
- ⚠ CHECK on chunk: 1.12 MB compressed is below the 2.5 MB target
- ✓ OK on shard: 324 MB is within the 100-600 MB range
- Spatial slice costs ~1.16 GB and touches 8 shards
- Full time series for one pixel requires 26 shards
- The template config code block can be copied directly into your `template_config.py`

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
  Chunk: (1000, 64, 64) → 3.12 MB compressed
  Shard: (3000, 384, 384) → 337.50 MB compressed
  Total shards: 200 | Evenness: 0.910 | Squareness: 1.000

  [RECOMMENDED - Best balance of size, evenness, and spatial squareness]

#2
  Chunk: (1000, 64, 64) → 3.12 MB compressed
  Shard: (7000, 256, 256) → 350.00 MB compressed
  Total shards: 198 | Evenness: 0.879 | Squareness: 1.000

#3
  Chunk: (1000, 64, 64) → 3.12 MB compressed
  Shard: (1000, 768, 768) → 450.00 MB compressed
  Total shards: 146 | Evenness: 0.967 | Squareness: 1.000

--------------------------------------------------------------------------------
Run this command to see detailed analysis of top recommendation:

uv run python src/scripts/chunk_shard_size.py \
    --time 73000:3.0:hour \
    --latitude 721:0.25:degrees \
    --longitude 1440:0.25:degrees \
    --chunk_shape 1000,64,64 \
    --shard_shape 3000,384,384

================================================================================
```

**Interpretation:**
- All top configurations achieve perfect spatial squareness (1.000) with 64×64 chunks
- Chunk size (3.12 MB) is near the 3.5 MB sweet spot
- The tool now outputs a ready-to-run command for the top recommendation

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
  Chunk: (1, 31, 128, 32, 30) → 2.91 MB compressed
  Shard: (1, 31, 256, 256, 240) → 372.00 MB compressed
  Total shards: 6,570 | Evenness: 0.954 | Squareness: 0.938

  [RECOMMENDED - Best balance of size, evenness, and spatial squareness]

#2
  Chunk: (1, 31, 128, 32, 36) → 3.49 MB compressed
  Shard: (1, 31, 256, 256, 252) → 390.60 MB compressed
  Total shards: 6,570 | Evenness: 0.898 | Squareness: 0.889

#3
  Chunk: (1, 31, 128, 32, 32) → 3.10 MB compressed
  Shard: (1, 31, 256, 256, 256) → 396.80 MB compressed
  Total shards: 6,570 | Evenness: 0.879 | Squareness: 1.000

--------------------------------------------------------------------------------
Run this command to see detailed analysis of top recommendation:

uv run python src/scripts/chunk_shard_size.py \
    --init_time 365 \
    --ensemble_member 31 \
    --lead_time 181:3.0:hour \
    --latitude 721:0.25:degrees \
    --longitude 1440:0.25:degrees \
    --chunk_shape 1,31,128,32,30 \
    --shard_shape 1,31,256,256,240

================================================================================
```

**Interpretation:**
- All configurations follow forecast mode constraint: 1 init_time per shard
- All 31 ensemble members kept together in chunks and shards
- Spatial dimensions now prefer squarer shapes (32×30 is close to square)
- The detailed analysis includes a ready-to-paste template config code block

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
