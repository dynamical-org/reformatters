# Source Data Exploration Guide

Before integrating a dataset, thoroughly explore the source data to understand what's available, how it's structured, and how to access it. This information drives your `TemplateConfig` and `RegionJob` implementations.

## Exploration Template

Copy and fill out this template as you explore. It captures everything needed for integration.

---

## Dataset: [Provider] [Model] [Variant]

### Source Information
- **Primary archive URL**:
- **Additional/historical archives**:
- **File format**: (e.g., GRIB2, NetCDF4, GeoTIFF)
- **Temporal coverage**: [start] to [end/present]
- **Update frequency**: (e.g., hourly, every 6 hours, daily)
- **Latency**: (typical delay from valid time to availability)

### File Naming Pattern
```
[Example file names and URL pattern]
```

### GRIB Index (if applicable)
- **Index files available**: Yes/No
- **Index style**: NOAA (colon-separated) / ECMWF (JSON) / Other
- **Example line**:
```
[paste one line from index file]
```

### Coordinate Reference System
- **CRS**: (e.g., WGS84 geographic, Lambert Conformal Conic, etc.)
- **PROJ string or EPSG**:

### Dimensions & Dimension Coordinates

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| time / init_time | | | | |
| lead_time | | | | |
| latitude / y | | | | |
| longitude / x | | | | |
| [other] | | | | |

### Data Variables (Sample)

Common variables observed (not exhaustive):

| Variable | Units | Available from | Notes |
|----------|-------|----------------|-------|
| temperature_2m | K | [start date] | |
| wind_u_10m | m/s | [start date] | |
| wind_v_10m | m/s | [start date] | |
| ... | | | |

**Temporal availability changes**:
- [Variable X] only available from [date] onward
- [Variable Y] discontinued after [date]
- [Other changes in variable availability]

### Sample Files Examined

- **Early archive**: [date, URL]
- **Recent data**: [date, URL]
- **[Other key periods]**: [date, URL, reason]

### Notable Observations
- [Coordinate system changes, metadata variations, file structure differences, accumulation periods, etc.]

---

## Exploration Process

### 1. Find the longest possible archive

Search aggressively for data sources. Archives may be split across:
- Operational systems (recent data, shorter retention)
- Historical archives (older data, may be years behind)
- Different providers or mirror sites

Check documentation, data catalogs, and existing integrations for similar datasets to find leads.

### 2. Get sample files from multiple time periods

Download files from:
- **Start of archive** (earliest available data)
- **End of archive** (most recent data)
- **Any known transition points** (model upgrades, resolution changes, etc.)

Data structure, file naming, variable availability, and metadata all change over time. You must examine files from different periods to catch these changes.

### 3. Use rasterio to examine files

```python
import rasterio

with rasterio.open('path/to/file.grib2') as src:
    print(src.profile)  # CRS, dimensions, dtype
    print(src.tags())   # Metadata
    print(src.indexes)  # Band/variable indexes

    # Read a band
    data = src.read(1)

    # Get coordinate bounds
    print(src.bounds)
    print(src.transform)
```

For multi-band files (common in GRIB), iterate bands to catalog variables:

```python
with rasterio.open('file.grib2') as src:
    for i in src.indexes:
        print(f"Band {i}: {src.tags(i)}")
```

### 4. Download GRIB index files (if available)

If the source provides GRIB index files (`.idx`), download samples. Index files list which bytes contain which variables, enabling efficient partial downloads.

**Identify index style**:
- **NOAA style**: Each line is colon-separated values
  ```
  1:0:d=2024010100:TMP:2 m above ground:anl:
  ```
- **ECMWF style**: Each line is a JSON object
  ```
  {"_type":"field","stream":"oper","levtype":"sfc","param":"2t",...}
  ```

Index style affects how you parse them in `RegionJob.generate_source_file_coords()`. See existing implementations for examples.

### 5. Trust only what you see

Use prior knowledge and documentation as a guide for what to look for, but trust only what you observe in actual data files. Documentation is often outdated or incomplete. Verify:
- Actual coordinate values (min, max, step)
- Variable names and units as they appear in files
- CRS and projection parameters
- File naming patterns and URL structures

### 6. Document changes over time

As you compare files from different periods, note:
- Structure changes (dimension sizes, new dimensions)
- Variable additions/removals
- Metadata changes (attribute names/values)
- File naming pattern changes
- Coordinate shifts or resolution changes

### 7. Fill out the template

Record everything as you discover it. This becomes your reference when implementing `TemplateConfig` and `RegionJob`.
