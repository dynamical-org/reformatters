# Source Data Exploration Guide

The most open-ended and often most challenging part of integrating a dataset is finding reliable sources, determining exactly what is available, how it is structured, and how to access it. This guide outlines a process to approach this work, along with a template to capture a concise summary of the information needed to integrate a dataset successfully.

Copy and fill out the template below as you explore. If you cannot verify a piece of information from the source files, note that rather than making a best guess. Leave blank any parts that are not relevant for your dataset, and add extra notes for unique details related to the questions below that aren't captured in the template.

Multiple sources: Our goal is to create the longest possible archive. Sometimes this requires combining data from the same underlying model or system stored in multiple locations. If using multiple sources provides better combined coverage than a single one, repeat the `Source Information` section for each.

Priorities for choosing a good source:
1. Completeness of data
2. Reliability and operational support (prefer sourcing directly from the producer over a mirror)
3. Access throughput (object storage is great)

The last two priorities can sometimes be in tension. In those cases, we often want to integrate with both sources, using code that first tries the high-throughput source and falls back to the high-reliability source if necessary.


## Exploration Template

---

## Dataset: [Provider] [Model]

### Source Information
- **Summary of data organization**: (e.g. "One file with all variables per lead time and init time", "One file for each init time, ensemble member, and variable, including all lead times")
- **File format**: (e.g. GRIB2, NetCDF4, HDF5)
- **Temporal coverage**: [start] to [end/present]
- **Temporal frequency**: (e.g. One initialization time every 6 hours, with a 1 hour forecast step for hours 0-90 and a 3 hour step for hours 93-384.)
- **Latency**: (e.g. Files for lead time 0 are available ~60 minutes after init time, and the last step is published 120 to 123 minutes after init time.)
- **Access notes**:
- **Browse root**: (link to browsable file listing, if available)
- **URL format**:
```
https://example.com/data/{YYYY}/{MM}/{DD}/model.t{HH}z.pgrb2.0p25.f{FFF}
https://example.com/data/{YYYY}/{MM}/{DD}/model.t{HH}z.pgrb2.0p25.f{FFF}.idx - if index files are available
```
- **Example URLs**:
```
https://example.com/data/2024/01/15/model.t00z.pgrb2.0p25.anl
https://example.com/data/2024/01/15/model.t00z.pgrb2.0p25.f000
https://example.com/data/2024/01/15/model.t00z.pgrb2.0p25.f003.idx
```

### GRIB Index (if applicable)
- **Index files available**: Yes/No
- **Index style**: NOAA (colon-separated) / ECMWF (JSON) / Other
- **Example line**:
```
[paste one line from index file]
```

### Coordinate Reference System
- **Common name**: (e.g. WGS84 geographic, Lambert Conformal Conic, etc.)
- **PROJ string or EPSG**:

### Dimensions & Dimension Coordinates

| Dimension | Min | Max | Step | Notes |
|-----------|-----|-----|------|-------|
| time / init_time | | | | |
| [lead_time] | | | | |
| latitude / y | | | | |
| longitude / x | | | | |
| [ensemble_member] | | | | |
| [model_level / pressure_level] | | | | |
| [other] | | | | |

We use pixel centers for spatial coordinates.

### Data Variables

Check availability for these core variables. If they aren't relevant, list the key variables available. We're generally interested in widely used parameters near the earth's surface.

| Variable name | Level | Units | Available from | Notes |
|---------------|-------|-------|----------------|-------|
| temperature_2m | 2 m | K or C | | |
| wind_u_10m | 10 m | m/s | | |
| wind_v_10m | 10 m | m/s | | |
| wind_u_100m | 100 m | m/s | | |
| wind_v_100m | 100 m | m/s | | |
| precipitation_surface | surface | mm or kg/m2 | | Note accumulation behavior |
| downward_short_wave_radiation_flux_surface | surface | W/m2 | | |
| downward_long_wave_radiation_flux_surface | surface | W/m2 | | |
| pressure_surface | surface | Pa | | |
| pressure_reduced_to_mean_sea_level | MSL | Pa | | |
| total_cloud_cover_atmosphere | atmosphere | % | | |
| relative_humidity_2m | 2 m | % | | |
| specific_humidity_2m | 2 m | kg/kg | | |
| dew_point_temperature_2m | 2 m | K or C | | |

**Temporal availability changes**:
- [Variable X] only available from [timestamp] onward
- [Variable Y] discontinued after [timestamp]
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

Search widely for data sources. Archives may be split across:
- Operational systems (recent data, shorter retention)
- Historical archives (older data, may be years behind)
- Different providers or mirror sites

Search online, review documentation, browse data catalogs, and examine existing integrations for related datasets to identify potential sources.

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

    # Get per-band metadata
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