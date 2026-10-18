# Source data exploration

Find the longest available archive, combining sources where needed. Prefer completeness, then reliability and operational support (prefer the producer over a mirror), then throughput; use a reliable fallback when the fastest source is less dependable.

Inspect actual files from the beginning, recent end, and known transitions of each source. Use documentation to guide investigation, but verify claims against files and mark uncertainty. Record source-specific findings in the exploration report, not this guide.

## Report template

Repeat source information for each source; omit fields that do not apply.

### Dataset: [provider] [model]

**Source information**

- Organization, file format, browse root, URL pattern, and example URLs
- Temporal coverage, frequency, forecast steps, publication latency, and access requirements
- License or terms link; flag restrictions and missing terms before integration
- GRIB index availability, format, and a sample entry

**Coordinates**

- CRS and projection parameters
- Dimensions: minimum, maximum, spacing, ordering, and changes over time
- Spatial coordinates use pixel centers

**Variables**

| Variable | Level | Units | Availability | Notes |
|---|---|---|---|---|
| | | | | |

Check common near-surface temperature, wind, precipitation, radiation, pressure, cloud, and humidity quantities, or the product's relevant variables. Record accumulation windows, missing-value conventions, and availability changes.

**Evidence and gaps**

- Sample dates and URLs from early, recent, and transition periods
- Observed changes in structure, coordinates, metadata, naming, and variables
- Unverified claims and open scope questions

Use a format-appropriate reader such as `rasterio` or `xarray` to inspect coordinates, metadata, and values. Inspect GRIB indexes where available to identify byte ranges for partial reads. Carry the report into [dataset scoping and implementation](dataset_development_guide.md).
