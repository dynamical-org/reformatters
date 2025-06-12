# Dataset Integration Guide

## Overview

Integrating a new dataset into this framework involves three core components:

1. **TemplateConfig**  
   Defines the on-disk template (Zarr metadata and initial empty arrays) and how it is expanded for processing.

2. **RegionJob**  
   Encapsulates the logic to slice the template along the append dimension, download and read source files, apply any transformations, and write chunks into the Zarr store.

3. **DynamicalDataset**  
   The top-level class that ties together a `TemplateConfig` and a `RegionJob` subclass, exposes CLI commands (`update-template`, `reformat-local`, `reformat-kubernetes`, etc.), and manages operational updates and validation.

## Suggested File Tree

Place your integration under `src/reformatters/<agency_or_producer>/<model>/<variant>/`:

```bash
tree src/reformatters/<agency>/<model>/<variant>
├── dataset.py
├── region_job.py
└── template_config.py
```

## 1. TemplateConfig

### Required Overrides

- `dims: tuple[str, ...]`  
  Names of all dataset dimensions.

- `append_dim: str`  
  The dimension along which new data is appended (e.g., `"time"`).

- `append_dim_start: Timestamp`  
  The starting coordinate for the append dimension.

- `append_dim_frequency: Timedelta`  
  Frequency between append-dim coordinates.

- `@property dataset_attributes -> DatasetAttributes`  
  Return a `DatasetAttributes` instance with `dataset_id`, `dataset_version`, global attributes, encoding defaults.

- `@property coords -> Sequence[Coordinate]`  
  Define all coordinates (both dimension and auxiliary).

- `@property data_vars -> Sequence[DataVar]`  
  Define all data variables, their dtypes, chunks, encoding, and any internal attrs.

- `dimension_coordinates(self) -> dict[str, Any]`  
  Map each dimension name to its coordinate values (e.g., a `DatetimeIndex` for time, or numpy arrays).

### Optional Overrides

- `derive_coordinates(self, ds: xr.Dataset) -> dict[str, xr.DataArray | tuple]`  
  Compute non-dimension coordinates (default returns only `spatial_ref`).

- Other helpers (`append_dim_coordinate_chunk_size`, `template_path`) can use the defaults.

## 2. RegionJob

Subclass `RegionJob[YourDataVar, YourSourceFileCoord]`:

### Required Overrides

- `generate_source_file_coords(self, processing_region_ds, data_var_group)`  
  Return one `SourceFileCoord` per source file needed for that region and variable group.

- `download_file(self, coord: YourSourceFileCoord) -> Path`  
  Fetch the file (e.g., HTTP, S3) and return its local path.

- `read_data(self, coord: YourSourceFileCoord, data_var: YourDataVar) -> ArrayFloat32`  
  Load the data array for a single variable from a source file.

- `@classmethod operational_update_jobs(...) -> (jobs, template_ds)`  
  For operational updates, compute time range, read existing Zarr, expand template, and return one or more jobs.

### Optional Overrides

- `apply_data_transformations(self, data_array, data_var)`  
  Default applies binary rounding; override for deaccumulation, interpolation, etc.

- `get_processing_region(self) -> slice`  
  Expand the region slice if you need overlap (e.g., for interpolation).

- `source_groups(cls, data_vars)`  
  Group variables by source file type or other logic.

- Class attributes: `max_vars_per_backfill_job`, `max_vars_per_download_group`, `download_parallelism`.

## 3. DynamicalDataset

Subclass `DynamicalDataset[YourDataVar, YourSourceFileCoord]`:

### Required Attributes & Overrides

- `template_config: TemplateConfig[...]`  
  An instance of your `TemplateConfig` subclass.

- `region_job_class: type[RegionJob[...]]`  
  Your `RegionJob` subclass.

- `validate_zarr(self) -> None`  
  Implement dataset-specific validation (e.g., no missing data).

- `operational_kubernetes_resources(self, image_tag: str) -> Iterable[Job]`  
  Define a `ReformatCronJob` for updates and a validation `CronJob`, with CPU/memory/cron schedule.

### Optional Overrides

- All remaining methods (`reformat_kubernetes`, `reformat_local`, `process_region_jobs`) use the default orchestration; override only for custom workflows.

```bash
# Preview the guide in your editor or documentation site
xdg-open docs/dataset_integration_guide.md
```
