# RegionJob Refactor Plan

This document outlines the detailed refactor plan to consolidate the existing `reformat_time_i_slices` and `reformat_init_time_i_slices` functions into a unified `RegionJob` base class.

## 1. Orchestration in `RegionJob.process()`

Define a single `process()` method on the `RegionJob` base class to manage the overall pipeline:

1. **Determine Processing Region**  
   - Call `get_processing_region(orig_slice: slice) -> slice`  
   - Default: returns the user-requested region.  
   - Override in subclasses (e.g. GEFS analysis) to add padding.

2. **Extract Template Subset**  
   - Use `self.template_ds.isel({self.append_dim: proc_slice})` to get an `xarray.Dataset` limited to the processing region.

3. **Group Data Variables**  
   - Call `group_data_vars(chunk_ds: Dataset) -> Sequence[Sequence[DATA_VAR]]`  
   - Default: batch `self.data_vars` present in `chunk_ds` into size `self.max_vars_per_backfill_job`.  
   - Override to implement file-type or statistic grouping.

4. **Setup Executors & Shared Buffer**  
   - Create:  
     - `ThreadPoolExecutor` for download coordination.  
     - `ThreadPoolExecutor` for I/O.  
     - `ThreadPoolExecutor` for CPU-bounded reads.  
     - `ProcessPoolExecutor` for parallel shard writes.  
     - Shared memory buffer of size `_calc_shared_buffer_size(chunk_ds)`.

5. **Process Each Group**  
   For `vars_in_group` in `var_groups`:

   a. **Download Stage**  
      - Call `_download_group(chunk_ds, vars_in_group, io_executor) -> List[SourceFileCoord]`  
      - Internally uses two hooks:  
        1. `generate_source_file_coords(chunk_ds) -> List[SourceFileCoord]`  
        2. `download_file(coord) -> Path|None`  
      - Each `SourceFileCoord` has a `status` field updated to `DownloadedFailed` on exception.

   b. **Metadata Collection**  
      - Call `get_max_lead_times(coords_and_paths)` to gather any needed time-metadata per data variable.

   c. **Read & Write Stage**  
      For each `dv` in `vars_in_group`:  
      i.   Call `create_data_array_and_template(chunk_ds, dv, shared_buffer)` to build the shared-memory output array.  
      ii.  Call `read_into_data_array(data_array, dv, coords, cpu_executor)`:  
             - Submits `self._read_and_write_one(coord, dv.name, out_values)` for each coord with status `Processing`.  
             - `_read_and_write_one` calls `self.read_data(coord, dv.name)`, writes the returned numpy chunk into the shared array at `coord.target_loc()`, and updates `coord.status` to `Succeeded` or `ReadFailed`.  
      iii. Call `apply_data_transformations(data_array, dv)` to run deaccumulation, interpolation, and binary rounding.  
      iv.  Call `write_shards(data_array_template, shared_buffer, chunk_ds, store, proc_executor)` to write zarr shards in parallel.

   d. **Cleanup**  
      - Delete any local files returned by `download_file()`.

6. **Yield Results**  
   - The `process()` generator yields `(DATA_VAR, metadata_dict)` for each successfully processed variable.

## 2. Subclass Hooks

Subclasses of `RegionJob[DATA_VAR]` need only override:

1. **`get_processing_region(orig_slice: slice) -> slice`**  
2. **`group_data_vars(chunk_ds: xr.Dataset) -> Sequence[Sequence[DATA_VAR]]`**  
3. **`generate_source_file_coords(chunk_ds: xr.Dataset) -> List[SourceFileCoord]`**  
4. **`download_file(coord: SourceFileCoord) -> Path|None`**  
5. **`read_data(coord: SourceFileCoord, var_name: str) -> numpy.ndarray`**

All other methods (`_download_group`, `_read_and_write_one`, `read_into_data_array`, `create_data_array_and_template`, `apply_data_transformations`, `write_shards`, etc.) are implemented in the base class.

## 3. `SourceFileCoord` & Status

### `SourceFileStatus` Enum

