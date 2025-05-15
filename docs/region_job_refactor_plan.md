# RegionJob Refactor Plan

This document outlines the refactor plan to consolidate the existing `reformat_time_i_slices` and `reformat_init_time_i_slices` functions into a unified `RegionJob` base class. It's being kept in docs/ because some of it may be useful for documenting how to integrate a new dataset once the refactor is complete.

## 1. Orchestration in `RegionJob.process()`

Define a single `process()` method on the `RegionJob` base class to manage the overall pipeline:

1. **Extract Processing Region**
   - Call `get_processing_region(original_slice: slice) -> slice`
   - Default: returns the user-requested region.
   - Override in subclasses (e.g. GEFS analysis) to add padding.
   - Use `self.template_ds.isel({self.append_dim: processing_slice})` to get an `xarray.Dataset` limited to the processing region we'll call `processing_template_ds`.

2. **Group Data Variables**
   - Call `group_data_vars(chunk_ds: Dataset) -> Sequence[Sequence[DATA_VAR]]`
   - Default: batch `self.data_vars` present in `chunk_ds` into size `self.max_vars_per_backfill_job`.
   - Override to implement file-type or statistic grouping.

3. **Setup Executors & Shared Buffer**
   - Create:
     - `ThreadPoolExecutor` for download coordination.
     - `ThreadPoolExecutor` for I/O.
     - `ThreadPoolExecutor` for CPU-bounded reads.
     - `ProcessPoolExecutor` for parallel shard writes.
     - Shared memory buffer of size `_calc_shared_buffer_size(processing_template_ds)`.

4. **Process Each Group**
   For `data_vars_in_group` in `data_var_groups`:

   a. **Download Stage**
      - Call `_download_group(chunk_ds, vars_in_group, io_executor) -> List[SourceFileCoord]`
      - Internally uses two hooks:
        1. `generate_source_file_coords(chunk_ds) -> List[SourceFileCoord]`
        2. `download_file(coord) -> Path|None`
      - Each `SourceFileCoord` has a `status` field updated to `DownloadedFailed` on exception.

   c. **Read & Write Stage**
      For each `data_var` in `vars_in_group`:
      i.   Call `create_data_array_and_template(chunk_ds, data_var, shared_buffer)` to build the shared-memory output array.
      ii.  Call `read_into_data_array(data_array, data_var, coords, cpu_executor)`:
             - Submits `self._read_and_write_one(coord, data_var.name, out_values)` for each coord with status `Processing`.
             - `_read_and_write_one` calls `self.read_data(coord, data_var.name)`, writes the returned numpy chunk into the shared array at `coord.out_loc()`, and updates `coord.status` to `Succeeded` or `ReadFailed`.
      iii. Call `apply_data_transformations(data_array, data_var)` to run deaccumulation, interpolation, and binary rounding. This method works inplace (does not copy data_array).
      iv.  Call `write_shards(data_array_template, shared_buffer, chunk_ds, store, proc_executor)` to write zarr shards in parallel.
      v. Metadata Collection. Call `summarize_processing_state(coords_and_paths) -> Any` to gather any needed time-metadata per data variable.

   d. **Cleanup**
      - Delete any local files returned by `download_file()`.

   e. **Return**
      - Return a dict of data var name to the output of `summarize_processing_state`


### Internal RegionJob Methods
- `_calc_shared_buffer_size(chunk_ds: Dataset) -> int`
- `_make_shared_buffer(size: int) -> ContextManager[SharedMemory]`
- `_download_processing_group(...)`
- `_create_data_array_and_template(...)`
- `_read_into_data_array(...)`
- `_read_and_write_one(coord, var_name, out_values)`
- `_write_shards(...)`


## 2. Subclass Hooks

Subclasses of `RegionJob[DATA_VAR]` need only override:

1. **`get_processing_region(orig_slice: slice) -> slice`** (optional, base class implementation just returns `self.region`)
2. **`group_data_vars(chunk_ds: xr.Dataset) -> Sequence[Sequence[DATA_VAR]]`**  (optional, base class returns all in a single group)
3. **`generate_source_file_coords(chunk_ds: xr.Dataset) -> List[SourceFileCoord]`**
4. **`download_file(coord: SourceFileCoord) -> Path|None`**
5. **`read_data(coord: SourceFileCoord) -> numpy.ndarray`**
6. **`apply_data_transformations(data_array: xr.DataArray, data_var: DATA_VAR) -> None`** (optional, base class applies binary rounding)

All other methods (e.g. `_download_processing_group`, `_create_data_array_and_template`, `_read_into_data_array`, `_write_shards`, etc.) are implemented in the base class.


## 3. `SourceFileCoord` & Status

### `SourceFileStatus` Enum

```
class SourceFileStatus(Enum):
   Processing     = auto()
   DownloadFailed = auto()
   ReadFailed     = auto()
   Succeeded      = auto()
```

### Base `SourceFileCoord` Class

- Few fixed fields; dataset subclasses define attributes (e.g. `init_time`, `lead_time`, `ensemble_member`, `file_type` relevant to how each ).

- Implements:

  - `get_url() -> str`
  - `out_loc() -> Dict[str, slice|int|float|str]`

- Base class fields:
  - `status: SourceFileStatus`
  - `downloaded_path: Path|None`


## 4. Example Subclass

```
class GEFSAnalysisRegionJob(RegionJob[GEFSDataVar]):

   def get_processing_region(self, original: slice):
      return buffer_slice(original, buffer_size=2)

   def group_data_vars(self, chunk_ds):
      return batched(group_data_vars_by_gefs_file_type(self.data_vars), self.max_vars_per_backfill_job)

   def generate_source_file_coords(self, chunk_ds):
      # produce list of GEFSAnalysisSourceFileCoord(...)
      ...

   def download_file(self, coord) -> Path:
      url = coord.get_url()
      path = # download to local path
      return path


   def read_data(self, coord, var_name) -> np.array[Any, Any]:
      ...
```


## 5. Future Extension
- Add third stage: `upload_to_final_store(tmp_store, final_store)` and initially write shards to a tmp local disk store
