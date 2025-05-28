# Operational Update Refactor

## End State

The refactored operational update system will provide a clean interface where dataset authors subclass `RegionJob` to define dataset-specific processing logic. The system will support:

1. **Automatic operational update job determination** via `RegionJob.operational_update_jobs()`
2. **Pipelined processing** with concurrent download, read/compress/write, and upload steps
3. **Incremental metadata updates** after each region job to make data available quickly
4. **Progress tracking and resumption** via `UpdateProgressTracker`
5. **Temporary store pattern** to ensure readers always see valid datasets

### Key Interfaces

#### RegionJob
- `operational_update_jobs(final_store, template_ds, append_dim, all_data_vars)` - class method to determine what jobs need processing
- `process()` - enhanced to write to tmp_store first, then upload chunks to final_store with pipelining
- `update_template_with_results(process_results)` - instance method to update template based on what was actually processed

#### DynamicalDataset
- `reformat_operational_update()` - orchestrates the operational update flow
- `_tmp_store()` - provides local temporary store

### Processing Pipeline

For each region job:
1. **Download** source files (concurrent)
2. **Read/Compress/Write** to tmp_store (concurrent with upload of previous variable)
3. **Upload** chunk data to final_store (concurrent with read/compress of next variable)
4. **Update metadata** incrementally after each job

Step 4 only happens in the operational update case and is orchestrated by DynamicalDataset, while steps 1, 2 and 3 happen in RegionJob.process.

### Temporary Store Pattern

The temporary store pattern is essential for maintaining dataset validity:
- Write full metadata (including updated dataset size) to tmp_store for zarr's `region="auto"` to work
- Process all data variables and write chunk data to tmp_store
- Upload only chunk data (not metadata) from tmp_store to final_store
- Only after all chunk data is uploaded, copy the updated metadata to final_store
- This ensures readers always see a valid view of the dataset during updates

### Pipelining for Performance

Within `RegionJob.process()`, the three main steps should be pipelined:
- While variable N is uploading chunk data to final_store
- Variable N+1 should be reading/compressing/writing to tmp_store
- Variable group N+2 should be downloading source files

This maximizes resource utilization and minimizes total processing time.

## Implementation Phases

### Phase 1: Enhanced RegionJob.process with tmp_store support

**Goal**: Modify `RegionJob.process()` to write to a temporary store first, then upload chunk data to the final store with pipelining.

**Changes needed**:
- Add `tmp_store: zarr.abc.store.Store | Path` parameter to `RegionJob.__init__()`
- Modify `RegionJob.process()` to:
  1. Write zarr metadata to tmp_store (using `template_utils.write_metadata`)
  2. Process data variables and write chunk data to tmp_store
  3. Upload chunk data (not metadata) from tmp_store to final_store using `copy_data_var` from `reformatters.common.zarr`
  4. Pipeline the upload step with processing of subsequent variables
- Update `RegionJob._write_shards()` to write to tmp_store instead of final_store
- Add pipelining logic using `ThreadPoolExecutor` for concurrent upload while processing next variable
- Integrate UpdateProgressTracker into RegionJob.process for resumption support

**Key considerations**:
- Use `get_local_tmp_store()` from `reformatters.common.zarr` for temporary storage
- Use `copy_data_var()` to upload only chunk data, not metadata
- Ensure proper cleanup of temporary files

### Phase 2: Operational update job determination

**Goal**: Add base class method to determine which region jobs need processing for operational updates.

**Changes needed**:
- Add `RegionJob.operational_update_jobs()` class method with signature:
  ```python
  @classmethod
  def operational_update_jobs(
      cls,
      final_store: zarr.abc.store.Store,
      template_ds: xr.Dataset,
      append_dim: AppendDim,
      all_data_vars: Sequence[DATA_VAR],
  ) -> Sequence["RegionJob[DATA_VAR, SOURCE_FILE_COORD]"]:
  ```
- Subclassers must implement their own logic that does something like this:
  1. Reads existing data from final_store to determine what's already processed
  2. Determines what new data is available (dataset-specific logic)
  3. Optionally identifies recent incomplete data for reprocessing
  4. Returns appropriate `RegionJob` instances
- In this implementation phase, just implement the base class method that raises NotImplementedError
- Update all method signatures to use `final_store` consistently instead of `store`
- Subclasses can override this method for dataset-specific logic (e.g., GEFS analysis vs forecast_35_day have different update patterns)

### Phase 3: Template update interface

**Goal**: Add base class interface for updating template dataset based on processing results.

**Changes needed**:
- Modify `RegionJob.process()` return type to `dict[str, Sequence[SOURCE_FILE_COORD]]`
  - Key: variable name
  - Value: sequence of source file coords with their final processing status
- Add `RegionJob.update_template_with_results()` instance method:
  ```python
  def update_template_with_results(
      self, 
      process_results: dict[str, Sequence[SOURCE_FILE_COORD]]
  ) -> xr.Dataset:
  ```
- Subclassers of this this method should:
  1. Make a copy of `self.template_ds`
  2. Apply dataset-specific adjustments based on `process_results`
  3. Examples of adjustments:
     - Trim dataset along append_dim to only include successfully processed data (GEFS analysis pattern)
     - Load existing coordinate values from `self.final_store`, update them based on results (GEFS forecast_35_day `ingested_forecast_length` pattern)
  4. Return the updated template dataset
- Base class should raise NotImplementedError

### Phase 4: DynamicalDataset operational update orchestration

**Goal**: Implement the main operational update orchestration in `DynamicalDataset`.

**Changes needed**:
- Add `DynamicalDataset._tmp_store()` method that returns `get_local_tmp_store()`
- Implement `DynamicalDataset.reformat_operational_update()` method that:
  1. Calls `RegionJob.operational_update_jobs()` to determine what needs processing
  2. For each region job:
     a. Creates region job with both final_store and tmp_store
     b. Calls `region_job.process()` to get processing results
     c. Calls `region_job.update_template_with_results()` to get updated template
     d. Writes updated metadata to tmp_store using `template_utils.write_metadata`
     e. Copies updated metadata to final_store using `copy_zarr_metadata`
  3. Handles cleanup of temporary stores
- Ensure incremental metadata updates happen after each region job
- Add test for simple path of this to `dynamical_dataset_test.py`

### Phase 5: Migration of existing datasets

**Goal**: Migrate existing GEFS datasets to use the new interface and remove old code.

**Changes needed**:
- Create GEFS-specific `RegionJob` subclasses that implement:
  - `operational_update_jobs()` with GEFS-specific logic from existing `reformat_operational_update` functions
  - `update_template_with_results()` with GEFS-specific metadata update logic
- Update GEFS `DynamicalDataset` subclasses to use new operational update interface
- Remove old `reformat_operational_update` functions from:
  - `src/reformatters/noaa/gefs/analysis/reformat.py`
  - `src/reformatters/noaa/gefs/forecast_35_day/reformat.py`
- Update CLI interfaces to use new `DynamicalDataset.reformat_operational_update()`
- Update Kubernetes cron job definitions to use new interface

## Technical Details

### Error Handling

- Download failures should be logged (and status should be updated in source file coords) but not stop processing (existing pattern)
- Read failures should be logged but not stop processing (existing pattern)
- Use Sentry monitoring decorators for operational update functions
- Ensure proper cleanup of temporary files and stores

### Performance Considerations

- Pipeline download → read/compress/write → upload steps for maximum throughput
- Use appropriate thread pool sizes for I/O vs CPU-bound operations
- Minimize memory usage by processing variables sequentially within region jobs
- Use shared memory buffers for efficient data transfer between processes

## Notes

- The temporary store pattern is essential: write full metadata to tmp_store for region="auto", but only copy chunk data (not metadata) to final_store until all chunks are written
- Pipelining is critical for performance: while variable N is uploading, variable N+1 should be reading/compressing
- Progress tracking allows resumption if operational update jobs are interrupted
- Incremental metadata updates make data available to readers as quickly as possible
- All existing operational update logic should be preserved, just refactored into the new interface
- The `RegionJob` interface should be flexible enough to handle different dataset patterns (analysis vs forecast, different file structures, etc.)
