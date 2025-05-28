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
4. **Update metadata** incrementally after each job.

Step 4 only happens in the operational update case and is orchestrated by DynamicalDataset, while steps 1, 2 and 3 happen in RegionJob.process.

## Implementation Phases

### Phase 1: Enhanced RegionJob.process with tmp_store support
- Add `tmp_store` parameter to `RegionJob`
- Modify `RegionJob.process()` to write to tmp_store first
- Implement pipelined upload of chunk data to final_store
- Add concurrent processing: download → read/compress/write → upload
- **TODO: Integrate UpdateProgressTracker into RegionJob.process for resumption support**

### Phase 2: Operational update job determination
- Add `RegionJob.operational_update_jobs()` class method
- Update method signatures to use `final_store` consistently

### Phase 3: Template update interface
- Add `RegionJob.update_template_with_results()` instance method
- Modify `RegionJob.process()` to return `dict[str, Sequence[SOURCE_FILE_COORD]]`
- Implement base template update logic

### Phase 4: DynamicalDataset operational update orchestration
- Add `DynamicalDataset._tmp_store()` method
- Implement `DynamicalDataset.reformat_operational_update()`
- Integrate incremental metadata updates
- Handle temporary store cleanup

## Notes

- The temporary store pattern is essential: write full metadata to tmp_store for region="auto", but only copy chunk data (not metadata) to final_store until all chunks are written
- Pipelining is critical for performance: while variable N is uploading, variable N+1 should be reading/compressing
- Progress tracking allows resumption if operational update jobs are interrupted
- Incremental metadata updates make data available to readers as quickly as possible
