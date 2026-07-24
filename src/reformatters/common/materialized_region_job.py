import concurrent.futures
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import suppress
from copy import deepcopy
from http import HTTPStatus
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, ClassVar, Generic, cast

import numpy as np
import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common import storage, template_utils
from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.download import http_status_code
from reformatters.common.iterating import split_groups
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import replace
from reformatters.common.region_job import (
    DATA_VAR,
    SOURCE_FILE_COORD,
    RegionJob,
    SourceFileResult,
    SourceFileStatus,
)
from reformatters.common.shared_memory_utils import (
    create_data_array_and_template,
    make_shared_buffer,
    write_shards,
)
from reformatters.common.types import ArrayND
from reformatters.common.zarr import copy_data_var

log = get_logger(__name__)


class MaterializedRegionJob(
    RegionJob[DATA_VAR, SOURCE_FILE_COORD], Generic[DATA_VAR, SOURCE_FILE_COORD]
):
    # Reformats source data into rechunked (materialized) chunk data: download
    # source files, read them into shared-memory arrays, write output shards.
    # Base class for all existing rechunked datasets.

    # Limit the number of variables processed in each download group if set.
    # If value is less than len(data_vars), downloading, reading/recompressing, and writing steps
    # will be pipelined within a region job.
    max_vars_per_download_group: ClassVar[int | None] = None

    # Subclasses can override this to control download parallelism
    # This particularly useful of the data source cannot handle a large number of concurrent requests
    download_parallelism: int = (os.cpu_count() or 1) * 2

    # Subclasses can override this to control read parallelism.
    # This is useful in cases where many threads reading data into shared memory
    # causes deadlocks or resource contention.
    read_parallelism: int = os.cpu_count() or 1

    def download_file(self, coord: SOURCE_FILE_COORD) -> Path:
        """Download the file for the given coordinate and return the local path."""
        raise NotImplementedError(
            "Download the file for the given coordinate and return the local path."
        )

    def read_data(
        self,
        coord: SOURCE_FILE_COORD,
        data_var: DATA_VAR,
    ) -> ArrayND[np.generic]:
        """
        Read and return the data chunk for the given variable and source file coordinate.

        Subclasses must implement this to load the data (e.g., from a file or remote source)
        for the specified coord and data_var. The returned array will be written into the shared
        output array by the base class.

        Parameters
        ----------
        coord : SOURCE_FILE_COORD
            The coordinate specifying which file and region to read.
        data_var : DATA_VAR
            The data variable metadata.

        Returns
        -------
        ArrayND[np.generic]
            The loaded data.
        """
        raise NotImplementedError(
            "Read and return data for the given variable and source file coordinate."
        )

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: DATA_VAR
    ) -> None:
        """
        Apply in-place data transformations to the output data array for a given data variable.

        This method is called after reading all data for a variable into the shared-memory array,
        and before writing shards to the output store. The default implementation applies binary
        rounding to float32 arrays if `data_var.internal_attrs.keep_mantissa_bits` is set.

        Subclasses may override this method to implement additional transformations such as
        deaccumulation, interpolation or other custom logic. All transformations should be
        performed in-place (don't copy `data_array`, it's large).

        Parameters
        ----------
        data_array : xr.DataArray
            The output data array to be transformed in-place.
        data_var : DATA_VAR
            The data variable metadata object, which may contain transformation parameters.
        """
        keep_mantissa_bits = data_var.internal_attrs.keep_mantissa_bits
        if isinstance(keep_mantissa_bits, int):
            round_float32_inplace(
                data_array.values, keep_mantissa_bits=keep_mantissa_bits
            )

    @classmethod
    def process_worker_jobs(
        cls,
        worker_jobs: Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]],
        store_factory: storage.StoreFactory,
        branch_name: str,
        worker_index: int,
    ) -> dict[str, list[SourceFileResult]]:
        """Write all of this worker's jobs to ``branch_name`` in a single commit."""
        # One commit per worker; an empty job set would make an empty icechunk
        # commit (which raises), so the caller must filter empty workers out.
        assert worker_jobs, "process_worker_jobs requires at least one job"
        primary_store = store_factory.primary_store(writable=True, branch=branch_name)
        replica_stores = store_factory.replica_stores(writable=True, branch=branch_name)

        assert all(isinstance(job, MaterializedRegionJob) for job in worker_jobs)
        jobs = cast(
            "Sequence[MaterializedRegionJob[DATA_VAR, SOURCE_FILE_COORD]]", worker_jobs
        )
        del worker_jobs

        worker_results: dict[str, list[SourceFileResult]] = {}
        for job in jobs:
            template_utils.write_metadata(job.template_ds, job.tmp_store)
            results = job.process(
                primary_store=primary_store, replica_stores=replica_stores
            )
            for var_name, coords in results.items():
                worker_results.setdefault(var_name, []).extend(
                    SourceFileResult(
                        status=c.status,
                        out_loc={**c.out_loc()},
                        url=c.get_url(),
                    )
                    for c in coords
                )

        now = pd.Timestamp.now(tz="UTC")
        storage.commit_if_icechunk(
            f"Update worker {worker_index} at {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            primary_store,
            replica_stores,
        )
        return worker_results

    def process(
        self,
        primary_store: Store,
        replica_stores: list[Store],
    ) -> Mapping[str, Sequence[SOURCE_FILE_COORD]]:
        """
        Orchestrate the full region job processing pipeline.

        1. Group data variables for efficient processing (e.g., by file type or batch size)
        2. Write zarr metadata to tmp store for region="auto" support
        3. For each group of data variables:
            a. Download all required source files
            b. For each variable in the group:
                i.   Read data from source files into the shared array
                ii.  Apply any required data transformations (e.g., rounding, deaccumulation)
                iii. Write output shards to the tmp_store
                iv.  Upload chunk data from tmp_store to the primary store

        Returns
        -------
        Mapping[str, Sequence[SOURCE_FILE_COORD]]
            Mapping from variable names to their source file coordinates with final processing status.
        """
        processing_region_ds, output_region_ds = self._get_region_datasets()

        data_var_groups = self.source_file_var_groups(self.data_vars)
        if self.max_vars_per_download_group is not None:
            data_var_groups = split_groups(
                data_var_groups, self.max_vars_per_download_group
            )

        results: dict[str, Sequence[SOURCE_FILE_COORD]] = {}
        upload_futures: list[Any] = []

        with (
            make_shared_buffer(processing_region_ds) as shared_buffer,
            ThreadPoolExecutor(max_workers=1) as download_executor,
            ThreadPoolExecutor(max_workers=1) as upload_executor,
            ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as write_executor,
        ):
            log.info(f"Starting {self!r}")

            # Submit all download tasks to the executor
            download_futures = {}
            for data_var_group in data_var_groups:
                data_var_names = [v.name for v in data_var_group]
                source_file_coords = self.generate_source_file_coords(
                    processing_region_ds, data_var_group
                )
                download_future = download_executor.submit(
                    self._download_processing_group, source_file_coords, data_var_names
                )
                download_futures[download_future] = data_var_group

            # Process downloaded data var groups as they complete
            for download_future in concurrent.futures.as_completed(download_futures):
                data_var_group = download_futures[download_future]
                source_file_coords = download_future.result()
                log.info(f"Downloaded: {[v.name for v in data_var_group]}")

                # Process one data variable at a time to ensure a single user of
                # the shared buffer at a time and to reduce peak memory usage
                for data_var in data_var_group:
                    # Copy so we have a unique status per variable, not per variable group
                    data_var_source_file_coords = deepcopy(source_file_coords)
                    data_array, data_array_template = create_data_array_and_template(
                        processing_region_ds,
                        data_var.name,
                        shared_buffer,
                        fill_value=data_var.encoding.fill_value,
                    )
                    data_var_source_file_coords = self._read_into_data_array(
                        data_array,
                        data_var,
                        data_var_source_file_coords,
                    )
                    self.apply_data_transformations(
                        data_array,
                        data_var,
                    )
                    self._write_shards(
                        data_array_template,
                        shared_buffer,
                        output_region_ds,
                        self.tmp_store,
                        write_executor,
                    )

                    upload_futures.append(
                        upload_executor.submit(
                            copy_data_var,
                            data_var.name,
                            self.region,
                            output_region_ds,
                            self.append_dim,
                            self.tmp_store,
                            primary_store,
                            replica_stores=replica_stores,
                        )
                    )

                    results[data_var.name] = data_var_source_file_coords

                self._cleanup_local_files(source_file_coords)  # after _group_ is done

        for future in concurrent.futures.as_completed(upload_futures):
            if (e := future.exception()) is not None:
                raise e

        return results

    def _get_region_datasets(self) -> tuple[xr.Dataset, xr.Dataset]:
        # Materialized datasets are single-level (DynamicalDataset enforces it), so the
        # root node holds every var; subset it to this job's vars by name.
        ds: xr.Dataset = self.template_ds.to_dataset()[[v.name for v in self.data_vars]]  # ty: ignore[invalid-assignment]
        processing_region = self.get_processing_region()
        processing_region_ds = ds.isel({self.append_dim: processing_region})
        output_region_ds = ds.isel({self.append_dim: self.region})
        return processing_region_ds, output_region_ds

    def _download_processing_group(
        self,
        source_file_coords: Sequence[SOURCE_FILE_COORD],
        data_var_names: Sequence[str],
    ) -> list[SOURCE_FILE_COORD]:
        """
        Download specified source files in parallel.

        Returns
        -------
        list[SOURCE_FILE_COORD]
            List of SourceFileCoord objects with updated download status and path.
        """

        def _call_download_file(coord: SOURCE_FILE_COORD) -> SOURCE_FILE_COORD:
            try:
                path = self.download_file(coord)
                return replace(coord, downloaded_path=path)
            except Exception as e:
                updated_coord = replace(coord, status=SourceFileStatus.DownloadFailed)

                # For recent files, we expect some files to not exist yet, just log the path
                # else, log exception so it is caught by error reporting but doesn't stop processing
                append_dim_coord = coord.append_dim_coord
                two_days_ago = pd.Timestamp.now() - pd.Timedelta(hours=48)
                is_not_found = isinstance(e, FileNotFoundError) or (
                    http_status_code(e) in (HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND)
                )
                if (
                    is_not_found
                    and isinstance(append_dim_coord, np.datetime64 | pd.Timestamp)
                    and append_dim_coord > two_days_ago
                ):
                    log.info(" ".join(str(e).split("\n")[:2]))
                else:
                    log.exception(f"Download failed {coord.get_url()}")

                return updated_coord

        log.info(f"Downloading {data_var_names} in {len(source_file_coords)} files...")
        with ThreadPoolExecutor(
            max_workers=self.download_parallelism
        ) as download_executor:
            return list(download_executor.map(_call_download_file, source_file_coords))

    def _read_into_data_array(
        self,
        out: xr.DataArray,
        data_var: DATA_VAR,
        source_file_coords: Sequence[SOURCE_FILE_COORD],
    ) -> list[SOURCE_FILE_COORD]:
        """
        Reads data from source files into `out`.
        Returns a list of coords with the final status.
        """
        # Skip coords where the download failed
        read_coords = (
            c for c in source_file_coords if c.status == SourceFileStatus.Processing
        )

        def _read_and_write_one(coord: SOURCE_FILE_COORD) -> ArrayND[np.generic]:
            return self.read_data(coord, data_var)

        # Index is used to maintain order of coords
        updated_coords: dict[int, SOURCE_FILE_COORD] = {}

        log.info(f"Reading {data_var.name}...")
        with ThreadPoolExecutor(max_workers=self.read_parallelism) as executor:
            futures = {}
            for i, coord in enumerate(read_coords):
                future = executor.submit(_read_and_write_one, coord)
                futures[future] = (i, coord)

            for future in concurrent.futures.as_completed(futures):
                index, coord = futures[future]
                try:
                    out.loc[coord.out_loc()] = future.result()
                    updated_coords[index] = replace(
                        coord, status=SourceFileStatus.Succeeded
                    )
                except Exception:
                    log.exception(f"Read failed {coord.downloaded_path}")
                    updated_coords[index] = replace(
                        coord, status=SourceFileStatus.ReadFailed
                    )
                finally:
                    # as_completed retains futures and results; clear to avoid ~2x peak memory
                    future._result = None  # noqa: SLF001

        sorted_updated_coords = [updated_coords[i] for i in sorted(updated_coords)]
        return sorted_updated_coords

    def _write_shards(
        self,
        processing_region_da_template: xr.DataArray,
        shared_buffer: SharedMemory,
        output_region_ds: xr.Dataset,
        tmp_store: Path,
        write_executor: ProcessPoolExecutor,
    ) -> None:
        write_shards(
            processing_region_da_template,
            shared_buffer,
            self.append_dim,
            output_region_ds,
            tmp_store,
            write_executor,
        )

    def _cleanup_local_files(
        self, source_file_coords: Sequence[SOURCE_FILE_COORD]
    ) -> None:
        for coord in source_file_coords:
            if coord.downloaded_path:
                with suppress(FileNotFoundError):
                    coord.downloaded_path.unlink()
