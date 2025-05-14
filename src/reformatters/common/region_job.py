import os
from collections.abc import Generator, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum, auto
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Annotated, Any, Generic, TypeVar

import pydantic
import xarray as xr
import zarr
from pydantic import AfterValidator

from reformatters.common.config_models import DataVar
from reformatters.common.template_config import AppendDim


class SourceFileStatus(Enum):
    Processing = auto()
    DownloadFailed = auto()
    ReadFailed = auto()
    Succeeded = auto()


class SourceFileCoord(pydantic.BaseModel):
    status: SourceFileStatus = SourceFileStatus.Processing
    downloaded_path: Path | None = None

    def get_url(self) -> str:
        raise NotImplementedError("Subclasses must implement get_url")

    def out_loc(self) -> Mapping[str, Any]:
        raise NotImplementedError("Subclasses must implement out_loc")


DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])


class RegionJob(pydantic.BaseModel, Generic[DATA_VAR]):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    store: zarr.storage.FsspecStore
    template_ds: xr.Dataset
    data_vars: Sequence[DATA_VAR]
    append_dim: AppendDim
    region: Annotated[
        slice,
        AfterValidator(
            lambda s: isinstance(s.start, int)
            and isinstance(s.stop, int)
            and s.step is None
        ),
    ]
    max_vars_per_backfill_job: int

    def process(self) -> dict[str, Any]:
        """
        Orchestrate the full region job processing pipeline.

        This method manages the end-to-end workflow for processing a region of a dataset,
        including grouping variables, downloading required files, reading and transforming data,
        and writing output shards. The steps are:

        1. Extract the processing region from the template dataset using `get_processing_region`.
        2. Group data variables for efficient processing (e.g., by file type or batch size).
        3. Set up shared resources, including a shared memory buffer and thread/process executors.
        4. For each group of data variables:
            a. Download all required source files in parallel, updating their status.
            b. For each variable in the group:
                i.   Create a shared-memory-backed output array and template.
                ii.  Read data from source files into the shared array, in parallel.
                iii. Apply any required data transformations (e.g., rounding, deaccumulation).
                iv.  Write output shards to the Zarr store in parallel.
                v.   Collect and summarize processing metadata for the variable.
            c. Clean up any temporary local files.
        5. Return a dictionary mapping each variable name to its processing summary.

        Returns:
            dict[str, Any]: Mapping from data variable name to the output of `summarize_processing_state`.
        """
        processing_ds = self.processing_ds()
        with self._make_shared_buffer(processing_ds) as shared_buffer:
            # TODO: any more specific type hints or some summary Type that could make this clearer?
            #       Is Any correct here? Do we allow users to define the form of a summary and not just its contents?
            results: dict[str, Any] = {}

            # Group vars and process each group
            for data_var_group in self.group_data_vars(processing_ds):
                source_file_coords = self._download_processing_group(processing_ds)
                for data_var in data_var_group:
                    data_array, data_array_template = (
                        self._create_data_array_and_template(
                            processing_ds, data_var, shared_buffer
                        )
                    )
                    self._read_into_data_array(data_array, data_var, source_file_coords)
                    self.apply_data_transformations(data_array, data_var)
                    self._write_shards(
                        data_array_template, shared_buffer, processing_ds, self.store
                    )
                    results[data_var.name] = self.summarize_processing_state(
                        data_var, source_file_coords
                    )
                self._cleanup_local_files(source_file_coords)
            return results

    # TODO: Consider calling this region_ds?
    # TODO: Consider making this a cached_property or just memoize the result to simplify our other method signatures?
    def processing_ds(self) -> xr.Dataset:
        region_slice = self.get_processing_region(self.region)
        var_names = [v.name for v in self.data_vars]
        return self.template_ds[var_names].isel({self.append_dim: region_slice})

    def get_processing_group_size(self) -> int:
        match len(self.data_vars):
            case n if n > 6:
                return 4
            case n if n > 3:
                return 2
            case _:
                return 1

    def get_processing_region(self, original_slice: slice) -> slice:
        return original_slice

    def group_data_vars(self, chunk_ds: xr.Dataset) -> Iterator[Sequence[DATA_VAR]]:
        from itertools import batched

        return batched(self.data_vars, self.max_vars_per_backfill_job)

    def generate_source_file_coords(
        self, chunk_ds: xr.Dataset
    ) -> Sequence[SourceFileCoord]:
        raise NotImplementedError(
            "Subclasses must implement generate_source_file_coords"
        )

    def download_file(self, coord: SourceFileCoord) -> Path | None:
        raise NotImplementedError("Subclasses must implement download_file")

    def read_data(
        self,
        coord: SourceFileCoord,
        *,
        out: xr.DataArray,
        data_var: DATA_VAR,
    ) -> Any:
        raise NotImplementedError("Subclasses must implement read_data")

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: DATA_VAR
    ) -> None:
        from reformatters.common.binary_rounding import round_float32_inplace

        keep_mantissa_bits = data_var.internal_attrs.keep_mantissa_bits
        if isinstance(keep_mantissa_bits, int):
            round_float32_inplace(
                data_array.values, keep_mantissa_bits=keep_mantissa_bits
            )

    def summarize_processing_state(
        self,
        data_var: DATA_VAR,
        source_file_coords: Sequence[SourceFileCoord],
    ) -> Any:
        raise NotImplementedError(
            "Subclasses must implement summarize_processing_state"
        )

    def _calc_shared_buffer_size(self, chunk_ds: xr.Dataset) -> int:
        return max(var.nbytes for var in chunk_ds.data_vars.values())

    @contextmanager
    def _make_shared_buffer(
        self, processing_ds: xr.Dataset
    ) -> Generator[SharedMemory, None, None]:
        buffer_size = self._calc_shared_buffer_size(processing_ds)
        shared_memory = SharedMemory(create=True, size=buffer_size)
        try:
            yield shared_memory
        finally:
            shared_memory.close()
            shared_memory.unlink()

    def _download_processing_group(
        self,
        chunk_ds: xr.Dataset,
    ) -> list[SourceFileCoord]:
        coords = self.generate_source_file_coords(chunk_ds)
        from concurrent.futures import as_completed

        io_executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)
        futures = {
            io_executor.submit(self.download_file, coord): coord for coord in coords
        }
        results: list[SourceFileCoord] = []
        for future in as_completed(futures):
            coord = futures[future]
            try:
                path = future.result()
                coord.downloaded_path = path
                coord.status = (
                    SourceFileStatus.Succeeded
                    if path
                    else SourceFileStatus.DownloadFailed
                )
            except Exception:
                coord.status = SourceFileStatus.DownloadFailed
                coord.downloaded_path = None
            results.append(coord)
        io_executor.shutdown(wait=True)
        return results

    def _create_data_array_and_template(
        self,
        chunk_ds: xr.Dataset,
        data_var: DATA_VAR,
        shared_buffer: SharedMemory,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        from reformatters.common.reformat_utils import create_data_array_and_template

        return create_data_array_and_template(chunk_ds, data_var.name, shared_buffer)

    def _read_into_data_array(
        self,
        out: xr.DataArray,
        data_var: DATA_VAR,
        source_file_coords: Sequence[SourceFileCoord],
    ) -> None:
        from functools import partial

        from reformatters.common.iterating import consume

        cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
        consume(
            cpu_executor.map(
                partial(self.read_data, out=out, data_var=data_var),
                source_file_coords,
            )
        )
        cpu_executor.shutdown(wait=True)

    def _write_shards(
        self,
        data_array_template: xr.DataArray,
        shared_buffer: SharedMemory,
        chunk_ds: xr.Dataset,
        store: zarr.storage.FsspecStore,
    ) -> None:
        from reformatters.common.reformat_utils import write_shards

        write_shards(
            data_array_template,
            store,
            shared_buffer,
            ProcessPoolExecutor(max_workers=os.cpu_count() or 1),
        )

    def _cleanup_local_files(
        self, source_file_coords: Sequence[SourceFileCoord]
    ) -> None:
        # TODO: Could make this a method on SourceFileCoord and just do
        # for coord in source_file_coords:
        #     coord.cleanup()
        for coord in source_file_coords:
            if coord.downloaded_path:
                coord.downloaded_path.unlink()
