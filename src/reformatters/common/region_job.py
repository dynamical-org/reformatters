import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from enum import Enum, auto
from itertools import batched
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Annotated, Any, ClassVar, Generic, TypeVar

import pandas as pd
import pydantic
import xarray as xr
import zarr
from numpy.typing import NDArray
from pydantic import AfterValidator

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.config_models import DataVar
from reformatters.common.logging import get_logger
from reformatters.common.reformat_utils import (
    create_data_array_and_template,
    make_shared_buffer,
    write_shards,
)
from reformatters.common.template_config import AppendDim, Dim

logger = get_logger(__name__)

type CoordinateValueOrRange = slice | int | float | pd.Timestamp | pd.Timedelta | str
DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])


class SourceFileStatus(Enum):
    Processing = auto()
    DownloadFailed = auto()
    ReadFailed = auto()
    Succeeded = auto()


class SourceFileCoord(pydantic.BaseModel):
    """
    Base class representing the coordinates and status of a single source file required for processing.

    Subclasses should define dataset-specific fields (e.g., init_time, lead_time, file_type) and
    implement the `get_url` and `out_loc` methods.

    Attributes
    ----------
    status : SourceFileStatus
        The current processing status of this file (Processing, DownloadFailed, ReadFailed, Succeeded).
    downloaded_path : Path | None
        Local filesystem path to the downloaded file, or None if not downloaded.
    """

    status: SourceFileStatus = SourceFileStatus.Processing
    downloaded_path: Path | None = None

    def get_url(self) -> str:
        """
        Return the remote URL for this source file.

        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement get_url")

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValueOrRange]:
        """
        Return the output location mapping for this file's data within the output array.

        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement out_loc")


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
    max_vars_per_backfill_job: ClassVar[int]

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
        region_ds = self._region_ds()
        with make_shared_buffer(region_ds) as shared_buffer:
            results: dict[str, Any] = {}

            # Group vars and process each group
            for data_var_group in self._data_var_download_groups(region_ds):
                # TODO: Does _download_processing_group need to be called for each group?
                #       Should it be taking data_var_group?
                source_file_coords = self._download_processing_group(region_ds)
                for data_var in data_var_group:
                    data_array, data_array_template = create_data_array_and_template(
                        region_ds,
                        data_var.name,
                        shared_buffer,
                    )
                    self._read_into_data_array(
                        data_array,
                        data_var,
                        source_file_coords,
                    )
                    self.apply_data_transformations(
                        data_array,
                        data_var,
                    )
                    self._write_shards(
                        data_array_template,
                        shared_buffer,
                        region_ds,
                        self.store,
                    )
                    results[data_var.name] = self.summarize_processing_state(
                        data_var,
                        source_file_coords,
                    )
                self._cleanup_local_files(source_file_coords)
            return results

    def _region_ds(self) -> xr.Dataset:
        region_slice = self.get_processing_region(self.region)
        var_names = [v.name for v in self.data_vars]
        return self.template_ds[var_names].isel({self.append_dim: region_slice})

    def _data_var_download_groups(
        self, region_ds: xr.Dataset
    ) -> Sequence[Sequence[DATA_VAR]]:
        """
        Possibly split groups of data variables into group sizes ideal for downloading.

        When the batch_size is smaller than len(self.data_vars), downloading and reading/recompressing
        can begin on variables can begin on an already downloaded group, while other variables finish
        downloading.
        """
        data_var_groups = self.group_data_vars(region_ds)

        if len(self.data_vars) > 6:
            batch_size = 4
        elif len(self.data_vars) > 3:
            batch_size = 2
        else:
            batch_size = 1

        return [
            tuple(download_group)
            for group in data_var_groups
            for download_group in batched(group, batch_size)
        ]

    def get_processing_region(self, original_slice: slice) -> slice:
        return original_slice

    def group_data_vars(self, region_ds: xr.Dataset) -> Sequence[Sequence[DATA_VAR]]:
        """Return groups of variables, where all variables in a group can be retrieived from the same source file."""

        return [self.data_vars]

    def generate_source_file_coords(
        self, region_ds: xr.Dataset
    ) -> Sequence[SourceFileCoord]:
        raise NotImplementedError(
            "Subclasses must implement generate_source_file_coords"
        )

    def download_file(self, coord: SourceFileCoord) -> Path:
        raise NotImplementedError("Subclasses must implement download_file")

    def read_data(
        self,
        coord: SourceFileCoord,
        data_var: DATA_VAR,
    ) -> NDArray[Any]:  # TODO: A more specific type and then fix the docstring
        """
        Read and return the data chunk for the given variable and source file coordinate.

        Subclasses must implement this to load the data (e.g., from a file or remote source)
        for the specified coord and data_var. The returned array will be written into the shared
        output array by the base class.

        Parameters
        ----------
        coord : SourceFileCoord
            The coordinate specifying which file and region to read.
        data_var : DATA_VAR
            The data variable metadata.

        Returns
        -------
        NDArray[Any]
            The loaded data
        """
        raise NotImplementedError("Subclasses must implement read_data")

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: DATA_VAR
    ) -> None:
        """
        Apply in-place data transformations to the output data array for a given data variable.

        This method is called after reading all data for a variable into the shared-memory array,
        and before writing shards to the output store. The default implementation applies binary
        rounding to float32 arrays if `data_var.internal_attrs.keep_mantissa_bits` is set.

        Subclasses may override this method to implement additional transformations such as
        deaccumulation, interpolation, or other custom logic. All transformations should be
        performed in-place (i.e., do not copy data_array).

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

    def summarize_processing_state(
        self,
        data_var: DATA_VAR,
        source_file_coords: Sequence[SourceFileCoord],
    ) -> Any:
        raise NotImplementedError(
            "Subclasses must implement summarize_processing_state"
        )

    def _download_processing_group(
        self,
        region_ds: xr.Dataset,
    ) -> list[SourceFileCoord]:
        """
        Download all required source files for the given region dataset in parallel.

        This method generates the list of source file coordinates needed for the specified
        region and time chunk, attempts to download each file, and updates the download
        status and path for each coordinate.

        Parameters
        ----------
        region_ds : xr.Dataset
            The dataset representing the region and time chunk to process.

        Returns
        -------
        list[SourceFileCoord]
            List of SourceFileCoord objects with updated download status and path.
        """
        coords = self.generate_source_file_coords(region_ds)

        with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2) as io_executor:
            futures = {
                io_executor.submit(self.download_file, coord): coord for coord in coords
            }
            results: list[SourceFileCoord] = []
            for future in as_completed(futures):
                coord = futures[future]
                try:
                    path = future.result()
                    coord.downloaded_path = path
                except Exception as e:
                    coord.status = SourceFileStatus.DownloadFailed
                    if isinstance(e, FileNotFoundError) and getattr(
                        coord, self.append_dim, pd.Timestamp.min
                    ) > (pd.Timestamp.now() - pd.Timedelta(hours=24)):
                        # For recent files, we expect some files to not exist yet, just log the path
                        logger.info(" ".join(str(e).split("\n")[:2]))
                    else:
                        logger.exception("Download failed")

                results.append(coord)
        return results

    def _read_into_data_array(
        self,
        out: xr.DataArray,
        data_var: DATA_VAR,
        source_file_coords: Sequence[SourceFileCoord],
    ) -> None:
        """
        For each coord with status Processing, submit a job to:
          - call self.read_data(coord, data_var)
          - write the result into the shared array at coord.out_loc()
          - update coord.status to Succeeded or ReadFailed
        """

        def _read_and_write_one(coord: SourceFileCoord) -> None:
            try:
                # read_data should return a numpy array chunk
                chunk = self.read_data(coord, data_var=data_var)
                # Write the chunk into the correct location in the shared array
                out.loc[coord.out_loc()] = chunk
                coord.status = SourceFileStatus.Succeeded
            except Exception:
                coord.status = SourceFileStatus.ReadFailed

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            futures = [
                executor.submit(_read_and_write_one, coord)
                for coord in source_file_coords
                if coord.status == SourceFileStatus.Processing
            ]
            for future in as_completed(futures):
                future.result()  # propagate exceptions if any

    def _write_shards(
        self,
        data_array_template: xr.DataArray,
        shared_buffer: SharedMemory,
        chunk_ds: xr.Dataset,
        store: zarr.storage.FsspecStore,
    ) -> None:
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as process_executor:
            write_shards(
                data_array_template,
                store,
                shared_buffer,
                process_executor,
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
