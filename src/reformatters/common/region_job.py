import os
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import suppress
from enum import Enum, auto
from itertools import batched
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Annotated, Any, ClassVar, Generic, TypeVar

import numpy as np
import pandas as pd
import pydantic
import xarray as xr
import zarr
from pydantic import AfterValidator, Field

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.config_models import DataVar
from reformatters.common.iterating import dimension_slices, get_worker_jobs
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel, replace
from reformatters.common.reformat_utils import (
    create_data_array_and_template,
)
from reformatters.common.shared_memory_utils import make_shared_buffer, write_shards
from reformatters.common.types import AppendDim, ArrayFloat32, Dim, Timestamp

logger = get_logger(__name__)

type CoordinateValueOrRange = slice | int | float | pd.Timestamp | pd.Timedelta | str


class SourceFileStatus(Enum):
    Processing = auto()
    DownloadFailed = auto()
    ReadFailed = auto()
    Succeeded = auto()


class SourceFileCoord(FrozenBaseModel):
    """
    Base class representing the coordinates and status of a single source file required for processing.

    Subclasses should define dataset-specific fields (e.g., data_vars, init_time, lead_time, file_type) required
    to uniquely identify a source file and implement the `get_url` and `out_loc` methods.

    Attributes
    ----------
    status : SourceFileStatus
        The current processing status of this file (Processing, DownloadFailed, ReadFailed, Succeeded).
    downloaded_path : Path | None
        Local filesystem path to the downloaded file, or None if not downloaded.
    """

    status: SourceFileStatus = Field(default=SourceFileStatus.Processing, frozen=False)
    downloaded_path: Path | None = Field(default=None, frozen=False)

    def get_url(self) -> str:
        """Return the URL for this source file."""
        raise NotImplementedError("Return the URL of the source file.")

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValueOrRange]:
        """
        Return a data array indexer which identifies the region in the output dataset
        to write the data from the source file. The indexer is a dict from dimension
        names to coordinate values or slices.

        If the names of the coordinate attributes of your SourceFileCoord subclass are also all
        dimension names in the output dataset, use the default implementation of this method.

        Examples where you would override this method:
        - For an analysis dataset created from forecast data: {"time": self.init_time + self.lead_time}
        """
        # .model_dump() returns a dict from attribute names to values
        return self.model_dump(exclude=["status", "downloaded_path"])  # type: ignore


DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD", bound=SourceFileCoord)


def region_slice(s: slice) -> slice:
    if not (isinstance(s.start, int) and isinstance(s.stop, int) and s.step is None):
        raise ValueError("region must be integer slice")
    return s


class RegionJob(pydantic.BaseModel, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    store: zarr.abc.store.Store
    template_ds: xr.Dataset
    data_vars: Sequence[DATA_VAR]
    append_dim: AppendDim
    # integer slice along append_dim
    region: Annotated[slice, AfterValidator(region_slice)]

    # Limit the number of variables processed in each backfill job if set.
    max_vars_per_backfill_job: ClassVar[int | None] = None
    # Limit the number of variables processed in each download group if set.
    # If value is less than len(data_vars), downloading, reading/recompressing, and writing steps
    # will be pipelined within a region job.
    max_vars_per_download_group: ClassVar[int | None] = None

    # Subclasses can override this to control download parallelism
    # This particularly useful of the data source cannot handle a large number of concurrent requests
    download_parallelism: int = (os.cpu_count() or 1) * 2

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[DATA_VAR],
    ) -> Sequence[Sequence[DATA_VAR]]:
        """
        Return groups of variables, where all variables in a group can be retrieived from the same source file.

        This is a class method so it can be called by RegionJob factory methods.
        """
        return [data_vars]

    def get_processing_region(self) -> slice:
        """
        Return a slice of integer offsets into self.template_ds along self.append_dim that identifies
        the region to process. In most cases this is exactly self.region, but if additional data outside
        the region is required, for example for correct interpolation or deaccumulation, this method can
        return a modified slice (e.g. `slice(self.region.start - 1, self.region.stop + 1)`).
        """
        return self.region

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[DATA_VAR]
    ) -> Sequence[SOURCE_FILE_COORD]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds."""
        raise NotImplementedError(
            "Return a sequence of SourceFileCoord objects, one for each source file required to process the data covered by processing_region_ds."
        )

    def download_file(self, coord: SOURCE_FILE_COORD) -> Path:
        """Download the file for the given coordinate and return the local path."""
        raise NotImplementedError(
            "Download the file for the given coordinate and return the local path."
        )

    def read_data(
        self,
        coord: SOURCE_FILE_COORD,
        data_var: DATA_VAR,
    ) -> ArrayFloat32:
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
        ArrayFloat32
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
        source_file_coords: Sequence[SOURCE_FILE_COORD],
    ) -> Any:
        return None

    # ----- Most subclasses will not need to override the attributes and methods below -----

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True, frozen=True, strict=True
    )

    @classmethod
    def get_backfill_jobs(
        cls,
        store: zarr.abc.store.Store,
        template_ds: xr.Dataset,
        append_dim: AppendDim,
        all_data_vars: Sequence[DATA_VAR],
        worker_index: int = 0,
        workers_total: int = 1,
        filter_start: Timestamp | None = None,
        filter_end: Timestamp | None = None,
        filter_variable_names: list[str] | None = None,
    ) -> Sequence["RegionJob[DATA_VAR, SOURCE_FILE_COORD]"]:
        """
        Return a sequence of RegionJob instances for backfill processing.

        If `workers_total` and `worker_index` are provided the returned jobs are
        filtered to only include jobs which should be processed by `worker_index`.

        If any of the `filter_*` arguments are provided, the returned jobs are filtered
        to only include jobs which intersect the filter. Complete jobs are always returned,
        so regions may extend outside `filter_start` and `filter_end`.

        Parameters
        ----------
        store : zarr.abc.store.Store
            The Zarr store to write into.
        template_ds : xr.Dataset
            Dataset template defining structure and metadata.
        append_dim : AppendDim
            The dimension along which data is appended (e.g., "time").
        all_data_vars : Sequence[DATA_VAR]
            Sequence of all data variable configs for this dataset.
            Provided so that grouping and RegionJob made access DataVar.internal_attrs.

        worker_index : int, default 0
        workers_total : int, default 1

        filter_start : Timestamp | None, default None
        filter_end : Timestamp | None, default None
        filter_variable_names : list[str] | None, default None

        Returns
        -------
        Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]]
            RegionJob instances assigned to this worker.
        """

        # Data variables -- filter and group
        assert {v.name for v in all_data_vars} == set(template_ds.data_vars)

        data_vars: Sequence[DATA_VAR]
        if filter_variable_names:
            data_vars = [v for v in all_data_vars if v.name in filter_variable_names]
        else:
            data_vars = all_data_vars

        data_var_groups = cls.source_groups(data_vars)
        if cls.max_vars_per_backfill_job is not None:
            data_var_groups = cls._maybe_split_groups(
                data_var_groups, cls.max_vars_per_backfill_job
            )

        # Regions along append dimension
        regions = dimension_slices(template_ds, append_dim, kind="shards")

        # Filter regions
        append_dim_coords = template_ds.coords[append_dim]
        if filter_start is not None:
            regions = [
                region
                for region in regions
                if append_dim_coords[region].max() >= filter_start  # type: ignore[operator]
            ]
        if filter_end is not None:
            regions = [
                region
                for region in regions
                if append_dim_coords[region].min() < filter_end  # type: ignore[operator]
            ]

        all_jobs = [
            cls(
                store=store,
                template_ds=template_ds,
                data_vars=data_var_group,
                append_dim=append_dim,
                region=region,
            )
            for region in regions
            for data_var_group in data_var_groups
        ]

        return get_worker_jobs(all_jobs, worker_index, workers_total)

    def process(self) -> dict[str, Any]:
        """
        Orchestrate the full region job processing pipeline.

        1. Group data variables for efficient processing (e.g., by file type or batch size).
        2. For each group of data variables:
            a. Download all required source files
            b. For each variable in the group:
                i.   Read data from source files into the shared array
                ii.  Apply any required data transformations (e.g., rounding, deaccumulation).
                iii. Write output shards to the Zarr store in parallel.

        Returns:
            dict[str, Any]: Mapping from data variable name to the output of `self.summarize_processing_state`.
        """
        processing_region_ds, output_region_ds = self._get_region_datasets()
        data_var_groups = self.source_groups(self.data_vars)
        if self.max_vars_per_download_group is not None:
            data_var_groups = self._maybe_split_groups(
                data_var_groups, self.max_vars_per_download_group
            )

        results: dict[str, Any] = {}
        with make_shared_buffer(processing_region_ds) as shared_buffer:
            for data_var_group in data_var_groups:
                source_file_coords = self.generate_source_file_coords(
                    processing_region_ds,
                    data_var_group,
                )
                source_file_coords = self._download_processing_group(source_file_coords)

                for data_var in data_var_group:
                    data_array, data_array_template = create_data_array_and_template(
                        processing_region_ds,
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
                        output_region_ds,
                        self.store,
                    )
                    results[data_var.name] = self.summarize_processing_state(
                        data_var,
                        source_file_coords,
                    )
                self._cleanup_local_files(source_file_coords)
        return results

    def _get_region_datasets(self) -> tuple[xr.Dataset, xr.Dataset]:
        ds = self.template_ds[[v.name for v in self.data_vars]]
        processing_region = self.get_processing_region()
        processing_region_ds = ds.isel({self.append_dim: processing_region})
        output_region_ds = ds.isel({self.append_dim: self.region})
        return processing_region_ds, output_region_ds

    @classmethod
    def _maybe_split_groups(
        cls, data_var_groups: Sequence[Sequence[DATA_VAR]], batch_size: int
    ) -> Sequence[Sequence[DATA_VAR]]:
        """Splits inner groups into smaller groups of at most batch_size."""
        return [
            tuple(split_group)
            for group in data_var_groups
            for split_group in batched(group, batch_size)
        ]

    def _download_processing_group(
        self, source_file_coords: Iterable[SOURCE_FILE_COORD]
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
                append_dim_coord = getattr(coord, self.append_dim, pd.Timestamp.min)
                if isinstance(append_dim_coord, slice):
                    append_dim_coord = append_dim_coord.start
                day_ago = pd.Timestamp.now() - pd.Timedelta(hours=24)
                if (
                    isinstance(e, FileNotFoundError)
                    and isinstance(append_dim_coord, np.datetime64 | pd.Timestamp)
                    and append_dim_coord > day_ago
                ):
                    logger.info(" ".join(str(e).split("\n")[:2]))
                else:
                    logger.exception(f"Download failed {coord.get_url()}")

                return updated_coord

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

        def _read_and_write_one(coord: SOURCE_FILE_COORD) -> SOURCE_FILE_COORD:
            try:
                out.loc[coord.out_loc()] = self.read_data(coord, data_var)
                return replace(coord, status=SourceFileStatus.Succeeded)
            except Exception:
                logger.exception(f"Read failed {coord.downloaded_path}")
                return replace(coord, status=SourceFileStatus.ReadFailed)

        # Skip coords where the download failed
        read_coords = (
            c for c in source_file_coords if c.status == SourceFileStatus.Processing
        )
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            return list(executor.map(_read_and_write_one, read_coords))

    def _write_shards(
        self,
        processing_region_da_template: xr.DataArray,
        shared_buffer: SharedMemory,
        output_region_ds: xr.Dataset,
        store: zarr.abc.store.Store,
    ) -> None:
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as process_executor:
            write_shards(
                processing_region_da_template,
                shared_buffer,
                self.append_dim,
                output_region_ds,
                store,
                process_executor,
            )

    def _cleanup_local_files(
        self, source_file_coords: Sequence[SOURCE_FILE_COORD]
    ) -> None:
        for coord in source_file_coords:
            if coord.downloaded_path:
                with suppress(FileNotFoundError):
                    coord.downloaded_path.unlink()

    def summary(self) -> str:
        return f"({self.region.start} - {self.region.stop}) {[d.name for d in self.data_vars]}"
