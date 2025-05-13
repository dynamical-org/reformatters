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
from pydantic.functional_validators import AfterValidator
from zarr.storage import FsspecStore

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
    store: FsspecStore
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
        Orchestrate the processing pipeline.

        Returns:
            Dict mapping data_var.name to summarize_processing_state output.
        """
        processing_slice = self.get_processing_region(self.region)
        processing_ds = self.template_ds.isel({self.append_dim: processing_slice})
        buffer_size = self._calc_shared_buffer_size(processing_ds)
        with self._make_shared_buffer(buffer_size) as shared_buffer:
            results: dict[str, Any] = {}
            # Group vars and process each group
            for data_var_group in self.group_data_vars(processing_ds):
                coords_and_paths = self._download_processing_group(
                    processing_ds, data_var_group
                )
                for data_var in data_var_group:
                    data_array, data_array_template = (
                        self._create_data_array_and_template(
                            processing_ds, data_var, shared_buffer
                        )
                    )
                    self._read_into_data_array(data_array, data_var, coords_and_paths)
                    self.apply_data_transformations(data_array, data_var)
                    self._write_shards(
                        data_array_template, shared_buffer, processing_ds, self.store
                    )
                    results[data_var.name] = self.summarize_processing_state(
                        data_var, coords_and_paths
                    )
                # cleanup local files
                for _coord, path in coords_and_paths:
                    if path is not None:
                        path.unlink()
            return results

    def region_template_ds(self) -> xr.Dataset:
        var_names = [v.name for v in self.data_vars]
        return self.template_ds[var_names].isel({self.append_dim: self.region})

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
        coords_and_paths: Sequence[tuple[SourceFileCoord, Path | None]],
    ) -> Any:
        raise NotImplementedError(
            "Subclasses must implement summarize_processing_state"
        )

    def _calc_shared_buffer_size(self, chunk_ds: xr.Dataset) -> int:
        return max(var.nbytes for var in chunk_ds.data_vars.values())

    @contextmanager
    def _make_shared_buffer(self, size: int) -> Generator[SharedMemory, None, None]:
        shared_memory = SharedMemory(create=True, size=size)
        try:
            yield shared_memory
        finally:
            shared_memory.close()
            shared_memory.unlink()

    def _download_processing_group(
        self,
        chunk_ds: xr.Dataset,
        data_vars: Sequence[DATA_VAR],
    ) -> list[tuple[SourceFileCoord, Path | None]]:
        coords = self.generate_source_file_coords(chunk_ds)
        from concurrent.futures import as_completed

        io_executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)
        futures = {
            io_executor.submit(self.download_file, coord): coord for coord in coords
        }
        results: list[tuple[SourceFileCoord, Path | None]] = []
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
                results.append((coord, path))
            except Exception:
                coord.status = SourceFileStatus.DownloadFailed
                results.append((coord, None))
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
        coords_and_paths: Sequence[tuple[SourceFileCoord, Path | None]],
    ) -> None:
        from functools import partial

        from reformatters.common.iterating import consume

        cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
        consume(
            cpu_executor.map(
                partial(self.read_data, out=out, data_var=data_var),
                *zip(*coords_and_paths, strict=True),
            )
        )
        cpu_executor.shutdown(wait=True)

    def _write_shards(
        self,
        data_array_template: xr.DataArray,
        shared_buffer: SharedMemory,
        chunk_ds: xr.Dataset,
        store: FsspecStore,
    ) -> None:
        from reformatters.common.iterating import consume
        from reformatters.common.reformat_utils import write_shards

        write_shards(
            data_array_template,
            store,
            shared_buffer,
            ProcessPoolExecutor(max_workers=os.cpu_count() or 1),
        )
