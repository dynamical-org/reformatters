import warnings
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, contextmanager
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from pydantic import BaseModel

from reformatters.common.iterating import consume, shard_slice_indexers
from reformatters.common.logging import get_logger
from reformatters.common.types import ArrayFloat32

logger = get_logger(__name__)


class ChunkFilters(BaseModel):
    """
    Filters for controlling which chunks of data to process.
    A value of None means no filtering.
    """

    time_dim: str
    time_start: str | None = None
    time_end: str | None = None
    variable_names: list[str] | None = None


def create_data_array_and_template(
    chunk_template_ds: xr.Dataset,
    data_var_name: str,
    shared_buffer: SharedMemory,
) -> tuple[xr.DataArray, xr.DataArray]:
    # This template is small and we will pass it between processes
    data_array_template = chunk_template_ds[data_var_name]

    # This data array will be assigned actual, shared memory
    data_array = data_array_template.copy()

    # Drop all non-dimension coordinates from the template only,
    # they are already written by write_metadata.
    data_array_template = data_array_template.drop_vars(
        [
            coord
            for coord in data_array_template.coords
            if coord not in data_array_template.dims
        ]
    )

    shared_array: ArrayFloat32 = np.ndarray(
        data_array.shape,
        dtype=data_array.dtype,
        buffer=shared_buffer.buf,
    )
    # Important:
    # We rely on initializing with nans so failed reads (eg. corrupt source data)
    # leave nan and to reuse the same shared buffer for each variable.
    shared_array[:] = np.nan
    data_array.data = shared_array

    return data_array, data_array_template


def write_shards(
    data_array_template: xr.DataArray,
    store: zarr.storage.FsspecStore | Path,
    shared_buffer: SharedMemory,
    cpu_process_executor: ProcessPoolExecutor,
) -> None:
    shard_indexers = tuple(shard_slice_indexers(data_array_template))
    chunk_init_times_str = ", ".join(
        data_array_template.init_time.dt.strftime("%Y-%m-%dT%H:%M").values
    )
    logger.info(
        f"Writing {data_array_template.name} {chunk_init_times_str} in {len(shard_indexers)} shards"
    )

    # Use ProcessPoolExecutor for parallel writing of shards.
    # Pass only a lightweight template and the name of the shared buffer
    # to avoid ProcessPool pickling very large arrays.
    consume(
        cpu_process_executor.map(
            partial(
                write_shard_to_zarr,
                data_array_template,
                shared_buffer.name,
                store,
            ),
            shard_indexers,
        )
    )


def write_shard_to_zarr(
    data_array_template: xr.DataArray,
    shared_buffer_name: str,
    store: zarr.storage.FsspecStore | Path,
    shard_indexer: tuple[slice, ...],
) -> None:
    """Write a shard of data to zarr storage using shared memory."""
    with (
        warnings.catch_warnings(),
        closing(SharedMemory(name=shared_buffer_name)) as shared_memory,
    ):
        shared_array: ArrayFloat32 = np.ndarray(
            data_array_template.shape,
            dtype=data_array_template.dtype,
            buffer=shared_memory.buf,
        )

        data_array = data_array_template.copy()
        data_array.data = shared_array

        warnings.filterwarnings(
            "ignore",
            message="In a future version of xarray decode_timedelta will default to False rather than None.",
            category=FutureWarning,
        )
        data_array[shard_indexer].to_zarr(store, region="auto")  # type: ignore[call-overload]


# TODO: Delete this version once all datasets are using _make_shared_buffer
@contextmanager
def create_shared_buffer(size: int) -> Generator[SharedMemory, None, None]:
    try:
        shared_memory = SharedMemory(create=True, size=size)
        yield shared_memory
    finally:
        shared_memory.close()
        shared_memory.unlink()
    # TODO: Make common utility alongside write_shards


@contextmanager
def make_shared_buffer(ds: xr.Dataset) -> Generator[SharedMemory, None, None]:
    """
    Context manager to create and manage a shared memory buffer sized to fit the largest variable in the dataset.

    Arguments
    ----------
    ds : xr.Dataset
        The xarray Dataset whose data variables are used to determine the required buffer size.
        This will most likely be a particular region of a dataset, rather than the entire dataset.

    Yields
    ------
    SharedMemory
        A shared memory object that can be used for inter-process communication.
    """
    buffer_size = max(data_var.nbytes for data_var in ds.data_vars.values())

    shared_memory = SharedMemory(create=True, size=buffer_size)
    try:
        yield shared_memory
    finally:
        shared_memory.close()
        shared_memory.unlink()
