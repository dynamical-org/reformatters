"""
NOTE: We intend to delete this file once the RegionJob refactor is complete.
"""

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
from reformatters.common.shared_memory_utils import (
    create_data_array_and_template as create_data_array_and_template,
)
from reformatters.common.types import ArrayFloat32

log = get_logger(__name__)


# NOTE: superseded by in region_job.py by just passing keyword arguments
class ChunkFilters(BaseModel):
    """
    Filters for controlling which chunks of data to process.
    A value of None means no filtering.
    """

    time_dim: str
    time_start: str | None = None
    time_end: str | None = None
    variable_names: list[str] | None = None


# NOTE: This can be deleted once we complete the RegionJob refactor
# It is superceded by utilities in zarr.py
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
    log.info(
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


# NOTE: This can be deleted once we complete the RegionJob refactor
# It is superceded by utilities in zarr.py
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


# TODO: Delete this version once all datasets are using make_shared_buffer
@contextmanager
def create_shared_buffer(size: int) -> Generator[SharedMemory, None, None]:
    shared_memory = SharedMemory(create=True, size=size)
    try:
        yield shared_memory
    finally:
        shared_memory.close()
        shared_memory.unlink()
