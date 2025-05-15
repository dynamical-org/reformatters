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

from reformatters.common.iterating import consume, shard_slice_indexers
from reformatters.common.logging import get_logger
from reformatters.common.template_config import AppendDim
from reformatters.common.types import ArrayFloat32

logger = get_logger(__name__)


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


def create_data_array_and_template(
    processing_region_ds: xr.Dataset,
    data_var_name: str,
    shared_buffer: SharedMemory,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Prepare an xarray.DataArray backed by shared memory for writing,
    and a lightweight template of the same data array.

    The returned template and the _name_ of the shared buffer are passed
    between processes, avoiding costly pickling of the large shared memory buffer.

    Parameters
    ----------
    processing_region_ds : xr.Dataset
        Template dataset covering the region of the dataset to process.
    data_var_name : str
        Name of the variable within `processing_region_ds` to use.
    shared_buffer : SharedMemory
        A shared memory buffer to use as backing for the data array.

    Returns
    -------
    data_array : xr.DataArray
        A data array whose `.data` is a NumPy view
        into the shared memory, initialized to NaN.
    data_array_template : xr.DataArray
        processing_region_ds[data_var_name] with non-dimension coordinates dropped.
        This is lightweight and can be passed between processes.
    """

    # This template is small and we will pass it between processes
    data_array_template = processing_region_ds[data_var_name]

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
    processing_region_da_template: xr.DataArray,
    shared_buffer: SharedMemory,
    append_dim: AppendDim,
    output_region_ds: xr.Dataset,
    store: zarr.storage.FsspecStore | Path,
    cpu_process_executor: ProcessPoolExecutor,
) -> None:
    """
    Write the shards of a data array as zarr to `store`. The data array is
    reconstructed from `processing_region_da_template` and the `shared_buffer`.

    `processing_region_da_template` may include additional, padded steps along `append_dim`,
    while `output_region_ds` has exactly the output size and coordinates along
    `append_dim` of the shards to write.
    """

    shard_indexers = tuple(
        shard_slice_indexers(output_region_ds[processing_region_da_template.name])
    )
    chunk_times_str = " - ".join(
        processing_region_da_template[append_dim]
        .isel({append_dim: [0, -1]})
        .dt.strftime("%Y-%m-%dT%H:%M")
        .values
    )
    logger.info(
        f"Writing {processing_region_da_template.name} {chunk_times_str} in {len(shard_indexers)} shards"
    )
    # Use ProcessPoolExecutor for parallel writing of shards.
    # Pass only a lightweight template and the name of the shared buffer
    # to avoid ProcessPool pickling very large arrays.
    consume(
        cpu_process_executor.map(
            partial(
                write_shard_to_zarr,
                processing_region_da_template,
                shared_buffer.name,
                append_dim,
                output_region_ds,
                store,
            ),
            shard_indexers,
        )
    )


def write_shard_to_zarr(
    processing_region_da_template: xr.DataArray,
    shared_buffer_name: str,
    append_dim: AppendDim,
    output_region_ds: xr.Dataset,
    store: zarr.storage.FsspecStore | Path,
    shard_indexer: tuple[slice, ...],
) -> None:
    """Write a shard of data held in shared memory to a zarr store."""
    with (
        warnings.catch_warnings(),
        closing(SharedMemory(name=shared_buffer_name)) as shared_memory,
    ):
        shared_array: ArrayFloat32 = np.ndarray(
            processing_region_da_template.shape,
            dtype=processing_region_da_template.dtype,
            buffer=shared_memory.buf,
        )

        data_array = processing_region_da_template.copy()
        data_array.data = shared_array

        # Data array may have been padded along `append_dim` to support interpolation and deaccumulation.
        # Trim to the exact shard length provided by chunk_template_ds.
        append_dim_coords = output_region_ds[append_dim]
        # coords.max() implies an _inclusive_ slice endpoint which only happens with time indexes in pandas
        assert np.issubdtype(append_dim_coords.dtype, np.datetime64)
        append_dim_slice = slice(append_dim_coords.min(), append_dim_coords.max())
        data_array = data_array.sel({append_dim: append_dim_slice})

        warnings.filterwarnings(
            "ignore",
            message="In a future version of xarray decode_timedelta will default to False rather than None.",
            category=FutureWarning,
        )
        data_array[shard_indexer].to_zarr(store, region="auto")  # type: ignore[call-overload]
