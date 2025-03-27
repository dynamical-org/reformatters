import concurrent
import os
import warnings
from collections.abc import Generator, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing, contextmanager
from functools import partial
from itertools import batched, groupby, product
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from reformatters.common.iterating import consume, shard_slice_indexers
from reformatters.common.logging import get_logger
from reformatters.common.types import ArrayFloat32
from reformatters.common.zarr import create_data_array_and_template
from reformatters.noaa.gfs.forecast import template
from reformatters.noaa.gfs.read_data import SourceFileCoords, download_file, read_into
from reformatters.noaa.noaa_config_models import NOAADataVar

logger = get_logger(__name__)

type CoordAndPath = tuple[SourceFileCoords, Path | None]
type CoordsAndPaths = list[CoordAndPath]


def reformat_time_i_slices(
    jobs: Sequence[tuple[slice, list[str]]],
    template_ds: xr.Dataset,
    store: zarr.storage.FsspecStore | Path,
    var_download_group_size: int,
) -> Generator[tuple[NOAADataVar, dict[pd.Timestamp, pd.Timedelta]], None, None]:
    # The only effective way we've found to fully utilize cpu resources
    # while writing to zarr is to parallelize across processes (not threads).
    # Use shared memory to avoid pickling large arrays to share between processes.
    shared_buffer_size = max(
        data_var.nbytes
        for data_var in template_ds.isel(
            {template.APPEND_DIMENSION: jobs[0][0]}
        ).values()
    )
    with (
        ThreadPoolExecutor(max_workers=1) as wait_executor,
        ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2) as io_executor,
        ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as cpu_executor,
        ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as cpu_process_executor,
        create_shared_buffer(shared_buffer_size) as shared_buffer,
    ):
        for init_time_i_slice, data_var_names in jobs:
            chunk_template_ds = template_ds[data_var_names].isel(
                init_time=init_time_i_slice
            )

            chunk_init_times = pd.to_datetime(chunk_template_ds["init_time"].values)
            chunk_lead_times = pd.to_timedelta(chunk_template_ds["lead_time"].values)
            chunk_coords = generate_chunk_coordinates(
                chunk_init_times,
                chunk_lead_times,
            )

            chunk_init_times_str = ", ".join(
                chunk_init_times.strftime("%Y-%m-%dT%H:%M")
            )
            logger.info(f"Starting chunk with init times {chunk_init_times_str}")

            download_var_group_futures = get_download_var_group_futures(
                chunk_template_ds,
                chunk_coords,
                io_executor,
                wait_executor,
                var_download_group_size,
            )
            for future in concurrent.futures.as_completed(download_var_group_futures):
                if (e := future.exception()) is not None:
                    raise e

                coords_and_paths = future.result()
                data_vars = download_var_group_futures[future]

                max_lead_times = get_max_lead_times(coords_and_paths)

                # Write variable by variable to avoid blowing up memory usage
                for data_var in data_vars:
                    data_array, data_array_template = create_data_array_and_template(
                        chunk_template_ds, data_var.name, shared_buffer
                    )
                    var_coords_and_paths = filter_coords_and_paths(
                        data_var, coords_and_paths
                    )

                    read_into_data_array(
                        data_array, data_var, var_coords_and_paths, cpu_executor
                    )
                    # apply_data_transformations_inplace(data_array, data_var)
                    write_shards(
                        data_array_template,
                        store,
                        shared_buffer,
                        cpu_process_executor,
                    )

                    yield (data_var, max_lead_times)

                # Reclaim space once done.
                for _, filepath in coords_and_paths:
                    if filepath is not None:
                        filepath.unlink()


def get_max_lead_times(
    coords_and_paths: list[tuple[SourceFileCoords, Path | None]],
) -> dict[pd.Timestamp, pd.Timedelta]:
    max_lead_times: dict[pd.Timestamp, pd.Timedelta] = {}

    sorted_coords_and_paths = sorted(
        coords_and_paths, key=lambda coords_and_path: coords_and_path[0]["init_time"]
    )
    groups = groupby(
        sorted_coords_and_paths,
        key=lambda coords_and_path: coords_and_path[0]["init_time"],
    )
    for init_time, init_time_coords_and_paths in groups:
        ingested_lead_times = [
            coord["lead_time"]
            for coord, path in init_time_coords_and_paths
            if path is not None
        ]

        max_lead_times[init_time] = max(
            ingested_lead_times, default=pd.Timedelta("NaT")
        )
    return max_lead_times


# TODO: Basically common except for the data_var type
def filter_coords_and_paths(
    data_var: NOAADataVar, coords_and_paths: CoordsAndPaths
) -> CoordsAndPaths:
    # Skip reading the 0-hour for accumulated or last N hours avg values
    if data_var.attrs.step_type in ("accum", "avg"):
        return [
            coords_and_path
            for coords_and_path in coords_and_paths
            if coords_and_path[0]["lead_time"] > pd.Timedelta(hours=0)
        ]
    else:
        return coords_and_paths


# TODO: Basically common except for the data_var type
def read_into_data_array(
    out: xr.DataArray,
    data_var: NOAADataVar,
    var_coords_and_paths: CoordsAndPaths,
    cpu_executor: ThreadPoolExecutor,
) -> None:
    logger.info(f"Reading {data_var.name}")
    consume(
        cpu_executor.map(
            partial(
                read_into,
                out,
                data_var=data_var,
            ),
            *zip(*var_coords_and_paths, strict=True),
        )
    )


# def apply_data_transformations_inplace(
#     data_array: xr.DataArray, data_var: DataVar[Any]
# ) -> None:
#     if data_var.internal_attrs.deaccumulate_to_rates:
#         logger.info(f"Converting {data_var.name} from accumulations to rates")
#         try:
#             deaccumulate_to_rates_inplace(data_array, dim="lead_time")
#         except ValueError:
#             # Log exception so we are notified if deaccumulation errors are larger than expected.
#             logger.exception(f"Error deaccumulating {data_var.name}")

#     keep_mantissa_bits = data_var.internal_attrs.keep_mantissa_bits
#     if isinstance(keep_mantissa_bits, int):
#         round_float32_inplace(
#             data_array.values,
#             keep_mantissa_bits=keep_mantissa_bits,
#         )


# TODO: Probably move to common zarr module
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


# TODO: Probably move to common zarr module
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


type ChunkCoordinates = Sequence[SourceFileCoords]


def generate_chunk_coordinates(
    chunk_init_times: Iterable[pd.Timestamp],
    chunk_lead_times: Iterable[pd.Timedelta],
) -> ChunkCoordinates:
    return [
        {"init_time": init_time, "lead_time": lead_time}
        for init_time, lead_time in product(chunk_init_times, chunk_lead_times)
    ]


def download_var_group_files(
    idx_data_vars: Iterable[NOAADataVar],
    chunk_coords: Iterable[SourceFileCoords],
    io_executor: ThreadPoolExecutor,
) -> list[tuple[SourceFileCoords, Path | None]]:
    logger.info(f"Downloading {[d.name for d in idx_data_vars]}")
    done, not_done = concurrent.futures.wait(
        [
            io_executor.submit(
                download_file,
                coord,
                gfs_idx_data_vars=idx_data_vars,
            )
            for coord in chunk_coords
        ]
    )
    assert len(not_done) == 0

    for future in done:
        if (e := future.exception()) is not None:
            raise e

    logger.info(f"Completed download for {[d.name for d in idx_data_vars]}")
    return [f.result() for f in done]


type DownloadVarGroupFutures = dict[
    Future[list[tuple[SourceFileCoords, Path | None]]],
    tuple[NOAADataVar, ...],
]


def get_download_var_group_futures(
    chunk_template_ds: xr.Dataset,
    chunk_coords: ChunkCoordinates,
    io_executor: ThreadPoolExecutor,
    wait_executor: ThreadPoolExecutor,
    group_size: int,
) -> DownloadVarGroupFutures:
    download_var_group_futures: DownloadVarGroupFutures = {}

    chunk_data_vars = [
        d for d in template.DATA_VARIABLES if d.name in chunk_template_ds
    ]
    chunk_data_vars = sorted(
        chunk_data_vars, key=lambda data_var: data_var.internal_attrs.index_position
    )
    data_var_groups = batched(chunk_data_vars, group_size)

    for data_vars in data_var_groups:
        download_var_group_futures[
            wait_executor.submit(
                download_var_group_files,
                data_vars,
                chunk_coords,
                io_executor,
            )
        ] = data_vars

    return download_var_group_futures


# TODO: Probably move to common zarr module
@contextmanager
def create_shared_buffer(size: int) -> Generator[SharedMemory, None, None]:
    try:
        shared_memory = SharedMemory(create=True, size=size)
        yield shared_memory
    finally:
        shared_memory.close()
        shared_memory.unlink()
