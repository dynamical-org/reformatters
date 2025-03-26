import concurrent.futures
import os
import warnings
from collections import defaultdict
from collections.abc import Generator, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing, contextmanager
from functools import partial
from itertools import batched, groupby
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.config_models import DataVar, EnsembleStatistic
from reformatters.common.iterating import (
    consume,
    shard_slice_indexers,
)
from reformatters.common.logging import get_logger
from reformatters.common.types import ArrayFloat32
from reformatters.noaa.gefs.analysis import template
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar, GEFSFileType
from reformatters.noaa.gefs.read_data import (
    SourceFileCoords,
    download_file,
    is_v12_index,
    read_into,
)

logger = get_logger(__name__)

# Integer ensemble member or an ensemble statistic
type EnsOrStat = int | np.integer[Any] | str

type CoordAndPath = tuple[SourceFileCoords, Path | None]
type CoordsAndPaths = Sequence[CoordAndPath]

INIT_TIME_FREQUENCY: Final[pd.Timedelta] = pd.Timedelta("6h")


def reformat_time_i_slices(
    jobs: Sequence[tuple[slice, list[str]]],
    template_ds: xr.Dataset,
    store: zarr.storage.FsspecStore | Path,
    var_download_group_size: int,
) -> Generator[
    tuple[GEFSDataVar, dict[tuple[pd.Timestamp, EnsOrStat], pd.Timedelta]], None, None
]:
    """
    Do the chunk data reformatting work - download files, read into memory, write to zarr.
    Yields the data variable/time slice combinations as it processes.
    """
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
        for append_dim_i_slice, data_var_names in jobs:
            chunk_template_ds = template_ds[data_var_names].isel(
                {template.APPEND_DIMENSION: append_dim_i_slice}
            )

            chunk_times = pd.to_datetime(chunk_template_ds["time"].values)

            chunk_times_str = f"{chunk_times[0].strftime('%Y-%m-%dT%H:%M')} - {chunk_times[-1].strftime('%Y-%m-%dT%H:%M')}"
            logger.info(f"Starting chunk with times {chunk_times_str}")

            download_var_group_futures = get_download_var_group_futures(
                chunk_template_ds,
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
                        chunk_template_ds, data_var, shared_buffer
                    )
                    var_coords_and_paths = filter_coords_and_paths(
                        data_var, coords_and_paths
                    )
                    read_into_data_array(
                        data_array, data_var, var_coords_and_paths, cpu_executor
                    )
                    apply_data_transformations_inplace(data_array, data_var)
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


type DownloadVarGroupFutures = dict[
    Future[Sequence[tuple[SourceFileCoords, Path | None]]],
    tuple[GEFSDataVar, ...],
]


def get_download_var_group_futures(
    chunk_template_ds: xr.Dataset,
    io_executor: ThreadPoolExecutor,
    wait_executor: ThreadPoolExecutor,
    group_size: int,
) -> DownloadVarGroupFutures:
    download_var_group_futures: DownloadVarGroupFutures = {}
    data_var_groups = group_data_vars_by_gefs_file_type(
        [d for d in template.DATA_VARIABLES if d.name in chunk_template_ds],
        group_size=group_size,
    )
    for gefs_file_type, ensemble_statistic, has_hour_0, data_vars in data_var_groups:
        chunk_coords_by_type = generate_chunk_coordinates(
            chunk_template_ds["time"],
            template.ANALYSIS_ENSEMBLE_MEMBER,
            init_time_frequency=INIT_TIME_FREQUENCY,
            var_has_hour_0_values=has_hour_0,
        )

        chunk_coords: Iterable[SourceFileCoords]
        if ensemble_statistic is None:
            chunk_coords = chunk_coords_by_type["ensemble"]
        else:
            chunk_coords = chunk_coords_by_type["statistic"]

        download_var_group_futures[
            wait_executor.submit(
                download_var_group_files,
                data_vars,
                chunk_coords,
                gefs_file_type,
                io_executor,
            )
        ] = data_vars

    return download_var_group_futures


def download_var_group_files(
    idx_data_vars: Iterable[GEFSDataVar],
    chunk_coords: Iterable[SourceFileCoords],
    gefs_file_type: GEFSFileType,
    io_executor: ThreadPoolExecutor,
) -> Sequence[tuple[SourceFileCoords, Path | None]]:
    logger.info(f"Downloading {[d.name for d in idx_data_vars]}")
    done, not_done = concurrent.futures.wait(
        [
            io_executor.submit(
                download_file,
                coord,
                gefs_file_type=gefs_file_type,
                gefs_idx_data_vars=idx_data_vars,
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


def generate_chunk_coordinates(
    chunk_times: Iterable[pd.Timestamp],
    ensemble_member: int,
    init_time_frequency: pd.Timedelta,
    var_has_hour_0_values: bool,
) -> dict[str, Sequence[SourceFileCoords]]:
    """Construct the init time and lead time coordinates which correspond to each time in `chunk_times`."""
    times = pd.to_datetime(chunk_times)  # type: ignore[call-overload]

    times = filter_available_times(times)

    init_times = times.floor(init_time_frequency)

    # If var doesn't have hour 0 values we have to go back one forecast.
    # eg. Get the 6th hour rather than the 0th hour for 6-hourly init times.
    if not var_has_hour_0_values:
        is_hour_0 = times == init_times
        init_times = init_times.where(~is_hour_0, init_times - init_time_frequency)

    lead_times = times - init_times

    ensemble_coords: Sequence[SourceFileCoords] = [
        {
            "init_time": init_time,
            "ensemble_member": ensemble_member,
            "lead_time": lead_time,
        }
        for init_time, lead_time in zip(init_times, lead_times, strict=True)
    ]
    return {"ensemble": ensemble_coords}


def filter_available_times(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Before v12, GEFS files had a 6 hour step."""
    # pre-v12 data is all we have for the 9 month period after the v12 reforecast ends
    # 2019-12-31 and before the v12 forecast archive starts 2020-10-01.
    return times[is_v12_index(times) | (times.hour % 6 == 0)]


def data_var_has_hour_0_values(data_var: DataVar[Any]) -> bool:
    return data_var.attrs.step_type not in ("accum", "avg")


def group_data_vars_by_gefs_file_type(
    data_vars: Iterable[GEFSDataVar], *, group_size: int
) -> list[tuple[GEFSFileType, EnsembleStatistic | None, bool, tuple[GEFSDataVar, ...]]]:
    """
    Group data variables by the things which determine which source file they come from:
    1. their GEFS file type (a, b, or s) and
    2. their ensemble statistic if present or None if they are an ensemble trace.

    Then, within each group, chunk them into groups of size `group_size`. We download
    all variables in a group together which can reduce the number of tiny requests if
    they are nearby in the source grib file. By breaking groups into `group_size`, we
    allow reading/writing to begin before we've downloaded _all_ variables from a file.
    """
    grouper = defaultdict(list)
    for data_var in data_vars:
        gefs_file_type = data_var.internal_attrs.gefs_file_type
        ensemble_statistic = data_var.attrs.ensemble_statistic
        has_hour_0_values = data_var_has_hour_0_values(data_var)
        grouper[(gefs_file_type, ensemble_statistic, has_hour_0_values)].append(
            data_var
        )
    chunks = []
    for (
        file_type,
        ensemble_statistic,
        has_hour_0_values,
    ), idx_data_vars in grouper.items():
        idx_data_vars = sorted(
            idx_data_vars, key=lambda data_var: data_var.internal_attrs.index_position
        )
        chunks.extend(
            [
                (file_type, ensemble_statistic, has_hour_0_values, data_vars_chunk)
                for data_vars_chunk in batched(idx_data_vars, group_size)
            ]
        )
    # Consistent group order is required for correct job distribution between workers
    return list(sorted(chunks, key=str))  # noqa: C413


def get_max_lead_times(
    coords_and_paths: Sequence[tuple[SourceFileCoords, Path | None]],
) -> dict[tuple[pd.Timestamp, EnsOrStat], pd.Timedelta]:
    max_lead_times: dict[tuple[pd.Timestamp, EnsOrStat], pd.Timedelta] = {}

    def group_by_key(
        v: tuple[SourceFileCoords, Path | None],
    ) -> tuple[pd.Timestamp, EnsOrStat]:
        coords, _ = v

        ensemble_portion: EnsOrStat
        if isinstance(
            ensemble_member := coords.get("ensemble_member"),
            int | np.integer,
        ):
            ensemble_portion = ensemble_member
        elif isinstance(statistic := coords.get("statistic"), str):
            ensemble_portion = statistic
        return (coords["init_time"], ensemble_portion)

    sorted_coords_and_paths = sorted(coords_and_paths, key=group_by_key)
    groups = groupby(sorted_coords_and_paths, key=group_by_key)

    for (init_time, ensemble_member), init_time_coords_and_paths in groups:
        ingested_lead_times = [
            coord["lead_time"]
            for coord, path in init_time_coords_and_paths
            if path is not None
        ]
        max_lead_times[(init_time, ensemble_member)] = max(
            ingested_lead_times, default=pd.Timedelta("NaT")
        )
    return max_lead_times


def create_data_array_and_template(
    chunk_template_ds: xr.Dataset, data_var: GEFSDataVar, shared_buffer: SharedMemory
) -> tuple[xr.DataArray, xr.DataArray]:
    # This template is small and we will pass it between processes
    data_array_template = chunk_template_ds[data_var.name]

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


def filter_coords_and_paths(
    data_var: GEFSDataVar, coords_and_paths: CoordsAndPaths
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


def read_into_data_array(
    out: xr.DataArray,
    data_var: GEFSDataVar,
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


def apply_data_transformations_inplace(
    data_array: xr.DataArray, data_var: DataVar[Any]
) -> None:
    # TODO refactor deaccumulation to work with the `time` not lead_time dimension
    # if data_var.internal_attrs.deaccumulate_to_rates:
    #     logger.info(f"Converting {data_var.name} from accumulations to rates")
    #     try:
    #         deaccumulate_to_rates_inplace(data_array, dim="time")
    #     except ValueError:
    #         # Log exception so we are notified if deaccumulation errors are larger than expected.
    #         logger.exception(f"Error deaccumulating {data_var.name}")

    keep_mantissa_bits = data_var.internal_attrs.keep_mantissa_bits
    if isinstance(keep_mantissa_bits, int):
        round_float32_inplace(
            data_array.values,
            keep_mantissa_bits=keep_mantissa_bits,
        )


def write_shards(
    data_array_template: xr.DataArray,
    store: zarr.storage.FsspecStore | Path,
    shared_buffer: SharedMemory,
    cpu_process_executor: ProcessPoolExecutor,
) -> None:
    shard_indexers = tuple(shard_slice_indexers(data_array_template))
    chunk_times_str = " - ".join(
        data_array_template[template.APPEND_DIMENSION]
        .isel({template.APPEND_DIMENSION: [0, -1]})
        .dt.strftime("%Y-%m-%dT%H:%M")
        .values
    )
    logger.info(
        f"Writing {data_array_template.name} {chunk_times_str} in {len(shard_indexers)} shards"
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


@contextmanager
def create_shared_buffer(size: int) -> Generator[SharedMemory, None, None]:
    try:
        shared_memory = SharedMemory(create=True, size=size)
        yield shared_memory
    finally:
        shared_memory.close()
        shared_memory.unlink()
