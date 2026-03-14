# ruff: noqa: INP001
"""
Prototype: Virtual Icechunk dataset from NOAA GFS GRIB files on S3.

Creates a virtual zarr dataset backed by icechunk where data variable chunks
point to byte ranges within remote GRIB2 files on S3, decoded at read time
by GribberishCodec. Coordinate arrays are stored as real (non-virtual) data.

Phases demonstrated:
1. Initialize and backfill 3 init times with all lead times
2. Resize the dataset to add a 4th init time, fill 3 of its lead times
3. Fill in 2 more lead times for the 4th init time
"""

import shutil
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from gribberish.zarr.codec import GribberishCodec

from reformatters.common.download import http_download_to_disk
from reformatters.common.logging import get_logger
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index

log = get_logger(__name__)

S3_BUCKET = "noaa-gfs-bdp-pds"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.amazonaws.com"
S3_VIRTUAL_PREFIX = f"s3://{S3_BUCKET}/"

# Use a small subset of variables for the prototype: 3 instant variables
PROTOTYPE_VAR_NAMES = frozenset(
    {
        "temperature_2m",
        "pressure_surface",
        "wind_u_10m",
    }
)

# Use recent init times that are reliably available on S3
INIT_TIMES = pd.to_datetime(
    [
        "2026-03-10T00:00",
        "2026-03-10T06:00",
        "2026-03-10T12:00",
        "2026-03-10T18:00",
    ]
)

LEAD_TIMES = pd.timedelta_range("0h", "6h", freq="1h")

# GFS native grid: 721 lat x 1440 lon at 0.25 degree resolution
N_LAT = 721
N_LON = 1440

OUTPUT_DIR = Path("prototype_output/gfs_icechunk_virtual")


def get_prototype_data_vars() -> list[NoaaDataVar]:
    """Get the subset of data variables we want for this prototype."""
    config = NoaaGfsForecastTemplateConfig()
    return [v for v in config.data_vars if v.name in PROTOTYPE_VAR_NAMES]


def gfs_s3_path(init_time: pd.Timestamp, lead_time: pd.Timedelta) -> str:
    """Return the S3 key for a GFS GRIB file."""
    init_date = init_time.strftime("%Y%m%d")
    init_hour = init_time.strftime("%H")
    lead_hours = int(lead_time.total_seconds() // 3600)
    return f"gfs.{init_date}/{init_hour}/atmos/gfs.t{init_hour}z.pgrb2.0p25.f{lead_hours:03d}"


def gfs_s3_url(init_time: pd.Timestamp, lead_time: pd.Timedelta) -> str:
    return f"{S3_VIRTUAL_PREFIX}{gfs_s3_path(init_time, lead_time)}"


def gfs_http_url(init_time: pd.Timestamp, lead_time: pd.Timedelta) -> str:
    return f"{S3_BASE_URL}/{gfs_s3_path(init_time, lead_time)}"


def download_and_parse_index(
    init_time: pd.Timestamp,
    lead_time: pd.Timedelta,
    data_vars: list[NoaaDataVar],
) -> dict[str, tuple[int, int]]:
    """Download a GRIB index file and parse byte ranges for each variable.

    Returns a dict mapping variable name to (offset, length).
    """
    idx_url = f"{gfs_http_url(init_time, lead_time)}.idx"
    idx_path = http_download_to_disk(idx_url, "gfs-icechunk-prototype")

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    result: dict[str, tuple[int, int]] = {}
    for var, start, end in zip(data_vars, starts, ends, strict=True):
        result[var.name] = (start, end - start)

    return result


def create_repository(output_dir: Path) -> icechunk.Repository:
    """Create a new icechunk repository on local disk."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    storage = icechunk.local_filesystem_storage(str(output_dir))

    config = icechunk.RepositoryConfig.default()
    s3_store = icechunk.s3_store(region="us-east-1")
    container = icechunk.VirtualChunkContainer(S3_VIRTUAL_PREFIX, s3_store)
    config.set_virtual_chunk_container(container)

    repo = icechunk.Repository.create(
        storage,
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials(
            {S3_VIRTUAL_PREFIX: icechunk.s3_anonymous_credentials()}
        ),
    )
    return repo


def initialize_zarr_metadata(
    store: icechunk.IcechunkStore,
    init_times: pd.DatetimeIndex,
    lead_times: pd.TimedeltaIndex,
    data_vars: list[NoaaDataVar],
) -> None:
    """Write zarr v3 metadata and coordinate arrays to the store.

    Data variable arrays use GribberishCodec with chunk shape matching
    individual GRIB messages: (1, 1, 721, 1440).
    """
    root = zarr.open_group(store, mode="w")

    # Write coordinate arrays with real data (not virtual)
    init_time_seconds = np.array(
        (init_times - pd.Timestamp("1970-01-01")).total_seconds(), dtype="int64"
    )
    arr = root.create_array(
        "init_time",
        data=init_time_seconds,
        chunks=(len(init_times),),
        fill_value=0,
        dimension_names=("init_time",),
    )
    arr.attrs.update(
        {
            "calendar": "proleptic_gregorian",
            "units": "seconds since 1970-01-01 00:00:00",
            "long_name": "Forecast initialization time",
        }
    )

    lead_time_seconds = np.array(lead_times.total_seconds(), dtype="int64")
    arr = root.create_array(
        "lead_time",
        data=lead_time_seconds,
        chunks=(len(lead_times),),
        fill_value=-1,
        dimension_names=("lead_time",),
    )
    arr.attrs.update(
        {
            "units": "seconds",
            "long_name": "Forecast lead time",
        }
    )

    lat_values = np.flip(np.arange(-90, 90.25, 0.25))
    arr = root.create_array(
        "latitude",
        data=lat_values,
        chunks=(N_LAT,),
        fill_value=np.nan,
        dimension_names=("latitude",),
    )
    arr.attrs.update(
        {
            "units": "degree_north",
            "long_name": "Latitude",
            "standard_name": "latitude",
        }
    )

    lon_values = np.arange(-180, 180, 0.25)
    arr = root.create_array(
        "longitude",
        data=lon_values,
        chunks=(N_LON,),
        fill_value=np.nan,
        dimension_names=("longitude",),
    )
    arr.attrs.update(
        {
            "units": "degree_east",
            "long_name": "Longitude",
            "standard_name": "longitude",
        }
    )

    # Create data variable arrays with GribberishCodec
    # Each chunk = one GRIB message = one (init_time, lead_time) slice for a variable
    for var in data_vars:
        codec = GribberishCodec(var=var.internal_attrs.grib_element)
        arr = root.create_array(
            var.name,
            shape=(len(init_times), len(lead_times), N_LAT, N_LON),
            chunks=(1, 1, N_LAT, N_LON),
            dtype="float32",
            fill_value=np.nan,
            serializer=codec,
            compressors=(),
            filters=(),
            dimension_names=("init_time", "lead_time", "latitude", "longitude"),
        )
        var_attrs: dict[str, str] = {
            "long_name": var.attrs.long_name,
            "units": var.attrs.units,
        }
        if var.attrs.standard_name:
            var_attrs["standard_name"] = var.attrs.standard_name
        arr.attrs.update(var_attrs)


def set_virtual_refs_for_file(
    store: icechunk.IcechunkStore,
    init_time: pd.Timestamp,
    lead_time: pd.Timedelta,
    init_time_idx: int,
    lead_time_idx: int,
    data_vars: list[NoaaDataVar],
    byte_ranges: dict[str, tuple[int, int]],
) -> None:
    """Set virtual references for all variables in a single GRIB file."""
    s3_url = gfs_s3_url(init_time, lead_time)

    for var in data_vars:
        offset, length = byte_ranges[var.name]
        chunk_key = f"{var.name}/c/{init_time_idx}/{lead_time_idx}/0/0"
        store.set_virtual_ref(
            chunk_key,
            location=s3_url,
            offset=offset,
            length=length,
        )


def report_dataset_state(repo: icechunk.Repository, label: str) -> None:
    """Open the dataset read-only and report its shape and NaN fractions."""
    session = repo.readonly_session(branch="main")
    store = session.store

    ds = xr.open_zarr(store, consolidated=False)

    log.info(f"\n{'=' * 60}")
    log.info(f"Dataset state: {label}")
    log.info(f"{'=' * 60}")
    log.info(f"Dimensions: {dict(ds.sizes)}")

    init_times = pd.to_datetime(ds["init_time"].values, unit="s", origin="unix")
    lead_times = pd.to_timedelta(ds["lead_time"].values, unit="s")
    log.info(f"Init times: {init_times.tolist()}")
    log.info(f"Lead times: {lead_times.tolist()}")
    log.info(f"Data variables: {list(ds.data_vars)}")

    # Report NaN fraction per init_time for each variable
    for var_name in ds.data_vars:
        log.info(f"\n  {var_name}:")
        var_data = ds[var_name]
        for i, it in enumerate(init_times):
            slice_data = var_data.isel(init_time=i)
            total = slice_data.size
            # Count non-NaN: load one lead time at a time to manage memory
            non_nan_count = 0
            for lt_idx in range(len(lead_times)):
                lt_slice = slice_data.isel(lead_time=lt_idx).values
                non_nan_count += int(np.count_nonzero(~np.isnan(lt_slice)))
            fraction = non_nan_count / total
            log.info(
                f"    init_time={it}: non-NaN fraction = {fraction:.4f} ({non_nan_count}/{total})"
            )

    ds.close()


def resize_init_time_dimension(
    store: icechunk.IcechunkStore,
    new_init_time: pd.Timestamp,
    new_init_time_count: int,
    lead_time_count: int,
    data_vars: list[NoaaDataVar],
) -> None:
    """Grow the init_time dimension by one, updating coordinate and data arrays."""
    root = zarr.open_group(store, mode="r+")

    # Resize and append to the init_time coordinate array
    init_time_arr = root["init_time"]
    assert isinstance(init_time_arr, zarr.Array)
    init_time_arr.resize((new_init_time_count,))
    new_value = int((new_init_time - pd.Timestamp("1970-01-01")).total_seconds())
    init_time_arr[new_init_time_count - 1] = new_value

    # Resize all data variable arrays
    for var in data_vars:
        arr = root[var.name]
        assert isinstance(arr, zarr.Array)
        arr.resize((new_init_time_count, lead_time_count, N_LAT, N_LON))


def backfill_init_times(
    repo: icechunk.Repository,
    backfill_init_times_idx: pd.DatetimeIndex,
    lead_times: pd.TimedeltaIndex,
    data_vars: list[NoaaDataVar],
) -> None:
    """Phase 1: Initialize metadata and backfill 3 init times with all lead times."""
    log.info("\n%s", "=" * 60)
    log.info("PHASE 1: Initialize and backfill")
    log.info("=" * 60)

    session = repo.writable_session("main")
    store = session.store

    initialize_zarr_metadata(store, backfill_init_times_idx, lead_times, data_vars)
    log.info("Zarr metadata written")

    for init_idx, init_time in enumerate(backfill_init_times_idx):
        log.info(f"Backfilling init_time={init_time} (all lead times)")
        for lt_idx, lead_time in enumerate(lead_times):
            byte_ranges = download_and_parse_index(init_time, lead_time, data_vars)
            set_virtual_refs_for_file(
                store, init_time, lead_time, init_idx, lt_idx, data_vars, byte_ranges
            )

    snapshot_id = session.commit("Phase 1: Initialize and backfill 3 init times")
    log.info(f"Phase 1 committed: {snapshot_id}")
    report_dataset_state(repo, "After Phase 1 (backfill)")


def add_new_init_time(
    repo: icechunk.Repository,
    new_init_time: pd.Timestamp,
    new_init_time_idx: int,
    lead_times: pd.TimedeltaIndex,
    partial_lead_times: pd.TimedeltaIndex,
    data_vars: list[NoaaDataVar],
) -> None:
    """Phase 2: Resize the dataset to add a 4th init time, fill 3 of its lead times."""
    log.info("\n%s", "=" * 60)
    log.info("PHASE 2: Add new init time (partial)")
    log.info("=" * 60)

    session = repo.writable_session("main")
    store = session.store

    resize_init_time_dimension(
        store, new_init_time, new_init_time_idx + 1, len(lead_times), data_vars
    )
    log.info(f"Resized dataset: init_time dimension is now {new_init_time_idx + 1}")

    log.info(
        f"Adding init_time={new_init_time} (lead times {partial_lead_times[0]}-{partial_lead_times[-1]})"
    )
    for lead_time in partial_lead_times:
        lt_idx_raw = lead_times.get_loc(lead_time)
        assert isinstance(lt_idx_raw, int)
        byte_ranges = download_and_parse_index(new_init_time, lead_time, data_vars)
        set_virtual_refs_for_file(
            store,
            new_init_time,
            lead_time,
            new_init_time_idx,
            lt_idx_raw,
            data_vars,
            byte_ranges,
        )

    snapshot_id = session.commit("Phase 2: Add 4th init time with partial lead times")
    log.info(f"Phase 2 committed: {snapshot_id}")
    report_dataset_state(repo, "After Phase 2 (new init time, partial)")


def fill_missing_lead_times(
    repo: icechunk.Repository,
    init_time: pd.Timestamp,
    init_time_idx: int,
    lead_times: pd.TimedeltaIndex,
    missing_lead_times: pd.TimedeltaIndex,
    data_vars: list[NoaaDataVar],
) -> None:
    """Phase 3: Fill in remaining lead times for the 4th init time."""
    log.info("\n%s", "=" * 60)
    log.info("PHASE 3: Fill missing lead times")
    log.info("=" * 60)

    session = repo.writable_session("main")
    store = session.store

    log.info(
        f"Filling in lead times {missing_lead_times.tolist()} for init_time={init_time}"
    )
    for lead_time in missing_lead_times:
        lt_idx_raw = lead_times.get_loc(lead_time)
        assert isinstance(lt_idx_raw, int)
        byte_ranges = download_and_parse_index(init_time, lead_time, data_vars)
        set_virtual_refs_for_file(
            store,
            init_time,
            lead_time,
            init_time_idx,
            lt_idx_raw,
            data_vars,
            byte_ranges,
        )

    snapshot_id = session.commit("Phase 3: Fill remaining lead times for 4th init time")
    log.info(f"Phase 3 committed: {snapshot_id}")
    report_dataset_state(repo, "After Phase 3 (fill missing lead times)")


def run_prototype() -> None:
    data_vars = get_prototype_data_vars()
    log.info(f"Prototype variables: {[v.name for v in data_vars]}")
    log.info(f"All init times: {INIT_TIMES.tolist()}")
    log.info(f"Lead times: {LEAD_TIMES.tolist()}")

    output_dir = OUTPUT_DIR
    repo = create_repository(output_dir)

    # Phase 1: Backfill first 3 init times with all lead times
    backfill_init_times(repo, INIT_TIMES[:3], LEAD_TIMES, data_vars)

    # Phase 2: Resize to add 4th init time, fill only first 3 lead times (0h-2h)
    add_new_init_time(
        repo,
        INIT_TIMES[3],
        new_init_time_idx=3,
        lead_times=LEAD_TIMES,
        partial_lead_times=LEAD_TIMES[:3],
        data_vars=data_vars,
    )

    # Phase 3: Fill in lead times 3h-4h for the 4th init time
    fill_missing_lead_times(
        repo,
        init_time=INIT_TIMES[3],
        init_time_idx=3,
        lead_times=LEAD_TIMES,
        missing_lead_times=LEAD_TIMES[3:5],
        data_vars=data_vars,
    )

    log.info("\n%s", "=" * 60)
    log.info("Snapshot history")
    log.info("=" * 60)
    for info in repo.ancestry(branch="main"):
        log.info(f"  {info.id[:12]}  {info.written_at}  {info.message}")

    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    log.info(f"\nOn-disk repository size: {total_size / 1024:.1f} KB")
    log.info("(Data variable chunks are virtual references to S3, not stored locally)")


if __name__ == "__main__":
    run_prototype()
