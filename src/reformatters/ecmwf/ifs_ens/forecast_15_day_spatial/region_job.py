import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar

import obstore
import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.download import s3_download_to_disk, s3_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValue,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, DatetimeLike, Dim, Timedelta, Timestamp
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfDataVar,
    has_hour_0_values,
    vars_available,
)
from reformatters.ecmwf.ecmwf_grib_index import (
    grib_message_byte_ranges_from_index_by_member,
)

log = get_logger(__name__)

_S3_LOCATION_PREFIX = "s3://ecmwf-forecasts/"
_S3_BUCKET_REGION = "eu-central-1"


def _vars_in_file(
    data_vars: Sequence[EcmwfDataVar], init_time: pd.Timestamp, lead_time: pd.Timedelta
) -> list[EcmwfDataVar]:
    """The subset of data_vars an open-data file for (init_time, lead_time) contains.

    Excludes variables not yet available at init_time and max/min variables that
    have no value at lead_time=0h.
    """
    return [
        var
        for var in data_vars
        if vars_available([var], init_time)
        and (lead_time != pd.Timedelta(0) or has_hour_0_values(var))
    ]


class IfsEnsForecast15DaySpatialSourceFileCoord(SourceFileCoord):
    """One open-data GRIB file: (init_time, lead_time) for a set of ensemble members.

    Post IFS-50r1 the control member (0) lives in the oper-fc file and perturbed
    members (1-50) in the enfo-ef file, so one (init_time, lead_time) maps to two
    files; each is its own coord with its own member set.
    """

    init_time: Timestamp
    lead_time: Timedelta
    ensemble_members: tuple[int, ...]
    data_vars: Sequence[EcmwfDataVar]

    @property
    def is_control(self) -> bool:
        return self.ensemble_members[0] == 0

    def _base_url(self) -> str:
        date_str = self.init_time.strftime("%Y%m%d")
        hour_str = self.init_time.strftime("%H")
        lead_hours = whole_hours(self.lead_time)
        stream, kind = ("oper", "oper-fc") if self.is_control else ("enfo", "enfo-ef")
        return (
            f"{_S3_LOCATION_PREFIX}{date_str}/{hour_str}z/ifs/0p25/{stream}/"
            f"{date_str}{hour_str}0000-{lead_hours}h-{kind}"
        )

    def get_url(self) -> str:
        return self._base_url() + ".grib2"

    def get_index_url(self) -> str:
        return self._base_url() + ".index"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        # A single chunk location (the first member) for the filter's presence probe;
        # the whole file commits atomically, so any member stands in for all of them.
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_members[0],
        }


class EcmwfIfsEnsForecast15DaySpatialRegionJob(
    VirtualRegionJob[EcmwfDataVar, IfsEnsForecast15DaySpatialSourceFileCoord]
):
    """RegionJob for the ECMWF IFS ENS 15-day spatial (virtual) forecast dataset."""

    # Reprocess this span of recent init times each operational update. Each fire
    # re-sweeps the window, catching stragglers earlier fires gave up on.
    operational_window: ClassVar[Timedelta] = pd.Timedelta("48h")
    # When polling, pace bucket listings to at most one sweep per tick.
    tick_interval: ClassVar[Timedelta] = pd.Timedelta("1s")
    # Concurrent index file downloads (one ~2MB .index per source file).
    download_concurrency: ClassVar[int] = 32

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[EcmwfDataVar]
    ) -> Sequence[IfsEnsForecast15DaySpatialSourceFileCoord]:
        members = [int(m) for m in processing_region_ds["ensemble_member"].values]
        member_groups = [
            group
            for group in (
                tuple(m for m in members if m == 0),  # control -> oper-fc
                tuple(m for m in members if m != 0),  # perturbed -> enfo-ef
            )
            if group
        ]
        coords = []
        for init_time in processing_region_ds["init_time"].values:
            for lead_time in processing_region_ds["lead_time"].values:
                vars_in_file = _vars_in_file(
                    data_var_group, pd.Timestamp(init_time), pd.Timedelta(lead_time)
                )
                if not vars_in_file:
                    continue
                coords.extend(
                    IfsEnsForecast15DaySpatialSourceFileCoord(
                        init_time=pd.Timestamp(init_time),
                        lead_time=pd.Timedelta(lead_time),
                        ensemble_members=member_group,
                        data_vars=vars_in_file,
                    )
                    for member_group in member_groups
                )
        return coords

    def process_virtual_refs(
        self, remaining: Sequence[IfsEnsForecast15DaySpatialSourceFileCoord]
    ) -> Iterator[
        Sequence[tuple[IfsEnsForecast15DaySpatialSourceFileCoord, Sequence[VirtualRef]]]
    ]:
        """Each tick: list the bucket, fetch newly available files' indexes, yield their refs.

        One yield (= one commit) per tick, containing every file that became
        available since the last tick.
        """
        pending = {
            coord.get_url().removeprefix(_S3_LOCATION_PREFIX): coord
            for coord in remaining
        }
        with ThreadPoolExecutor(self.download_concurrency) as pool:
            while pending:
                tick_start = time.monotonic()
                available = self._discover_available(pending)
                if available:
                    coords = [pending[key] for key in available]
                    refs_per_file = pool.map(
                        self._file_refs_or_skip, coords, available.values()
                    )
                    # Files with no refs were skipped (unreadable source); drop them
                    # so a batch never carries a file with no chunks.
                    batch = [
                        (coord, refs)
                        for coord, refs in zip(coords, refs_per_file, strict=True)
                        if refs
                    ]
                    for key in available:
                        del pending[key]
                    skipped = len(coords) - len(batch)
                    log.info(
                        f"Ingesting {len(batch)} files"
                        f"{f' ({skipped} skipped)' if skipped else ''}, "
                        f"{len(pending)} still pending"
                    )
                    if batch:
                        yield batch
                if self.processing_mode == "backfill":
                    if pending:
                        log.info(
                            f"{len(pending)} source files not present, skipping (first: {next(iter(pending))})"
                        )
                    return
                if pending:
                    elapsed = time.monotonic() - tick_start
                    time.sleep(max(0.0, self.tick_interval.total_seconds() - elapsed))

    def _discover_available(
        self, pending: Mapping[str, IfsEnsForecast15DaySpatialSourceFileCoord]
    ) -> dict[str, int]:
        """The pending files retrievable now, mapped to their data file's size in bytes.

        A file is available once the bucket lists both its .grib2 and .index objects
        (refs need the index, and the index lands alongside the data).
        """
        listed: dict[str, int] = {}
        for prefix in sorted({key.rsplit("/", 1)[0] + "/" for key in pending}):
            listed |= _list_objects(prefix)
        return {
            key: listed[key]
            for key in pending
            if key in listed and _index_key(key) in listed
        }

    def _file_refs_or_skip(
        self, coord: IfsEnsForecast15DaySpatialSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        # Skip a file we can't turn into refs (a decode surprise, a transient read
        # error) rather than sink the whole job; its chunks stay fill and validation
        # surfaces the gap. AssertionError is our own invariant, so it propagates.
        # PR 7 lifts this onto VirtualRegionJob; see docs/plans/virtual_icechunk_datasets.md.
        try:
            return self._file_refs(coord, file_size)
        except AssertionError:
            raise
        except Exception:
            log.exception(f"Skipping {coord.get_url()}: could not build virtual refs")
            return []

    def _file_refs(
        self, coord: IfsEnsForecast15DaySpatialSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        index_path = s3_download_to_disk(
            coord.get_index_url(), self.dataset_id, region=_S3_BUCKET_REGION
        )
        try:
            ranges_by_member = grib_message_byte_ranges_from_index_by_member(
                index_path, coord.data_vars, coord.ensemble_members
            )
        finally:
            # Index files accumulate by the millions in backfills; never keep them.
            index_path.unlink()

        location = coord.get_url()
        refs: list[VirtualRef] = []
        for member, (starts, ends) in ranges_by_member.items():
            out_loc: Mapping[Dim, CoordinateValue] = {
                "init_time": coord.init_time,
                "lead_time": coord.lead_time,
                "ensemble_member": member,
            }
            for var, start, end in zip(coord.data_vars, starts, ends, strict=True):
                # A matching index never references bytes past the data file. When it
                # does, the source was re-published without regenerating the index and
                # the offsets are unusable, so skip the whole file (its chunks stay fill).
                if end > file_size or end <= start:
                    log.warning(
                        f"Skipping {coord.get_url()}: index byte ranges fall outside "
                        f"the {file_size}-byte data file; stale or mismatched index"
                    )
                    return []
                refs.append(
                    VirtualRef(
                        data_var=var,
                        out_loc=out_loc,
                        location=location,
                        offset=start,
                        length=end - start,
                    )
                )
        return refs

    def representative_var(
        self, coord: IfsEnsForecast15DaySpatialSourceFileCoord
    ) -> EcmwfDataVar:
        # A coord's file contains only its own vars (lead-0 coords drop max/min vars),
        # so probe one of those rather than the base class's default drawn from all
        # of self.data_vars.
        return next(
            (v for v in coord.data_vars if v.attrs.step_type == "instant"),
            coord.data_vars[0],
        )

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,  # noqa: ARG003 - the icechunk manifest, not a coordinate, tracks ingested data
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[EcmwfDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[EcmwfDataVar, IfsEnsForecast15DaySpatialSourceFileCoord]],
        xr.Dataset,
    ]:
        """Return a single polling job spanning the active window of recent init times.

        The cron fires just before an init's publication window opens; the job
        polls until all expected files are ingested, with the pod deadline
        bounding how long it waits on a file that never publishes.
        filter_already_present derives the remaining work from the icechunk
        manifest. See "Operational updates" in docs/virtual_datasets.md.
        """
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)
        init_times = template_ds.get_index(append_dim)
        window_start = int(
            init_times.searchsorted(append_dim_end - cls.operational_window)
        )
        job = cls(
            tmp_store=tmp_store,
            template_ds=template_ds,
            data_vars=all_data_vars,
            append_dim=append_dim,
            region=slice(window_start, len(init_times)),
            reformat_job_name=reformat_job_name,
            processing_mode="update",
        )
        return [job], template_ds


def _index_key(data_key: str) -> str:
    return data_key.removesuffix(".grib2") + ".index"


def _list_objects(prefix: str) -> dict[str, int]:
    """All object keys under `prefix` in the ECMWF bucket, mapped to size in bytes."""
    store = s3_store(_S3_LOCATION_PREFIX, region=_S3_BUCKET_REGION)
    return {
        meta["path"]: meta["size"]
        for batch in obstore.list(store, prefix=prefix, chunk_size=10_000)
        for meta in batch
    }
