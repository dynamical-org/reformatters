from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import ClassVar, Literal

import icechunk
import obstore
import obstore.store
import pandas as pd
import pydantic
import xarray as xr
from icechunk.store import IcechunkStore
from zarr.abc.store import Store

from reformatters.common import storage
from reformatters.common.download import download_to_disk, get_local_path
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, DatetimeLike, Timedelta, Timestamp
from reformatters.common.virtual_region_job import SourceFileResult, VirtualRef
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
    NoaaHrrrDomain,
    NoaaHrrrFileType,
)
from reformatters.noaa.hrrr.nomads_mirror import (
    MIRROR_LOCATION_PREFIX,
    mirror_key,
    mirror_store,
    parse_mirror_key,
)
from reformatters.noaa.hrrr.region_job import DownloadSource
from reformatters.noaa.hrrr.virtual_region_job import (
    S3_LOCATION_PREFIX,
    NoaaHrrrForecastVirtualRegionJob,
    NoaaHrrrForecastVirtualSourceFileCoord,
    hrrr_virtual_chunk_containers,
)

log = get_logger(__name__)

type SourceBucket = Literal["nodd", "mirror"]

# One empty coordination file per data file whose refs point at the mirror, in the
# dataset's own bucket; deleted once the refs are rewritten from NODD.
PENDING_REPOINT_JOB_NAME = "pending-repoint"


def hrrr_18_hour_virtual_fast_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    """NODD plus the NOMADS mirror's public domain."""
    return (
        *hrrr_virtual_chunk_containers(),
        icechunk.VirtualChunkContainer(MIRROR_LOCATION_PREFIX, icechunk.http_store()),
    )


class NoaaHrrrForecast18HourVirtualFastSourceFileCoord(
    NoaaHrrrForecastVirtualSourceFileCoord
):
    """An HRRR file resolved against NODD or the NOMADS mirror (identical keys). The
    two routing fields are assignable; the fields naming the file stay frozen.
    """

    model_config = pydantic.ConfigDict(frozen=False, strict=True)

    init_time: Timestamp = pydantic.Field(frozen=True)
    lead_time: Timedelta = pydantic.Field(frozen=True)
    domain: NoaaHrrrDomain = pydantic.Field(frozen=True)
    file_type: NoaaHrrrFileType = pydantic.Field(frozen=True)
    data_vars: Sequence[NoaaHrrrDataVar] = pydantic.Field(frozen=True)

    bucket: SourceBucket = "nodd"
    # The store already holds refs for this file, possibly into the mirror; only
    # NODD may supply it again.
    already_present: bool = False

    def route_to(self, bucket: SourceBucket) -> None:
        self.bucket = bucket  # ty: ignore[invalid-assignment] - frozen=False here

    def mark_present(self) -> None:
        self.already_present = True  # ty: ignore[invalid-assignment] - frozen=False here

    def get_url(self, source: DownloadSource = "s3") -> str:
        url = super().get_url(source=source)
        if self.bucket == "mirror":
            return MIRROR_LOCATION_PREFIX + url.removeprefix(S3_LOCATION_PREFIX)
        return url


class NoaaHrrrForecast18HourVirtualFastRegionJob(NoaaHrrrForecastVirtualRegionJob):
    """Takes a file from NODD, or from the NOMADS mirror while NODD does not have it,
    and records each mirror-sourced file as a pending repoint so its refs are
    rewritten from NODD on a later fire, for as long as it takes NODD to publish
    it. See "NOMADS mirror" in docs/virtual_datasets.md."""

    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("6h")
    source_file_coord_class: ClassVar[
        type[NoaaHrrrForecast18HourVirtualFastSourceFileCoord]
    ] = NoaaHrrrForecast18HourVirtualFastSourceFileCoord
    # Re-offering pending repoints is update work only: operational_update_jobs turns
    # it on, and the validation job keeps the plain manifest probe.
    repoint_mirrored: bool = False
    # A pending repoint waits on NODD for minutes to days; probing it every tick would
    # list its old date's NODD prefix each second ahead of the current init.
    repoint_probe_every: ClassVar[int] = 60
    # After an outage the pending repoints can outnumber an init's files many times
    # over. Each fire takes the oldest half of this many, nearest the mirror's expiry,
    # and the newest half, so records NODD never satisfies cannot starve new ones.
    max_repoints_per_fire: ClassVar[int] = 500

    _ticks: int = pydantic.PrivateAttr(default=0)
    _store_factory: storage.StoreFactory | None = pydantic.PrivateAttr(default=None)
    _mirror_store: obstore.store.ObjectStore | None = pydantic.PrivateAttr(default=None)

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaHrrrDataVar],
        reformat_job_name: str,
        job_fire_time: Timestamp | None = None,
    ) -> tuple[
        Sequence[RegionJob[NoaaHrrrDataVar, NoaaHrrrForecastVirtualSourceFileCoord]],
        xr.DataTree,
    ]:
        jobs, template_ds = super().operational_update_jobs(
            primary_store,
            tmp_store,
            get_template_fn,
            append_dim,
            all_data_vars,
            reformat_job_name,
            job_fire_time,
        )
        return [job.model_copy(update={"repoint_mirrored": True}) for job in jobs], (
            template_ds
        )

    @classmethod
    def process_worker_jobs(
        cls,
        worker_jobs: Sequence[
            RegionJob[NoaaHrrrDataVar, NoaaHrrrForecastVirtualSourceFileCoord]
        ],
        store_factory: storage.StoreFactory,
        branch_name: str,
        worker_index: int,
        *,
        overwrite_chunks: bool,
    ) -> dict[str, list[SourceFileResult]]:
        for job in worker_jobs:
            _fast(job).bind_store_factory(store_factory)
        return super().process_worker_jobs(
            worker_jobs,
            store_factory,
            branch_name,
            worker_index,
            overwrite_chunks=overwrite_chunks,
        )

    def bind_store_factory(self, store_factory: storage.StoreFactory) -> None:
        """The pending-repoint records live in the dataset's coordination area."""
        self._store_factory = store_factory

    def mirror_store(self) -> obstore.store.ObjectStore:
        store = self._mirror_store
        if store is None:
            store = mirror_store()
            self._mirror_store = store  # ty: ignore[invalid-assignment] - private cache
        return store

    def download_index(self, coord: NoaaHrrrForecastVirtualSourceFileCoord) -> Path:
        if _routed(coord).bucket != "mirror":
            return super().download_index(coord)
        key = mirror_key(coord) + ".idx"
        local_path = get_local_path(self.dataset_id, key)
        download_to_disk(self.mirror_store(), key, local_path)
        return local_path

    def read_data_bytes(
        self, coord: NoaaHrrrForecastVirtualSourceFileCoord, start: int, end: int
    ) -> bytes:
        if _routed(coord).bucket != "mirror":
            return super().read_data_bytes(coord, start, end)
        return bytes(
            obstore.get_range(
                self.mirror_store(), mirror_key(coord), start=start, end=end
            )
        )

    def filter_already_present(
        self,
        candidates: Sequence[NoaaHrrrForecastVirtualSourceFileCoord],
        store: IcechunkStore,
    ) -> list[NoaaHrrrForecastVirtualSourceFileCoord]:
        if not self.repoint_mirrored:
            return super().filter_already_present(candidates, store)
        by_key = {mirror_key(c): _routed(c) for c in candidates}
        pending = self.pending_repoints()
        # A record whose file is outside the update window is rebuilt from the
        # template; a record whose refs never got committed is a fresh ingest again.
        outside_window = [key for key in pending if key not in by_key]
        half = self.max_repoints_per_fire // 2
        selected = dict.fromkeys([*outside_window[:half], *outside_window[-half:]])
        recovered = [
            coord for key in selected if (coord := self._coord_for_mirror_key(key))
        ]
        all_candidates = [*candidates, *recovered]
        absent = super().filter_already_present(all_candidates, store)
        absent_ids = {id(coord) for coord in absent}
        recorded = set(pending)
        repoints = [
            coord
            for coord in all_candidates
            if id(coord) not in absent_ids and mirror_key(coord) in recorded
        ]
        for coord in repoints:
            _routed(coord).mark_present()
        return [*absent, *repoints]

    def pending_repoints(self) -> list[str]:
        """Data-file keys whose refs point at the mirror, oldest init first."""
        names = self._bound_store_factory().list_coordination_files(
            PENDING_REPOINT_JOB_NAME, ""
        )
        return sorted(name.replace("__", "/") for name in names)

    def _bound_store_factory(self) -> storage.StoreFactory:
        assert self._store_factory is not None, "bind_store_factory first"
        return self._store_factory

    def _check_refs_complete(
        self, coord: NoaaHrrrForecastVirtualSourceFileCoord, refs: list[VirtualRef]
    ) -> None:
        """Skip a file whose index lacks any message the template expects of it. A
        partial file, mirrored before NOMADS finished writing it or truncated on NODD,
        would otherwise be ingested as such, and its NODD rewrite would clear the
        pending record while the messages it lacks stay pointed at the mirror."""
        lookup = self._message_lookup(coord.data_vars, whole_hours(coord.lead_time))
        expected = {
            (var.path, tuple(sorted(level.items())))
            for entries in lookup.values()
            for var, level in entries
        }
        filled = {
            (
                ref.data_var.path,
                tuple(
                    sorted(
                        (dim, value)
                        for dim, value in ref.out_loc.items()
                        if dim not in ("init_time", "lead_time")
                    )
                ),
            )
            for ref in refs
        }
        missing = sorted(expected - filled)
        if missing:
            raise ValueError(
                f"{coord.get_url()} lacks {len(missing)} of {len(expected)} expected "
                f"messages, e.g. {missing[:3]}; not ingesting a partial file"
            )

    def _coord_for_mirror_key(
        self, key: str
    ) -> NoaaHrrrForecast18HourVirtualFastSourceFileCoord | None:
        """The coord for a mirrored data file this job could write, or None for a key
        outside the template's coordinates or with no variables in the file."""
        parsed = parse_mirror_key(key)
        if parsed is None:
            return None
        init_time, lead_time, file_type = parsed
        root = self.template_ds.to_dataset()
        if init_time not in root.get_index(
            "init_time"
        ) or lead_time not in root.get_index("lead_time"):
            return None
        region_ds = root[["init_time", "lead_time"]].sel(
            init_time=[init_time], lead_time=[lead_time]
        )
        coords = self.generate_source_file_coords(
            region_ds,
            [v for v in self.data_vars if v.internal_attrs.hrrr_file_type == file_type],
        )
        return _routed(coords[0]) if coords else None

    def discover_available(
        self, pending: list[NoaaHrrrForecastVirtualSourceFileCoord]
    ) -> list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]:
        if self.processing_mode != "update":
            return super().discover_available(pending)
        self._ticks += 1
        assert self.repoint_probe_every >= 1
        probe_present = (self._ticks - 1) % self.repoint_probe_every == 0
        routed = [_routed(coord) for coord in pending]
        for coord in routed:
            coord.route_to("nodd")
        nodd_pending: list[NoaaHrrrForecastVirtualSourceFileCoord] = [
            c for c in routed if probe_present or not c.already_present
        ]
        on_nodd = super().discover_available(nodd_pending)
        found_ids = {id(coord) for coord, _ in on_nodd}
        candidates = [
            c for c in routed if id(c) not in found_ids and not c.already_present
        ]
        for coord in candidates:
            coord.route_to("mirror")
        try:
            on_mirror = discover_available_by_obstore_listing(
                candidates,
                store=self.mirror_store(),
                location_prefix=MIRROR_LOCATION_PREFIX,
                require_index=True,
            )
        except Exception:
            # The mirror is an accelerator; NODD stays the floor when it is unreachable.
            log.exception("Cannot list the NOMADS mirror this tick; using NODD only")
            on_mirror = []
        mirrored_ids = {id(coord) for coord, _ in on_mirror}
        for coord in candidates:
            if id(coord) not in mirrored_ids:
                coord.route_to("nodd")
        return [*on_nodd, *on_mirror]

    def process_virtual_refs(
        self,
        remaining: Sequence[NoaaHrrrForecastVirtualSourceFileCoord],
    ) -> Iterator[
        Sequence[tuple[NoaaHrrrForecastVirtualSourceFileCoord, Sequence[VirtualRef]]]
    ]:
        # The base loop commits each batch between the yield and the resume, so a
        # pending-repoint record is written before the refs it describes exist and
        # deleted only after NODD's refs have replaced them.
        store_factory = self._bound_store_factory()
        for batch in super().process_virtual_refs(remaining):
            for name in _pending_repoint_names(batch, bucket="mirror"):
                store_factory.write_coordination_file(
                    PENDING_REPOINT_JOB_NAME, name, b""
                )
            yield batch
            for name in _pending_repoint_names(batch, already_present=True):
                store_factory.delete_coordination_file(PENDING_REPOINT_JOB_NAME, name)


def _pending_repoint_names(
    batch: Sequence[
        tuple[NoaaHrrrForecastVirtualSourceFileCoord, Sequence[VirtualRef]]
    ],
    *,
    bucket: SourceBucket | None = None,
    already_present: bool | None = None,
) -> list[str]:
    """The record name (the NODD key, flattened) of each distinct file in the batch
    matching the given routing."""
    return sorted(
        {
            mirror_key(coord).replace("/", "__")
            for coord, _ in batch
            if (bucket is None or _routed(coord).bucket == bucket)
            and (already_present is None or _routed(coord).already_present)
        }
    )


def _fast(
    job: RegionJob[NoaaHrrrDataVar, NoaaHrrrForecastVirtualSourceFileCoord],
) -> NoaaHrrrForecast18HourVirtualFastRegionJob:
    assert isinstance(job, NoaaHrrrForecast18HourVirtualFastRegionJob), type(job)
    return job


def _routed(
    coord: NoaaHrrrForecastVirtualSourceFileCoord,
) -> NoaaHrrrForecast18HourVirtualFastSourceFileCoord:
    assert isinstance(coord, NoaaHrrrForecast18HourVirtualFastSourceFileCoord), type(
        coord
    )
    return coord
