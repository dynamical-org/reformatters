from collections.abc import Callable, Sequence
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

from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.types import AppendDim, DatetimeLike, Timedelta, Timestamp
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
    NoaaHrrrDomain,
    NoaaHrrrFileType,
)
from reformatters.noaa.hrrr.nomads_mirror import (
    MIRROR_BUCKET_REGION,
    MIRROR_LOCATION_PREFIX,
    mirror_key,
    mirror_store,
    parse_mirror_key,
)
from reformatters.noaa.hrrr.region_job import NODD_BUCKET_REGION, DownloadSource
from reformatters.noaa.hrrr.virtual_region_job import (
    S3_LOCATION_PREFIX,
    NoaaHrrrForecastVirtualRegionJob,
    NoaaHrrrForecastVirtualSourceFileCoord,
    hrrr_virtual_chunk_containers,
)

log = get_logger(__name__)

type SourceBucket = Literal["nodd", "mirror"]


def hrrr_18_hour_virtual_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    """NODD plus the NOMADS mirror."""
    return (
        *hrrr_virtual_chunk_containers(),
        icechunk.VirtualChunkContainer(
            MIRROR_LOCATION_PREFIX, icechunk.s3_store(region=MIRROR_BUCKET_REGION)
        ),
    )


class NoaaHrrrForecast18HourVirtualSourceFileCoord(
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


class NoaaHrrrForecast18HourVirtualRegionJob(NoaaHrrrForecastVirtualRegionJob):
    """Takes a file from NODD, or from the NOMADS mirror while NODD does not have it,
    and keeps offering every file the mirror still holds so its refs are rewritten
    from NODD once NODD publishes it (an idempotent rewrite when they already point
    there). See "NOMADS mirror" in docs/virtual_datasets.md."""

    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("6h")
    source_file_coord_class: ClassVar[
        type[NoaaHrrrForecast18HourVirtualSourceFileCoord]
    ] = NoaaHrrrForecast18HourVirtualSourceFileCoord
    # Re-offering present files is update work only: operational_update_jobs turns it
    # on, and the validation job keeps the plain manifest probe.
    repoint_mirrored: bool = False
    # Files the store holds wait on NODD for minutes to days; probing them every tick
    # would list every old date's NODD prefix each second ahead of the current init.
    repoint_probe_every: ClassVar[int] = 60

    _ticks: int = pydantic.PrivateAttr(default=0)

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

    def source_region(self, coord: NoaaHrrrForecastVirtualSourceFileCoord) -> str:
        return (
            MIRROR_BUCKET_REGION
            if _routed(coord).bucket == "mirror"
            else NODD_BUCKET_REGION
        )

    def mirror_store(self) -> obstore.store.ObjectStore:
        return mirror_store(write=False)

    def filter_already_present(
        self,
        candidates: Sequence[NoaaHrrrForecastVirtualSourceFileCoord],
        store: IcechunkStore,
    ) -> list[NoaaHrrrForecastVirtualSourceFileCoord]:
        if not self.repoint_mirrored:
            return super().filter_already_present(candidates, store)
        mirrored = self._mirror_listing()
        # A file the mirror still holds can be repointed for as long as the mirror
        # keeps it, so the work is not bounded by the update window's candidates.
        known = {mirror_key(c) for c in candidates}
        recovery = [
            coord
            for key in sorted(mirrored)
            if key not in known and (coord := self._coord_for_mirror_key(key))
        ]
        all_candidates = [*candidates, *recovery]
        absent = super().filter_already_present(all_candidates, store)
        absent_ids = {id(coord) for coord in absent}
        present_in_mirror = []
        for coord in all_candidates:
            if id(coord) not in absent_ids and mirror_key(coord) in mirrored:
                _routed(coord).mark_present()
                present_in_mirror.append(coord)
        return [*absent, *present_in_mirror]

    def _coord_for_mirror_key(
        self, key: str
    ) -> NoaaHrrrForecast18HourVirtualSourceFileCoord | None:
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

    def _mirror_listing(self) -> set[str]:
        """Every data-file key the mirror holds with its index."""
        try:
            listing = {
                meta["path"]
                for batch in obstore.list(self.mirror_store(), chunk_size=10_000)
                for meta in batch
            }
        except Exception:
            log.exception(
                "Cannot list the NOMADS mirror; repoints wait for a later fire"
            )
            return set()
        return {key for key in listing if key + ".idx" in listing}


def _routed(
    coord: NoaaHrrrForecastVirtualSourceFileCoord,
) -> NoaaHrrrForecast18HourVirtualSourceFileCoord:
    assert isinstance(coord, NoaaHrrrForecast18HourVirtualSourceFileCoord), type(coord)
    return coord
