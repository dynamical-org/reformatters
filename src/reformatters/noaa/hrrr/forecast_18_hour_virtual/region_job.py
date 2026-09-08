from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import icechunk
import obstore.store
import pandas as pd
import pydantic
from icechunk.store import IcechunkStore

from reformatters.common.download import s3_download_to_disk
from reformatters.common.logging import get_logger
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Timedelta, Timestamp
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
    NoaaHrrrDomain,
    NoaaHrrrFileType,
)
from reformatters.noaa.hrrr.nomads_cache import (
    NOMADS_CACHE_BUCKET_REGION,
    NOMADS_CACHE_LOCATION_PREFIX,
    cache_key,
    list_cache,
    mark_repointed,
    nomads_cache_store,
    parse_cache_key,
    unrepointed_data_files,
)
from reformatters.noaa.hrrr.region_job import NODD_BUCKET_REGION, DownloadSource
from reformatters.noaa.hrrr.virtual_region_job import (
    S3_LOCATION_PREFIX,
    NoaaHrrrForecastVirtualRegionJob,
    NoaaHrrrForecastVirtualSourceFileCoord,
    hrrr_virtual_chunk_containers,
)
from reformatters.noaa.noaa_grib_index import parse_grib_index_lines

log = get_logger(__name__)

type SourceBucket = Literal["nodd", "cache"]


def hrrr_18_hour_virtual_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    """NODD plus the NOMADS cache; fresh objects per call like the NODD-only helper."""
    return (
        *hrrr_virtual_chunk_containers(),
        icechunk.VirtualChunkContainer(
            NOMADS_CACHE_LOCATION_PREFIX,
            icechunk.s3_store(region=NOMADS_CACHE_BUCKET_REGION),
        ),
    )


class NoaaHrrrForecast18HourVirtualSourceFileCoord(
    NoaaHrrrForecastVirtualSourceFileCoord
):
    """An HRRR file resolved against NODD or the NOMADS cache (identical keys).

    Discovery records where it found the file on the object the write loop hands it
    (the loop drops coords by identity), so the three routing fields are assignable;
    the fields naming the file stay frozen.
    """

    model_config = pydantic.ConfigDict(frozen=False, strict=True)

    init_time: Timestamp = pydantic.Field(frozen=True)
    lead_time: Timedelta = pydantic.Field(frozen=True)
    domain: NoaaHrrrDomain = pydantic.Field(frozen=True)
    file_type: NoaaHrrrFileType = pydantic.Field(frozen=True)
    data_vars: Sequence[NoaaHrrrDataVar] = pydantic.Field(frozen=True)

    bucket: SourceBucket = "nodd"
    # A second pass over an already-ingested cache file, rewriting its refs from NODD.
    repoint: bool = False
    # The cache's copy did not carry every variable this coord needs; use NODD.
    cache_rejected: bool = False

    def route_to(self, bucket: SourceBucket) -> None:
        self.bucket = bucket  # ty: ignore[invalid-assignment] - frozen=False here

    def reject_cache(self) -> None:
        self.cache_rejected = True  # ty: ignore[invalid-assignment] - frozen=False here

    def get_url(self, source: DownloadSource = "s3") -> str:
        url = super().get_url(source=source)
        if self.bucket == "cache":
            return NOMADS_CACHE_LOCATION_PREFIX + url.removeprefix(S3_LOCATION_PREFIX)
        return url

    def file_key(
        self,
    ) -> tuple[pd.Timestamp, pd.Timedelta, NoaaHrrrDomain, NoaaHrrrFileType]:
        return (self.init_time, self.lead_time, self.domain, self.file_type)

    def repoint_twin(self) -> NoaaHrrrForecast18HourVirtualSourceFileCoord:
        return NoaaHrrrForecast18HourVirtualSourceFileCoord(
            init_time=self.init_time,
            lead_time=self.lead_time,
            domain=self.domain,
            file_type=self.file_type,
            data_vars=self.data_vars,
            repoint=True,
        )


class NoaaHrrrForecast18HourVirtualRegionJob(NoaaHrrrForecastVirtualRegionJob):
    """Reads the NOMADS cache before NODD during operational updates, then rewrites
    each cache-sourced file's refs from NODD once NODD publishes it. See "NOMADS
    cache" in docs/virtual_datasets.md."""

    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("6h")
    source_file_coord_class: ClassVar[
        type[NoaaHrrrForecast18HourVirtualSourceFileCoord]
    ] = NoaaHrrrForecast18HourVirtualSourceFileCoord
    # Rollback knob: False stops selecting the cache for new files while repoints,
    # markers and the container registration keep working.
    cache_first: ClassVar[bool] = True
    # Repoint twins wait on NODD for minutes to days; probing them every tick would
    # list every old date's NODD prefix each second ahead of the current init's files.
    repoint_probe_every: ClassVar[int] = 60

    _ticks: int = pydantic.PrivateAttr(default=0)

    def source_region(self, coord: NoaaHrrrForecastVirtualSourceFileCoord) -> str:
        return (
            NOMADS_CACHE_BUCKET_REGION
            if _routed(coord).bucket == "cache"
            else NODD_BUCKET_REGION
        )

    def cache_store(self) -> obstore.store.ObjectStore:
        return nomads_cache_store(write=False)

    def cache_writer(self) -> obstore.store.ObjectStore:
        return nomads_cache_store(write=True)

    def unfinished_work(
        self, store: IcechunkStore
    ) -> list[NoaaHrrrForecastVirtualSourceFileCoord]:
        if self.processing_mode != "update":
            return super().unfinished_work(store)
        candidates = [_routed(coord) for coord in self.source_file_coords()]
        try:
            unrepointed = set(unrepointed_data_files(list_cache(self.cache_store())))
        except Exception:
            log.exception(
                "Cannot list the NOMADS cache; repoints wait for a later fire"
            )
            unrepointed = set()
        known = {coord.file_key() for coord in candidates}
        for key in unrepointed:
            parsed = parse_cache_key(key)
            assert parsed is not None
            init_time, lead_time, file_type = parsed
            if (init_time, lead_time, "conus", file_type) in known:
                continue
            coord = self.source_file_coord(
                init_time, lead_time, file_type, self.data_vars
            )
            if coord is not None:
                candidates.append(_routed(coord))
        absent = {id(coord) for coord in self.filter_already_present(candidates, store)}
        work: list[NoaaHrrrForecastVirtualSourceFileCoord] = []
        for coord in candidates:
            if id(coord) in absent:
                work.append(coord)
            elif cache_key(coord) in unrepointed:
                work.append(coord.repoint_twin())
        return work

    def discover_available(
        self, pending: list[NoaaHrrrForecastVirtualSourceFileCoord]
    ) -> list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]:
        if self.processing_mode != "update" or not self.cache_first:
            return super().discover_available(pending)
        self._ticks += 1
        routed = [_routed(coord) for coord in pending]
        probe_twins = self._ticks % self.repoint_probe_every == 1
        cache_candidates = [
            coord for coord in routed if not (coord.repoint or coord.cache_rejected)
        ]
        for coord in cache_candidates:
            coord.route_to("cache")
        found: list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]] = []
        try:
            in_cache = discover_available_by_obstore_listing(
                cache_candidates,
                store=self.cache_store(),
                location_prefix=NOMADS_CACHE_LOCATION_PREFIX,
                require_index=True,
            )
            for coord, size in in_cache:
                if self._cache_index_covers(coord):
                    found.append((coord, size))
                else:
                    coord.reject_cache()
                    log.warning(
                        f"Cache index for {cache_key(coord)} does not cover every "
                        "variable the file supplies; waiting for NODD"
                    )
        except Exception:
            # The cache is an accelerator; NODD stays the floor when it is unreachable.
            log.exception("Cannot read the NOMADS cache this tick; using NODD only")
            found = []
        found_ids = {id(coord) for coord, _ in found}
        rest = [
            coord
            for coord in routed
            if id(coord) not in found_ids and (probe_twins or not coord.repoint)
        ]
        for coord in rest:
            coord.route_to("nodd")
        nodd_pending: list[NoaaHrrrForecastVirtualSourceFileCoord] = list(rest)
        return [*found, *super().discover_available(nodd_pending)]

    def committed(
        self,
        batch: Sequence[
            tuple[NoaaHrrrForecastVirtualSourceFileCoord, Sequence[VirtualRef]]
        ],
    ) -> Sequence[NoaaHrrrForecastVirtualSourceFileCoord]:
        follow_ups = []
        for coord, _ in batch:
            routed = _routed(coord)
            if routed.repoint:
                mark_repointed(cache_key(routed), self.cache_writer())
            elif routed.bucket == "cache":
                follow_ups.append(routed.repoint_twin())
        return follow_ups

    def _check_refs_complete(
        self,
        coord: NoaaHrrrForecastVirtualSourceFileCoord,
        refs: list[VirtualRef],
    ) -> None:
        # A cache file or a repoint must fill every cell the coord names: a partial
        # repoint would leave refs on the cache under a marker that says otherwise.
        routed = _routed(coord)
        if routed.bucket != "cache" and not routed.repoint:
            return
        missing = self._expected_cells(coord) - {
            (ref.data_var.name, tuple(sorted(ref.out_loc.items()))) for ref in refs
        }
        if missing:
            raise ValueError(
                f"{coord.get_url()} filled {len(refs)} of "
                f"{len(missing) + len(refs)} expected cells; missing e.g. {sorted(missing)[:3]}"
            )

    def _cache_index_covers(
        self, coord: NoaaHrrrForecast18HourVirtualSourceFileCoord
    ) -> bool:
        index_path = s3_download_to_disk(
            coord.get_index_url(), self.dataset_id, region=NOMADS_CACHE_BUCKET_REGION
        )
        try:
            index_lines = parse_grib_index_lines(index_path)
        finally:
            index_path.unlink()
        lookup = self._message_lookup(coord.data_vars, whole_hours(coord.lead_time))
        out_loc_base = dict(coord.out_loc())
        covered = {
            (var.name, tuple(sorted({**out_loc_base, **level_label}.items())))
            for _, element, level, window in index_lines
            for var, level_label in lookup.get((element, level, window), [])
        }
        return self._expected_cells(coord) <= covered

    def _expected_cells(
        self, coord: NoaaHrrrForecastVirtualSourceFileCoord
    ) -> set[tuple[str, tuple[tuple[str, Any], ...]]]:
        lookup = self._message_lookup(coord.data_vars, whole_hours(coord.lead_time))
        out_loc_base = dict(coord.out_loc())
        return {
            (var.name, tuple(sorted({**out_loc_base, **level_label}.items())))
            for matches in lookup.values()
            for var, level_label in matches
        }


def _routed(
    coord: NoaaHrrrForecastVirtualSourceFileCoord,
) -> NoaaHrrrForecast18HourVirtualSourceFileCoord:
    assert isinstance(coord, NoaaHrrrForecast18HourVirtualSourceFileCoord), type(coord)
    return coord
