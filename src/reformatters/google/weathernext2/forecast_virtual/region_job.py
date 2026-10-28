import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import ClassVar

import pandas as pd
import xarray as xr
from pydantic import PrivateAttr
from zarr.abc.store import Store

from reformatters.common.config_models import ROOT
from reformatters.common.region_job import (
    CoordinateValue,
    RegionJob,
)
from reformatters.common.types import (
    AppendDim,
    DatetimeLike,
    Dim,
    Timedelta,
    Timestamp,
)
from reformatters.google.weathernext_virtual.listing import (
    PROXY_LOCATION_PREFIX,
    NativeObjectMetadata,
    ObjectListingQuery,
)
from reformatters.google.weathernext_virtual.region_job import (
    NativeSourceChunk,
    WeatherNextSourceFileCoord,
    WeatherNextVirtualRegionJob,
)
from reformatters.google.weathernext_virtual.region_job import utc_now as _utc_now

from .template_config import (
    INIT_TIME_FREQUENCY,
    LEAD_TIMES,
    PER_INIT_STORE_DATE,
    PRESSURE_LEVELS,
    GoogleWeathernext2DataVar,
    SourceLayout,
)

SOURCE_LOCATION_PREFIX = "gs://weathernext/"
_SOURCE_ZARR_PREFIX = f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0/zarr/"
_SOURCE_LEVEL_INDEX = {level: index for index, level in enumerate(PRESSURE_LEVELS)}
_OPERATIONAL_MEMBER_GLOB = "{" + ",".join(map(str, range(64))) + "}"
OPERATIONAL_SOURCE_LANDING_DELAY = pd.Timedelta("7h")
# The two layouts pack chunks differently, so splits are sized per product and per
# array group by ref density; see docs/virtual_datasets.md.
HISTORICAL_MANIFEST_INIT_SPLIT = 128
OPERATIONAL_ROOT_MANIFEST_INIT_SPLIT = 32
OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT = 4

# An operational update usually fires before Google has published the whole init, then
# polls. Each tick lists one probe per pending init (one member of one surface variable,
# all leads: a single page) and lists only leads the probe newly shows, rather than
# re-listing every pending (lead, variable) directory each tick.
_PROBE_MEMBER = 63
_OPERATIONAL_CHUNK_KEY = re.compile(r"\d+\.(?P<lead>\d+)(\.\d+)+")
# A variable with probed leads still incomplete ("lagging") gets its own probe each
# tick, and its probed leads are re-listed with this backoff in case members land out
# of order.
_INCOMPLETE_RETRY_INITIAL_SECONDS = 3.0
_INCOMPLETE_RETRY_MAX_SECONDS = 15.0
# Every pending coord is re-listed this often, whatever the probe shows.
_FULL_LISTING_INTERVAL_SECONDS = 600.0
# A per-lead query pages through its variable's whole directory (keys are member-first),
# so for this many leads of one variable a single all-leads listing costs less.
_ALL_LEADS_LISTING_MIN_LEADS = 5


class GoogleWeathernext2ForecastVirtualSourceFileCoord(WeatherNextSourceFileCoord):
    """One forecast lead from one native annual or per-init source Zarr store."""

    source_layout: SourceLayout
    data_vars: Sequence[GoogleWeathernext2DataVar]

    def get_url(self) -> str:
        if self.source_layout == "operational":
            assert self.init_time >= PER_INIT_STORE_DATE
            return (
                f"{_SOURCE_ZARR_PREFIX}2025_to_present/"
                f"{self.init_time:%Y%m%d}_{self.init_time:%H}hr_01_preds/predictions.zarr"
            )
        assert self.init_time < PER_INIT_STORE_DATE
        year = self.init_time.year
        return f"{_SOURCE_ZARR_PREFIX}{year}_to_{year + 1}/predictions.zarr"

    @property
    def lead_index(self) -> int:
        return int(self.lead_time // pd.Timedelta("6h")) - 1

    @property
    def annual_init_index(self) -> int:
        assert self.source_layout == "historical"
        year_start = pd.Timestamp(f"{self.init_time.year}-01-01")
        return int((self.init_time - year_start) // pd.Timedelta("6h"))

    def chunk_key(
        self,
        var: GoogleWeathernext2DataVar,
        ensemble_member: int,
        pressure_level: int | None,
    ) -> str:
        if self.source_layout == "operational":
            indices = [ensemble_member, self.lead_index]
            if pressure_level is not None:
                indices.append(_SOURCE_LEVEL_INDEX[pressure_level])
        else:
            indices = [self.annual_init_index, ensemble_member // 4, self.lead_index]
            if pressure_level is not None:
                indices.append(0)
        indices.extend((0, 0))
        return f"{var.internal_attrs.source_name}/" + ".".join(map(str, indices))

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


@dataclass
class _InitPoll:
    """One init's polling state across an operational update's ticks."""

    next_full_listing: float = float("-inf")
    # Variable path -> lead indices its probe member has been listed with.
    probed_leads: dict[str, set[int]] = field(default_factory=dict)
    # Lagging variable path -> (next re-listing time, current backoff seconds).
    retry_at: dict[str, tuple[float, float]] = field(default_factory=dict)

    def known_leads(self, var_path: str, probe_var_path: str) -> set[int]:
        return self.probed_leads.get(probe_var_path, set()) | self.probed_leads.get(
            var_path, set()
        )

    def update_retry(
        self, var_path: str, now: float, *, lagging: bool, progressed: bool
    ) -> None:
        previous = self.retry_at.get(var_path)
        if not lagging:
            self.retry_at.pop(var_path, None)
            return
        if previous is None:
            delay = _INCOMPLETE_RETRY_INITIAL_SECONDS
        elif now < previous[0]:
            return
        elif progressed:
            delay = previous[1]
        else:
            delay = min(previous[1] * 2, _INCOMPLETE_RETRY_MAX_SECONDS)
        self.retry_at[var_path] = (now + delay, delay)


def _variable_prefix(
    coord: GoogleWeathernext2ForecastVirtualSourceFileCoord,
    var: GoogleWeathernext2DataVar,
) -> str:
    return f"{_store_key(coord.get_url())}/{var.internal_attrs.source_name}/"


def _required(
    objects: dict[str, NativeObjectMetadata] | None, query: ObjectListingQuery
) -> dict[str, NativeObjectMetadata]:
    """The proxy answers a listing of a valid init with a (possibly empty) page, even
    before the store exists, so a 403/404 means it refused the query, e.g. a proxy
    deployed without support for these listing forms. Fail rather than wait silently."""
    if objects is None:
        raise RuntimeError(
            f"wn.dynamical.org refused listing prefix={query.prefix} "
            f"matchGlob={query.match_glob}; deploy proxy support for it first"
        )
    return objects


def _listed_leads(objects: Mapping[str, NativeObjectMetadata]) -> set[int]:
    """Lead indices of listed operational chunks, whose keys are `member.lead[...]`."""
    leads = set()
    for location in objects:
        match = _OPERATIONAL_CHUNK_KEY.fullmatch(location.rsplit("/", 1)[-1])
        if match is not None:
            leads.add(int(match["lead"]))
    return leads


def _monotonic() -> float:
    return time.monotonic()


class GoogleWeathernext2ForecastVirtualRegionJob(
    WeatherNextVirtualRegionJob[
        GoogleWeathernext2DataVar, GoogleWeathernext2ForecastVirtualSourceFileCoord
    ]
):
    source_layout: ClassVar[SourceLayout]

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[GoogleWeathernext2DataVar],
        reformat_job_name: str,
        job_fire_time: Timestamp | None = None,
    ) -> tuple[
        Sequence[
            RegionJob[
                GoogleWeathernext2DataVar,
                GoogleWeathernext2ForecastVirtualSourceFileCoord,
            ]
        ],
        xr.DataTree,
    ]:
        if cls.source_layout == "historical":
            return super().operational_update_jobs(
                primary_store=primary_store,
                tmp_store=tmp_store,
                get_template_fn=get_template_fn,
                append_dim=append_dim,
                all_data_vars=all_data_vars,
                reformat_job_name=reformat_job_name,
                job_fire_time=PER_INIT_STORE_DATE,
            )
        job_fire_time = job_fire_time or _utc_now()
        newest_init = (job_fire_time - OPERATIONAL_SOURCE_LANDING_DELAY).floor(
            INIT_TIME_FREQUENCY
        )
        return super().operational_update_jobs(
            primary_store=primary_store,
            tmp_store=tmp_store,
            get_template_fn=get_template_fn,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            job_fire_time=newest_init + INIT_TIME_FREQUENCY,
        )

    def _available_lead_times(
        self, init_time: Timestamp, processing_region_ds: xr.Dataset
    ) -> Sequence[Timedelta]:
        if (init_time >= PER_INIT_STORE_DATE) != (self.source_layout == "operational"):
            return []
        return [
            pd.Timedelta(value) for value in processing_region_ds["lead_time"].values
        ]

    def _coords_for_step(
        self,
        init_time: Timestamp,
        lead_time: Timedelta,
        data_var_group: Sequence[GoogleWeathernext2DataVar],
    ) -> Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord]:
        return [
            GoogleWeathernext2ForecastVirtualSourceFileCoord(
                source_layout=self.source_layout,
                init_time=init_time,
                lead_time=lead_time,
                data_vars=(data_var,),
            )
            for data_var in data_var_group
        ]

    def _source_chunk_axes(
        self, var: GoogleWeathernext2DataVar
    ) -> tuple[Sequence[int], Sequence[int | None]]:
        ensemble_members = [
            int(value)
            for value in self.template_ds.to_dataset().get_index("ensemble_member")
        ]
        if self.source_layout == "historical":
            return ensemble_members[::4], (
                [None] if var.group is ROOT else [PRESSURE_LEVELS[0]]
            )
        return ensemble_members, [None] if var.group is ROOT else PRESSURE_LEVELS

    def _refs_per_step(self, var: GoogleWeathernext2DataVar) -> int:
        return prod(len(axis) for axis in self._source_chunk_axes(var))

    def _source_chunks(
        self, coord: GoogleWeathernext2ForecastVirtualSourceFileCoord
    ) -> list[NativeSourceChunk[GoogleWeathernext2DataVar]]:
        assert coord.source_layout == self.source_layout
        store_key_prefix = _store_key(coord.get_url()) + "/"
        chunks = []
        for var in coord.data_vars:
            members, levels = self._source_chunk_axes(var)
            for member in members:
                for level in levels:
                    key = coord.chunk_key(var, member, level)
                    out_loc: dict[Dim, CoordinateValue] = {
                        "init_time": coord.init_time,
                        "ensemble_member": member,
                        "lead_time": coord.lead_time,
                    }
                    if level is not None:
                        out_loc["pressure_level"] = level
                    chunks.append(
                        NativeSourceChunk(
                            data_var=var,
                            out_loc=out_loc,
                            location=f"{PROXY_LOCATION_PREFIX}{store_key_prefix}{key}",
                        )
                    )
        return chunks

    def _listing_queries(
        self, coord: GoogleWeathernext2ForecastVirtualSourceFileCoord
    ) -> list[ObjectListingQuery]:
        store_key_prefix = _store_key(coord.get_url()) + "/"
        queries = []
        for var in coord.data_vars:
            prefix = f"{store_key_prefix}{var.internal_attrs.source_name}/"
            if self.source_layout == "historical":
                queries.append(
                    ObjectListingQuery(f"{prefix}{coord.annual_init_index}.")
                )
            else:
                queries.append(
                    ObjectListingQuery(
                        prefix=prefix,
                        match_glob=(
                            f"{prefix}{_OPERATIONAL_MEMBER_GLOB}.{coord.lead_index}.*"
                        ),
                        delimiter="/",
                    )
                )
        return queries


class GoogleWeathernext2ForecastHistoricalVirtualRegionJob(
    GoogleWeathernext2ForecastVirtualRegionJob
):
    source_layout: ClassVar[SourceLayout] = "historical"
    manifest_init_split: ClassVar[int] = HISTORICAL_MANIFEST_INIT_SPLIT
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("1D")


class GoogleWeathernext2ForecastOperationalVirtualRegionJob(
    GoogleWeathernext2ForecastVirtualRegionJob
):
    source_layout: ClassVar[SourceLayout] = "operational"
    tick_interval: ClassVar[Timedelta] = pd.Timedelta("3s")
    # A 32-init batch would construct about 11.2 million virtual refs in memory.
    manifest_init_split: ClassVar[int] = OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT
    operational_update_window: ClassVar[Timedelta] = LEAD_TIMES[-1] + pd.Timedelta("2D")
    _polls: dict[Timestamp, _InitPoll] = PrivateAttr(default_factory=dict)

    def discover_available(
        self, pending: list[GoogleWeathernext2ForecastVirtualSourceFileCoord]
    ) -> list[tuple[GoogleWeathernext2ForecastVirtualSourceFileCoord, int]]:
        now = _monotonic()
        probe_var = self._probe_var().path
        by_init: dict[
            Timestamp, list[GoogleWeathernext2ForecastVirtualSourceFileCoord]
        ] = {}
        for coord in pending:
            by_init.setdefault(coord.init_time, []).append(coord)
        polls = {init: self._polls.setdefault(init, _InitPoll()) for init in by_init}
        full = {init for init, poll in polls.items() if now >= poll.next_full_listing}
        probes = {
            (init, var_path): self._probe_query(by_init[init][0], var_path)
            for init, poll in polls.items()
            if init not in full
            for var_path in {probe_var, *poll.retry_at}
        }
        probed = self._list(set(probes.values())) if probes else {}
        new_leads: dict[tuple[Timestamp, str], set[int]] = {}
        for (init, var_path), query in probes.items():
            leads = _listed_leads(_required(probed[query], query))
            seen = polls[init].probed_leads.setdefault(var_path, set())
            new_leads[init, var_path] = leads - seen
            seen |= leads

        due = [
            coord
            for init, coords in by_init.items()
            for coord in self._due_coords(
                coords, polls[init], now, full=init in full, new_leads=new_leads
            )
        ]
        if not due:
            return []

        queries = self._due_queries(due)
        objects: dict[str, NativeObjectMetadata] = {}
        for query, query_objects in self._list(queries).items():
            objects.update(_required(query_objects, query))
        for init in full:
            self._record_probe_leads(by_init[init][0], polls[init], objects)
        available = []
        ready_ids = set()
        for coord in due:
            locations = {chunk.location for chunk in self._source_chunks(coord)}
            if locations <= objects.keys():
                coord.chunk_metadata.clear()
                coord.chunk_metadata.update(
                    {location: objects[location] for location in locations}
                )
                available.append((coord, 0))
                ready_ids.add(id(coord))
        for init, coords in by_init.items():
            self._update_lagging(coords, polls[init], now, ready_ids)
        return available

    def _due_coords(
        self,
        coords: Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord],
        poll: _InitPoll,
        now: float,
        *,
        full: bool,
        new_leads: Mapping[tuple[Timestamp, str], set[int]],
    ) -> list[GoogleWeathernext2ForecastVirtualSourceFileCoord]:
        if full:
            poll.next_full_listing = now + _FULL_LISTING_INTERVAL_SECONDS
            return list(coords)
        probe_var = self._probe_var().path
        due = []
        for coord in coords:
            var_path = coord.data_vars[0].path
            lead = coord.lead_index
            retry = poll.retry_at.get(var_path)
            if retry is None:
                is_due = lead in new_leads.get((coord.init_time, probe_var), set())
            else:
                # A lagging variable waits for its own probe or its batched retry.
                is_due = lead in new_leads.get((coord.init_time, var_path), set()) or (
                    now >= retry[0] and lead in poll.known_leads(var_path, probe_var)
                )
            if is_due:
                due.append(coord)
        return due

    def _record_probe_leads(
        self,
        coord: GoogleWeathernext2ForecastVirtualSourceFileCoord,
        poll: _InitPoll,
        objects: Mapping[str, NativeObjectMetadata],
    ) -> None:
        """A full listing also covers each probe, so its leads aren't "new" next tick."""
        for var in self.data_vars:
            prefix = self._probe_query(coord, var.path).prefix
            poll.probed_leads.setdefault(var.path, set()).update(
                _listed_leads(
                    {
                        location: metadata
                        for location, metadata in objects.items()
                        if location.startswith(f"{PROXY_LOCATION_PREFIX}{prefix}")
                    }
                )
            )

    def _update_lagging(
        self,
        coords: Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord],
        poll: _InitPoll,
        now: float,
        ready_ids: set[int],
    ) -> None:
        """A variable lags while any pending coord of it has a probed lead."""
        probe_var = self._probe_var().path
        outcomes: dict[str, tuple[bool, bool]] = {}  # var -> (lagging, progressed)
        for coord in coords:
            var_path = coord.data_vars[0].path
            lagging, progressed = outcomes.get(var_path, (False, False))
            if id(coord) in ready_ids:
                progressed = True
            elif coord.lead_index in poll.known_leads(var_path, probe_var):
                lagging = True
            outcomes[var_path] = (lagging, progressed)
        for var_path, (lagging, progressed) in outcomes.items():
            poll.update_retry(var_path, now, lagging=lagging, progressed=progressed)

    def _probe_var(self) -> GoogleWeathernext2DataVar:
        return next(
            (var for var in self.data_vars if var.group is ROOT), self.data_vars[0]
        )

    def _probe_query(
        self, coord: GoogleWeathernext2ForecastVirtualSourceFileCoord, var_path: str
    ) -> ObjectListingQuery:
        var = next(var for var in self.data_vars if var.path == var_path)
        prefix = f"{_variable_prefix(coord, var)}{_PROBE_MEMBER}."
        return ObjectListingQuery(prefix=prefix, match_glob=f"{prefix}*", delimiter="/")

    def _due_queries(
        self, due: Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord]
    ) -> set[ObjectListingQuery]:
        by_variable: dict[
            str, list[GoogleWeathernext2ForecastVirtualSourceFileCoord]
        ] = {}
        for coord in due:
            for var in coord.data_vars:
                by_variable.setdefault(_variable_prefix(coord, var), []).append(coord)
        queries: set[ObjectListingQuery] = set()
        for prefix, coords in by_variable.items():
            if (
                len({coord.lead_index for coord in coords})
                >= _ALL_LEADS_LISTING_MIN_LEADS
            ):
                queries.add(
                    ObjectListingQuery(
                        prefix=prefix,
                        match_glob=f"{prefix}{_OPERATIONAL_MEMBER_GLOB}.*",
                        delimiter="/",
                    )
                )
            else:
                queries.update(
                    query for coord in coords for query in self._listing_queries(coord)
                )
        return queries


def _store_key(url: str) -> str:
    return url.removeprefix(SOURCE_LOCATION_PREFIX)
