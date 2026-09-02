from base64 import b64decode
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import ClassVar, NamedTuple, Self

import httpx
import icechunk
import pandas as pd
import xarray as xr
from pydantic import Field
from zarr.abc.store import Store

from reformatters.common.config_models import ROOT
from reformatters.common.logging import get_logger
from reformatters.common.region_job import CoordinateValue, RegionJob, SourceFileCoord
from reformatters.common.retry import retry
from reformatters.common.types import (
    AppendDim,
    DatetimeLike,
    Dim,
    Timedelta,
    Timestamp,
)
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob

from .template_config import (
    PER_INIT_STORE_DATE,
    PRESSURE_LEVELS,
    GoogleWeathernext2DataVar,
    SourceLayout,
)

SOURCE_LOCATION_PREFIX = "gs://weathernext/"
PROXY_LOCATION_PREFIX = "https://wn.dynamical.org/chunks/"
OBJECTS_LOCATION = "https://wn.dynamical.org/objects"
_SOURCE_ZARR_PREFIX = f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0/zarr/"
_SOURCE_LEVEL_INDEX = {level: index for index, level in enumerate(PRESSURE_LEVELS)}
_OPERATIONAL_MEMBER_GLOB = "{" + ",".join(map(str, range(64))) + "}"
_PUBLICATION_LAG = pd.Timedelta("48h")
# The two layouts pack chunks differently, so splits are sized per product and per
# array group by ref density; see docs/virtual_datasets.md.
HISTORICAL_MANIFEST_INIT_SPLIT = 128
OPERATIONAL_ROOT_MANIFEST_INIT_SPLIT = 32
OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT = 4

log = get_logger(__name__)


class NativeObjectMetadata(NamedTuple):
    size: int
    etag_checksum: str


def weathernext2_virtual_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    return (
        icechunk.VirtualChunkContainer(PROXY_LOCATION_PREFIX, icechunk.http_store()),
    )


class GoogleWeathernext2ForecastVirtualSourceFileCoord(SourceFileCoord):
    """One forecast lead from one native annual or per-init source Zarr store."""

    source_layout: SourceLayout
    init_time: Timestamp
    lead_time: Timedelta
    data_vars: Sequence[GoogleWeathernext2DataVar]
    chunk_metadata: dict[str, NativeObjectMetadata] = Field(
        default_factory=dict, frozen=False
    )

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


class NativeSourceChunk(NamedTuple):
    data_var: GoogleWeathernext2DataVar
    out_loc: Mapping[Dim, CoordinateValue]
    location: str


class ObjectListingQuery(NamedTuple):
    prefix: str
    match_glob: str | None = None


class GoogleWeathernext2ForecastVirtualRegionJob(
    VirtualRegionJob[
        GoogleWeathernext2DataVar, GoogleWeathernext2ForecastVirtualSourceFileCoord
    ]
):
    source_layout: ClassVar[SourceLayout]
    manifest_init_split: ClassVar[int]
    publication_cutoff: Timestamp = pd.Timestamp.max

    @classmethod
    def get_jobs(
        cls,
        tmp_store: Path,
        template_ds: xr.DataTree,
        append_dim: AppendDim,
        all_data_vars: Sequence[GoogleWeathernext2DataVar],
        reformat_job_name: str,
        filter_start: Timestamp | None = None,
        filter_end: Timestamp | None = None,
        filter_contains: list[Timestamp] | None = None,
        filter_variable_names: list[str] | None = None,
    ) -> Sequence[Self]:
        jobs = super().get_jobs(
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=filter_start,
            filter_end=filter_end,
            filter_contains=filter_contains,
            filter_variable_names=filter_variable_names,
        )
        if cls.source_layout == "historical":
            return jobs
        cutoff = _current_publication_cutoff()
        return [job.model_copy(update={"publication_cutoff": cutoff}) for job in jobs]

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
        fire_time = job_fire_time or _utc_now()
        publication_cutoff = fire_time - _PUBLICATION_LAG
        jobs, template_ds = super().operational_update_jobs(
            primary_store=primary_store,
            tmp_store=tmp_store,
            get_template_fn=get_template_fn,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            job_fire_time=publication_cutoff,
        )
        (job,) = jobs
        assert isinstance(job, cls)
        return [
            job.model_copy(update={"publication_cutoff": publication_cutoff})
        ], template_ds

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[GoogleWeathernext2DataVar],
    ) -> Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord]:
        coords = []
        for init_time_value in processing_region_ds["init_time"].values:
            init_time = pd.Timestamp(init_time_value)
            if (init_time >= PER_INIT_STORE_DATE) != (
                self.source_layout == "operational"
            ):
                continue
            if (
                self.source_layout == "operational"
                and init_time >= self.publication_cutoff
            ):
                continue
            for lead_time_value in processing_region_ds["lead_time"].values:
                lead_time = pd.Timedelta(lead_time_value)
                coords.extend(
                    GoogleWeathernext2ForecastVirtualSourceFileCoord(
                        source_layout=self.source_layout,
                        init_time=init_time,
                        lead_time=lead_time,
                        data_vars=(data_var,),
                    )
                    for data_var in data_var_group
                )
        return coords

    def _source_chunks(
        self, coord: GoogleWeathernext2ForecastVirtualSourceFileCoord
    ) -> list[NativeSourceChunk]:
        assert coord.source_layout == self.source_layout
        store_key_prefix = _store_key(coord.get_url()) + "/"
        ensemble_members = [
            int(value)
            for value in self.template_ds.to_dataset().get_index("ensemble_member")
        ]
        chunks = []
        for var in coord.data_vars:
            if self.source_layout == "historical":
                members = ensemble_members[::4]
                levels: Sequence[int | None] = (
                    [None] if var.group is ROOT else [PRESSURE_LEVELS[0]]
                )
            else:
                members = ensemble_members
                levels = [None] if var.group is ROOT else PRESSURE_LEVELS
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
                    )
                )
        return queries

    def discover_available(
        self, pending: list[GoogleWeathernext2ForecastVirtualSourceFileCoord]
    ) -> list[tuple[GoogleWeathernext2ForecastVirtualSourceFileCoord, int]]:
        queries = sorted(
            {query for coord in pending for query in self._listing_queries(coord)}
        )
        with (
            httpx.Client(timeout=30) as client,
            ThreadPoolExecutor(self.download_concurrency) as pool,
        ):
            listed = dict(
                zip(
                    queries,
                    pool.map(partial(_list_objects, client), queries),
                    strict=True,
                )
            )

        available = []
        for coord in pending:
            coord_objects: dict[str, NativeObjectMetadata] = {}
            for query in self._listing_queries(coord):
                objects = listed[query]
                if objects is None:
                    break
                coord_objects.update(objects)
            else:
                locations = {chunk.location for chunk in self._source_chunks(coord)}
                if locations <= coord_objects.keys():
                    coord.chunk_metadata.clear()
                    coord.chunk_metadata.update(
                        {location: coord_objects[location] for location in locations}
                    )
                    available.append((coord, 0))
                else:
                    missing = locations - coord_objects.keys()
                    log.debug(
                        f"{len(missing)} source chunks unavailable for "
                        f"{coord.get_url()} {coord.data_vars[0].path}; "
                        f"first: {min(missing)}"
                    )
        return available

    def process_virtual_refs(
        self,
        remaining: Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord],
    ) -> Iterator[
        Sequence[
            tuple[
                GoogleWeathernext2ForecastVirtualSourceFileCoord,
                Sequence[VirtualRef],
            ]
        ]
    ]:
        if self.processing_mode == "update":
            yield from super().process_virtual_refs(remaining)
            return

        coords_by_manifest: dict[
            int, list[GoogleWeathernext2ForecastVirtualSourceFileCoord]
        ] = {}
        init_times = self.template_ds.to_dataset().get_index("init_time")
        for coord in remaining:
            init_index = init_times.get_loc(coord.init_time)
            assert isinstance(init_index, int)
            manifest_index = init_index // self.manifest_init_split
            coords_by_manifest.setdefault(manifest_index, []).append(coord)
        for coords in coords_by_manifest.values():
            coords.sort(key=lambda coord: (coord.init_time, coord.lead_time))
            yield from super().process_virtual_refs(coords)

    def file_refs(
        self,
        coord: GoogleWeathernext2ForecastVirtualSourceFileCoord,
        file_size: int,  # noqa: ARG002 - each coord covers several native objects
    ) -> list[VirtualRef]:
        chunks = self._source_chunks(coord)
        assert set(coord.chunk_metadata) == {chunk.location for chunk in chunks}
        return [
            VirtualRef(
                data_var=chunk.data_var,
                out_loc=chunk.out_loc,
                location=chunk.location,
                offset=0,
                length=coord.chunk_metadata[chunk.location].size,
                etag_checksum=coord.chunk_metadata[chunk.location].etag_checksum,
            )
            for chunk in chunks
        ]


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
    # A 32-init batch would construct about 11.2 million virtual refs in memory.
    manifest_init_split: ClassVar[int] = OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("24h")


def _store_key(url: str) -> str:
    return url.removeprefix(SOURCE_LOCATION_PREFIX)


def _utc_now() -> Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _current_publication_cutoff() -> Timestamp:
    return _utc_now() - _PUBLICATION_LAG


def _list_objects(
    client: httpx.Client, query: ObjectListingQuery
) -> dict[str, NativeObjectMetadata] | None:
    objects: dict[str, NativeObjectMetadata] = {}
    page_token: str | None = None
    while True:
        params = {"prefix": query.prefix, "maxResults": "1000"}
        if query.match_glob is not None:
            params.update({"matchGlob": query.match_glob, "delimiter": "/"})
        if page_token is not None:
            params["pageToken"] = page_token

        def get_page(params: dict[str, str] = params) -> httpx.Response:
            response = client.get(OBJECTS_LOCATION, params=params)
            if response.status_code in {408, 429} or response.status_code >= 500:
                response.raise_for_status()
            return response

        response = retry(
            get_page,
            retryable_exceptions=(httpx.RequestError, httpx.HTTPStatusError),
        )
        if response.status_code in {403, 404}:
            return None
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("items", []):
            key = str(item["name"])
            assert key.startswith(query.prefix), (
                f"listed object escaped prefix {query.prefix}: {key}"
            )
            size = int(item["size"])
            assert size > 0, f"invalid object size for {key}: {size}"
            location = f"{PROXY_LOCATION_PREFIX}{key}"
            assert location not in objects, f"duplicate listed object: {key}"
            md5 = b64decode(str(item["md5Hash"]), validate=True)
            assert len(md5) == 16, f"invalid object MD5 for {key}"
            objects[location] = NativeObjectMetadata(
                size=size, etag_checksum=f'"{md5.hex()}"'
            )
        page_token = payload.get("nextPageToken")
        if page_token is None:
            return objects
