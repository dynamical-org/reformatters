from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, ClassVar, Generic, NamedTuple, TypeVar

import httpx
import pandas as pd
import xarray as xr
from pydantic import Field

from reformatters.common.config_models import DataVar
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    DATA_VAR,
    CoordinateValue,
    InitLeadSourceFileCoord,
)
from reformatters.common.types import Dim, Timedelta, Timestamp
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob

from .holdback import is_publishable
from .listing import NativeObjectMetadata, ObjectListingQuery, list_objects

log = get_logger(__name__)


class WeatherNextSourceFileCoord(InitLeadSourceFileCoord):
    data_vars: Sequence[DataVar[Any]]
    chunk_metadata: dict[str, NativeObjectMetadata] = Field(
        default_factory=dict, frozen=False
    )


class NativeSourceChunk(NamedTuple, Generic[DATA_VAR]):
    data_var: DATA_VAR
    out_loc: Mapping[Dim, CoordinateValue]
    location: str


SOURCE_COORD = TypeVar("SOURCE_COORD", bound=WeatherNextSourceFileCoord)


class WeatherNextVirtualRegionJob(
    VirtualRegionJob[DATA_VAR, SOURCE_COORD], Generic[DATA_VAR, SOURCE_COORD]
):
    manifest_init_split: ClassVar[int]
    publication_cutoff: Timestamp

    @abstractmethod
    def _available_lead_times(
        self, init_time: Timestamp, processing_region_ds: xr.Dataset
    ) -> Sequence[Timedelta]:
        raise NotImplementedError

    @abstractmethod
    def _coords_for_step(
        self,
        init_time: Timestamp,
        lead_time: Timedelta,
        data_var_group: Sequence[DATA_VAR],
    ) -> Sequence[SOURCE_COORD]:
        raise NotImplementedError

    @abstractmethod
    def _source_chunks(
        self, coord: SOURCE_COORD
    ) -> Sequence[NativeSourceChunk[DATA_VAR]]:
        raise NotImplementedError

    @abstractmethod
    def _listing_queries(self, coord: SOURCE_COORD) -> Sequence[ObjectListingQuery]:
        raise NotImplementedError

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[DATA_VAR],
    ) -> Sequence[SOURCE_COORD]:
        return self._step_coords(processing_region_ds, data_var_group, publishable=True)

    def held_back_source_file_coords(
        self,
    ) -> Sequence[SOURCE_COORD]:
        """Return every held-back source step in this job's processing region."""
        return self._step_coords(
            self._processing_region_ds(), self.data_vars, publishable=False
        )

    def _step_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[DATA_VAR],
        *,
        publishable: bool,
    ) -> Sequence[SOURCE_COORD]:
        coords = []
        for init_time_value in processing_region_ds["init_time"].values:
            init_time = pd.Timestamp(init_time_value)
            for lead_time in self._available_lead_times(
                init_time, processing_region_ds
            ):
                eligible = is_publishable(init_time, lead_time, self.publication_cutoff)
                if eligible is not publishable:
                    continue
                coords.extend(
                    self._coords_for_step(init_time, lead_time, data_var_group)
                )
        return coords

    def discover_available(
        self, pending: list[SOURCE_COORD]
    ) -> list[tuple[SOURCE_COORD, int]]:
        queries = sorted(
            {query for coord in pending for query in self._listing_queries(coord)},
            key=lambda query: (
                query.prefix,
                query.match_glob or "",
                query.delimiter or "",
            ),
        )
        with (
            httpx.Client(timeout=30) as client,
            ThreadPoolExecutor(self.download_concurrency) as pool,
        ):
            listed = dict(
                zip(
                    queries,
                    pool.map(partial(list_objects, client), queries),
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
        remaining: Sequence[SOURCE_COORD],
    ) -> Iterator[
        Sequence[
            tuple[
                SOURCE_COORD,
                Sequence[VirtualRef],
            ]
        ]
    ]:
        if self.processing_mode == "update":
            yield from super().process_virtual_refs(remaining)
            return

        coords_by_manifest: dict[int, list[SOURCE_COORD]] = {}
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
        coord: SOURCE_COORD,
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
