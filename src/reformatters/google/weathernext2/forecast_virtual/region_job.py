from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import ClassVar

import httpx
import icechunk
import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.config_models import ROOT
from reformatters.common.region_job import CoordinateValue, RegionJob, SourceFileCoord
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
)

SOURCE_LOCATION_PREFIX = "gs://weathernext/"
AVAILABILITY_LOCATION_PREFIX = "https://wn.dynamical.org/available/"
PROXY_LOCATION_PREFIX = "https://wn.dynamical.org/chunks/"
_SOURCE_ZARR_PREFIX = f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0/zarr/"
_SOURCE_LEVEL_INDEX = {
    level: index for index, level in enumerate(sorted(PRESSURE_LEVELS))
}
_NO_SINGLE_FILE_SIZE = 0
_PUBLICATION_LAG = pd.Timedelta("48h")
OUTPUT_CHUNK_LENGTH = 721 * 1440 * 4
ROOT_MANIFEST_INIT_SPLIT = 32
PRESSURE_MANIFEST_INIT_SPLIT = 4


def weathernext2_virtual_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    return (
        icechunk.VirtualChunkContainer(PROXY_LOCATION_PREFIX, icechunk.http_store()),
    )


class GoogleWeathernext2ForecastVirtualSourceFileCoord(SourceFileCoord):
    """One forecast lead from an annual or per-init source Zarr store."""

    init_time: Timestamp
    lead_time: Timedelta
    data_vars: Sequence[GoogleWeathernext2DataVar]

    @property
    def is_per_init_store(self) -> bool:
        return self.init_time >= PER_INIT_STORE_DATE

    def get_url(self) -> str:
        if self.is_per_init_store:
            return (
                f"{_SOURCE_ZARR_PREFIX}2025_to_present/"
                f"{self.init_time:%Y%m%d}_{self.init_time:%H}hr_01_preds/predictions.zarr"
            )
        year = self.init_time.year
        return f"{_SOURCE_ZARR_PREFIX}{year}_to_{year + 1}/predictions.zarr"

    def get_success_marker_url(self) -> str:
        return self.get_url().removesuffix("predictions.zarr") + "success"

    def get_availability_url(self) -> str:
        return AVAILABILITY_LOCATION_PREFIX + _store_key(self.get_success_marker_url())

    @property
    def lead_index(self) -> int:
        return int(self.lead_time // pd.Timedelta("6h")) - 1

    @property
    def annual_init_index(self) -> int:
        assert not self.is_per_init_store
        year_start = pd.Timestamp(f"{self.init_time.year}-01-01")
        return int((self.init_time - year_start) // pd.Timedelta("6h"))

    def chunk_key(
        self,
        var: GoogleWeathernext2DataVar,
        ensemble_member: int,
        pressure_level: int | None,
    ) -> str:
        if self.is_per_init_store:
            indices = [ensemble_member, self.lead_index]
            if pressure_level is not None:
                indices.append(_SOURCE_LEVEL_INDEX[pressure_level])
        else:
            indices = [self.annual_init_index, ensemble_member // 4, self.lead_index]
            if pressure_level is not None:
                indices.append(0)
        indices.extend((0, 0))
        return f"{var.internal_attrs.source_name}/" + ".".join(map(str, indices))

    def plane_index(
        self,
        var: GoogleWeathernext2DataVar,
        ensemble_member: int,
        pressure_level: int | None,
    ) -> int:
        if self.is_per_init_store:
            return 0
        member_plane = ensemble_member % 4
        if var.group is ROOT:
            return member_plane
        assert pressure_level is not None
        return member_plane * len(PRESSURE_LEVELS) + _SOURCE_LEVEL_INDEX[pressure_level]

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class GoogleWeathernext2ForecastVirtualRegionJob(
    VirtualRegionJob[
        GoogleWeathernext2DataVar, GoogleWeathernext2ForecastVirtualSourceFileCoord
    ]
):
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("18D")
    publication_cutoff: Timestamp = pd.Timestamp.max

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
        jobs, template_ds = super().operational_update_jobs(
            primary_store=primary_store,
            tmp_store=tmp_store,
            get_template_fn=get_template_fn,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            job_fire_time=job_fire_time,
        )
        (job,) = jobs
        assert isinstance(job, cls)
        fire_time = job_fire_time or pd.Timestamp.now()
        return [
            job.model_copy(update={"publication_cutoff": fire_time - _PUBLICATION_LAG})
        ], template_ds

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[GoogleWeathernext2DataVar],
    ) -> Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord]:
        coords = []
        for init_time_value in processing_region_ds["init_time"].values:
            init_time = pd.Timestamp(init_time_value)
            for lead_time_value in processing_region_ds["lead_time"].values:
                lead_time = pd.Timedelta(lead_time_value)
                if init_time + lead_time >= self.publication_cutoff:
                    continue
                coords.append(
                    GoogleWeathernext2ForecastVirtualSourceFileCoord(
                        init_time=init_time,
                        lead_time=lead_time,
                        data_vars=data_var_group,
                    )
                )
        return coords

    def discover_available(
        self, pending: list[GoogleWeathernext2ForecastVirtualSourceFileCoord]
    ) -> list[tuple[GoogleWeathernext2ForecastVirtualSourceFileCoord, int]]:
        availability_urls = sorted({coord.get_availability_url() for coord in pending})
        with (
            httpx.Client(timeout=30) as client,
            ThreadPoolExecutor(self.download_concurrency) as pool,
        ):
            available_leads = dict(
                zip(
                    availability_urls,
                    pool.map(partial(_available_lead_count, client), availability_urls),
                    strict=True,
                )
            )
        return [
            (coord, _NO_SINGLE_FILE_SIZE)
            for coord in pending
            if (lead_count := available_leads[coord.get_availability_url()]) is not None
            and coord.lead_index < lead_count
        ]

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
            manifest_index = init_index // PRESSURE_MANIFEST_INIT_SPLIT
            coords_by_manifest.setdefault(manifest_index, []).append(coord)
        for coords in coords_by_manifest.values():
            coords.sort(key=lambda coord: (coord.init_time, coord.lead_time))
            yield from super().process_virtual_refs(coords)

    def file_refs(
        self,
        coord: GoogleWeathernext2ForecastVirtualSourceFileCoord,
        file_size: int,  # noqa: ARG002 - the coord spans many source chunk objects
    ) -> list[VirtualRef]:
        refs = []
        store_key_prefix = _store_key(coord.get_url()) + "/"
        ensemble_members = self.template_ds.to_dataset().get_index("ensemble_member")
        for var in coord.data_vars:
            template_var = self.template_ds[var.path]
            pressure_levels: Sequence[int | None] = (
                [None]
                if var.group is ROOT
                else [int(level) for level in template_var.get_index("pressure_level")]
            )
            for member_value in ensemble_members:
                member = int(member_value)
                for level in pressure_levels:
                    key = coord.chunk_key(var, member, level)
                    out_loc: dict[Dim, CoordinateValue] = {
                        "init_time": coord.init_time,
                        "ensemble_member": member,
                        "lead_time": coord.lead_time,
                    }
                    if level is not None:
                        out_loc["pressure_level"] = level
                    plane = coord.plane_index(var, member, level)
                    refs.append(
                        VirtualRef(
                            data_var=var,
                            out_loc=out_loc,
                            location=(
                                f"{PROXY_LOCATION_PREFIX}plane/{plane}/"
                                f"{store_key_prefix}{key}"
                            ),
                            offset=0,
                            length=OUTPUT_CHUNK_LENGTH,
                        )
                    )
        return refs


def _store_key(url: str) -> str:
    return url.removeprefix(SOURCE_LOCATION_PREFIX)


def _available_lead_count(client: httpx.Client, url: str) -> int | None:
    response = client.head(url)
    if response.status_code in {403, 404}:
        return None
    response.raise_for_status()
    lead_count = int(response.headers["X-WeatherNext-Available-Lead-Count"])
    assert 1 <= lead_count <= 60, f"invalid available lead count: {lead_count}"
    return lead_count
