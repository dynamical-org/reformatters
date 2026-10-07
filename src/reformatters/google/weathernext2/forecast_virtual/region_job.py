from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
import xarray as xr
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
from reformatters.google.weathernext_virtual.holdback import (
    PUBLICATION_HOLDBACK,
)
from reformatters.google.weathernext_virtual.holdback import (
    utc_now as _utc_now,
)
from reformatters.google.weathernext_virtual.listing import (
    PROXY_LOCATION_PREFIX,
    ObjectListingQuery,
)
from reformatters.google.weathernext_virtual.region_job import (
    NativeSourceChunk,
    WeatherNextSourceFileCoord,
    WeatherNextVirtualRegionJob,
)

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
# The two layouts pack chunks differently, so splits are sized per product and per
# array group by ref density; see docs/virtual_datasets.md.
HISTORICAL_MANIFEST_INIT_SPLIT = 128
OPERATIONAL_ROOT_MANIFEST_INIT_SPLIT = 32
OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT = 4


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


class GoogleWeathernext2ForecastVirtualRegionJob(
    WeatherNextVirtualRegionJob[
        GoogleWeathernext2DataVar, GoogleWeathernext2ForecastVirtualSourceFileCoord
    ]
):
    requires_scan_provenance: ClassVar[bool] = False
    source_layout: ClassVar[SourceLayout]
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
        reference_time: Timestamp | None = None,
        append_dim_end: Timestamp | None = None,
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
            reference_time=reference_time,
            append_dim_end=append_dim_end,
        )
        if cls.source_layout == "historical":
            return jobs
        cutoff = (reference_time or _utc_now()) - PUBLICATION_HOLDBACK
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
        publication_cutoff = (job_fire_time or _utc_now()) - PUBLICATION_HOLDBACK
        latest_publishable_valid_time = (
            publication_cutoff
            - (publication_cutoff - PER_INIT_STORE_DATE) % INIT_TIME_FREQUENCY
        )
        newest_publishable_init = latest_publishable_valid_time - LEAD_TIMES[0]
        jobs, template_ds = super().operational_update_jobs(
            primary_store=primary_store,
            tmp_store=tmp_store,
            get_template_fn=get_template_fn,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            job_fire_time=newest_publishable_init + INIT_TIME_FREQUENCY,
        )
        (job,) = jobs
        assert isinstance(job, cls)
        return [
            job.model_copy(update={"publication_cutoff": publication_cutoff})
        ], template_ds

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

    def _source_chunks(
        self, coord: GoogleWeathernext2ForecastVirtualSourceFileCoord
    ) -> list[NativeSourceChunk[GoogleWeathernext2DataVar]]:
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
    # A 32-init batch would construct about 11.2 million virtual refs in memory.
    manifest_init_split: ClassVar[int] = OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT
    operational_update_window: ClassVar[Timedelta] = LEAD_TIMES[-1] + pd.Timedelta("2D")


def _store_key(url: str) -> str:
    return url.removeprefix(SOURCE_LOCATION_PREFIX)
