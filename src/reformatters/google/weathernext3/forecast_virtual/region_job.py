from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.config_models import DataVar
from reformatters.common.region_job import CoordinateValue, RegionJob
from reformatters.common.types import AppendDim, DatetimeLike, Dim, Timedelta, Timestamp
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.google.weathernext_virtual.holdback import (
    PUBLICATION_HOLDBACK,
    is_publishable,
    utc_now,
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

from .source import source_horizon, source_store_key, source_store_url
from .template_config import STATISTICS, GoogleWeathernext3DataVar


class GoogleWeathernext3ForecastVirtualSourceFileCoord(WeatherNextSourceFileCoord):
    data_vars: Sequence[GoogleWeathernext3DataVar]

    def get_url(self) -> str:
        return source_store_url(self.init_time)

    @property
    def lead_index(self) -> int:
        assert self.lead_time % pd.Timedelta("1h") == pd.Timedelta(0)
        assert pd.Timedelta("1h") <= self.lead_time <= source_horizon(self.init_time)
        return int(self.lead_time // pd.Timedelta("1h")) - 1

    def chunk_location(self, var: GoogleWeathernext3DataVar, statistic: str) -> str:
        assert statistic in STATISTICS
        return f"{PROXY_LOCATION_PREFIX}{source_store_key(self.init_time)}/{var.internal_attrs.source_name}_{statistic}/c/{self.lead_index}/0/0"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class GoogleWeathernext3ForecastVirtualRegionJob(
    WeatherNextVirtualRegionJob[
        GoogleWeathernext3DataVar, GoogleWeathernext3ForecastVirtualSourceFileCoord
    ]
):
    init_frequency: ClassVar[Timedelta]
    tick_interval: ClassVar[Timedelta] = pd.Timedelta("30s")

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,  # noqa: ARG003
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[GoogleWeathernext3DataVar],
        reformat_job_name: str,
        job_fire_time: Timestamp | None = None,
    ) -> tuple[
        Sequence[
            RegionJob[
                GoogleWeathernext3DataVar,
                GoogleWeathernext3ForecastVirtualSourceFileCoord,
            ]
        ],
        xr.DataTree,
    ]:
        reference_time = job_fire_time or utc_now()
        cutoff = reference_time - PUBLICATION_HOLDBACK
        newest_init = (cutoff.floor("h") - pd.Timedelta("1h")).floor(cls.init_frequency)
        append_dim_end = newest_init + cls.init_frequency
        template = get_template_fn(append_dim_end)
        inits = template.to_dataset().get_index(append_dim)
        window_start = int(
            inits.searchsorted(append_dim_end - cls.operational_update_window)
        )
        job = cls(
            tmp_store=tmp_store,
            template_ds=template,
            append_dim=append_dim,
            data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            region=slice(window_start, len(inits)),
            processing_mode="update",
            reference_time=reference_time,
            publication_cutoff=cutoff,
        )
        return [job], template

    def _available_lead_times(
        self, init_time: Timestamp, processing_region_ds: xr.Dataset
    ) -> Sequence[Timedelta]:
        return [
            pd.Timedelta(value)
            for value in processing_region_ds["lead_time"].values
            if pd.Timedelta(value) <= source_horizon(init_time)
        ]

    def _coords_for_step(
        self,
        init_time: Timestamp,
        lead_time: Timedelta,
        data_var_group: Sequence[GoogleWeathernext3DataVar],
    ) -> Sequence[GoogleWeathernext3ForecastVirtualSourceFileCoord]:
        return [
            GoogleWeathernext3ForecastVirtualSourceFileCoord(
                init_time=init_time, lead_time=lead_time, data_vars=(var,)
            )
            for var in data_var_group
        ]

    def representative_probe_loc(
        self, coord: GoogleWeathernext3ForecastVirtualSourceFileCoord, var: DataVar[Any]
    ) -> Mapping[Dim, CoordinateValue]:
        return {**super().representative_probe_loc(coord, var), "statistic": "mean"}

    def _source_chunks(
        self, coord: GoogleWeathernext3ForecastVirtualSourceFileCoord
    ) -> Sequence[NativeSourceChunk[GoogleWeathernext3DataVar]]:
        return [
            NativeSourceChunk(
                var,
                {**coord.out_loc(), "statistic": statistic},
                coord.chunk_location(var, statistic),
            )
            for var in coord.data_vars
            for statistic in STATISTICS
        ]

    def _listing_queries(
        self, coord: GoogleWeathernext3ForecastVirtualSourceFileCoord
    ) -> Sequence[ObjectListingQuery]:
        queries = []
        for var in coord.data_vars:
            prefix = (
                f"{source_store_key(coord.init_time)}/{var.internal_attrs.source_name}_"
            )
            whole_variable = is_publishable(
                coord.init_time,
                source_horizon(coord.init_time),
                self.publication_cutoff,
            )
            queries.append(
                ObjectListingQuery(
                    prefix=prefix,
                    match_glob=None
                    if whole_variable
                    else f"{prefix}{{{','.join(STATISTICS)}}}/c/{coord.lead_index}/0/0",
                )
            )
        return queries

    def file_refs(
        self, coord: GoogleWeathernext3ForecastVirtualSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        refs = super().file_refs(coord, file_size)
        assert is_publishable(coord.init_time, coord.lead_time, self.publication_cutoff)
        for ref in refs:
            assert ref.data_var in coord.data_vars
            assert ref.out_loc["init_time"] == coord.init_time
            assert ref.out_loc["lead_time"] == coord.lead_time
            statistic = ref.out_loc["statistic"]
            assert isinstance(statistic, str)
            assert statistic in STATISTICS
            assert isinstance(ref.data_var, GoogleWeathernext3DataVar)
            assert ref.location == coord.chunk_location(ref.data_var, statistic)
            target = self.template_ds[ref.data_var.path]
            station = ref.data_var.internal_attrs.source_name.startswith(
                "station_head_"
            )
            assert (target.sizes["y"], target.sizes["x"]) == (
                (3601, 7200) if station else (1801, 3600)
            )
            assert coord.init_time in target.get_index("init_time")
            assert coord.lead_time in target.get_index("lead_time")
            assert statistic in target.get_index("statistic")
        return refs
