import re
from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd
import xarray as xr

from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim, Timedelta
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.noaa.noaa_virtual_region_job import (
    NoaaVirtualRegionJob,
    NoaaVirtualSourceFileCoord,
)
from reformatters.noaa.rrfs.region_job import SOURCE_PREFIX, SOURCE_REGION

from .models import NoaaRefsDataVar, ProductFamily


class NoaaRefsSourceFileCoord(NoaaVirtualSourceFileCoord[NoaaRefsDataVar]):
    source_family: ProductFamily

    def get_url(self) -> str:
        hour = int(self.lead_time / pd.Timedelta("1h"))
        assert self.lead_time == pd.Timedelta(hours=hour)
        cycle = self.init_time.strftime("%Y%m%d/%H")
        prefix = self.init_time.strftime("refs.t%Hz")
        return f"{SOURCE_PREFIX}refs.{cycle}/ensprod/{prefix}.{self.source_family}.f{hour:02}.conus.grib2"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        loc: dict[Dim, CoordinateValue] = {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }
        if self.source_family in ("mean", "sprd"):
            loc["statistic"] = (
                "mean" if self.source_family == "mean" else "standard_deviation"
            )
        return loc

    def index_selectors(self, selectors: tuple[str, ...]) -> tuple[str, ...]:
        if self.source_family in ("mean", "sprd"):
            tag = "wt ens mean" if self.source_family == "mean" else "ens spread"
            assert selectors.count(tag) == 1, (self.get_url(), selectors)
            return tuple(s for s in selectors if s != tag)
        if self.source_family in ("prob", "eas", "ffri"):
            counts = [s for s in selectors if re.fullmatch(r"prob fcst \d+/\d+", s)]
            assert len(counts) == 1, (self.get_url(), selectors)
            return tuple(s for s in selectors if s not in counts)
        return selectors


class NoaaRefsRegionJob(NoaaVirtualRegionJob[NoaaRefsDataVar, NoaaRefsSourceFileCoord]):
    source_location_prefix: ClassVar[str] = SOURCE_PREFIX
    source_bucket_region: ClassVar[str] = SOURCE_REGION
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("18h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaRefsDataVar],
    ) -> Sequence[NoaaRefsSourceFileCoord]:
        return [
            NoaaRefsSourceFileCoord(
                init_time=init,
                lead_time=lead,
                source_family=family,
                data_vars=variables,
            )
            for init in pd.to_datetime(processing_region_ds.init_time.values)
            for family in sorted(
                {f for v in data_var_group for f in v.internal_attrs.source_families}
            )
            for lead in pd.to_timedelta(processing_region_ds.lead_time.values)
            if (
                variables := [
                    v
                    for v in data_var_group
                    if family in v.internal_attrs.source_families
                    and v.available_at(lead)
                ]
            )
        ]

    def _check_refs_complete(
        self, coord: NoaaRefsSourceFileCoord, refs: list[VirtualRef]
    ) -> None:
        expected = {v.path for v in coord.data_vars}
        actual = {r.data_var.path for r in refs}
        assert len(refs) == len(actual), (
            f"{coord.get_url()}: duplicate output references"
        )
        assert actual == expected, (
            f"{coord.get_url()}: required GRIB messages absent from index: {sorted(expected - actual)}"
        )
