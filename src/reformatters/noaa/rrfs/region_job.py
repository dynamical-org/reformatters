from collections.abc import Mapping, Sequence
from typing import ClassVar

import icechunk
import pandas as pd
import xarray as xr

from reformatters.common.config_models import ROOT
from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim, Timedelta
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.noaa.noaa_virtual_region_job import (
    NoaaVirtualRegionJob,
    NoaaVirtualSourceFileCoord,
)
from reformatters.noaa.rrfs.models import NoaaRrfsDataVar, RrfsSourceFamily

SOURCE_PREFIX = "s3://noaa-rrfs-ops-pds/"
SOURCE_REGION = "us-east-1"


def rrfs_virtual_chunk_containers() -> tuple[icechunk.VirtualChunkContainer, ...]:
    return (
        icechunk.VirtualChunkContainer(
            SOURCE_PREFIX, icechunk.s3_store(region=SOURCE_REGION)
        ),
    )


class NoaaRrfsSourceFileCoord(NoaaVirtualSourceFileCoord[NoaaRrfsDataVar]):
    source_family: RrfsSourceFamily
    ensemble_member: int | None = None

    def get_url(self) -> str:
        hour = int(self.lead_time / pd.Timedelta("1h"))
        assert self.lead_time == pd.Timedelta(hours=hour)
        cycle = self.init_time.strftime("%Y%m%d/%H")
        prefix = self.init_time.strftime("rrfs.t%Hz")
        member = self.ensemble_member
        if member is not None and member > 0:
            assert self.source_family != "subh"
            key = f"rrfsens.{cycle}/m{member:03}/{prefix}.m{member:03}.{self.source_family}nomads.3km.f{hour:03}.conus.grib2"
        else:
            family = (
                "2dfld.3km.subh"
                if self.source_family == "subh"
                else f"{self.source_family}.3km"
            )
            key = f"rrfs.{cycle}/{prefix}.{family}.f{hour:03}.conus.grib2"
        return SOURCE_PREFIX + key

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        loc: dict[Dim, CoordinateValue] = {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }
        if self.ensemble_member is not None:
            loc["ensemble_member"] = self.ensemble_member
        return loc

    def message_lead_times(self) -> Sequence[Timedelta]:
        if self.source_family != "subh":
            return super().message_lead_times()
        return tuple(self.lead_time - pd.Timedelta(minutes=m) for m in (45, 30, 15, 0))

    def index_selectors(self, selectors: tuple[str, ...]) -> tuple[str, ...]:
        if self.ensemble_member is None or self.ensemble_member == 0:
            return selectors
        ensemble = f"ENS=+{self.ensemble_member}"
        assert ensemble in selectors, (self.get_url(), selectors)
        return tuple(s for s in selectors if s != ensemble)


class NoaaRrfsRegionJob(NoaaVirtualRegionJob[NoaaRrfsDataVar, NoaaRrfsSourceFileCoord]):
    source_location_prefix: ClassVar[str] = SOURCE_PREFIX
    source_bucket_region: ClassVar[str] = SOURCE_REGION
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("18h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaRrfsDataVar],
    ) -> Sequence[NoaaRrfsSourceFileCoord]:
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        leads = pd.to_timedelta(processing_region_ds["lead_time"].values)
        members = (
            processing_region_ds["ensemble_member"].values.tolist()
            if "ensemble_member" in processing_region_ds
            else [None]
        )
        coords = []
        for init in init_times:
            for family in sorted(
                {v.internal_attrs.source_family for v in data_var_group}
            ):
                file_leads = sorted(
                    {lead.ceil("1h") if family == "subh" else lead for lead in leads}
                )
                for lead in file_leads:
                    variables = [
                        v
                        for v in data_var_group
                        if v.internal_attrs.source_family == family
                        and v.available_at(lead)
                    ]
                    if not variables:
                        continue
                    coords.extend(
                        NoaaRrfsSourceFileCoord(
                            init_time=init,
                            lead_time=lead,
                            source_family=family,
                            ensemble_member=member,
                            data_vars=variables,
                        )
                        for member in members
                    )
        return coords

    def _check_refs_complete(
        self, coord: NoaaRrfsSourceFileCoord, refs: list[VirtualRef]
    ) -> None:
        expected = set()
        for lead in coord.message_lead_times():
            for var in coord.data_vars:
                if not var.available_at(lead):
                    continue
                levels = (
                    [None]
                    if var.group is ROOT
                    else self.template_ds[var.path].get_index(var.group)
                )
                for level in levels:
                    expected.add((var.path, lead, level))
        actual = {
            (
                r.data_var.path,
                r.out_loc["lead_time"],
                None if r.data_var.group is ROOT else r.out_loc[r.data_var.group],
            )
            for r in refs
        }
        missing = expected - actual
        assert not missing, (
            f"{coord.get_url()}: required GRIB messages absent from index: {sorted(missing, key=str)}"
        )
