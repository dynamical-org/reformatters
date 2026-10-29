from collections.abc import Mapping, Sequence
from typing import ClassVar

import icechunk
import pandas as pd
import xarray as xr

from reformatters.common.region_job import CoordinateValue
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim, Timedelta
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

    def get_url(self) -> str:
        assert self.source_family != "subh"
        hour = whole_hours(self.lead_time)
        cycle = self.init_time.strftime("%Y%m%d/%H")
        prefix = self.init_time.strftime("rrfs.t%Hz")
        key = f"rrfs.{cycle}/{prefix}.{self.source_family}.3km.f{hour:03}.conus.grib2"
        return SOURCE_PREFIX + key

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }


class NoaaRrfsRegionJob(NoaaVirtualRegionJob[NoaaRrfsDataVar, NoaaRrfsSourceFileCoord]):
    source_location_prefix: ClassVar[str] = SOURCE_PREFIX
    source_bucket_region: ClassVar[str] = SOURCE_REGION
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("18h")


class NoaaRrfsHourlyRegionJob(NoaaRrfsRegionJob):
    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaRrfsDataVar],
    ) -> Sequence[NoaaRrfsSourceFileCoord]:
        coords = []
        for init in pd.to_datetime(processing_region_ds["init_time"].values):
            for family in sorted(
                {v.internal_attrs.source_family for v in data_var_group}
            ):
                assert family != "subh"
                for lead in pd.to_timedelta(processing_region_ds["lead_time"].values):
                    variables = [
                        v
                        for v in data_var_group
                        if v.internal_attrs.source_family == family
                        and v.available_at(lead)
                    ]
                    if variables:
                        coords.append(
                            NoaaRrfsSourceFileCoord(
                                init_time=init,
                                lead_time=lead,
                                source_family=family,
                                data_vars=variables,
                            )
                        )
        return coords
