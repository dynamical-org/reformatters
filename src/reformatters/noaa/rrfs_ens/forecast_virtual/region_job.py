from collections.abc import Mapping, Sequence
from typing import Annotated

import pandas as pd
import xarray as xr
from pydantic import Field

from reformatters.common.region_job import CoordinateValue
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim
from reformatters.noaa.rrfs.models import NoaaRrfsDataVar
from reformatters.noaa.rrfs.region_job import (
    SOURCE_PREFIX,
    NoaaRrfsRegionJob,
    NoaaRrfsSourceFileCoord,
)


class NoaaRrfsEnsSourceFileCoord(NoaaRrfsSourceFileCoord):
    ensemble_member: Annotated[int, Field(ge=0, le=5)]

    def get_url(self) -> str:
        if self.ensemble_member == 0:
            return super().get_url()
        assert self.source_family != "subh"
        hour = whole_hours(self.lead_time)
        cycle = self.init_time.strftime("%Y%m%d/%H")
        prefix = self.init_time.strftime("rrfs.t%Hz")
        member = self.ensemble_member
        return f"{SOURCE_PREFIX}rrfsens.{cycle}/m{member:03}/{prefix}.m{member:03}.{self.source_family}nomads.3km.f{hour:03}.conus.grib2"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {**super().out_loc(), "ensemble_member": self.ensemble_member}

    def index_selectors(self, selectors: tuple[str, ...]) -> tuple[str, ...]:
        if self.ensemble_member == 0:
            return selectors
        ensemble = f"ENS=+{self.ensemble_member}"
        assert ensemble in selectors, (self.get_url(), selectors)
        return tuple(s for s in selectors if s != ensemble)


class NoaaRrfsEnsForecastVirtualRegionJob(NoaaRrfsRegionJob):
    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaRrfsDataVar],
    ) -> Sequence[NoaaRrfsEnsSourceFileCoord]:
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
                        coords.extend(
                            NoaaRrfsEnsSourceFileCoord(
                                init_time=init,
                                lead_time=lead,
                                source_family=family,
                                ensemble_member=member,
                                data_vars=variables,
                            )
                            for member in processing_region_ds[
                                "ensemble_member"
                            ].values.tolist()
                        )
        return coords
