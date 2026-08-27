from collections import Counter
from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd
import xarray as xr

from reformatters.common.iterating import group_by, item
from reformatters.common.region_job import CoordinateValue
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim, Timedelta, Timestamp
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrVirtualRegionJob,
    NoaaHrrrVirtualSourceFileCoord,
)


class NoaaHrrrAnalysisVirtualSourceFileCoord(NoaaHrrrVirtualSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.valid_time()}

    def valid_time(self) -> Timestamp:
        return self.init_time + self.lead_time


class NoaaHrrrAnalysisVirtualRegionJob(
    NoaaHrrrVirtualRegionJob[NoaaHrrrAnalysisVirtualSourceFileCoord]
):
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("12h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaHrrrDataVar],
    ) -> Sequence[NoaaHrrrAnalysisVirtualSourceFileCoord]:
        """Use the shortest present lead for each variable: f00 if available, f01 otherwise.

        A variable contributes no coord before its `analysis_usable_from`, so the store holds no
        ref there and readers get NaN.
        """
        times = pd.to_datetime(processing_region_ds["time"].values)
        var_groups = group_by(
            data_var_group,
            lambda v: (v.internal_attrs.hrrr_file_type, v.has_hour_0_values()),
        )
        coords = []
        for vars_in_file in var_groups:
            file_type = item({v.internal_attrs.hrrr_file_type for v in vars_in_file})
            has_hour_0_values = item({v.has_hour_0_values() for v in vars_in_file})
            lead_time = pd.Timedelta("0h") if has_hour_0_values else pd.Timedelta("1h")
            for time in times:
                usable_vars = [
                    var
                    for var in vars_in_file
                    if var.internal_attrs.analysis_usable_from is None
                    or time >= var.internal_attrs.analysis_usable_from
                ]
                if not usable_vars:
                    continue
                coords.append(
                    NoaaHrrrAnalysisVirtualSourceFileCoord(
                        init_time=time - lead_time,
                        lead_time=lead_time,
                        domain="conus",
                        file_type=file_type,
                        data_vars=usable_vars,
                    )
                )
        return coords

    def representative_var(
        self, coord: NoaaHrrrAnalysisVirtualSourceFileCoord
    ) -> NoaaHrrrDataVar:
        """Specific representative vars known to be present continuously throughout the archive."""
        match (coord.file_type, whole_hours(coord.lead_time)):
            case ("sfc", 0):
                paths = ("composite_reflectivity", "temperature_2m")
            case ("sfc", 1):
                paths = ("categorical_rain_surface", "total_precipitation_surface")
            case ("prs", 0):
                paths = ("pressure_level/temperature", "pressure_level/wind_u")
            case ("nat", 0):
                paths = ("model_level/temperature", "model_level/wind_u")
            case unexpected:
                raise AssertionError(f"No representative variable for {unexpected}")
        by_path = {var.path: var for var in coord.data_vars}
        # A variable-filtered job may carry none of them; the write loop's
        # probe-coverage assert catches a pick the file doesn't hold.
        return next(
            (by_path[path] for path in paths if path in by_path),
            coord.data_vars[0],
        )

    def discover_available(
        self, pending: list[NoaaHrrrAnalysisVirtualSourceFileCoord]
    ) -> list[tuple[NoaaHrrrAnalysisVirtualSourceFileCoord, int]]:
        """Extend `time` only as far as the newest time holding all of its f00 files.

        A time's f01 files come from the previous init and publish an hour before its
        own f00 files, so ungated discovery would extend `time` to an hour carrying only
        the variables that have no hour-0 value. Files past the limit stay pending and
        are offered again next tick, so an hour is first visible complete.

        A time the store already covers is never withheld: nothing extends there, and an
        f00 the archive never published must not block the f01 beside it forever. A
        backfill's branch is pre-sized, so nothing it writes extends anything either.
        """
        available = super().discover_available(pending)
        hour_0 = pd.Timedelta("0h")
        pending_hour_0 = Counter(
            coord.valid_time() for coord in pending if coord.lead_time == hour_0
        )
        ready_hour_0 = Counter(
            coord.valid_time() for coord, _ in available if coord.lead_time == hour_0
        )
        authorized = [
            time
            for time, ready in ready_hour_0.items()
            if ready == pending_hour_0[time]
        ]
        if self.ingested_through is not None:
            authorized.append(self.ingested_through)
        if not authorized:
            return []
        limit = max(authorized)
        return [
            (coord, size) for coord, size in available if coord.valid_time() <= limit
        ]
