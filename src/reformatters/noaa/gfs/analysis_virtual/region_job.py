from collections import Counter
from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd
import xarray as xr

from reformatters.common.region_job import CoordinateValue
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim, Timedelta, Timestamp
from reformatters.noaa.gfs.analysis.region_job import NOAA_GFS_INIT_FREQUENCY
from reformatters.noaa.gfs.virtual_region_job import (
    GFS_FILE_TYPES,
    NoaaGfsFileType,
    NoaaGfsVirtualRegionJob,
    NoaaGfsVirtualSourceFileCoord,
)
from reformatters.noaa.models import NoaaDataVar

# The variables a source file's ingestion is probed by, in preference order. Each is
# carried only by its own product, so the probe can never land on a chunk the file does
# not fill, and each is published in every era of the archive. A coord holding only the
# variables without hour 0 values falls through to the second entry.
_REPRESENTATIVE_VARS: dict[NoaaGfsFileType, tuple[str, ...]] = {
    "pgrb2": ("temperature_2m", "total_precipitation_surface"),
    "pgrb2b": ("temperature_305m_amsl", "uv_b_downward_solar_flux_surface"),
}


class NoaaGfsAnalysisVirtualSourceFileCoord(NoaaGfsVirtualSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.valid_time()}

    def valid_time(self) -> Timestamp:
        return self.init_time + self.lead_time


class NoaaGfsAnalysisVirtualRegionJob(
    NoaaGfsVirtualRegionJob[NoaaGfsAnalysisVirtualSourceFileCoord]
):
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("12h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaDataVar],
    ) -> Sequence[NoaaGfsAnalysisVirtualSourceFileCoord]:
        """Take each variable at the shortest lead it is published at, matching the
        materialized noaa-gfs-analysis: a variable with hour 0 values comes from the
        cycle its time falls in (leads 0-5) and any other from the cycle before the
        preceding hour (leads 1-6), so a windowed variable's window opens at the most
        recent synoptic hour strictly before its time.
        """
        times = pd.to_datetime(processing_region_ds["time"].values)
        init_frequency = f"{whole_hours(NOAA_GFS_INIT_FREQUENCY)}h"
        lead_offsets = {
            pd.Timedelta(0): [v for v in data_var_group if v.has_hour_0_values()],
            pd.Timedelta("1h"): [
                v for v in data_var_group if not v.has_hour_0_values()
            ],
        }

        coords = []
        for time in times:
            # Away from a synoptic hour both offsets land on one cycle, so the two
            # variable sets share a file and are ingested by one coord.
            vars_by_init: dict[Timestamp, list[NoaaDataVar]] = {}
            for offset, offset_vars in lead_offsets.items():
                if not offset_vars:
                    continue
                init_time = (time - offset).floor(init_frequency)
                vars_by_init.setdefault(init_time, []).extend(offset_vars)
            for init_time, init_vars in vars_by_init.items():
                coords += [
                    NoaaGfsAnalysisVirtualSourceFileCoord(
                        init_time=init_time,
                        lead_time=time - init_time,
                        file_type=file_type,
                        data_vars=init_vars,
                    )
                    for file_type in GFS_FILE_TYPES
                ]
        return coords

    def representative_var(
        self, coord: NoaaGfsAnalysisVirtualSourceFileCoord
    ) -> NoaaDataVar:
        by_name = {var.name: var for var in coord.data_vars}
        # A variable-filtered job may carry none of them; the write loop's
        # probe-coverage assert catches a pick the file doesn't hold.
        return next(
            (
                by_name[name]
                for name in _REPRESENTATIVE_VARS[coord.file_type]
                if name in by_name
            ),
            coord.data_vars[0],
        )

    def discover_available(
        self, pending: list[NoaaGfsAnalysisVirtualSourceFileCoord]
    ) -> list[tuple[NoaaGfsAnalysisVirtualSourceFileCoord, int]]:
        """Extend `time` only as far as the newest time holding all of its files.

        At a synoptic hour the windowed variables come from the previous cycle and
        publish about six hours before that hour's own instantaneous files, so ungated
        discovery would extend `time` to an hour carrying only some of its variables.
        Files past the limit stay pending and are offered again next tick, so an hour is
        first visible complete.

        A time the store already covers is never withheld: nothing extends there, and a
        file the archive never published must not block the files beside it forever. A
        backfill's branch is pre-sized, so nothing it writes extends anything either.
        """
        available = super().discover_available(pending)
        pending_per_time = Counter(coord.valid_time() for coord in pending)
        ready_per_time = Counter(coord.valid_time() for coord, _ in available)
        authorized = [
            time
            for time, ready in ready_per_time.items()
            if ready == pending_per_time[time]
        ]
        if self.ingested_through is not None:
            authorized.append(self.ingested_through)
        if not authorized:
            return []
        limit = max(authorized)
        return [
            (coord, size) for coord, size in available if coord.valid_time() <= limit
        ]
