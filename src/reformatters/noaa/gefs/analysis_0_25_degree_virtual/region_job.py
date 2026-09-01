from collections import Counter
from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd
import xarray as xr

from reformatters.common.region_job import CoordinateValue
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim, Timedelta, Timestamp
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_CURRENT_ARCHIVE_START,
    GEFS_INIT_TIME_FREQUENCY,
    GEFSSourceFileType,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.gefs.virtual_region_job import (
    NoaaGefsVirtualRegionJob,
    NoaaGefsVirtualSourceFileCoord,
)


class NoaaGefsAnalysis025DegreeVirtualSourceFileCoord(NoaaGefsVirtualSourceFileCoord):
    ensemble_member: int = 0  # Control member for analysis
    source_file_type: GEFSSourceFileType = "s"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.valid_time()}

    def valid_time(self) -> Timestamp:
        return self.init_time + self.lead_time


class NoaaGefsAnalysis025DegreeVirtualRegionJob(
    NoaaGefsVirtualRegionJob[NoaaGefsAnalysis025DegreeVirtualSourceFileCoord]
):
    # Three update cron fires' span, so two consecutive missed runs still self-heal.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("18h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaGefsVirtualDataVar],
    ) -> Sequence[NoaaGefsAnalysis025DegreeVirtualSourceFileCoord]:
        """Use the shortest lead holding each variable, mirroring noaa-gefs-analysis.

        A time resolves to the cycle it falls in at lead 0 or 3; a variable with no
        hour-0 values shifts back one cycle to lead 6 rather than reading the degenerate
        zero-length window the source publishes at lead 0. Variables landing on the same
        file share one coord, so each file is listed and indexed once.

        A variable contributes no coord before its `available_from`, so the store holds
        no ref there and readers get NaN.
        """
        cycle_frequency = f"{whole_hours(GEFS_INIT_TIME_FREQUENCY)}h"
        grouped: dict[
            tuple[Timestamp, Timedelta, GEFSSourceFileType],
            list[NoaaGefsVirtualDataVar],
        ] = {}
        for time in pd.to_datetime(processing_region_ds["time"].values):
            cycle = time.floor(cycle_frequency)
            for var in data_var_group:
                init_time = cycle
                if cycle == time and not var.has_hour_0_values():
                    init_time -= GEFS_INIT_TIME_FREQUENCY
                if init_time < GEFS_CURRENT_ARCHIVE_START:
                    continue
                # Whether a variable exists is a property of the cycle that produced the
                # file, not of the valid time it lands on.
                available_from = var.internal_attrs.available_from
                if available_from is not None and init_time < available_from:
                    continue
                key = (
                    init_time,
                    time - init_time,
                    var.internal_attrs.source_file_type,
                )
                grouped.setdefault(key, []).append(var)

        return [
            NoaaGefsAnalysis025DegreeVirtualSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                source_file_type=source_file_type,
                data_vars=data_vars,
            )
            for (init_time, lead_time, source_file_type), data_vars in grouped.items()
        ]

    def discover_available(
        self, pending: list[NoaaGefsAnalysis025DegreeVirtualSourceFileCoord]
    ) -> list[tuple[NoaaGefsAnalysis025DegreeVirtualSourceFileCoord, int]]:
        """Extend `time` only as far as the newest time holding its own cycle's file.

        A time's windowed variables come from the previous cycle and publish hours before
        its own cycle's file, so ungated discovery would extend `time` to a step carrying
        only the windowed half. Files past the limit stay pending and are offered again
        next tick, so a time is first visible complete.

        A time the store already covers is never withheld: nothing extends there, and a
        cycle the archive never published must not block the file beside it forever. A
        backfill's branch is pre-sized, so nothing it writes extends anything either.
        """
        available = super().discover_available(pending)
        pending_own_cycle = Counter(
            coord.valid_time() for coord in pending if _from_own_cycle(coord)
        )
        ready_own_cycle = Counter(
            coord.valid_time() for coord, _ in available if _from_own_cycle(coord)
        )
        authorized = [
            time
            for time, ready in ready_own_cycle.items()
            if ready == pending_own_cycle[time]
        ]
        if self.ingested_through is not None:
            authorized.append(self.ingested_through)
        if not authorized:
            return []
        limit = max(authorized)
        return [
            (coord, size) for coord, size in available if coord.valid_time() <= limit
        ]


def _from_own_cycle(coord: NoaaGefsAnalysis025DegreeVirtualSourceFileCoord) -> bool:
    """Whether this file belongs to the cycle its valid time falls in, rather than the
    previous one. It is the last of a time's files to publish, so it fixes arrival."""
    cycle_frequency = f"{whole_hours(GEFS_INIT_TIME_FREQUENCY)}h"
    return coord.init_time == coord.valid_time().floor(cycle_frequency)
