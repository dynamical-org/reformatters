import re
from collections.abc import Mapping, Sequence

from reformatters.common.iterating import item
from reformatters.common.region_job import CoordinateValueOrRange, SourceFileCoord
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    Dim,
    Timedelta,
    Timestamp,
)
from reformatters.noaa.models import NoaaDataVar


class NoaaGfsSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process."""

    init_time: Timestamp
    lead_time: Timedelta
    data_vars: Sequence[NoaaDataVar]

    def get_url(self) -> str:
        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")
        lead_hours = whole_hours(self.lead_time)
        base_path = f"gfs.{init_date_str}/{init_hour_str}/atmos/gfs.t{init_hour_str}z.pgrb2.0p25.f{lead_hours:03d}"
        return f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/{base_path}"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


def parse_grib_index(
    index_contents: str,
    coord: NoaaGfsSourceFileCoord,
) -> tuple[list[int], list[int]]:
    """Parse byte ranges from GRIB index file for the given coordinate."""
    starts: list[int] = []
    ends: list[int] = []

    lead_hours = whole_hours(coord.lead_time)

    # All accumulation reset frequencies are the same for GFS, `item(set(...))` will ensure that
    reset_freq = item(
        {v.internal_attrs.window_reset_frequency for v in coord.data_vars}
    )
    if reset_freq is not None:
        reset_hours = whole_hours(reset_freq)
    else:
        reset_hours = None

    for var in coord.data_vars:
        # Handle how the lead time or accumulation time is included in the element name in the grib index file
        if lead_hours == 0:
            hours_str_prefix = "anl"
        elif var.attrs.step_type == "instant":
            hours_str_prefix = f"{lead_hours} hour fcst"
        elif reset_hours is not None:
            diff = lead_hours % reset_hours
            reset_hour = lead_hours - diff if diff != 0 else lead_hours - reset_hours
            step_type = (
                "acc"
                if var.internal_attrs.deaccumulate_to_rate
                else var.attrs.step_type
            )
            hours_str_prefix = f"{reset_hour}-{lead_hours} hour {step_type} fcst"
        else:
            raise ValueError(f"Unhandled grib lead/accumulation hours: {var.name}")

        var_match_str = re.escape(
            f"{var.internal_attrs.grib_element}:{var.internal_attrs.grib_index_level}:{hours_str_prefix}"
        )
        matches = re.findall(
            rf"\d+:(\d+):.+:{var_match_str}:(?:\n\d+:(\d+))?",
            index_contents,
        )
        assert len(matches) == 1, (
            f"Expected exactly 1 match, found {matches}, {var.name}"
        )

        m0, m1 = matches[0]
        start = int(m0)
        # If this is the last message in the file, m1 will be empty.
        # We fall back to adding a large value (+10 GiB) to get an end point since obstore
        # doesn't support omitting the end byte but will return bytes up to the end of the file.
        end = int(m1) if m1 else start + 10 * (2**30)
        starts.append(start)
        ends.append(end)

    return starts, ends
