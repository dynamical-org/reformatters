import re
from collections.abc import Sequence
from os import PathLike

import pandas as pd

from reformatters.common.config_models import DataVar
from reformatters.common.time_utils import whole_hours
from reformatters.noaa.models import NoaaInternalAttrs


def grib_message_byte_ranges_from_index(
    index_path: PathLike[str],
    data_vars: Sequence[DataVar[NoaaInternalAttrs]],
    init_time: pd.Timestamp,
    lead_time: pd.Timedelta,
) -> tuple[list[int], list[int]]:
    """
    Parse byte ranges from GRIB index file for the given data variables.
    returns a tuple of two lists: (start_byte_offsets, end_byte_offsets)
    """
    with open(index_path) as f:
        index_contents = f.read()

    starts: list[int] = []
    ends: list[int] = []

    lead_hours = whole_hours(lead_time)
    init_time_str = init_time.strftime("d=%Y%m%d%H")

    for var in data_vars:
        # Handle how the lead time or accumulation time is included in the element name in the grib index file
        if (reset_freq := var.internal_attrs.window_reset_frequency) is not None:
            reset_hours = whole_hours(reset_freq)
            diff = lead_hours % reset_hours
            reset_hour = lead_hours - diff if diff != 0 else lead_hours - reset_hours

            if var.internal_attrs.deaccumulate_to_rate:
                step_type = "acc"
            elif var.attrs.step_type == "avg":
                step_type = "ave"  # yep
            else:
                step_type = var.attrs.step_type

            if lead_hours == 0:
                lead_time_str = f"0-0 day {step_type} fcst"
            else:
                lead_time_str = f"{reset_hour}-{lead_hours} hour {step_type} fcst"

        elif lead_hours == 0:
            lead_time_str = "anl"
        elif var.attrs.step_type == "instant":
            lead_time_str = f"{lead_hours} hour fcst"
        else:
            raise ValueError(f"Unhandled grib lead/accumulation hours: {var.name}")

        grib_elements = (
            var.internal_attrs.grib_element,
            *var.internal_attrs.grib_element_alternatives,
        )
        element_pattern = "|".join(re.escape(e) for e in grib_elements)
        var_match_str = (
            re.escape(f"{init_time_str}:")
            + f"(?:{element_pattern})"
            + re.escape(f":{var.internal_attrs.grib_index_level}:{lead_time_str}")
        )
        # The format of a NOAA grib index line is
        # variable number:byte offset start:init timestamp:element:level:lead time and accum:ensemble info
        # This regex captures the byte start from the line that matches, and has an optional
        # 2nd capture which wraps to capture the byte start of the next line to get our end byte.
        # 2nd capture is optional because we need to support matching the final variable in the index.
        matches = re.findall(
            rf"\d+:(\d+):"  # row number : start byte offset (captured)
            rf"{var_match_str}:"  # uniquely identify the variable we want
            rf"(?:.*\n\d+:(\d+))?",  # end of line and wrap to capture next line's byte offset to get end byte (optional capture to handle last line of index)
            index_contents,
        )
        assert len(matches) == 1, (
            f"Expected exactly one match for {var.name}, found {matches}"
        )

        start_match, end_match = matches[0]
        start = int(start_match)
        # If this is the last message in the file, end_match will be empty.
        # We fall back to adding a large value (+10 GiB) to get an end point since obstore
        # doesn't support omitting the end byte but will return bytes up to the end of the file.
        end = int(end_match) if end_match else start + 10 * (2**30)
        starts.append(start)
        ends.append(end)

    return starts, ends
