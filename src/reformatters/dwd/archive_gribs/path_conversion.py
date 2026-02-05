import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Final


def convert_src_path_to_dst_path(
    src_path_starting_with_nwp_var: PurePosixPath,
) -> PurePosixPath:
    """
    Args:
        src_path_starting_with_nwp_var: Should have exactly 2 parts: NWP_var/filename.grib2.bz2.
    Returns:
        dst_path: In the form NWP_init_datetime/NWP_var/filename.grib2.bz2.
    """
    grib_filename = src_path_starting_with_nwp_var.name
    nwp_init_datetime = extract_nwp_init_datetime_from_grib_filename(grib_filename)
    nwp_init_datetime_str = format_datetime_for_dst_path(nwp_init_datetime)
    nwp_variable_name = extract_nwp_var_from_src_path(src_path_starting_with_nwp_var)
    dst_path = PurePosixPath(nwp_init_datetime_str) / nwp_variable_name / grib_filename
    return dst_path


def extract_nwp_var_from_src_path(src_path: PurePosixPath) -> str:
    """Check number of parts in the path & extract NWP variable name.

    Args:
        src_path: A path on the DWD HTTPs server in the form:
            alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026013003_000_ALB_RAD.grib2.bz2
    """
    n_expected_parts: Final[int] = 2
    if len(src_path.parts) != n_expected_parts:
        raise RuntimeError(
            f"Expected {n_expected_parts} parts in the DWD path, found "
            f"{len(src_path.parts)} parts. Path: {src_path}"
        )
    return src_path.parts[0]


class DateExtractionError(Exception):
    pass


DWD_NWP_INIT_DATE_REGEX: Final[re.Pattern[str]] = re.compile(r"_(\d{10})(?=_)")


def extract_nwp_init_datetime_from_grib_filename(
    grib_filename: str,
) -> datetime:
    nwp_init_date_matches = DWD_NWP_INIT_DATE_REGEX.findall(grib_filename)
    if len(nwp_init_date_matches) == 0:
        raise DateExtractionError(f"No date found in file: {grib_filename}")
    elif len(nwp_init_date_matches) > 1:
        raise DateExtractionError(
            "Expected exactly one 10-digit number in the filename (the NWP init date"
            f" represented as YYYYMMDDHH), but instead found {len(nwp_init_date_matches)}"
            f" 10-digit numbers in path {grib_filename}"
        )
    nwp_init_date_str = nwp_init_date_matches[0]
    return datetime.strptime(nwp_init_date_str, "%Y%m%d%H")


def format_datetime_for_dst_path(nwp_init_datetime: datetime) -> str:
    return nwp_init_datetime.strftime("%Y-%m-%dT%H")
