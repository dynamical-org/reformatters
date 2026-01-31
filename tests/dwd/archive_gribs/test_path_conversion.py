from datetime import datetime
from pathlib import PurePosixPath

import pytest

from reformatters.dwd.archive_gribs.path_conversion import (
    DateExtractionError,
    convert_src_path_to_dst_path,
    extract_nwp_init_datetime_from_grib_filename,
    extract_nwp_var_from_src_path,
    format_datetime_for_dst_path,
)


def test_extract_nwp_init_datetime_from_grib_filename() -> None:
    filename = (
        "icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2"
    )
    expected = datetime(2026, 1, 14, 0)
    assert extract_nwp_init_datetime_from_grib_filename(filename) == expected


def test_extract_nwp_init_datetime_from_grib_filename_invalid() -> None:
    with pytest.raises(DateExtractionError, match="No date found"):
        extract_nwp_init_datetime_from_grib_filename("no_date_here.grib2")

    with pytest.raises(DateExtractionError, match="Expected exactly one"):
        extract_nwp_init_datetime_from_grib_filename(
            "two_dates_2026011400_2026011401_.grib2"
        )


def test_extract_nwp_var_from_src_path() -> None:
    src_path = PurePosixPath("alb_rad/icon-eu_2026011400_000_ALB_RAD.grib2.bz2")
    assert extract_nwp_var_from_src_path(src_path) == "alb_rad"


def test_extract_nwp_var_from_src_path_invalid() -> None:
    with pytest.raises(RuntimeError, match="Expected 2 parts"):
        extract_nwp_var_from_src_path(PurePosixPath("too/many/parts/file.grib2"))


def test_format_datetime_for_dst_path() -> None:
    dt = datetime(2026, 1, 14, 3)
    assert format_datetime_for_dst_path(dt) == "2026-01-14T03"


def test_convert_src_path_to_dst_path() -> None:
    src_path = PurePosixPath(
        "alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2"
    )
    expected = PurePosixPath(
        "2026-01-14T00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2"
    )
    assert convert_src_path_to_dst_path(src_path) == expected
