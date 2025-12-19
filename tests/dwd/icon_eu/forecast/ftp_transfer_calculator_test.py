import csv
from collections.abc import Sequence
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, cast
from unittest.mock import MagicMock

import aioftp.client
import obstore.store
import pytest

from reformatters.common.ftp_transfer_calculator import (
    FtpPathAndInfo,
    PathAndSize,
)
from reformatters.dwd.icon_eu.forecast.archive_grib_files import (
    DwdFtpTransferCalculator,
)


@pytest.fixture
def dwd_ftp_listing_00z_fixture() -> Sequence[FtpPathAndInfo]:
    # Load the captured FTP listing from the CSV fixture
    converted_listing: list[FtpPathAndInfo] = []
    with open(
        "tests/dwd/icon_eu/forecast/fixtures/ftp_listing_00z.csv", newline=""
    ) as csvfile:
        csv_reader = csv.reader(csvfile)
        _ = next(csv_reader)  # Skip header row

        for row in csv_reader:
            if len(row) == 3:
                path_str, size_str, item_type = row
                ftp_path = PurePosixPath(path_str)
                file_size_bytes = int(size_str)

                # Create UnixListInfo with dummy datetimes,
                # because none of our code uses these datetimes.
                dummy_datetime = datetime(2000, 1, 1, 0, 0, 0)
                converted_info: aioftp.client.UnixListInfo = cast(
                    aioftp.client.UnixListInfo,
                    {
                        "name": ftp_path.name,
                        "size": file_size_bytes,
                        "type": cast(
                            Literal["dir", "file", "link"], item_type
                        ),  # Use actual type
                        "atime": dummy_datetime,
                        "mtime": dummy_datetime,
                        "ctime": dummy_datetime,
                    },
                )
                converted_listing.append((ftp_path, converted_info))
    return converted_listing


@pytest.fixture
def mock_object_store() -> MagicMock:
    return MagicMock(spec=obstore.store.ObjectStore)


@pytest.mark.asyncio
async def test_dwd_ftp_transfer_calculator_filtering(
    mock_object_store: MagicMock,
    dwd_ftp_listing_00z_fixture: Sequence[FtpPathAndInfo],
    mocker: MagicMock,
) -> None:
    """Objective: Verify that the _skip_ftp_item method correctly excludes directories etc.
    Also, confirm that the filename_filter regex accurately filters files based on
    a user-provided pattern."""

    calc_no_filter = DwdFtpTransferCalculator(
        dst_obstore=mock_object_store,
        dst_root_path=PurePosixPath("test_root"),
        ftp_host="opendata.dwd.de",
        filename_filter="",
    )

    mocker.patch.object(
        calc_no_filter,
        "_list_obstore_files_for_single_nwp_init",
        return_value=set(),
    )

    # Create a dictionary for efficient lookups within this test function
    ftp_listing_map = dict(dwd_ftp_listing_00z_fixture)

    filtered_jobs_no_filter = await calc_no_filter.calc_new_files_from_ftp_listing(
        dwd_ftp_listing_00z_fixture
    )

    for job in filtered_jobs_no_filter:
        # Look up original_ftp_info directly from the dictionary for O(1) lookup
        original_ftp_info = ftp_listing_map.get(job.src_ftp_path)

        assert original_ftp_info is not None
        assert original_ftp_info.get("type") == "file", (
            f"Directory found in filtered jobs: {job.src_ftp_path}"
        )
        assert "pressure-level" not in job.src_ftp_path.name
        assert job.src_ftp_path.name.endswith("grib2.bz2")

    # Scenario 2: With filename filter
    # Filter for 'single-level' and forecast step '000'
    filename_regex = "single-level_.*_000_"
    calc_with_filter = DwdFtpTransferCalculator(
        dst_obstore=mock_object_store,
        dst_root_path=PurePosixPath("test_root"),
        ftp_host="opendata.dwd.de",
        filename_filter=filename_regex,
    )
    mocker.patch.object(
        calc_with_filter,
        "_list_obstore_files_for_single_nwp_init",
        return_value=set(),
    )

    filtered_jobs_with_filter = await calc_with_filter.calc_new_files_from_ftp_listing(
        dwd_ftp_listing_00z_fixture
    )

    for job in filtered_jobs_with_filter:
        assert "single-level" in job.src_ftp_path.name
        assert "_000_" in job.src_ftp_path.name
        assert job.src_ftp_path.name.endswith("grib2.bz2")


@pytest.mark.asyncio
async def test_dwd_ftp_path_to_transfer_job_conversion(
    mock_object_store: MagicMock,
) -> None:
    """Objective: Ensure that a given FTP path and its associated UnixListInfo are correctly
    transformed into a TransferJob object, with accurate extraction of nwp_init_datetime
    and correct construction of dst_obstore_path."""

    test_ftp_path = PurePosixPath(
        "/weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025112600_004_ALB_RAD.grib2.bz2"
    )
    # Create a dummy UnixListInfo for the test case with type "file"
    dummy_datetime = datetime(2000, 1, 1, 0, 0, 0)
    test_ftp_info = cast(
        aioftp.client.UnixListInfo,
        {
            "name": test_ftp_path.name,
            "size": 1234567,
            "type": "file",
            "atime": dummy_datetime,
            "mtime": dummy_datetime,
            "ctime": dummy_datetime,
        },
    )

    # Instantiate with a dummy object store, it's not used in this specific method call
    calc = DwdFtpTransferCalculator(
        dst_obstore=mock_object_store,
        dst_root_path=PurePosixPath("dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/"),
        ftp_host="opendata.dwd.de",
    )

    transfer_job = calc._convert_ftp_path_to_transfer_job(test_ftp_path, test_ftp_info)

    assert transfer_job.src_ftp_path == test_ftp_path
    assert transfer_job.src_ftp_file_size_bytes == 1234567
    assert transfer_job.nwp_init_datetime == datetime(2025, 11, 26, 0, 0)
    expected_dst_obstore_path = PurePosixPath(
        "dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/2025-11-26T00Z/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025112600_004_ALB_RAD.grib2.bz2"
    )
    assert transfer_job.dst_obstore_path == expected_dst_obstore_path


@pytest.mark.asyncio
async def test_dwd_ftp_transfer_calculator_identifying_new_files(
    mock_object_store: MagicMock,
    dwd_ftp_listing_00z_fixture: Sequence[FtpPathAndInfo],
    mocker: MagicMock,
) -> None:
    """Objective: Validate that the calc_new_files_from_ftp_listing method accurately
    identifies files that need to be transferred by comparing the FTP listing with a
    simulated object store listing (considering both presence and file size for re-downloads)."""

    calc = DwdFtpTransferCalculator(
        dst_obstore=mock_object_store,
        dst_root_path=PurePosixPath("dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/"),
        ftp_host="opendata.dwd.de",
    )

    # Create a dictionary for efficient lookups within this test function
    ftp_listing_map = dict(dwd_ftp_listing_00z_fixture)

    # Determine the NWP init datetime from the fixture for consistent mocking.
    # We'll take the first valid file from the fixture to extract its NWP init datetime.
    first_valid_ftp_path = None
    first_valid_ftp_info = None

    for ftp_path, ftp_info in dwd_ftp_listing_00z_fixture:
        # _skip_ftp_item also checks for file type, so ensure the dummy UnixListInfo has type "file"
        if (
            not calc._skip_ftp_item(ftp_path, ftp_info)
            and ftp_info.get("type") == "file"
        ):
            first_valid_ftp_path = ftp_path
            first_valid_ftp_info = ftp_info
            break

    assert first_valid_ftp_path is not None, (
        "Could not find a valid FTP file in the fixture to determine NWP init datetime."
    )
    assert first_valid_ftp_info is not None, (
        "Could not find valid FTP info in the fixture to determine NWP init datetime."
    )

    fixture_nwp_init_datetime = calc._extract_init_datetime_from_ftp_path(
        first_valid_ftp_path
    )

    # Define paths using the determined fixture_nwp_init_datetime
    alb_rad_ftp_path_004 = PurePosixPath(
        f"/weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_{fixture_nwp_init_datetime.strftime('%Y%m%d%H')}_004_ALB_RAD.grib2.bz2"
    )
    clch_ftp_path_004 = PurePosixPath(
        f"/weather/nwp/icon-eu/grib/00/clch/icon-eu_europe_regular-lat-lon_single-level_{fixture_nwp_init_datetime.strftime('%Y%m%d%H')}_004_CLCH.grib2.bz2"
    )
    t_2m_ftp_path_004 = PurePosixPath(
        f"/weather/nwp/icon-eu/grib/00/t_2m/icon-eu_europe_regular-lat-lon_single-level_{fixture_nwp_init_datetime.strftime('%Y%m%d%H')}_004_T_2M.grib2.bz2"
    )

    # We need to get the actual sizes from the fixture for these specific files
    alb_rad_size = 0
    clch_size = 0
    # Use ftp_listing_map for efficient lookup
    alb_rad_info = ftp_listing_map.get(alb_rad_ftp_path_004)
    if alb_rad_info:
        alb_rad_size = int(alb_rad_info["size"])
    clch_info = ftp_listing_map.get(clch_ftp_path_004)
    if clch_info:
        clch_size = int(clch_info["size"])

    # Construct expected_mock_obstore_files with `PathAndSize` objects
    expected_mock_obstore_files = {
        PathAndSize(
            path=str(
                calc._calc_obstore_path(alb_rad_ftp_path_004, fixture_nwp_init_datetime)
            ),
            file_size_bytes=alb_rad_size,  # Correct size from fixture, should NOT be downloaded
        ),
        PathAndSize(
            path=str(
                calc._calc_obstore_path(clch_ftp_path_004, fixture_nwp_init_datetime)
            ),
            file_size_bytes=clch_size
            + 1,  # Incorrect size, should be downloaded (re-download)
        ),
    }

    # Custom mock function to handle dynamic date from calc_new_files_from_ftp_listing
    def mock_list_obstore_files_for_single_nwp_init(
        nwp_init: datetime,
    ) -> set[PathAndSize]:
        if nwp_init == fixture_nwp_init_datetime:
            return expected_mock_obstore_files
        return set()

    mocker.patch.object(
        calc,
        "_list_obstore_files_for_single_nwp_init",
        side_effect=mock_list_obstore_files_for_single_nwp_init,
    )

    files_to_download = await calc.calc_new_files_from_ftp_listing(
        dwd_ftp_listing_00z_fixture
    )

    # Assertions
    # 1. The file with an incorrect size (CLCH) should be in files_to_download.
    assert any(job.src_ftp_path == clch_ftp_path_004 for job in files_to_download)

    # 2. A file that is entirely new (not in mock_obstore_files at all) should be in files_to_download.
    assert any(job.src_ftp_path == t_2m_ftp_path_004 for job in files_to_download)

    # 3. The file with correct size (ALB_RAD) should NOT be in files_to_download.
    assert not any(
        job.src_ftp_path == alb_rad_ftp_path_004 for job in files_to_download
    )
