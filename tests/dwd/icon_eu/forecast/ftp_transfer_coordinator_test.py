from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, cast

import obstore
import pytest
from pytest_mock import MockerFixture

from reformatters.common.ftp_to_obstore_types import PathAndSize, TransferJob
from reformatters.common.ftp_transfer_coordinator import (
    FtpTransferCoordinator,
    set_difference,
)
from reformatters.common.obstore_manager import ObstoreManager
from reformatters.dwd.icon_eu.forecast.ftp_manager import DwdFtpManager

FTP_PATH = PurePosixPath(
    "/weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025112600_004_ALB_RAD.grib2.bz2"
)
FTP_INFO = cast(Any, {"type": "file", "size": 123})


@pytest.fixture
def coordinator() -> FtpTransferCoordinator:
    ext = DwdFtpManager(ftp_host="host")
    mgr = ObstoreManager(
        dst_obstore=obstore.store.MemoryStore(), dst_root_path=PurePosixPath("root")
    )
    return FtpTransferCoordinator(ftp_manager=ext, obstore_manager=mgr)


def test_skip_ftp_item() -> None:
    ext = DwdFtpManager(ftp_host="host")
    assert ext.skip_ftp_item(
        ftp_path=PurePosixPath("dir"), ftp_info=cast(Any, {"type": "dir"})
    )
    assert ext.skip_ftp_item(
        ftp_path=PurePosixPath("pressure-level.grib"),
        ftp_info=cast(Any, {"type": "file"}),
    )
    assert ext.skip_ftp_item(
        ftp_path=PurePosixPath("file.txt"), ftp_info=cast(Any, {"type": "file"})
    )
    assert not ext.skip_ftp_item(
        ftp_path=FTP_PATH, ftp_info=cast(Any, {"type": "file"})
    )


def test_extract_init_datetime() -> None:
    assert DwdFtpManager.extract_init_datetime_from_ftp_path(
        ftp_path=FTP_PATH
    ) == datetime(2025, 11, 26, 0)


def test_extract_variable_name() -> None:
    assert (
        DwdFtpManager.extract_nwp_variable_name_from_ftp_path(ftp_path=FTP_PATH)
        == "alb_rad"
    )


def test_calc_obstore_path() -> None:
    mgr = ObstoreManager(
        dst_obstore=obstore.store.MemoryStore(), dst_root_path=PurePosixPath("root")
    )
    expected = PurePosixPath(
        "root/2025-11-26T00Z/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025112600_004_ALB_RAD.grib2.bz2"
    )
    assert (
        mgr.calc_obstore_path(
            ftp_path=FTP_PATH,
            nwp_init_datetime=datetime(2025, 11, 26, 0),
            nwp_variable_name="alb_rad",
        )
        == expected
    )


def test_convert_ftp_path_to_transfer_job(coordinator: FtpTransferCoordinator) -> None:
    job = coordinator.convert_ftp_path_to_transfer_job(
        ftp_path=FTP_PATH, ftp_info=FTP_INFO
    )
    assert job.src_ftp_path == FTP_PATH
    assert job.src_ftp_file_size_bytes == 123
    assert job.nwp_init_datetime == datetime(2025, 11, 26, 0)
    assert job.dst_obstore_path.parts[-2:] == ("alb_rad", FTP_PATH.name)


def test_set_difference() -> None:
    job = TransferJob(
        src_ftp_path=FTP_PATH,
        src_ftp_file_size_bytes=100,
        dst_obstore_path=PurePosixPath("dst"),
        nwp_init_datetime=datetime(2025, 1, 1),
    )
    assert set_difference(
        ftp_transfer_jobs=[job], objects_already_downloaded=set()
    ) == [job]
    assert (
        set_difference(
            ftp_transfer_jobs=[job],
            objects_already_downloaded={PathAndSize(path="dst", file_size_bytes=100)},
        )
        == []
    )
    assert set_difference(
        ftp_transfer_jobs=[job],
        objects_already_downloaded={PathAndSize(path="dst", file_size_bytes=99)},
    ) == [job]


def test_filter_filenames_by_regex() -> None:
    ext = DwdFtpManager(ftp_host="host", filename_filter=".*_000_.*")
    listing = cast(
        Any, [(PurePosixPath("a_000_b"), {}), (PurePosixPath("a_001_b"), {})]
    )
    assert len(ext.filter_filenames_by_regex(ftp_listing=listing)) == 1


@pytest.mark.asyncio
async def test_calc_new_files_for_single_nwp_init_hour(
    coordinator: FtpTransferCoordinator, mocker: MockerFixture
) -> None:
    mocker.patch.object(
        coordinator.ftp_manager,
        "list_and_filter_for_init_hour",
        return_value=[(FTP_PATH, FTP_INFO)],
    )
    mocker.patch.object(
        coordinator.obstore_manager,
        "list_files_starting_at_nwp_init",
        return_value=set(),
    )
    jobs = await coordinator.calc_new_files_for_single_nwp_init_hour(init_hour=0)
    assert len(jobs) == 1
    assert jobs[0].src_ftp_path == FTP_PATH
