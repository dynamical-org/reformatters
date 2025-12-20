from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from aioftp.client import UnixListInfo

from reformatters.common.ftp_to_obstore_types import (
    PathAndSize,
    TransferJob,
)
from reformatters.common.logging import get_logger

if TYPE_CHECKING:
    from reformatters.common.ftp_manager import FtpManager
    from reformatters.common.obstore_manager import ObstoreManager

log = get_logger(__name__)


class FtpTransferCoordinator:
    """Coordinate FTP and object storage interactions to calculate transfer
    jobs.

    This class uses injected FtpManager and
    ObstoreManager instances to perform its operations, making
    it easier to test and extend.
    """

    def __init__(
        self,
        ftp_manager: "FtpManager",
        obstore_manager: "ObstoreManager",
    ) -> None:
        self.ftp_manager = ftp_manager
        self.obstore_manager = obstore_manager

    async def calc_new_files_for_multiple_nwp_init_hours(
        self,
        nwp_inits_to_transfer: Sequence[int] = (0, 6, 12, 18),
    ) -> list[TransferJob]:
        transfer_jobs: list[TransferJob] = []
        for init_hour in nwp_inits_to_transfer:
            transfer_jobs_for_init = await self.calc_new_files_for_single_nwp_init_hour(
                init_hour=init_hour
            )
            transfer_jobs.extend(transfer_jobs_for_init)
        return transfer_jobs

    async def calc_new_files_for_single_nwp_init_hour(
        self, init_hour: int
    ) -> list[TransferJob]:
        filtered_ftp_listing = await self.ftp_manager.list_and_filter_for_init_hour(
            init_hour=init_hour
        )
        ftp_transfer_jobs = [
            self.convert_ftp_path_to_transfer_job(ftp_path, ftp_info)
            for ftp_path, ftp_info in filtered_ftp_listing
        ]

        # Find the earliest NWP init datetime to use as the `offset` when listing objects on object storage.
        min_nwp_init_datetime = min(
            [transfer_job.nwp_init_datetime for transfer_job in ftp_transfer_jobs]
        )

        objects_already_downloaded = (
            self.obstore_manager.list_files_starting_at_nwp_init(min_nwp_init_datetime)
        )

        jobs_still_to_download = set_difference(
            ftp_transfer_jobs, objects_already_downloaded
        )

        log.info(
            "After filtering, we were planning to download %d files. Of those files, %d files already exist on object storage. We are now planning to download %d files.",
            len(ftp_transfer_jobs),
            len(ftp_transfer_jobs) - len(jobs_still_to_download),
            len(jobs_still_to_download),
        )

        return jobs_still_to_download

    def convert_ftp_path_to_transfer_job(
        self, ftp_path: PurePosixPath, ftp_info: UnixListInfo
    ) -> TransferJob:
        nwp_init_datetime = self.ftp_manager.extract_init_datetime_from_ftp_path(
            ftp_path
        )
        nwp_variable_name = self.ftp_manager.extract_nwp_variable_name_from_ftp_path(
            ftp_path
        )
        dst_obstore_path = self.obstore_manager.calc_obstore_path(
            ftp_path, nwp_init_datetime, nwp_variable_name
        )
        src_ftp_file_size_bytes = int(ftp_info["size"])
        return TransferJob(
            src_ftp_path=ftp_path,
            src_ftp_file_size_bytes=src_ftp_file_size_bytes,
            dst_obstore_path=dst_obstore_path,
            nwp_init_datetime=nwp_init_datetime,
        )


def set_difference(
    ftp_transfer_jobs: list[TransferJob], objects_already_downloaded: set[PathAndSize]
) -> list[TransferJob]:
    # Only download FTP files that aren't already on object storage:
    jobs_still_to_download: list[TransferJob] = []
    for transfer_job in ftp_transfer_jobs:
        candidate_to_transfer = PathAndSize(
            path=str(transfer_job.dst_obstore_path),
            file_size_bytes=transfer_job.src_ftp_file_size_bytes,
        )
        if candidate_to_transfer not in objects_already_downloaded:
            jobs_still_to_download.append(transfer_job)

    return jobs_still_to_download
