import re
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from aioftp.client import UnixListInfo

from reformatters.common.ftp_to_obstore_types import (
    FtpPathAndInfo,
    PathAndSize,
    TransferJob,
)
from reformatters.common.logging import get_logger

if TYPE_CHECKING:
    from reformatters.common.ftp_info_extractor import FtpInfoExtractor
    from reformatters.common.object_storage_info_manager import ObjectStorageInfoManager

log = get_logger(__name__)


class FtpTransferCoordinator:
    """Coordinate FTP and object storage interactions to calculate transfer
    jobs.

    This class uses injected FtpInfoExtractor and
    ObjectStorageInfoManager instances to perform its operations, making
    it easier to test and extend.
    """

    def __init__(
        self,
        ftp_info_extractor: "FtpInfoExtractor",
        obstore_info_manager: "ObjectStorageInfoManager",
        filename_filter: str = "",
    ) -> None:
        self.ftp_info_extractor = ftp_info_extractor
        self.obstore_info_manager = obstore_info_manager
        self.filename_filter = filename_filter

    async def calc_new_files_for_multiple_nwp_init_hours(
        self,
        nwp_inits_to_transfer: Sequence[int] = (0, 6, 12, 18),
    ) -> list[TransferJob]:
        transfer_jobs: list[TransferJob] = []
        for init_hour in nwp_inits_to_transfer:
            transfer_jobs_for_init = await self.calc_new_files_for_single_nwp_init_hour(
                init_hour
            )
            transfer_jobs.extend(transfer_jobs_for_init)
        return transfer_jobs

    async def calc_new_files_for_single_nwp_init_hour(
        self, init_hour: int
    ) -> list[TransferJob]:
        ftp_path = self.ftp_info_extractor.convert_nwp_init_hour_to_ftp_path(init_hour)
        ftp_host_and_path = f"ftp://{self.ftp_info_extractor.ftp_host}{ftp_path}"
        log.info("Recursively listing files below FTP path: %s ...", ftp_host_and_path)
        ftp_listing_for_nwp_init = await self.ftp_info_extractor.list(ftp_path)
        log.info(
            "Found %d items (prior to filtering) below FTP path: %s",
            len(ftp_listing_for_nwp_init),
            ftp_host_and_path,
        )
        return await self.calc_new_files_from_ftp_listing(ftp_listing_for_nwp_init)

    async def calc_new_files_from_ftp_listing(
        self, ftp_listing: Sequence[FtpPathAndInfo]
    ) -> list[TransferJob]:
        # Filter ftp_listing using skip_ftp_item and convert to TransferJobs.
        filtered_ftp_listing: list[TransferJob] = []
        for ftp_path, ftp_info in ftp_listing:
            if not self.ftp_info_extractor.skip_ftp_item(ftp_path, ftp_info):
                self.ftp_info_extractor.sanity_check_ftp_path(ftp_path)
                transfer_job = self._convert_ftp_path_to_transfer_job(
                    ftp_path, ftp_info
                )
                filtered_ftp_listing.append(transfer_job)
        log.info(
            "Filtering with skip_ftp_item reduced the number of files we plan to download from %d down to %d (a reduction of %d files).",
            len(ftp_listing),
            len(filtered_ftp_listing),
            len(ftp_listing) - len(filtered_ftp_listing),
        )

        if self.filename_filter:
            filtered_ftp_listing = self.filter_filenames_by_regex(filtered_ftp_listing)

        # Find the earliest NWP init datetime to use as the `offset` when listing objects on object storage.
        min_nwp_init_datetime = min(
            [transfer_job.nwp_init_datetime for transfer_job in filtered_ftp_listing]
        )

        set_of_objects_already_downloaded: set[PathAndSize] = (
            self.obstore_info_manager.list_obstore_files_for_single_nwp_init(
                min_nwp_init_datetime
            )
        )
        log.info(
            "Found a total of %d files on object store for NWP init time %s UTC and for subsequent init times.",
            len(set_of_objects_already_downloaded),
            min_nwp_init_datetime,
        )

        # Only download FTP files that aren't already on object storage:
        jobs_still_to_download: list[TransferJob] = []
        for transfer_job in filtered_ftp_listing:
            candidate_to_transfer = PathAndSize(
                path=str(transfer_job.dst_obstore_path),
                file_size_bytes=transfer_job.src_ftp_file_size_bytes,
            )
            if candidate_to_transfer not in set_of_objects_already_downloaded:
                jobs_still_to_download.append(transfer_job)

        log.info(
            "After filtering, we were planning to download %d files. Of those files, %d files already exist on object storage. We are now planning to download %d files.",
            len(filtered_ftp_listing),
            len(filtered_ftp_listing) - len(jobs_still_to_download),
            len(jobs_still_to_download),
        )

        return jobs_still_to_download

    def filter_filenames_by_regex(
        self, ftp_transfer_jobs: list[TransferJob]
    ) -> list[TransferJob]:
        pattern = re.compile(self.filename_filter)
        filtered_ftp_transfer_jobs = [
            job for job in ftp_transfer_jobs if pattern.search(str(job.src_ftp_path))
        ]
        log.info(
            "Filtering with user-supplied regex '%s' reduced the number of files we plan to download from %d down to %d (a reduction of %d files).",
            self.filename_filter,
            len(ftp_transfer_jobs),
            len(filtered_ftp_transfer_jobs),
            len(ftp_transfer_jobs) - len(filtered_ftp_transfer_jobs),
        )
        return filtered_ftp_transfer_jobs

    def _convert_ftp_path_to_transfer_job(
        self, ftp_path: PurePosixPath, ftp_info: UnixListInfo
    ) -> TransferJob:
        nwp_init_datetime = self.ftp_info_extractor.extract_init_datetime_from_ftp_path(
            ftp_path
        )
        nwp_variable_name = (
            self.ftp_info_extractor.extract_nwp_variable_name_from_ftp_path(ftp_path)
        )
        dst_obstore_path = self.obstore_info_manager.calc_obstore_path(
            ftp_path, nwp_init_datetime, nwp_variable_name
        )
        src_ftp_file_size_bytes = int(ftp_info["size"])
        return TransferJob(
            src_ftp_path=ftp_path,
            src_ftp_file_size_bytes=src_ftp_file_size_bytes,
            dst_obstore_path=dst_obstore_path,
            nwp_init_datetime=nwp_init_datetime,
        )
