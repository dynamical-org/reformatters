import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from pathlib import PurePosixPath
from typing import NamedTuple

from aioftp.client import UnixListInfo
from obstore.store import ObjectStore
from pydantic import BaseModel

from reformatters.common.logging import get_logger

log = get_logger(__name__)


class TransferJob(BaseModel):
    src_ftp_path: PurePosixPath
    src_ftp_file_size_bytes: int
    dst_obstore_path: PurePosixPath
    nwp_init_datetime: datetime


class PathAndSize(NamedTuple):
    # We have to use NamedTuple because we want to use PathAndSize objects in a `set`.
    path: str  # No slash at the start of this string!
    file_size_bytes: int


type FtpPathAndInfo = tuple[PurePosixPath, UnixListInfo]


class FtpTransferCalculator(ABC):
    """Calculate which files to download from FTP to object storage.

    The aim is to be as general as possible, and to be robust to NWP providers slightly changing
    their naming conventions, or adding new files.

    The implementation works like this:

    1. Loop through each NWP initialisation hour, one by one (e.g. 00, 06, 12, 18).
    2. For each NWP initialisation hour:
    3. List all the files available on the FTP server.
    4. Filter this list using:
        - the logic in `_skip_ftp_item`.
        - the `filename_filter` regex optionally provided by the user.
    5. Find the earliest NWP init datetime from the filtered list of FTP files.
    6. List all files on object storage, starting at the earliest NWP init time from the FTP files.
        (Object storage listing is in lexicographic order. And the NWP init time is included in the
        object key, e.g. .../icon-eu/regular-lat-lon/2025-12-17T00Z/.... So we avoid listing the
        entire archive by starting the listing at a given offset based on the NWP init datetime.)
    7. Put this list into a `set[PathAndSize]`. We include the filesize so the code will
        automatically re-download incomplete downloads.
    8. Return the set difference: filtered_files_on_ftp - files_on_object_storage.
    """

    def __init__(self, filename_filter: str = "") -> None:
        """
        Args:
            filename_filter: An optional regex pattern to filter filenames by.
                For example, to only download single-level files, for forecast steps 0 to 5
                then use a regex pattern like "single-level_.*_00[0-5]_".
        """
        self.filename_filter = filename_filter

    async def calc_new_files_for_all_nwp_init_hours(self) -> list[TransferJob]:
        transfer_jobs: list[TransferJob] = []
        for init_hour in self.nwp_init_hours:
            transfer_jobs_for_init = await self.calc_new_files_for_single_nwp_init_hour(
                init_hour
            )
            transfer_jobs.extend(transfer_jobs_for_init)
        return transfer_jobs

    async def calc_new_files_for_single_nwp_init_hour(
        self, init_hour: int
    ) -> list[TransferJob]:
        if init_hour not in self.nwp_init_hours:
            raise ValueError(
                f"init_hour must be one of {self.nwp_init_hours}, not {init_hour}."
            )
        ftp_path = self.convert_nwp_init_hour_to_ftp_path(init_hour)
        ftp_host_and_path = f"ftp://{self.ftp_host}{ftp_path}"
        log.info("Listing files in %s ...", ftp_host_and_path)
        ftp_listing_for_nwp_init = await self.list_ftp_files_for_single_nwp_init_path(
            ftp_path
        )
        log.info(
            "Found %d items (prior to filtering) in %s",
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
            if not self._skip_ftp_item(ftp_path, ftp_info):
                self._sanity_check_ftp_path(ftp_path)
                transfer_job = self._convert_ftp_path_to_transfer_job(
                    ftp_path, ftp_info
                )
                filtered_ftp_listing.append(transfer_job)
        log.info(
            "Skipping %d FTP items after filtering with _skip_ftp_item. Now planning to download %d FTP files.",
            len(ftp_listing) - len(filtered_ftp_listing),
            len(filtered_ftp_listing),
        )

        if self.filename_filter:
            filtered_ftp_listing = self.filter_filenames_by_regex(filtered_ftp_listing)

        # Find the earliest NWP init datetime to use as the `offset` when listing objects on object storage.
        min_nwp_init_datetime = min(
            [transfer_job.nwp_init_datetime for transfer_job in filtered_ftp_listing]
        )

        set_of_objects_already_downloaded: set[PathAndSize] = (
            self._list_obstore_files_for_single_nwp_init(min_nwp_init_datetime)
        )
        log.info(
            "Found %d files on obstore for NWP init time %s.",
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
        log.info(
            "Filtering FTP filenames using user-supplied regex pattern %s...",
            self.filename_filter,
        )
        pattern = re.compile(self.filename_filter)
        filtered_ftp_transfer_jobs = [
            job for job in ftp_transfer_jobs if pattern.search(str(job.src_ftp_path))
        ]
        log.info(
            "Skipping %d FTP items after filtering with user-supplied regex pattern %s. Now planning to download %d FTP files.",
            len(ftp_transfer_jobs) - len(filtered_ftp_transfer_jobs),
            self.filename_filter,
            len(filtered_ftp_transfer_jobs),
        )
        return filtered_ftp_transfer_jobs

    def _list_obstore_files_for_single_nwp_init(
        self, nwp_init: datetime
    ) -> set[PathAndSize]:
        nwp_init_datetime_str = nwp_init.strftime(
            self._format_string_for_nwp_init_datetime_in_obstore_path
        )

        obstore_listing = self.object_store.list(
            prefix=str(self._obstore_root_path),
            offset=str(self._obstore_root_path / nwp_init_datetime_str),
        ).collect()

        return {
            PathAndSize(path=item["path"], file_size_bytes=item["size"])
            for item in obstore_listing
        }

    def _convert_ftp_path_to_transfer_job(
        self, ftp_path: PurePosixPath, ftp_info: UnixListInfo
    ) -> TransferJob:
        nwp_init_datetime = self._extract_init_datetime_from_ftp_path(ftp_path)
        dst_obstore_path = self._calc_obstore_path(ftp_path, nwp_init_datetime)
        src_ftp_file_size_bytes = int(ftp_info["size"])
        return TransferJob(
            src_ftp_path=ftp_path,
            src_ftp_file_size_bytes=src_ftp_file_size_bytes,
            dst_obstore_path=dst_obstore_path,
            nwp_init_datetime=nwp_init_datetime,
        )

    ################# Methods that can (optionally) be overridden ######################
    def _calc_obstore_path(
        self,
        ftp_path: PurePosixPath,
        nwp_init_datetime: datetime,
    ) -> PurePosixPath:
        # Create dst_obstore_path:
        nwp_init_datetime_obstore_str = nwp_init_datetime.strftime(
            self._format_string_for_nwp_init_datetime_in_obstore_path
        )
        nwp_variable_name = self._extract_nwp_variable_name_from_ftp_path(ftp_path)
        dst_obstore_path: PurePosixPath = (
            self._obstore_root_path
            / nwp_init_datetime_obstore_str
            / nwp_variable_name
            / ftp_path.name
        )
        return dst_obstore_path

    @property
    def nwp_init_hours(self) -> Sequence[int]:
        """Return a sequence of NWP init hours like [0, 6, 12, 18]."""
        return (0, 6, 12, 18)

    @staticmethod
    def _skip_ftp_item(ftp_path: PurePosixPath, ftp_info: UnixListInfo) -> bool:
        """Skip FTP items that we don't need."""
        return (
            ftp_info["type"] == "dir"
            or "pressure-level" in ftp_path.name
            or not ftp_path.name.endswith("grib2.bz2")
        )

    @property
    def _format_string_for_nwp_init_datetime_in_obstore_path(self) -> str:
        return "%Y-%m-%dT%HZ"

    ################ Methods that must be overridden ###################################

    @property
    @abstractmethod
    def ftp_host(self) -> str:
        """The FTP host, without "ftp://" prefix."""

    @property
    @abstractmethod
    def _obstore_root_path(self) -> PurePosixPath:
        """No slash at the start of the path!"""

    @property
    @abstractmethod
    def object_store(self) -> ObjectStore:
        pass

    @abstractmethod
    async def list_ftp_files_for_single_nwp_init_path(
        self,
        ftp_path: PurePosixPath,
    ) -> Sequence[FtpPathAndInfo]:
        """List all files available on the FTP server for a single NWP init
        identified by the `ftp_path`.

        `ftp_path` must include the leading slash, and must not
        include the FTP host URL. For example, `ftp_path` could be /weather/nwp/icon-eu/grib/00
        """

    @staticmethod
    @abstractmethod
    def convert_nwp_init_hour_to_ftp_path(init_hour: int) -> PurePosixPath:
        pass

    @staticmethod
    @abstractmethod
    def _sanity_check_ftp_path(ftp_path: PurePosixPath) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _extract_init_datetime_from_ftp_path(ftp_path: PurePosixPath) -> datetime:
        pass

    @staticmethod
    @abstractmethod
    def _extract_nwp_variable_name_from_ftp_path(ftp_path: PurePosixPath) -> str:
        pass
