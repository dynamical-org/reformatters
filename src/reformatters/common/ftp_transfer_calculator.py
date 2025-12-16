from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from pathlib import PurePosixPath

from aioftp.client import UnixListInfo
from obstore.store import ObjectStore
from pydantic import BaseModel


class TransferJob(BaseModel):
    src_ftp_path: PurePosixPath
    src_ftp_file_size_bytes: int

    # The destination path in the object store.
    # Starts at the datetime part, e.g., '2025-12-12T00Z/alb_rad/...'.
    dst_obstore_path: PurePosixPath

    # The NWP initialization datetime, extracted from `src_ftp_path`.
    nwp_init_datetime: datetime


class PathAndSize(BaseModel):
    path: str
    file_size_bytes: int


type PathAndInfo = tuple[PurePosixPath, UnixListInfo]


class FtpTransferCalculator(ABC):
    async def transfer_new_files_for_all_nwp_inits(self) -> None:
        ftp_paths_for_nwp_inits: Sequence[
            PurePosixPath
        ] = await self._list_ftp_base_paths_for_all_required_nwp_inits()

        for ftp_path in ftp_paths_for_nwp_inits:
            ftp_listing_for_nwp_init = await self._list_ftp_files_for_single_nwp_init(
                ftp_path
            )
            await self.transfer_new_files_for_single_nwp_init(ftp_listing_for_nwp_init)

    async def transfer_new_files_for_single_nwp_init(
        self, ftp_listing_for_nwp_init: Sequence[PathAndInfo]
    ) -> None:
        # Collect list of TransferJobs from FTP server's listing:
        ftp_transfer_jobs: list[TransferJob] = []
        for ftp_path, ftp_info in ftp_listing_for_nwp_init:
            if not self._skip_ftp_item(ftp_path, ftp_info):
                self._sanity_check_ftp_path(ftp_path)
                transfer_job = self._convert_ftp_path_to_transfer_job(
                    ftp_path, ftp_info
                )
                ftp_transfer_jobs.append(transfer_job)

        # Find the earliest NWP init datetime.
        # We'll use this as the `offset` when listing objects on object storage.
        min_nwp_init_datetime = min(
            [transfer_job.nwp_init_datetime for transfer_job in ftp_transfer_jobs]
        )

        obstore_listing_set = self._list_obstore_files_for_single_nwp_init(
            min_nwp_init_datetime
        )

        jobs_still_to_download: list[TransferJob] = []

        for transfer_job in ftp_transfer_jobs:
            obstore_path = str(self._obstore_root_path / transfer_job.dst_obstore_path)
            obstore_path_and_size = PathAndSize(
                path=obstore_path, file_size_bytes=transfer_job.src_ftp_file_size_bytes
            )
            if obstore_path_and_size not in obstore_listing_set:
                jobs_still_to_download.append(transfer_job)

        # TODO(Jack): Send `jobs_still_to_download` to FTP download client

    def _list_obstore_files_for_single_nwp_init(
        self, nwp_init: datetime
    ) -> set[PathAndSize]:
        nwp_init_datetime_str = nwp_init.strftime(
            self._format_string_for_nwp_init_datetime_in_obstore_path
        )

        obstore_listing = self._object_store.list(
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
        """Converts the FTP path to a `TransferJob`.

        This function is designed to work with the style of DWD FTP ICON-EU path in use in 2025, such as:
        /weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025112600_004_ALB_RAD.grib2.bz2
        """
        nwp_init_datetime: datetime = self._extract_init_datetime_from_ftp_path(
            ftp_path
        )
        nwp_init_datetime_obstore_str = nwp_init_datetime.strftime(
            self._format_string_for_nwp_init_datetime_in_obstore_path
        )

        # Create dst_obstore_path:
        nwp_variable_name = self._extract_nwp_variable_name_from_ftp_path(ftp_path)
        dst_obstore_path = (
            PurePosixPath(nwp_init_datetime_obstore_str)
            / nwp_variable_name
            / ftp_path.name
        )

        file_size_bytes = int(ftp_info["size"])

        return TransferJob(
            src_ftp_path=ftp_path,
            src_ftp_file_size_bytes=file_size_bytes,
            dst_obstore_path=dst_obstore_path,
            nwp_init_datetime=nwp_init_datetime,
        )

    ################# Methods that can (optionally) be overridden ######################

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
    def _obstore_root_path(self) -> PurePosixPath:
        pass

    @property
    @abstractmethod
    def _object_store(self) -> ObjectStore:
        pass

    @staticmethod
    @abstractmethod
    async def _list_ftp_files_for_single_nwp_init(
        ftp_path: PurePosixPath,
    ) -> Sequence[PathAndInfo]:
        """List all files available on the FTP server for a single NWP init
        identified by the `ftp_path`."""

    @staticmethod
    @abstractmethod
    async def _list_ftp_base_paths_for_all_required_nwp_inits() -> Sequence[
        PurePosixPath
    ]:
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
