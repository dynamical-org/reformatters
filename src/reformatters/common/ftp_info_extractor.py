from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from pathlib import PurePosixPath

from aioftp.client import UnixListInfo

from reformatters.common.ftp_to_obstore_types import FtpPathAndInfo
from reformatters.common.logging import get_logger

log = get_logger(__name__)


class FtpInfoExtractor(ABC):
    """Abstract base class for extracting information from FTP servers and
    paths.

    The aim is to be as general as possible, and to be robust to NWP
    providers slightly changing their naming conventions, or adding new
    files.
    """

    def __init__(self, ftp_host: str) -> None:
        self.ftp_host = ftp_host

    @abstractmethod
    async def list(
        self,
        ftp_path: PurePosixPath,
    ) -> Sequence[FtpPathAndInfo]:
        """List all files available on the FTP server for a single NWP init
        identified by the `ftp_path`.

        `ftp_path` must start with the leading slash, and must not
        include the FTP host URL. For example, `ftp_path` could be /weather/nwp/icon-eu/grib/00
        """

    @staticmethod
    @abstractmethod
    def convert_nwp_init_hour_to_ftp_path(init_hour: int) -> PurePosixPath:
        pass

    @staticmethod
    @abstractmethod
    def sanity_check_ftp_path(ftp_path: PurePosixPath) -> None:
        pass

    @staticmethod
    @abstractmethod
    def extract_init_datetime_from_ftp_path(ftp_path: PurePosixPath) -> datetime:
        pass

    @staticmethod
    @abstractmethod
    def extract_nwp_variable_name_from_ftp_path(ftp_path: PurePosixPath) -> str:
        pass

    @staticmethod
    def skip_ftp_item(ftp_path: PurePosixPath, ftp_info: UnixListInfo) -> bool:
        """Skip FTP items that we don't need."""
        return (
            ftp_info["type"] == "dir"
            or "pressure-level" in ftp_path.name
            or not ftp_path.name.endswith("grib2.bz2")
        )
