import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final

import aioftp
from obstore.store import ObjectStore, S3Store

if TYPE_CHECKING:
    from obstore.store import S3Credential

from reformatters.common import kubernetes
from reformatters.common.ftp_transfer_calculator import (
    FtpPathAndInfo,
    FtpTransferCalculator,
)


class DwdFtpTransferCalculator(FtpTransferCalculator):
    """This class is designed to work with the style of DWD FTP ICON-EU path in
    use in 2025, such as:

    /weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025112600_004_ALB_RAD.grib2.bz2
    """

    @property
    def ftp_host(self) -> str:
        return "opendata.dwd.de"

    @property
    def _obstore_root_path(self) -> PurePosixPath:
        """*Without* the leading slash."""
        return PurePosixPath(
            "us-west-2.opendata.source.coop/dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/"
        )

    @property
    def object_store(self) -> ObjectStore:
        secret = kubernetes.load_secret("source-coop-storage-options-key")

        def get_credentials() -> "S3Credential":
            return {
                "access_key_id": secret["key"],
                "secret_access_key": secret["secret"],
                "expires_at": None,
            }

        # When running in prod, secret will {'key': 'xxx', 'secret': 'xxxx'}.
        # When not running in prod, secret will be empty.
        return S3Store(
            bucket="us-west-2.opendata.source.coop",
            region="us-west-2",
            credential_provider=get_credentials if secret else None,
        )

    @staticmethod
    def convert_nwp_init_hour_to_ftp_path(init_hour: int) -> PurePosixPath:
        return PurePosixPath(f"/weather/nwp/icon-eu/grib/{init_hour:02d}")

    async def list_ftp_files_for_single_nwp_init_path(
        self,
        ftp_path: PurePosixPath,
    ) -> list[FtpPathAndInfo]:
        """List all files on the FTP server for a single NWP init path.

        If `ftp_path` is a file, then result will be empty.

        This takes about 30 seconds for a single NWP init.
        """
        async with aioftp.Client.context(self.ftp_host) as ftp_client:
            ftp_listing = await ftp_client.list(ftp_path, recursive=True)
        return ftp_listing  # type: ignore[return-value]

    @staticmethod
    def _sanity_check_ftp_path(ftp_path: PurePosixPath) -> None:
        expected_number_of_parts: Final[int] = 8
        if len(ftp_path.parts) != expected_number_of_parts:
            raise ValueError(
                f"Expected the FTP path to have {expected_number_of_parts}, not {len(ftp_path.parts)}"
            )
        if ftp_path.parts[1:3] != ("weather", "nwp"):
            raise ValueError(
                f"Expected the start of the FTP path to be /weather/nwp/..., not {ftp_path}"
            )

    @staticmethod
    def _extract_init_datetime_from_ftp_path(ftp_path: PurePosixPath) -> datetime:
        # Extract the NWP init datetime string from the filename. For example, from this filename:
        #     "...lat-lon_single-level_2025112600_004_ALB_RAD.grib2.bz2"
        # Extract this:                ^^^^^^^^^^
        nwp_init_date_match = re.search(r"_(20\d{8})_", ftp_path.stem)
        if nwp_init_date_match:
            nwp_init_date_str = nwp_init_date_match.group(1)
        else:
            raise ValueError(f"Failed to match datetime string in {ftp_path.name=}")
        return datetime.strptime(nwp_init_date_str, "%Y%m%d%H")

    @staticmethod
    def _extract_nwp_variable_name_from_ftp_path(ftp_path: PurePosixPath) -> str:
        return ftp_path.parts[6]
