import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Final

import aioftp

from reformatters.common.ftp_manager import FtpManager
from reformatters.common.ftp_to_obstore_types import FtpPathAndInfo


class DwdFtpManager(FtpManager):
    """This class is designed to work with the style of DWD FTP ICON-EU path in
    use in 2025, such as:

    /weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025112600_004_ALB_RAD.grib2.bz2
    """

    @staticmethod
    def convert_nwp_init_hour_to_ftp_path(init_hour: int) -> PurePosixPath:
        return PurePosixPath(f"/weather/nwp/icon-eu/grib/{init_hour:02d}")

    async def list(
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
    def sanity_check_ftp_path(ftp_path: PurePosixPath) -> None:
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
    def extract_init_datetime_from_ftp_path(ftp_path: PurePosixPath) -> datetime:
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
    def extract_nwp_variable_name_from_ftp_path(ftp_path: PurePosixPath) -> str:
        return ftp_path.parts[6]
