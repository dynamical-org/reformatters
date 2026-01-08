from datetime import datetime
from pathlib import PurePosixPath
from typing import NamedTuple

from aioftp.client import UnixListInfo
from pydantic import BaseModel


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
