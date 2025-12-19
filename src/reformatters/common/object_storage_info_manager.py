from datetime import datetime
from pathlib import PurePosixPath

from obstore.store import ObjectStore

from reformatters.common.ftp_to_obstore_types import PathAndSize
from reformatters.common.logging import get_logger

log = get_logger(__name__)


class ObjectStorageInfoManager:
    def __init__(self, dst_obstore: ObjectStore, dst_root_path: PurePosixPath) -> None:
        self.dst_obstore = dst_obstore

        if dst_root_path.parts and dst_root_path.parts[0] == "/":
            # Strip the leading slash because self.dst_root_path must not start with a slash:
            self.dst_root_path = PurePosixPath(*dst_root_path.parts[1:])
        else:
            self.dst_root_path = dst_root_path

    def list_files_starting_at_nwp_init(self, nwp_init: datetime) -> set[PathAndSize]:
        nwp_init_datetime_str = nwp_init.strftime(
            self.format_string_for_nwp_init_datetime_in_obstore_path
        )

        obstore_listing = self.dst_obstore.list(
            prefix=str(self.dst_root_path),
            offset=str(self.dst_root_path / nwp_init_datetime_str),
        ).collect()

        log.info(
            "Found a total of %d files on object store for NWP init time %s UTC and for subsequent init times.",
            len(obstore_listing),
            nwp_init,
        )

        return {
            PathAndSize(path=item["path"], file_size_bytes=item["size"])
            for item in obstore_listing
        }

    def calc_obstore_path(
        self,
        ftp_path: PurePosixPath,
        nwp_init_datetime: datetime,
        nwp_variable_name: str,
    ) -> PurePosixPath:
        # Create dst_obstore_path:
        nwp_init_datetime_obstore_str = nwp_init_datetime.strftime(
            self.format_string_for_nwp_init_datetime_in_obstore_path
        )
        dst_obstore_path: PurePosixPath = (
            self.dst_root_path
            / nwp_init_datetime_obstore_str
            / nwp_variable_name
            / ftp_path.name
        )
        return dst_obstore_path

    @property
    def format_string_for_nwp_init_datetime_in_obstore_path(self) -> str:
        return "%Y-%m-%dT%HZ"
