import asyncio
from pathlib import PurePosixPath

import obstore

from reformatters.common.ftp_to_obstore import copy_files_from_ftp_to_obstore


def test_ftp_to_obstore() -> None:
    src_paths = [
        PurePosixPath(
            "/weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025111900_000_ALB_RAD.grib2.bz2"
        )
    ]
    dst_paths = ["/home/jack/data/test.grib2.bz2"]
    dst_store = obstore.store.LocalStore()

    asyncio.run(
        copy_files_from_ftp_to_obstore(
            "opendata.dwd.de", src_paths, dst_paths, dst_store
        )
    )
