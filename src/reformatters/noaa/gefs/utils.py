from collections.abc import Callable
from pathlib import Path

import pandas as pd
from obstore.exceptions import PermissionDeniedError

from reformatters.common.download import http_download_to_disk, httpx_download_to_disk
from reformatters.common.iterating import digest
from reformatters.noaa.gefs.gefs_config_models import GefsSourceFileCoord
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index

type _DownloadFn = Callable[..., Path]


def _download_file_from_gefs_source(
    dataset_id: str,
    coord: GefsSourceFileCoord,
    index_url: str,
    source_url: str,
    download: _DownloadFn,
) -> Path:
    idx_local_path = download(index_url, dataset_id)

    starts, ends = grib_message_byte_ranges_from_index(
        idx_local_path, coord.data_vars, coord.init_time, coord.lead_time
    )
    vars_suffix = digest(f"{s}-{e}" for s, e in zip(starts, ends, strict=True))
    return download(
        source_url,
        dataset_id,
        byte_ranges=(starts, ends),
        local_path_suffix=f"-{vars_suffix}",
    )


def gefs_download_file(
    dataset_id: str,
    coord: GefsSourceFileCoord,
) -> Path:
    """Download file from GEFS source with retry and fallback to alternative source."""
    try:
        return _download_file_from_gefs_source(
            dataset_id,
            coord,
            coord.get_index_url(),
            coord.get_url(),
            download=http_download_to_disk,
        )
    except FileNotFoundError:
        # if init time is within the last 4 days, try to download from the fallback source (NOMADS)
        if coord.init_time >= pd.Timestamp.now() - pd.Timedelta(days=4):
            try:
                return _download_file_from_gefs_source(
                    dataset_id,
                    coord,
                    coord.get_index_url(fallback=True),
                    coord.get_fallback_url(),
                    download=httpx_download_to_disk,
                )
            except PermissionDeniedError as e:
                raise FileNotFoundError(coord.get_url()) from e
        else:
            raise
