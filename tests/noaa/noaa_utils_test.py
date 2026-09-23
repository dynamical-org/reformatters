from pathlib import Path
from unittest.mock import Mock

import pytest

from reformatters.noaa import noaa_utils
from reformatters.noaa.noaa_utils import (
    NOMADS_RETRY_STATUS_CODES,
    nomads_download_to_disk,
    nomads_rate_limiter,
)


def test_nomads_download_to_disk_uses_the_nomads_limiter_and_retry_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = Mock(return_value=Path("downloaded.grib2"))
    monkeypatch.setattr(noaa_utils, "httpx_download_to_disk", download)

    result = nomads_download_to_disk(
        "https://nomads.ncep.noaa.gov/a.grib2",
        "dataset",
        byte_ranges=([0], [10]),
        local_path_suffix="-s",
        disk_cache=True,
    )

    assert result == Path("downloaded.grib2")
    download.assert_called_once_with(
        "https://nomads.ncep.noaa.gov/a.grib2",
        "dataset",
        byte_ranges=([0], [10]),
        local_path_suffix="-s",
        disk_cache=True,
        rate_limiter=nomads_rate_limiter,
        retry_status_codes=NOMADS_RETRY_STATUS_CODES,
    )
