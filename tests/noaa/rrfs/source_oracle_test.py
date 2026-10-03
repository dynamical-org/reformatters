import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import rasterio

from reformatters.common.logging import get_logger
from reformatters.noaa.rrfs.region_job import NoaaRrfsSourceFileCoord

log = get_logger(__name__)
SNAPSHOTS = json.loads(
    (Path(__file__).parent / "fixtures/value_snapshots.json").read_text()
)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("cycle", "member", "minutes"),
    [
        (0, None, 15),
        (0, None, 75),
        (1, None, 75),
        (12, 1, 120),
        (12, 5, 120),
        (18, 1, 120),
        (18, 5, 120),
    ],
)
def test_quarter_hour_and_member_snapshots_match_independent_gdal(
    cycle: int, member: int | None, minutes: int
) -> None:
    init = pd.Timestamp("2026-09-15") + pd.Timedelta(hours=cycle)
    coord = NoaaRrfsSourceFileCoord(
        init_time=init,
        lead_time=pd.Timedelta(minutes=minutes).ceil("1h"),
        source_family="subh" if member is None else "2dfld",
        ensemble_member=member,
        data_vars=[],
    )
    url = coord.get_url().replace(
        "s3://noaa-rrfs-ops-pds/", "https://noaa-rrfs-ops-pds.s3.amazonaws.com/"
    )
    window = f"{minutes} min fcst" if minutes % 60 else f"{minutes // 60} hour fcst"
    with httpx.Client(timeout=60) as client:
        response = client.get(url + ".idx")
        response.raise_for_status()
        matches = [
            line.split(":")
            for line in response.text.splitlines()
            if line.split(":")[3:6] == ["TMP", "2 m above ground", window]
        ]
        assert len(matches) == 1
        offset = int(matches[0][1])
        response = client.get(url, headers={"Range": f"bytes={offset}-{offset + 15}"})
        response.raise_for_status()
        assert response.status_code == 206
        length = int.from_bytes(response.content[8:16], "big")
        response = client.get(
            url, headers={"Range": f"bytes={offset}-{offset + length - 1}"}
        )
        response.raise_for_status()
        assert len(response.content) == length
    with rasterio.MemoryFile(response.content) as file, file.open() as source:
        actual = float(source.read(1, window=((635, 636), (1062, 1063)))[0, 0])
    dataset_id = (
        "noaa-rrfs-forecast-sub-hourly-virtual"
        if member is None
        else "noaa-refs-forecast-virtual"
    )
    index = minutes // 15 - 1 if member is None else member
    expected = SNAPSHOTS[dataset_id][init.isoformat()]["temperature_2m"][index]
    log.info(
        f"RAW GDAL ORACLE {url} offset={offset} bytes={length} value={actual!r} virtual={expected!r}"
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
