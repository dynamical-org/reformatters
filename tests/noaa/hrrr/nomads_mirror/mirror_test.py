from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import httpx
import obstore
import obstore.store
import pandas as pd
import pytest

from reformatters.noaa.hrrr.nomads_cache import cache_key
from reformatters.noaa.hrrr.nomads_mirror import mirror
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord

FIXTURE = Path(__file__).parents[2] / "fixtures/hrrr.t19z.wrfsfcf00.first2.grib2"
INIT = pd.Timestamp("2026-09-08T19:00Z")


def coord(lead: int) -> NoaaHrrrSourceFileCoord:
    return NoaaHrrrSourceFileCoord(
        init_time=INIT,
        lead_time=pd.Timedelta(hours=lead),
        domain="conus",
        file_type="sfc",
        data_vars=[],
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "published",
        "cached",
        "truncated",
        "fewer",
        "frontier",
        "deadline",
        "offsets",
        "download_error",
    ],
)
def test_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:  # noqa: PLR0915
    cache = obstore.store.LocalStore(tmp_path / "cache", mkdir=True)
    requested: list[int] = []
    uploaded: list[str] = []
    polls = 0
    leads = [5, 6] if scenario == "frontier" else [0, 1]
    if scenario == "cached":
        obstore.put(cache, cache_key(coord(0)) + ".idx", b"cached")
    original_put = obstore.put

    def put(store: obstore.store.ObjectStore, key: str, data: object) -> object:
        uploaded.append(key)
        return original_put(store, key, data)  # ty: ignore[invalid-argument-type]

    def sleep(seconds: float) -> None:
        nonlocal polls
        polls += 1

    def download(url: str) -> Path:
        lead = next(
            lead for lead in leads if coord(lead).get_url(source="nomads") == url
        )
        requested.append(lead)
        if (scenario == "frontier" and lead == 5) or (
            scenario == "published" and polls == 0
        ):
            raise httpx.HTTPStatusError(
                "404", request=httpx.Request("GET", url), response=httpx.Response(404)
            )
        if scenario == "download_error":
            raise RuntimeError("download failed")
        data = FIXTURE.read_bytes()
        if scenario == "truncated":
            data = data[:-10]
        if scenario == "fewer" and lead == 1:
            data = data[
                : int(
                    FIXTURE.with_suffix(".grib2.idx")
                    .read_text()
                    .splitlines()[1]
                    .split(":")[1]
                )
            ]
        path = tmp_path / f"{lead}.grib2"
        path.write_bytes(data)
        return path

    def build_index(path: Path, index: Path) -> None:
        lines = FIXTURE.with_suffix(".grib2.idx").read_text().splitlines(keepends=True)
        if scenario == "fewer" and path.name == "1.grib2":
            lines = lines[:1]
        if scenario == "offsets":
            lines = lines[:1]
        index.write_text("".join(lines))

    monkeypatch.setattr(mirror.time, "sleep", sleep)
    monkeypatch.setattr(obstore, "put", put)
    result = mirror.mirror_init_time(
        INIT,
        cache,
        deadline=pd.Timestamp.now("UTC")
        + timedelta(seconds=-1 if scenario == "deadline" else 20),
        lead_hours={"sfc": leads},
        poll_interval=timedelta(0),
        frontier_probe_every=2,
        download=download,
        build_index=build_index,
    )
    keys = [cache_key(coord(lead)) for lead in leads]
    if scenario == "deadline":
        assert result == ([], [])
        assert not requested
    elif scenario in {"truncated", "offsets", "download_error"}:
        assert result == ([], keys)
        assert requested == [0, 0, 1, 0, 1, 1]
        assert not uploaded
    elif scenario == "fewer":
        assert result == ([keys[0]], [keys[1]])
    elif scenario == "frontier":
        assert result == ([keys[1]], [keys[0]])
        assert requested == [5, 5, 6]
    elif scenario == "cached":
        assert result == ([keys[1]], [])
        assert requested == [1]
    else:
        assert result == (keys, [])
    assert uploaded == [name for key in result.mirrored for name in (key, key + ".idx")]
    for key in result.mirrored:
        assert bytes(obstore.get(cache, key).bytes()) == FIXTURE.read_bytes()
    assert not list(tmp_path.glob("*.grib2*"))


def test_download_from_nomads(monkeypatch: pytest.MonkeyPatch) -> None:
    download = Mock(return_value=Path("download"))
    monkeypatch.setattr(mirror, "httpx_download_to_disk", download)
    assert mirror.download_from_nomads("url") == Path("download")
    download.assert_called_once_with(
        "url",
        "noaa-hrrr-nomads-cache",
        rate_limiter=mirror.nomads_rate_limiter,
        retry_status_codes=mirror.NOMADS_RETRY_STATUS_CODES,
    )
