from datetime import timedelta
from itertools import count
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
        "transport_error",
        "late_frontier",
    ],
)
def test_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:  # noqa: PLR0915
    cache = obstore.store.LocalStore(tmp_path / "cache", mkdir=True)
    requested: list[int] = []
    uploaded: list[str] = []
    polls = 0
    leads = (
        [5, 6, 7]
        if scenario == "frontier"
        else [5, 6]
        if scenario == "late_frontier"
        else [0, 1]
    )
    if scenario == "cached":
        obstore.put(cache, cache_key(coord(0)) + ".idx", b"cached")
    original_put = obstore.put

    def put(store: obstore.store.ObjectStore, key: str, data: object) -> object:
        uploaded.append(key)
        return original_put(store, key, data)  # ty: ignore[invalid-argument-type]

    def sleep(seconds: float) -> None:
        nonlocal polls
        polls += 1

    def download(url: str, retry_timeout: float) -> Path:
        lead = next(
            lead for lead in leads if coord(lead).get_url(source="nomads") == url
        )
        requested.append(lead)
        if (
            (scenario == "frontier" and lead in {5, 6})
            or (scenario == "late_frontier" and lead == 5 and polls < 6)
            or (scenario == "published" and polls == 0)
        ):
            raise httpx.HTTPStatusError(
                "404", request=httpx.Request("GET", url), response=httpx.Response(404)
            )
        if scenario == "transport_error" and len(requested) <= 4:
            raise httpx.TransportError("download failed")
        data = FIXTURE.read_bytes()
        if scenario == "truncated" or (scenario == "late_frontier" and lead == 6):
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
        + timedelta(seconds=119 if scenario == "deadline" else 180),
        lead_hours={"sfc": leads},
        poll_interval=timedelta(0),
        frontier_probe_every=2,
        frontier_width=2,
        download=download,
        build_index=build_index,
    )
    keys = [cache_key(coord(lead)) for lead in leads]
    if scenario == "deadline":
        assert result == ([], [])
        assert not requested
    elif scenario in {"truncated", "offsets"}:
        assert result == ([], keys)
        assert requested == [0, 0, 1, 0, 1, 1]
        assert not uploaded
    elif scenario == "fewer":
        assert result == ([keys[0]], [keys[1]])
    elif scenario == "frontier":
        assert result == ([keys[2]], keys[:2])
        assert requested == [5, 5, 6, 7]
    elif scenario == "late_frontier":
        assert result == ([keys[0]], [keys[1]])
        assert requested.count(6) == 3
        assert requested.count(5) == 7
    elif scenario == "transport_error":
        assert result == (keys, [])
        assert len(requested) == 6
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
    assert mirror.download_from_nomads("url", 42.0) == Path("download")
    download.assert_called_once_with(
        "url",
        "noaa-hrrr-nomads-cache",
        rate_limiter=mirror.nomads_rate_limiter,
        retry_status_codes=mirror.NOMADS_RETRY_STATUS_CODES,
        retry_timeout=42.0,
    )


def test_mirror_passes_its_remaining_time_as_the_download_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budgets: list[float] = []

    def download(url: str, retry_timeout: float) -> Path:
        budgets.append(retry_timeout)
        raise httpx.HTTPStatusError(
            "404", request=httpx.Request("GET", url), response=httpx.Response(404)
        )

    monkeypatch.setattr(mirror.time, "sleep", lambda _s: None)
    start = pd.Timestamp("2026-09-08T20:00Z")
    clock = count()
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(
            lambda cls, *a, **k: start + pd.Timedelta(seconds=30 * next(clock))
        ),
    )
    mirror.mirror_init_time(
        INIT,
        obstore.store.LocalStore(tmp_path, mkdir=True),
        deadline=start + timedelta(minutes=5),
        lead_hours={"sfc": [0]},
        poll_interval=timedelta(0),
        stop_margin=timedelta(minutes=2),
        download=download,
        max_invalid_attempts=1,
    )
    # Left of the deadline minus the stop margin: about three minutes, shrinking.
    assert budgets
    assert 150 <= budgets[0] <= 180
    assert budgets[-1] <= budgets[0]


def test_frontier_attempts_walk_past_a_run_of_unpublished_leads() -> None:
    pending = [5, 6, 7, 8, 9]
    # First frontier poll: the lowest plus the two after it.
    assert mirror.frontier_attempts(pending, look_ahead=2, width=2) == [5, 6, 7]
    # Later polls walk the window up, then wrap to the front.
    assert mirror.frontier_attempts(pending, look_ahead=4, width=2) == [5, 8, 9]
    assert mirror.frontier_attempts(pending, look_ahead=6, width=2) == [5, 6, 7]
    assert mirror.frontier_attempts([5], look_ahead=2, width=2) == [5]
    assert mirror.frontier_attempts([], look_ahead=2, width=2) == []


def test_mirror_never_grants_more_than_the_operational_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budgets: list[float] = []

    def download(url: str, retry_timeout: float) -> Path:
        budgets.append(retry_timeout)
        raise httpx.HTTPStatusError(
            "404", request=httpx.Request("GET", url), response=httpx.Response(404)
        )

    monkeypatch.setattr(mirror.time, "sleep", lambda _s: None)
    start = pd.Timestamp("2026-09-08T19:49Z")
    clock = count()
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(
            lambda cls, *a, **k: start + pd.Timedelta(minutes=10 * next(clock))
        ),
    )
    mirror.mirror_init_time(
        INIT,
        obstore.store.LocalStore(tmp_path, mkdir=True),
        deadline=start + timedelta(minutes=54),
        lead_hours={"sfc": [0]},
        poll_interval=timedelta(0),
        download=download,
    )
    assert budgets
    assert max(budgets) <= mirror.DOWNLOAD_RETRY_TIMEOUT_SECONDS
