import httpx
import pytest

from reformatters.common import betterstack


def test_heartbeat_key_and_name() -> None:
    assert (
        betterstack.heartbeat_key("noaa-gfs-forecast", "update", "start")
        == "noaa-gfs-forecast_update_start"
    )
    assert (
        betterstack.heartbeat_name("noaa-gfs-forecast", "validate", "complete")
        == "reformatters noaa-gfs-forecast validate complete"
    )


def test_ping_fail_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(betterstack.httpx, "post", fake_post)

    betterstack.ping("https://hb.example/abc")
    betterstack.ping("https://hb.example/abc", failed=True)
    assert calls == ["https://hb.example/abc", "https://hb.example/abc/fail"]
